from __future__ import annotations
import csv
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.vae.factory import build_vae
from src.data.load_brain import buildMRIDataset
from src.vae.VAE import VAELossMRI

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None
    )

    return parser.parse_args()



def build_loaders(config):

    train_dataset = buildMRIDataset(config, split="train")

    val_dataset = buildMRIDataset(config, split="val")

    tconfig = config["training"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(tconfig.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(tconfig.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(
            int(tconfig.get("num_workers", 4)) > 0
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(tconfig.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(tconfig.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(
            int(tconfig.get("num_workers", 4)) > 0
        )
    )

    return train_loader, val_loader


def move_batch(batch, device):
    x = batch["image"].to(device, non_blocking=True)

    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)

    return x


def train_epoch(
    vae,
    loader,
    optimizer,
    loss_fn,
    device,
    scaler,
    use_amp,
    amp_dtype
):

    vae.train()

    totals = {
        "loss": torch.zeros((), device=device),
        "reconstruction": torch.zeros((), device=device),
        "kl": torch.zeros((), device=device),
        "perceptual": torch.zeros((), device=device)
    }

    num_samples = 0

    progress = tqdm(loader, desc="Train", leave=False)

    for step, batch in enumerate(progress):

        x = move_batch(batch, device)

        batch_size = x.shape[0]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp
        ):
            reconstruction, posterior = vae(x, sample=True)

            losses = loss_fn(
                target=x,
                reconstruction=reconstruction,
                posterior=posterior
            )

        if scaler.is_enabled():
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            optimizer.step()


        for key in totals:
            totals[key] += losses[key].detach() * batch_size
        
        num_samples += batch_size

        if step % 20 == 0:
            progress.set_postfix(
                loss=f"{losses['loss'].item():.4f}",
                recon=f"{losses['reconstruction'].item():.4f}",
                kl=f"{losses['kl'].item():.4f}"
            )

    return {
        key: value.item() / num_samples
        for key, value in totals.items()
    }


@torch.no_grad()
def validate(
    vae,
    loader,
    loss_fn,
    device,
    use_amp,
    amp_dtype
):

    vae.eval()

    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
        "perceptual": 0.0
    }

    num_samples = 0

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for batch in progress:

        x = move_batch(
            batch,
            device,
        )

        batch_size = x.shape[0]

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp
        ):
            reconstruction, posterior = vae(
                x,
                sample=False,
            )

            losses = loss_fn(
                target=x,
                reconstruction=reconstruction,
                posterior=posterior
            )

        for key in totals:
            totals[key] += (
                losses[key].detach().item()
                * batch_size
            )

        num_samples += batch_size

    return {
        key: value / num_samples
        for key, value in totals.items()
    }


@torch.no_grad()
def save_reconstructions(
    vae,
    loader,
    device,
    output_path,
    max_images=8
):

    vae.eval()

    batch = next(iter(loader))

    x = move_batch(batch,device)

    x = x[:max_images]

    reconstruction, _ = vae(x, sample=False)

    x = x.float().cpu()
    reconstruction = reconstruction.float().cpu()

    combined = torch.cat([x, reconstruction], dim=0)

    combined = (combined.clamp(-1, 1) + 1) / 2

    save_image(combined, output_path, nrow=max_images)


def save_checkpoint(
    path,
    vae,
    optimizer,
    epoch,
    best_val_loss,
    config
):
    model_to_save = vae._orig_mod if hasattr(vae, "_orig_mod") else vae

    state = {
        "vae": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": config
    }

    torch.save(state, path)


def main():
    args = parse_args()

    config = load_config(
        args.config
    )

    seed = int(config.get("seed", 42))

    set_seed(seed)

    device = get_device()
    tconfig = config["training"]

    best_val_loss = float("inf")
    start_epoch = 1

    # Speedup Add on
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print(f"Using device: {device}")

    vae = build_vae(config).to(device)

    optimizer = AdamW(
        vae.parameters(),
        lr=float(tconfig.get("learning_rate", 1e-4)),
        weight_decay=float(tconfig.get("weight_decay", 1e-4)),
        fused=False
    )

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device
        )
        vae.load_state_dict(checkpoint["vae"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        print(
            f"Resumed from epoch {checkpoint['epoch']} "
            f"with best val loss {best_val_loss:.6f}"
        )

    if device.type == "cuda":
        vae = vae.to(memory_format=torch.channels_last)
        vae = torch.compile(vae, mode="reduce-overhead")

    train_loader, val_loader = build_loaders(config)

    lconfig = config.get("vae_loss", {})

    loss_fn = VAELossMRI(
        reconstruction_weight=float(lconfig.get("reconstruction_weight", 1.0)),
        kl_weight=float(lconfig.get("kl_weight", 1e-6)),
        perceptual_weight=float(lconfig.get("perceptual_weight", 0.0))
    ).to(device)

    num_epochs = int(tconfig.get("epochs",100))

    use_amp = bool(tconfig.get("mixed_precision", True)) and device.type == "cuda"

    # Speedup Add on
    amp_dtype_name = tconfig.get("amp_dtype", "bf16")
    if amp_dtype_name == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_dtype_name == "fp16":
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unknown amp_dtype: {amp_dtype_name}")

    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    output_dir = Path(
        config.get("output_dir", "outputs/vae")
    )
    history_path = output_dir / "training_history.csv"

    checkpoint_dir = output_dir/ "checkpoints"

    reconstruction_dir = output_dir/ "reconstructions"

    checkpoint_dir.mkdir(parents=True,exist_ok=True)

    reconstruction_dir.mkdir(parents=True,exist_ok=True)

    

    print(f"Train samples: " f"{len(train_loader.dataset)}")
    print(f"Validation samples: "f"{len(val_loader.dataset)}")
    print("VAE parameters:",sum(p.numel() for p in vae.parameters()))

    

    for epoch in range(start_epoch, num_epochs + 1):

        print(f"\nEpoch " f"{epoch}/{num_epochs}")

        train_metrics = train_epoch(
            vae=vae,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype
        )

        val_metrics = validate(
            vae=vae,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype
        )

        print(
            "Train | "
            f"loss={train_metrics['loss']:.6f} "
            f"recon={train_metrics['reconstruction']:.6f} "
            f"kl={train_metrics['kl']:.6f} "
            f"perc={train_metrics['perceptual']:.6f}"
        )

        print(
            "Val   | "
            f"loss={val_metrics['loss']:.6f} "
            f"recon={val_metrics['reconstruction']:.6f} "
            f"kl={val_metrics['kl']:.6f} "
            f"perc={val_metrics['perceptual']:.6f}"
        )

        write_header = not history_path.exists()

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "train_reconstruction",
                    "train_kl",
                    "val_loss",
                    "val_reconstruction",
                    "val_kl"
                ])

            writer.writerow([
                epoch,
                train_metrics["loss"],
                train_metrics["reconstruction"],
                train_metrics["kl"],
                val_metrics["loss"],
                val_metrics["reconstruction"],
                val_metrics["kl"]
            ])

        save_reconstructions(
            vae=vae,
            loader=val_loader,
            device=device,
            output_path=(
                reconstruction_dir
                / f"epoch_{epoch:04d}.png"
            )
        )
        is_best = val_metrics["loss"] < best_val_loss

        if is_best:
            best_val_loss = val_metrics["loss"]

        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            vae=vae,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            config=config
        )

        if is_best:

            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                vae=vae,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config
            )

            print("Saved new best checkpoint")  

if __name__ == "__main__":
    main()