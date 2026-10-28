from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.load_masks import load_mask_path_splits, TumourMaskSliceDataset
from src.utils.config import load_config
from src.vae.factory import build_vae
from src.vae.VAE import VAELossSegmentation
from src.utils.seed import set_seed


# ----------------------- Datasets and loaders -----------------------

def build_loaders(config, overfit_samples=0):
    dconfig = config["data"]
    tconfig = config["training"]

    paths = load_mask_path_splits(
        split_path=dconfig["split_file"],
        manifest_path=dconfig["mask_manifest"]
    )

    dataset_options = {
        "image_size": int(dconfig.get("image_size", 256)),
        "slice_axis": int(dconfig.get("slice_axis", 2)),
        "tumour_only": bool(dconfig.get("tumour_only", True))
    }

    train_dataset = TumourMaskSliceDataset(
        mask_paths=paths["train"],
        **dataset_options
    )

    if overfit_samples > 0:
        generator = torch.Generator().manual_seed(
            int(config.get("seed", 42))
        )

        indices = torch.randperm(
            len(train_dataset),
            generator=generator,
        )[:overfit_samples].tolist()

        train_dataset = Subset(train_dataset, indices)
        val_dataset = train_dataset

        print(
            "OVERFIT MODE: training and validation use the "
            "same subset. KL is disabled."
        )

    else:
        val_dataset = TumourMaskSliceDataset(
            mask_paths=paths["val"],
            **dataset_options
        )

    workers = int(tconfig.get("num_workers", 4))

    loader_options = {
        "batch_size": int(tconfig.get("batch_size", 8)),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "drop_last": False
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        **loader_options
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_options
    )

    return train_loader, val_loader


def move_batch(batch, device):
    inputs = batch["image"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)

    return inputs, target


# ----------------------- Hard segmentation metrics -----------------------

class SegmentationMetrics:
    """
    Per-class Dice averaged over target-present slices.

    For target-absent slices, separately report the fraction
    containing any false-positive pixels.
    """

    class_names = {
        1: "label1",
        2: "edema",
        3: "enhancing"
    }

    def __init__(self):
        self.stats = {
            name: {
                "dice_sum": 0.0,
                "present": 0,
                "absent": 0,
                "false_positive": 0
            }
            for name in self.class_names.values()
        }

    def update(self, prediction, target):
        for class_idx, name in self.class_names.items():
            predicted = prediction == class_idx
            actual = target == class_idx

            predicted_area = predicted.sum(dim=(1, 2))
            target_area = actual.sum(dim=(1, 2))

            intersection = (
                predicted & actual
            ).sum(dim=(1, 2))

            present = target_area > 0

            dice = (2.0 * intersection[present] / (predicted_area[present] + target_area[present]))

            stats = self.stats[name]

            stats["dice_sum"] += dice.sum().item()
            stats["present"] += present.sum().item()
            stats["absent"] += (~present).sum().item()

            stats["false_positive"] += ((~present) & (predicted_area > 0)).sum().item()

    def compute(self):
        results = {}

        for name, stats in self.stats.items():
            results[name] = {
                "dice": (
                    stats["dice_sum"] / stats["present"]
                    if stats["present"] else None
                ),
                "absent_fp_fraction": (
                    stats["false_positive"] / stats["absent"]
                    if stats["absent"] else None
                ),
                "present_slices": stats["present"],
                "absent_slices": stats["absent"]
            }

        return results


# ----------------------- Training -----------------------

def train_epoch(
    vae,
    loader,
    optimizer,
    loss_fn,
    device,
    scaler,
    amp_dtype,
    use_amp,
    beta
):
    vae.train()

    totals = {}
    num_samples = 0

    progress = tqdm(loader, desc="Train", leave=False)

    for batch in progress:
        inputs, target = move_batch(batch, device)
        batch_size = inputs.shape[0]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp
        ):
            logits, posterior = vae(inputs, sample=True)

            losses = loss_fn(
                logits=logits,
                target=target,
                posterior=posterior,
                kl_weight=beta
            )

        if not torch.isfinite(losses["loss"]).item():
            raise FloatingPointError("Non-finite training loss")

        scaler.scale(losses["loss"]).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        for name, value in losses.items():
            totals[name] = (totals.get(name, 0.0) + value.detach().item() * batch_size)

        num_samples += batch_size

        progress.set_postfix(
            loss=f"{losses['loss'].item():.4f}",
            ce=f"{losses['ce'].item():.4f}",
            dice=f"{losses['dice_loss'].item():.4f}"
        )

    return {
        name: value / num_samples
        for name, value in totals.items()
    }


# ----------------------- Validation -----------------------

@torch.no_grad()
def validate(
    vae,
    loader,
    loss_fn,
    device,
    amp_dtype,
    use_amp,
    beta,
    seed
):
    vae.eval()

    totals = {}
    num_samples = 0

    mean_metrics = SegmentationMetrics()
    sampled_metrics = SegmentationMetrics()

    # Restore the training RNG state after validation.
    cuda_devices = (
        list(range(torch.cuda.device_count()))
        if device.type == "cuda" else []
    )

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)

        for batch in tqdm(
            loader,
            desc="Validation",
            leave=False
        ):
            inputs, target = move_batch(batch, device)
            batch_size = inputs.shape[0]

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp
            ):
                posterior = vae.encode(inputs)

                mean_logits = vae.decode(posterior.mode())

                losses = loss_fn(
                    logits=mean_logits,
                    target=target,
                    posterior=posterior,
                    kl_weight=beta
                )

                sampled_logits = vae.decode(
                    posterior.sample()
                )

            if not torch.isfinite(losses["loss"]).item():
                raise FloatingPointError(
                    "Non-finite validation loss"
                )

            mean_metrics.update(mean_logits.argmax(dim=1), target)

            sampled_metrics.update(sampled_logits.argmax(dim=1), target)

            for name, value in losses.items():
                totals[name] = (totals.get(name, 0.0) + value.item() * batch_size)

            num_samples += batch_size

    results = {
        name: value / num_samples
        for name, value in totals.items()
    }

    results["mean_metrics"] = mean_metrics.compute()
    results["sampled_metrics"] = sampled_metrics.compute()

    return results


# ----------------------- Fixed reconstruction examples -----------------------

def select_plot_examples(dataset, num_examples):
    num_examples = min(num_examples, len(dataset))

    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=num_examples,
        dtype=int
    )

    examples = [dataset[int(index)] for index in indices]

    return {
        "image": torch.stack([
            example["image"] for example in examples
        ]),
        "target": torch.stack([
            example["target"] for example in examples
        ]),
        "titles": [
            (
                Path(example["mask_path"]).name.removesuffix(
                    "_segm_preprocessed.nii.gz"
                )
                + f"\nz={example['slice_idx']}"
            )
            for example in examples
        ]
    }


@torch.no_grad()
def save_reconstructions(vae, examples, device, output_path):
    vae.eval()

    inputs = examples["image"].to(device)

    logits, _ = vae(inputs, sample=False)

    original = examples["target"].cpu().numpy()
    reconstructed = logits.argmax(dim=1).cpu().numpy()

    colors = [
        "#000000",  # Background
        "#E6AA1E",  # Label 1
        "#3CB45A",  # Oedema
        "#DC3C50"  # Enhancing tumour
    ]

    labels = [
        "Background",
        "Label 1",
        "Oedema",
        "Enhancing"
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5],
        cmap.N
    )

    count = len(original)

    fig, axes = plt.subplots(
        2,
        count,
        figsize=(3 * count, 6),
        squeeze=False
    )

    for column in range(count):
        axes[0, column].imshow(
            original[column],
            cmap=cmap,
            norm=norm,
            interpolation="nearest"
        )

        axes[1, column].imshow(
            reconstructed[column],
            cmap=cmap,
            norm=norm,
            interpolation="nearest"
        )

        axes[0, column].set_title(
            examples["titles"][column],
            fontsize=8
        )

        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruction")

    fig.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for color, label in zip(colors, labels)
        ],
        loc="lower center",
        ncol=4
    )

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ----------------------- Checkpoints -----------------------

def save_checkpoint(
    path,
    vae,
    optimizer,
    scaler,
    epoch,
    best_val_reconstruction,
    config,
    overfit_samples
):
    state = {
        "vae": vae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val_reconstruction": best_val_reconstruction,
        "config": config,
        "overfit_samples": overfit_samples,
        "label_values": [0, 1, 2, 4],
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        )
    }

    temporary_path = path.with_suffix(".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)



# ----------------------- Main -----------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--overfit-samples", type=int, default=0)

    return parser.parse_args()

def main():
    args = parse_args()

    if args.overfit_samples < 0:
        raise ValueError("--overfit-samples must be non-negative")

    config = load_config(args.config)

    tconfig = config["training"]
    lconfig = config["vae_loss"]
    econfig = config.get("evaluation", {})

    set_seed(int(config.get("seed", 42)))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    output_dir = Path(config["output_dir"])

    if args.overfit_samples > 0:
        output_dir = output_dir / "overfit"

    checkpoint_dir = output_dir / "checkpoints"
    reconstruction_dir = output_dir / "reconstructions"

    if (checkpoint_dir / "last.pt").exists() and not args.resume:
        raise FileExistsError(
            "An existing run was found. Use --resume or "
            "choose a different output_dir."
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(
        config,
        overfit_samples=args.overfit_samples
    )

    print("Training slices:", len(train_loader.dataset))
    print("Validation slices:", len(val_loader.dataset))

    examples = select_plot_examples(
        val_loader.dataset,
        num_examples=int(econfig.get("num_examples", 6))
    )

    vae = build_vae(config).to(device)

    if vae.in_channels != 4 or vae.out_channels != 4:
        raise ValueError("Mask VAE requires four input/output channels")

    image_size = int(config["data"]["image_size"])

    if image_size % vae.spatial_compression != 0:
        raise ValueError(
            "image_size must be divisible by spatial_compression"
        )

    latent_size = image_size // vae.spatial_compression

    print(
        "Latent shape:",
        (vae.latent_channels, latent_size, latent_size)
    )

    optimizer = AdamW(
        vae.parameters(),
        lr=float(tconfig.get("learning_rate", 1e-4)),
        weight_decay=float(tconfig.get("weight_decay", 1e-4))
    )

    loss_fn = VAELossSegmentation(
        ce_weight=float(lconfig.get("ce_weight", 1.0)),
        dice_weight=float(lconfig.get("dice_weight", 1.0)),
        kl_weight=float(lconfig.get("kl_weight", 1e-4))
    )

    use_amp = (device.type == "cuda" and bool(tconfig.get("mixed_precision", True)))

    amp_name = tconfig.get("amp_dtype", "bf16")

    if amp_name not in ("bf16", "fp16"):
        raise ValueError("amp_dtype must be bf16 or fp16")

    amp_dtype = (
        torch.bfloat16
        if amp_name == "bf16" else torch.float16
    )

    if (
        use_amp
        and amp_dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        amp_dtype = torch.float16

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    start_epoch = 1
    best_val_reconstruction = float("inf")

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=True
        )

        for section in ("vae", "vae_loss"):
            if checkpoint["config"][section] != config[section]:
                raise ValueError(
                    f"Resume configuration differs: {section}"
                )

        if checkpoint["overfit_samples"] != args.overfit_samples:
            raise ValueError("Resume overfit setting differs")

        vae.load_state_dict(checkpoint["vae"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_reconstruction = checkpoint["best_val_reconstruction"]

        torch.set_rng_state(checkpoint["torch_rng"].cpu())

        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([
                state.cpu()
                for state in checkpoint["cuda_rng"]
            ])

    epochs = int(tconfig.get("epochs", 100))
    warmup_epochs = int(lconfig.get("kl_warmup_epochs", 10))
    target_beta = float(lconfig.get("kl_weight", 1e-4))
    plot_every = int(econfig.get("plot_every", 1))

    if epochs < 1 or warmup_epochs < 0 or plot_every < 1:
        raise ValueError("Invalid epoch, warmup or plotting setting")

    history_path = output_dir / "history.jsonl"

    for epoch in range(start_epoch, epochs + 1):
        if args.overfit_samples > 0:
            beta = 0.0
        elif warmup_epochs > 0:
            beta = target_beta * min(
                1.0,
                (epoch - 1) / warmup_epochs,
            )
        else:
            beta = target_beta

        print(f"\nEpoch {epoch}/{epochs} | KL weight: {beta:.3g}")

        train_results = train_epoch(
            vae,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler,
            amp_dtype,
            use_amp,
            beta
        )

        val_results = validate(
            vae,
            val_loader,
            loss_fn,
            device,
            amp_dtype,
            use_amp,
            beta,
            seed=int(econfig.get("seed", 1234))
        )

        score = val_results["reconstruction"]
        improved = score < best_val_reconstruction

        if improved:
            best_val_reconstruction = score

        print(
            f"Train loss: {train_results['loss']:.5f} | "
            f"Validation reconstruction: {score:.5f}"
        )

        print(
            "Validation mean-code metrics:",
            val_results["mean_metrics"]
        )

        record = {
            "epoch": epoch,
            "beta": beta,
            "train": train_results,
            "val": val_results
        }

        with history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if epoch == 1 or epoch % plot_every == 0:
            save_reconstructions(
                vae,
                examples,
                device,
                reconstruction_dir / f"epoch_{epoch:04d}.png"
            )

        checkpoint_options = {
            "vae": vae,
            "optimizer": optimizer,
            "scaler": scaler,
            "epoch": epoch,
            "best_val_reconstruction": best_val_reconstruction,
            "config": config,
            "overfit_samples": args.overfit_samples
        }

        save_checkpoint(path=checkpoint_dir / "last.pt", **checkpoint_options)

        if improved:
            save_checkpoint(path=checkpoint_dir / "best.pt", **checkpoint_options)


if __name__ == "__main__":
    main()