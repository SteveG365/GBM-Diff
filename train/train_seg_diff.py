from __future__ import annotations

import argparse
import hashlib
import math
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm


from src.data.load_masks import load_mask_path_splits, TumourMaskSliceDataset, ShardBatchSampler, CachedMaskLatentDataset
from src.diffusion.sampler import sample_mask_ddpm
from src.diffusion.scheduler import DDPMSchedule
from src.diffusion.diff_model import MaskLatentDiffusionUNet
from src.vae.factory import build_vae
from src.utils.config import load_config
from src.utils.seed import set_seed

# --------------------- Data Loading / Checking ---------------------

def build_loaders(config):
    cache_dir = config["data"]["cache_dir"]
    tconfig = config["training"]

    train_dataset = CachedMaskLatentDataset(cache_dir, "train")
    val_dataset = CachedMaskLatentDataset(cache_dir, "val")

    metadata = train_dataset.metadata

    defaults = {
        "image_size": 256,
        "slice_axis": 2,
        "tumour_only": True
    }

    for key, default in defaults.items():
        expected = config["data"].get(key, default)
        cached = metadata["preprocessing"][key]

        if expected != cached:
            raise ValueError(f"Cache preprocessing mismatch: {key}")

    expected_channels = int(config["diffusion"]["latent_channels"])

    if metadata["vae"]["latent_shape"][0] != expected_channels:
        raise ValueError("Cache latent channels do not match diffusion")

    batch_size = int(tconfig.get("batch_size", 8))
    workers = int(tconfig.get("num_workers", 4))

    train_sampler = ShardBatchSampler(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=int(config.get("seed", 42))
    )

    val_sampler = ShardBatchSampler(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    loader_options = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0
    }

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **loader_options
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        **loader_options
    )

    return train_loader, val_loader


def check_batch(batch):
    mu = batch["mu"]
    logvar = batch["logvar"]
    circle = batch["circle"]
    conditions = batch["conditions"]

    batch_size = mu.shape[0]

    assert mu.ndim == 4
    assert logvar.shape == mu.shape
    assert circle.ndim == 4
    assert circle.shape[:2] == (batch_size, 1)
    assert conditions.shape == (batch_size, 4)

    for key in ("mu", "logvar", "circle", "conditions"):
        assert torch.isfinite(batch[key]).all()
        print(f"{key}: {tuple(batch[key].shape)}")

    assert ((conditions >= 0) & (conditions <= 1)).all()

    torch.testing.assert_close(
        conditions[:, 1:].sum(dim=1),
        torch.ones(batch_size)
    )

    print("Cached batch checks passed.", flush=True)

def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def check_cache_and_get_scale(config, metadata):
    checkpoint_path = Path(config["vae"]["checkpoint"])

    print("Verifying VAE checkpoint against cache...", flush=True)

    actual_hash = sha256_file(checkpoint_path)
    cached_hash = metadata["vae"]["checkpoint_sha256"]

    if actual_hash != cached_hash:
        raise ValueError("The selected VAE checkpoint differs from the one used to build this latent cache.")

    scaling = metadata.get("latent_scaling")

    if scaling is None:
        raise ValueError("Cache has no training-derived latent scale.")

    if scaling["source"] != "train":
        raise ValueError("Latent scale must come from training data")

    if scaling["centre_latents"]:
        raise ValueError(
            "This trainer expects scaling without mean subtraction"
        )

    latent_scale = float(scaling["scale"])

    if not math.isfinite(latent_scale) or latent_scale <= 0:
        raise ValueError("Invalid latent scale")

    print(f"Latent scale: {latent_scale:.6f}", flush=True)

    return latent_scale

# --------------------- Training / Validation Logic ---------------------

def build_diffusion_components(config, device):
    dconfig = config["diffusion"]
    tconfig = config["training"]

    model = MaskLatentDiffusionUNet(
        latent_channels=int(dconfig.get("latent_channels", 4)),
        base_channels=int(dconfig.get("base_channels", 128)),
        time_dim=int(dconfig.get("time_dim", 512)),
        norm_groups=int(dconfig.get("norm_groups", 32)),
        dropout=float(dconfig.get("dropout", 0.0))
    ).to(device)

    schedule = DDPMSchedule(
        num_timesteps=int(dconfig.get("num_timesteps", 1000)),
        beta_start=float(dconfig.get("beta_start", 1e-4)),
        beta_end=float(dconfig.get("beta_end", 0.02)),
        device=device
    )

    # Optimising only the diff model
    optimizer = AdamW(
        model.parameters(),
        lr=float(tconfig.get("learning_rate", 1e-4)),
        weight_decay=float(tconfig.get("weight_decay", 1e-4)),
    )

    use_amp = device.type == "cuda" and bool(tconfig.get("mixed_precision", True))
    amp_name = tconfig.get("amp_dtype", "bf16")
    if amp_name not in ("bf16", "fp16"):
        raise ValueError("amp_dtype must be bf16 or fp16")
    amp_dtype = (
        torch.bfloat16
        if amp_name == "bf16"
        else torch.float16
    )

    if (
        use_amp
        and amp_dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        amp_dtype = torch.float16

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and amp_dtype == torch.float16
    )

    print(
        "Diffusion parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
        flush=True
    )

    return model, schedule, optimizer, scaler, use_amp, amp_dtype

def sample_cached_latents(
    batch,
    device,
    latent_scale
):
    mu = batch["mu"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True
    )
    logvar = batch["logvar"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True
    )
    std = torch.exp(0.5 * logvar)

    z_0 = mu + std * torch.randn_like(mu)

    return z_0 * latent_scale

def train_epoch(
    model,
    loader,
    schedule,
    optimizer,
    scaler,
    device,
    latent_scale,
    condition_drop_probability,
    use_amp,
    amp_dtype
):
    if not 0.0 <= condition_drop_probability <= 1.0:
        raise ValueError("condition_drop_probability must be between 0 and 1")

    model.train()

    loss_sum = torch.zeros((), device=device)
    dropped_count = torch.zeros((), device=device)
    num_samples = 0

    progress = tqdm(loader, desc="Train", leave=False)

    for step, batch in enumerate(progress):
        optimizer.zero_grad(set_to_none=True)

        # 1) Sample the clean latent from the frozen VAE posterior.
        z_0 = sample_cached_latents(batch, device, latent_scale)

        batch_size = z_0.shape[0]

        circle = batch["circle"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True
        )

        conditions = batch["conditions"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True
        )

        # 2) Independently sample diffusion noise and timesteps.
        noise = torch.randn_like(z_0)

        timesteps = torch.randint(
            low=0,
            high=schedule.num_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long
        )

        z_t = schedule.add_noise(
            z_0,
            noise,
            timesteps
        )

        # 3) Drop both conditioning routes together for CFG training.
        conditions_present = (
            torch.rand(batch_size, device=device)
            >= condition_drop_probability
        )

        # 4) Predict the noise
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp
        ):
            noise_prediction = model(
                z_t,
                timesteps,
                circle,
                conditions,
                conditions_present=conditions_present
            )

        if noise_prediction.shape != noise.shape:
            raise ValueError(
                f"Noise prediction shape {noise_prediction.shape} "
                f"does not match target {noise.shape}"
            )

        loss = F.mse_loss(noise_prediction.float(), noise)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.detach() * batch_size
        dropped_count += (~conditions_present).sum()
        num_samples += batch_size

        if step % 20 == 0:
            progress.set_postfix(loss=f"{loss.item():.5f}")

    if num_samples == 0:
        raise ValueError("Training loader produced no samples")

    return {
        "loss": loss_sum.item() / num_samples,
        "condition_drop_fraction": (
            dropped_count.item() / num_samples
        ),
        "samples": num_samples
    }

@torch.no_grad()
def validate(
    model,
    loader,
    schedule,
    device,
    latent_scale,
    use_amp,
    amp_dtype,
    seed
):
    model.eval()

    conditional_loss_sum = torch.zeros((), device=device)
    unconditional_loss_sum = torch.zeros((), device=device)
    num_samples = 0

    cuda_devices = (
        list(range(torch.cuda.device_count()))
        if device.type == "cuda"
        else []
    )

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)

        progress = tqdm(
            loader,
            desc="Validation",
            leave=False
        )

        for batch in progress:
            z_0 = sample_cached_latents(
                batch,
                device,
                latent_scale
            )

            batch_size = z_0.shape[0]

            circle = batch["circle"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True
            )

            conditions = batch["conditions"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True
            )

            noise = torch.randn_like(z_0)

            timesteps = torch.randint(
                low=0,
                high=schedule.num_timesteps,
                size=(batch_size,),
                device=device,
                dtype=torch.long
            )

            z_t = schedule.add_noise(
                z_0,
                noise,
                timesteps
            )

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp
            ):
                conditional_prediction = model(
                    z_t,
                    timesteps,
                    circle,
                    conditions,
                    conditions_present=True
                )

                unconditional_prediction = model(
                    z_t,
                    timesteps,
                    circle,
                    conditions,
                    conditions_present=False
                )

            if (
                conditional_prediction.shape != noise.shape
                or unconditional_prediction.shape != noise.shape
            ):
                raise ValueError(
                    "Validation prediction shape differs from noise"
                )

            conditional_loss = F.mse_loss(conditional_prediction.float(), noise)

            unconditional_loss = F.mse_loss(unconditional_prediction.float(), noise)

            losses = torch.stack([conditional_loss, unconditional_loss])

            conditional_loss_sum += conditional_loss * batch_size
            unconditional_loss_sum += unconditional_loss * batch_size
            num_samples += batch_size

    return {
        "conditional_loss": conditional_loss_sum.item() / num_samples,
        "unconditional_loss": unconditional_loss_sum.item() / num_samples,
        "samples": num_samples
    }


# --------------------- Sampling / Visualisations ---------------------

def load_sampling_references(config, metadata):
    cache_dir = Path(config["data"]["cache_dir"])

    reference_path = cache_dir / metadata["splits"]["val"]["references"]["file"]

    references = torch.load(
        reference_path,
        map_location="cpu",
        weights_only=True
    )

    count = int(config.get("evaluation", {}).get("num_examples", 4))

    if count < 1:
        raise ValueError("evaluation.num_examples must be positive")

    references = references[:count]

    if not references:
        raise ValueError("No validation references were saved in the cache")

    image_size = metadata["preprocessing"]["image_size"]
    tolerance = metadata["conditioning"]["circle_boundary_tolerance"]

    rows, columns = torch.meshgrid(
        torch.arange(image_size, dtype=torch.float32),
        torch.arange(image_size, dtype=torch.float32),
        indexing="ij"
    )

    circles = []

    for reference in references:
        centre = reference["circle_centre_rc"]
        radius_squared = reference["circle_radius_squared"]

        circle = (
            (rows - centre[0]).square()
            + (columns - centre[1]).square()
            <= radius_squared + tolerance
        ).float().unsqueeze(0)

        circles.append(circle)

    return {
        "targets": torch.stack([reference["target"] for reference in references]),
        "circle": torch.stack(circles),
        "conditions": torch.stack([reference["conditions"] for reference in references]),
        "titles": [
            f"{reference['scan_id']}\nz={reference['slice_idx']}"
            for reference in references
        ]
    }

@torch.no_grad()
def save_sampling_plot(
    model,
    vae,
    schedule,
    references,
    latent_shape,
    latent_scale,
    device,
    guidance_scale,
    seed,
    output_path
):
    vae.eval()

    circle = references["circle"].to(device)
    conditions = references["conditions"].to(device)

    generated_latents = sample_mask_ddpm(
        model=model,
        schedule=schedule,
        shape=(circle.shape[0], *latent_shape),
        device=device,
        circle=circle,
        conditions=conditions,
        guidance_scale=guidance_scale,
        seed=seed
    )

    # Undo precisely the scaling used during diffusion training.
    logits = vae.decode(generated_latents.float() / latent_scale)

    generated = logits.argmax(dim=1).cpu().numpy()
    targets = references["targets"].numpy()
    circles = references["circle"][:, 0].numpy()

    colors = [
        "#000000",
        "#E6AA1E",
        "#3CB45A",
        "#DC3C50"
    ]

    labels = [
        "Background",
        "Necrotic",
        "Peritumoural Edema",
        "Enhancing Tumour"
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    count = len(targets)

    fig, axes = plt.subplots(
        2,
        count,
        figsize=(3.5 * count, 7),
        squeeze=False
    )

    for column in range(count):
        for row, masks in enumerate((targets, generated)):
            ax = axes[row, column]

            ax.imshow(
                masks[column],
                cmap=cmap,
                norm=norm,
                interpolation="nearest"
            )

            # Contours require values on both sides of the level.
            if circles[column].min() < 0.5 < circles[column].max():
                ax.contour(
                    circles[column],
                    levels=[0.5],
                    colors="cyan",
                    linewidths=0.8
                )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

        axes[0, column].set_title(
            references["titles"][column],
            fontsize=8
        )

    axes[0, 0].set_ylabel("Reference")
    axes[1, 0].set_ylabel("Generated")

    fig.suptitle(f"Conditional mask samples | CFG={guidance_scale:g}")

    fig.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for color, label in zip(colors, labels)
        ] + [
            Patch(
                facecolor="none",
                edgecolor="cyan",
                label="Requested circle"
            )
        ],
        loc="lower center",
        ncol=5
    )

    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# --------------------- Checkpointing ---------------------


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    best_val_loss,
    config,
    cache_identity,
    latent_scale
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": config,
        "cache_identity": cache_identity,
        "latent_scale": latent_scale,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )
    }

    path = Path(path)
    temporary_path = path.with_suffix(".tmp")

    torch.save(state, temporary_path)
    temporary_path.replace(path)

def restore_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    config,
    cache_identity,
    latent_scale,
    device
):
    # Avoid loading every checkpoint tensor onto the GPU immediately.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    saved_config = checkpoint["config"]

    if saved_config["diffusion"] != config["diffusion"]:
        raise ValueError(
            "Diffusion architecture or schedule differs from checkpoint"
        )

    if checkpoint["cache_identity"] != cache_identity:
        raise ValueError("The VAE or latent cache differs from the saved run")

    if not math.isclose(
        float(checkpoint["latent_scale"]),
        latent_scale,
        rel_tol=1e-10,
        abs_tol=0.0
    ):
        raise ValueError("Latent scale differs from checkpoint")

    saved_drop = float(saved_config["training"].get("condition_drop_probability", 0.1))

    current_drop = float(config["training"].get("condition_drop_probability", 0.1))

    if saved_drop != current_drop:
        raise ValueError("Condition-drop probability changed")

    if saved_config.get("seed", 42) != config.get("seed", 42):
        raise ValueError("Training seed changed")

    model.load_state_dict(checkpoint["model"], strict=True)

    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    torch.set_rng_state(checkpoint["torch_rng"].cpu())

    saved_cuda_rng = checkpoint["cuda_rng"]

    if device.type == "cuda" and saved_cuda_rng is not None:
        if len(saved_cuda_rng) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([state.cpu() for state in saved_cuda_rng])
        else:
            print(
                "GPU count changed: continuing with newly seeded "
                "CUDA RNG states.",
                flush=True
            )

    print(
        f"Resuming after epoch {checkpoint['epoch']}. "
        f"Restored learning rate: {optimizer.param_groups[0]['lr']}",
        flush=True
    )

    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_val_loss"]),
    )


# --------------------- Argument & VAE Loading ---------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)

    # Check data loading without starting model training.
    parser.add_argument("--check-data", action="store_true")

    return parser.parse_args()


def load_mask_vae(config, device):
    checkpoint_path = Path(config["vae"]["checkpoint"])

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    print(f"Loading mask VAE: {checkpoint_path}", flush=True)

    # Keep saved optimiser tensors off the GPU.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    saved_config = checkpoint["config"]
    architecture = saved_config["vae"]

    if (architecture["in_channels"] != 4 or architecture["out_channels"] != 4):
        raise ValueError("Expected a four-channel segmentation VAE checkpoint")

    # Paths may change between machines; preprocessing must agree.
    for key, default in {"image_size": 256, "slice_axis": 2}.items():
        saved = saved_config["data"].get(key, default)
        current = config["data"].get(key, default)
        if saved != current:
            raise ValueError(
                f"VAE preprocessing mismatch for {key}: "
                f"checkpoint={saved}, current={current}"
            )

    vae = build_vae({"vae": architecture})

    vae.load_state_dict(checkpoint["vae"], strict=True)

    expected_channels = int(config["diffusion"]["latent_channels"])

    if vae.latent_channels != expected_channels:
        raise ValueError(
            f"VAE has {vae.latent_channels} latent channels; "
            f"diffusion expects {expected_channels}"
        )

    image_size = int(config["data"].get("image_size", 256))

    if image_size % vae.spatial_compression != 0:
        raise ValueError(
            "image_size must be divisible by VAE spatial compression"
        )

    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to(device=device, dtype=torch.float32)

    return vae

# --------------------- Main Loop ---------------------

def main():
    args = parse_args()
    config = load_config(args.config)

    tconfig = config["training"]
    econfig = config.get("evaluation", {})

    epochs = int(tconfig.get("epochs", 100))
    sample_every = int(econfig.get("sample_every", 5))
    evaluation_seed = int(econfig.get("seed", 1234))
    condition_drop_probability = float(tconfig.get("condition_drop_probability", 0.1))
    guidance_scale = float(econfig.get("guidance_scale", 2.0))

    if epochs < 1 or sample_every < 1:
        raise ValueError("epochs and sample_every must be positive")
    if not 0 <= condition_drop_probability <= 1:
        raise ValueError("Invalid condition-drop probability")
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("Invalid guidance scale")
    if (guidance_scale != 1.0 and not 0 < condition_drop_probability < 1):
        raise ValueError("CFG requires training both conditional and unconditional examples")

    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    output_dir = Path(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    history_path = output_dir / "history.jsonl"

    if not args.check_data and not args.resume:
        if ((checkpoint_dir / "last.pt").exists() or history_path.exists()):
            raise FileExistsError("Run outputs already exist. Use --resume or choose a new output_dir.")

    train_loader, val_loader = build_loaders(config)

    print("Training slices:", len(train_loader.dataset))
    print("Validation slices:", len(val_loader.dataset))

    if args.check_data:
        print("\nChecking training batch:")
        check_batch(next(iter(train_loader)))

        print("\nChecking validation batch:")
        check_batch(next(iter(val_loader)))

        return

    metadata = train_loader.dataset.metadata

    latent_scale = check_cache_and_get_scale(config, metadata)

    vae = load_mask_vae(config, device)

    image_size = int(config["data"].get("image_size", 256))

    latent_shape = (
        vae.latent_channels,
        image_size // vae.spatial_compression,
        image_size // vae.spatial_compression
    )

    if tuple(metadata["vae"]["latent_shape"]) != latent_shape:
        raise ValueError("Cached latent shape differs from VAE")

    model, schedule, optimizer, scaler, use_amp, amp_dtype  = build_diffusion_components(config, device)

    cache_identity = {
        "metadata_sha256": sha256_file(
            Path(config["data"]["cache_dir"]) / "metadata.json"
        ),
        "vae_checkpoint_sha256": metadata["vae"]["checkpoint_sha256"]
    }
    references = load_sampling_references(config, metadata)

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, best_val_loss = restore_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            cache_identity=cache_identity,
            latent_scale=latent_scale,
            device=device
        )

    if start_epoch > epochs:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}", flush=True)

        train_loader.batch_sampler.set_epoch(epoch)

        train_results = train_epoch(
            model=model,
            loader=train_loader,
            schedule=schedule,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            latent_scale=latent_scale,
            condition_drop_probability=condition_drop_probability,
            use_amp=use_amp,
            amp_dtype=amp_dtype
        )

        val_results = validate(
            model=model,
            loader=val_loader,
            schedule=schedule,
            device=device,
            latent_scale=latent_scale,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            seed=evaluation_seed
        )

        score = val_results["conditional_loss"]
        improved = score < best_val_loss

        if improved:
            best_val_loss = score

        print(
            f"Train MSE: {train_results['loss']:.6f} | "
            f"Val conditional MSE: {score:.6f} | "
            f"Val unconditional MSE: "
            f"{val_results['unconditional_loss']:.6f}",
            flush=True
        )

        checkpoint_options = {
            "model": model,
            "optimizer": optimizer,
            "scaler": scaler,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
            "cache_identity": cache_identity,
            "latent_scale": latent_scale
        }

        save_checkpoint(path=checkpoint_dir / "last.pt", **checkpoint_options)

        if improved:
            save_checkpoint(path=checkpoint_dir / "best.pt", **checkpoint_options)

        record = {
            "epoch": epoch,
            "train": train_results,
            "val": val_results,
            "best_val_loss": best_val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"]
        }

        with history_path.open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")

        if (
            epoch == 1
            or epoch % sample_every == 0
            or epoch == epochs
        ):
            save_sampling_plot(
                model=model,
                vae=vae,
                schedule=schedule,
                references=references,
                latent_shape=latent_shape,
                latent_scale=latent_scale,
                device=device,
                guidance_scale=guidance_scale,
                seed=evaluation_seed,
                output_path=sample_dir / f"epoch_{epoch:04d}.png"
            )


if __name__ == "__main__":
    main()

