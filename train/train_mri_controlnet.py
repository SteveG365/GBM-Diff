import argparse
import copy
import csv
from pathlib import Path
import bisect

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from src.backbones.brainmri_backbone import BrainMRIBackbone
from src.diffusion.factory import build_diffusion
from src.control.controlnets.brainmri_controlnet import BrainMRIControlNet_v1, BrainMRIControlNet_v2
from src.vae.factory import build_vae, load_vae_state_dict


MODALITY_TO_ID = {
    "t1": 0,
    "t1gd": 1,
    "t2": 2,
    "flair": 3,
}


class DDPMSchedule:
    """The same forward diffusion schedule used by train_diff_backbone.py."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None,
    ):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            dtype=torch.float32,
            device=device,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def add_noise(
        self,
        z_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps].view(
            -1, 1, 1, 1
        )
        return sqrt_alpha_bar * z_0 + sqrt_one_minus_alpha_bar * noise


class ControlConditionedCachedDataset(Dataset):

    REQUIRED_KEYS = {
        "mu",
        "logvar",
        "condition",
        "modality",
        "subject"
    }

    def __init__(self, latent_dir):
        self.latent_dir = Path(latent_dir)
        self.shard_paths = sorted(
            self.latent_dir.glob("latents_*.pt")
        )

        if not self.shard_paths:
            raise FileNotFoundError(
                f"No latent shards found in {self.latent_dir}."
            )

        self.shard_sizes = []

        for path in self.shard_paths:
            shard = torch.load(
                path,
                map_location="cpu",
                weights_only=True
            )

            missing = self.REQUIRED_KEYS - set(shard)

            if missing:
                raise RuntimeError(
                    f"{path} is missing {sorted(missing)}. "
                    "Rebuild the ControlNet cache."
                )

            size = shard["mu"].shape[0]

            for key in ("logvar", "condition", "modality"):
                if shard[key].shape[0] != size:
                    raise RuntimeError(
                        f"Misaligned '{key}' in {path}."
                    )

            if len(shard["subject"]) != size:
                raise RuntimeError(
                    f"Misaligned 'subject' in {path}."
                )

            self.shard_sizes.append(size)

        self.cumulative_sizes = []
        total = 0

        for size in self.shard_sizes:
            total += size
            self.cumulative_sizes.append(total)

        self._cached_shard_idx = None
        self._cached_shard = None

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        shard_idx = bisect.bisect_right(
            self.cumulative_sizes,
            idx
        )

        previous = (
            0
            if shard_idx == 0
            else self.cumulative_sizes[shard_idx - 1]
        )

        local_idx = idx - previous

        if shard_idx != self._cached_shard_idx:
            self._cached_shard = torch.load(
                self.shard_paths[shard_idx],
                map_location="cpu",
                weights_only=True
            )
            self._cached_shard_idx = shard_idx

        shard = self._cached_shard

        return {
            "mu": shard["mu"][local_idx],
            "logvar": shard["logvar"][local_idx],
            "condition": shard["condition"][local_idx],
            "modality": shard["modality"][local_idx],
            "subject": shard["subject"][local_idx]
        }

def get_raw_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@torch.no_grad()
def load_vae(config, checkpoint_path, device):
    vae = build_vae(config).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    load_vae_state_dict(vae, checkpoint)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def select_visualization_examples(dataset, num_examples, max_candidates):
    """Choose fixed examples with the largest visible tumor masks."""
    candidate_count = min(len(dataset), max_candidates)
    if candidate_count < num_examples:
        raise RuntimeError(
            f"Need {num_examples} visualization examples, but the dataset "
            f"contains only {candidate_count}."
        )

    # Spread the scan across the dataset rather than considering only its start.
    candidate_indices = torch.linspace(
        0,
        len(dataset) - 1,
        steps=candidate_count,
        dtype=torch.long,
    ).tolist()
    scored = []
    for index in tqdm(candidate_indices, desc="Selecting large masks", leave=False):
        item = dataset[index]
        mask_fraction = (item["condition"] > 0).float().mean().item()
        scored.append((mask_fraction, index))

    selected = sorted(scored, reverse=True)[:num_examples]
    if selected[0][0] <= 0:
        raise RuntimeError(
            "No non-empty tumor masks were found for visualization."
        )

    print(
        "Visualization mask fractions: "
        + ", ".join(f"{fraction:.3%}" for fraction, _ in selected)
    )
    return [index for _, index in selected]


def load_visualization_batch(dataset, indices, device):
    items = [dataset[index] for index in indices]
    condition = torch.stack([item["condition"] for item in items]).to(
        device=device,
        dtype=torch.float32,
    )
    modality = torch.tensor(
        [item["modality"] for item in items],
        device=device,
        dtype=torch.long,
    )
    latent_shape = tuple(items[0]["mu"].shape)
    return condition, modality, latent_shape


@torch.no_grad()
def sample_controlled_ddpm(
    backbone,
    controlnet,
    schedule,
    condition,
    modality,
    latent_shape,
    device,
    seed,
):
    """Native ancestral DDPM sampling; no Diffusers dependency."""
    sampling_controlnet = get_raw_model(controlnet)
    sampling_controlnet.eval()
    backbone.model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    z = torch.randn(
        (condition.shape[0], *latent_shape),
        device=device,
        generator=generator,
    )

    for timestep in tqdm(
        reversed(range(schedule.num_timesteps)),
        total=schedule.num_timesteps,
        desc="Controlled sampling",
        leave=False,
    ):
        timesteps = torch.full(
            (z.shape[0],),
            timestep,
            device=device,
            dtype=torch.long,
        )
        epsilon_pred = predict_noise(
            backbone,
            sampling_controlnet,
            z,
            timesteps,
            condition,
            modality,
        )

        beta_t = schedule.betas[timestep]
        alpha_t = schedule.alphas[timestep]
        alpha_bar_t = schedule.alpha_bars[timestep]
        mean = torch.rsqrt(alpha_t) * (
            z
            - beta_t
            / torch.sqrt(1.0 - alpha_bar_t)
            * epsilon_pred
        )

        if timestep > 0:
            alpha_bar_previous = schedule.alpha_bars[timestep - 1]
            posterior_variance = (
                beta_t
                * (1.0 - alpha_bar_previous)
                / (1.0 - alpha_bar_t)
            )
            noise = torch.randn(
                z.shape,
                device=device,
                dtype=z.dtype,
                generator=generator,
            )
            z = mean + torch.sqrt(posterior_variance) * noise
        else:
            z = mean

    return z


@torch.no_grad()
def generate_conditioned_grid(
    backbone,
    controlnet,
    vae,
    schedule,
    condition,
    modality,
    latent_shape,
    latent_scale,
    device,
    seed,
    output_path,
):
    latents = sample_controlled_ddpm(
        backbone=backbone,
        controlnet=controlnet,
        schedule=schedule,
        condition=condition,
        modality=modality,
        latent_shape=latent_shape,
        device=device,
        seed=seed,
    )
    generated = vae.decode(latents / latent_scale).float()
    generated = generated.clamp(-1, 1).add(1.0).div(2.0)

    masks = (condition > 0).float()
    if masks.shape[-2:] != generated.shape[-2:]:
        masks = F.interpolate(masks, size=generated.shape[-2:], mode="nearest")

    # Interleave [mask, generated image] so every row is an obvious pair.
    paired = torch.stack(
        [tensor for pair in zip(masks, generated) for tensor in pair]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(paired, output_path, nrow=2)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    source = get_raw_model(model)
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in source.named_parameters():
        ema_parameters[name].mul_(decay).add_(
            parameter.detach(), alpha=1.0 - decay
        )

    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in source.named_buffers():
        ema_buffers[name].copy_(buffer)


def append_history(path, epoch, train_loss, val_loss, learning_rate):
    file_exists = path.exists()
    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "learning_rate"]
            )
        writer.writerow([epoch, train_loss, val_loss, learning_rate])


def save_checkpoint(
    path,
    controlnet,
    ema_controlnet,
    optimiser,
    scaler,
    epoch,
    best_val_loss,
    latent_scale,
    config,
    backbone_checkpoint,
):
    model_to_save = get_raw_model(controlnet)
    torch.save(
        {
            "controlnet": model_to_save.state_dict(),
            "ema_controlnet": (
                ema_controlnet.state_dict() if ema_controlnet is not None else None
            ),
            "optimizer": optimiser.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "latent_scale": latent_scale,
            "backbone_checkpoint": str(backbone_checkpoint),
            "config": config,
        },
        path,
    )


def load_controlnet_checkpoint(
    path,
    controlnet,
    ema_controlnet,
    optimiser,
    scaler,
    device,
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    controlnet.load_state_dict(checkpoint["controlnet"])

    if ema_controlnet is not None:
        ema_state = checkpoint.get("ema_controlnet")
        ema_controlnet.load_state_dict(
            ema_state if ema_state is not None else checkpoint["controlnet"]
        )

    optimiser.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    return (
        checkpoint["epoch"] + 1,
        checkpoint["best_val_loss"],
        checkpoint["latent_scale"],
    )


def read_backbone_latent_scale(path, device, config):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    latent_scale = checkpoint.get("latent_scale")
    if latent_scale is None:
        latent_scale = config.get("control_training", {}).get("latent_scale")
    if latent_scale is None:
        raise RuntimeError(
            "The backbone checkpoint has no latent_scale. Add "
            "control_training.latent_scale to the config explicitly."
        )
    return float(latent_scale)


def resolve_cached_dir(config, split, explicit_path):
    if explicit_path is not None:
        return explicit_path

    data_config = config["data"]
    candidates = (
        f"cached_control_{split}_dir",
        f"control_cached_{split}_dir",
    )
    for key in candidates:
        if key in data_config:
            return data_config[key]

    raise KeyError(
        f"No cached ControlNet {split} directory configured. Add "
        f"data.cached_control_{split}_dir or pass --{split}-latents."
    )


def build_dataset(config, split, latent_dir):
    del config, split
    return ControlConditionedCachedDataset(latent_dir)


def build_patient_subset(dataset, patient=None):
    patient_to_indices = {}
    for index in range(len(dataset.raw_dataset)):
        volume_idx, _, _ = dataset.raw_dataset.samples[index]
        row = dataset.raw_dataset._get_volume_row(volume_idx)
        subject = str(row["subject"])
        patient_to_indices.setdefault(subject, []).append(index)

    if not patient_to_indices:
        raise RuntimeError("No patients were found in the controlnet split.")

    if patient is None:
        patient = next(iter(patient_to_indices))
    if patient not in patient_to_indices:
        raise ValueError(f"Patient '{patient}' was not found in the controlnet split.")

    indices = patient_to_indices[patient]
    print(f"OVERFIT MODE: patient={patient}, samples={len(indices)}")
    return Subset(dataset, indices)


def prepare_batch(batch, schedule, device, latent_scale):
    mu = batch["mu"].to(device, non_blocking=True, dtype=torch.float32)
    logvar = batch["logvar"].to(
        device, non_blocking=True, dtype=torch.float32
    )
    condition = batch["condition"].to(
        device, non_blocking=True, dtype=torch.float32
    )
    modality = batch["modality"].to(
        device, non_blocking=True, dtype=torch.long
    )

    posterior_noise = torch.randn_like(mu)
    z_0 = (mu + torch.exp(0.5 * logvar) * posterior_noise) * latent_scale
    timesteps = torch.randint(
        0,
        schedule.num_timesteps,
        (z_0.shape[0],),
        device=device,
        dtype=torch.long,
    )
    diffusion_noise = torch.randn_like(z_0)
    z_t = schedule.add_noise(z_0, diffusion_noise, timesteps)

    z_t = z_t.contiguous(memory_format=torch.channels_last)
    condition = condition.contiguous(memory_format=torch.channels_last)
    return z_t, timesteps, diffusion_noise, condition, modality


def predict_noise(backbone, controlnet, z_t, timesteps, condition, modality):
    control_state = controlnet.get_control(
        z_t=z_t,
        timestep=timesteps,
        condition=condition,
        modality=modality,
    )
    return backbone.predict_noise(
        z_t=z_t,
        timestep=timesteps,
        control_state=control_state,
        modality=modality,
    )


def train_one_epoch(
    controlnet,
    ema_controlnet,
    backbone,
    loader,
    optimiser,
    schedule,
    device,
    latent_scale,
    scaler,
    use_amp,
    amp_dtype,
    ema_decay,
):
    controlnet.train()
    backbone.model.eval()
    total_loss = torch.zeros((), device=device)
    num_samples = 0
    progress = tqdm(loader, desc="Train", leave=False)

    for step, batch in enumerate(progress):
        z_t, timesteps, target, condition, modality = prepare_batch(
            batch, schedule, device, latent_scale
        )
        optimiser.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            prediction = predict_noise(
                backbone,
                controlnet,
                z_t,
                timesteps,
                condition,
                modality,
            )
            loss = F.mse_loss(prediction, target)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        if ema_controlnet is not None:
            update_ema(ema_controlnet, controlnet, ema_decay)

        batch_size = z_t.shape[0]
        total_loss += loss.detach() * batch_size
        num_samples += batch_size
        if step % 20 == 0:
            progress.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss.item() / num_samples


@torch.no_grad()
def validate(
    controlnet,
    backbone,
    loader,
    schedule,
    device,
    latent_scale,
    use_amp,
    amp_dtype,
):
    controlnet.eval()
    backbone.model.eval()
    total_loss = torch.zeros((), device=device)
    num_samples = 0

    for batch in tqdm(loader, desc="Validation", leave=False):
        z_t, timesteps, target, condition, modality = prepare_batch(
            batch, schedule, device, latent_scale
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            prediction = predict_noise(
                backbone,
                controlnet,
                z_t,
                timesteps,
                condition,
                modality,
            )
            loss = F.mse_loss(prediction, target)

        batch_size = z_t.shape[0]
        total_loss += loss * batch_size
        num_samples += batch_size

    return total_loss.item() / num_samples


def build_control_net(config, backbone, device):
    if config["diffusion"]["architecture"] == "latent_v1":
        controlnet = BrainMRIControlNet_v1(
            backbone=backbone,
            device=device,
            condition_channels=int(config["control"].get("condition_channels", 1))
        ).to(device)
    elif config["diffusion"]["architecture"] == "latent_v2":
        controlnet = BrainMRIControlNet_v2(
            backbone=backbone,
            device=device,
            condition_channels=int(config["control"].get("condition_channels", 1))
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture")
    return controlnet

def main(args):
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("MRI ControlNet training requires CUDA.")

    training_config = config["control_training"]
    control_config = config["control"]
    seed = int(training_config.get("seed", config.get("seed", 42)))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_latents = resolve_cached_dir(config, "train", args.train_latents)
    val_latents = resolve_cached_dir(config, "val", args.val_latents)
    train_dataset = build_dataset(config, "train", train_latents)
    val_dataset = build_dataset(config, "val", val_latents)

    if args.overfit:
        train_dataset = build_patient_subset(train_dataset, args.overfit_patient)
        val_dataset = train_dataset

    batch_size = 1 if args.overfit else int(training_config["batch_size"])
    num_workers = int(training_config.get("num_workers", 4))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
        "drop_last": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    diffusion_model = build_diffusion(config).to(device)
    backbone = BrainMRIBackbone(
        checkpoint_path=args.backbone_checkpoint,
        diff_model=diffusion_model,
        architecture=config["diffusion"]["architecture"],
        device=device
    )
    backbone.model.requires_grad_(False)
    backbone.model.eval()

    controlnet = build_control_net(config, backbone, device)

    # deepcopy preserves the frozen backbone flags, so explicitly make the
    # independently copied ControlNet branch trainable.
    controlnet.requires_grad_(True)
    controlnet = controlnet.to(memory_format=torch.channels_last)

    if any(parameter.requires_grad for parameter in backbone.model.parameters()):
        raise RuntimeError("The diffusion backbone must remain frozen.")
    if not all(parameter.requires_grad for parameter in controlnet.parameters()):
        raise RuntimeError("Every MRI ControlNet parameter should be trainable.")

    ema_enabled = bool(config.get("ema", {}).get("enabled", True))
    ema_decay = float(config.get("ema", {}).get("decay", 0.9999))
    if ema_enabled:
        ema_controlnet = build_control_net(config, backbone, device)
        ema_controlnet.load_state_dict(controlnet.state_dict())
        ema_controlnet.requires_grad_(False)
        ema_controlnet.eval()
        ema_controlnet = ema_controlnet.to(memory_format=torch.channels_last)
    else:
        ema_controlnet = None

    diffusion_config = config["diffusion"]
    schedule = DDPMSchedule(
        num_timesteps=int(diffusion_config.get("num_timesteps", 1000)),
        beta_start=float(diffusion_config.get("beta_start", 1e-4)),
        beta_end=float(diffusion_config.get("beta_end", 2e-2)),
        device=device,
    )

    optimiser = torch.optim.AdamW(
        controlnet.parameters(),
        lr=float(training_config.get("learning_rate", 1e-5)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
        fused=True,
    )

    use_amp = bool(training_config.get("mixed_precision", True))
    amp_dtype_name = training_config.get("amp_dtype", "bf16")
    if amp_dtype_name == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_dtype_name == "fp16":
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unknown amp_dtype: {amp_dtype_name}")
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("BF16 unsupported; falling back to FP16")
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16)
    )

    latent_scale = read_backbone_latent_scale(
        args.backbone_checkpoint, device, config
    )
    vae = load_vae(config, args.vae_checkpoint, device)

    evaluation_config = config.get("control_evaluation", {})
    visualization_count = int(evaluation_config.get("num_examples", 4))
    visualization_scan = int(evaluation_config.get("max_mask_candidates", 512))
    visualization_seed = int(evaluation_config.get("seed", 1234))
    if visualization_count < 1:
        raise ValueError("control_evaluation.num_examples must be at least 1.")
    if visualization_scan < visualization_count:
        raise ValueError(
            "control_evaluation.max_mask_candidates must be greater than or "
            "equal to num_examples."
        )
    visualization_indices = select_visualization_examples(
        val_dataset,
        num_examples=visualization_count,
        max_candidates=visualization_scan,
    )
    visualization_condition, visualization_modality, visualization_latent_shape = (
        load_visualization_batch(val_dataset, visualization_indices, device)
    )

    output_dir = Path(training_config.get("output_dir", "outputs/mri_controlnet"))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.csv"

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        start_epoch, best_val_loss, resumed_scale = load_controlnet_checkpoint(
            args.resume,
            controlnet,
            ema_controlnet,
            optimiser,
            scaler,
            device,
        )
        if abs(resumed_scale - latent_scale) > 1e-8:
            raise RuntimeError(
                "Resume checkpoint latent_scale does not match the backbone checkpoint."
            )
        print(f"Resumed from epoch {start_epoch}")

    if bool(training_config.get("compile", False)):
        controlnet = torch.compile(
            controlnet,
            mode=training_config.get("compile_mode", "reduce-overhead"),
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in get_raw_model(controlnet).parameters()
        if parameter.requires_grad
    )
    print(f"ControlNet parameters: {trainable_parameters / 1e6:.2f}M")
    print(f"Latent scale: {latent_scale:.6f}")
    print(f"AMP: {use_amp}, dtype: {amp_dtype}")

    num_epochs = int(training_config["num_epochs"])
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_one_epoch(
            controlnet,
            ema_controlnet,
            backbone,
            train_loader,
            optimiser,
            schedule,
            device,
            latent_scale,
            scaler,
            use_amp,
            amp_dtype,
            ema_decay,
        )
        val_loss = validate(
            controlnet,
            backbone,
            val_loader,
            schedule,
            device,
            latent_scale,
            use_amp,
            amp_dtype,
        )
        learning_rate = optimiser.param_groups[0]["lr"]
        print(f"Train loss: {train_loss:.6f}")
        print(f"Val loss:   {val_loss:.6f}")
        append_history(
            history_path, epoch + 1, train_loss, val_loss, learning_rate
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            controlnet,
            ema_controlnet,
            optimiser,
            scaler,
            epoch,
            best_val_loss,
            latent_scale,
            config,
            args.backbone_checkpoint,
        )
        if is_best:
            save_checkpoint(
                checkpoint_dir / "best.pt",
                controlnet,
                ema_controlnet,
                optimiser,
                scaler,
                epoch,
                best_val_loss,
                latent_scale,
                config,
                args.backbone_checkpoint,
            )
            print(f"New best validation loss: {best_val_loss:.6f}")

        # Use EMA when available because it is normally the cleaner sampling model.
        visualization_model = (
            ema_controlnet if ema_controlnet is not None else controlnet
        )
        sample_path = (
            output_dir
            / "samples"
            / f"epoch_{epoch + 1:04d}_mask_and_conditioned.png"
        )
        generate_conditioned_grid(
            backbone=backbone,
            controlnet=visualization_model,
            vae=vae,
            schedule=schedule,
            condition=visualization_condition,
            modality=visualization_modality,
            latent_shape=visualization_latent_shape,
            latent_scale=latent_scale,
            device=device,
            seed=visualization_seed,
            output_path=sample_path,
        )
        print(f"Saved mask/output examples to {sample_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backbone-checkpoint", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--train-latents", type=str, default=None)
    parser.add_argument("--val-latents", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit-patient", type=str, default=None)
    main(parser.parse_args())