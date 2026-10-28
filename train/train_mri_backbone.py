import argparse
import csv
import math
from pathlib import Path
import copy
import bisect
import random
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset, Sampler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

from src.data.load_brain import buildMRIDataset
from src.diffusion.sampler import sample_ddpm
from src.diffusion.scheduler import DDPMSchedule
from src.diffusion.factory import build_diffusion
from src.vae.factory import build_vae, load_vae_state_dict

# ----------------------- Dataset -----------------------
MODALITY_TO_ID = {
    "t1": 0,
    "t1gd": 1,
    "t2": 2,
    "flair": 3
}

class CachedBackboneDataset(Dataset):

    REQUIRED_KEYS = {
        "mu",
        "logvar",
        "modality",
        "subject"
    }

    def __init__(self, latent_dir):
        self.latent_dir = Path(latent_dir)

        self.shard_paths = sorted(self.latent_dir.glob( "latents_*.pt"))

        if not self.shard_paths:
            raise FileNotFoundError(
                "No latent shards found in "
                f"{self.latent_dir}."
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
                    f"{path} is missing "
                    f"{sorted(missing)}. "
                    "Rebuild the backbone cache."
                )
            size = shard["mu"].shape[0]

            for key in ("logvar", "modality"):
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
            self._cached_shard = (
                torch.load(
                    self.shard_paths[
                        shard_idx
                    ],
                    map_location="cpu",
                    weights_only=True
                )
            )

            self._cached_shard_idx = shard_idx
            
        shard = self._cached_shard

        return {
            "mu": shard["mu"][local_idx],
            "logvar": shard["logvar"][local_idx],
            "modality": shard["modality"][local_idx],
            "subject": shard["subject"][local_idx]
        }

class ShardShuffleSampler(Sampler):

    def __init__(
        self,
        dataset,
        seed=42
    ):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        shard_order = list(range(len(self.dataset.shard_sizes)))

        rng.shuffle(shard_order)

        starts = [0, *self.dataset.cumulative_sizes[:-1]]

        for shard_idx in shard_order:
            start = starts[shard_idx]

            stop = self.dataset.cumulative_sizes[shard_idx]

            indices = list(range(start, stop))
            rng.shuffle(indices)
            yield from indices

@torch.no_grad()
def estimate_latent_scale(
    loader,
    device,
    num_batches=100
):

    latent_sum = torch.zeros((), device=device, dtype=torch.float64)

    latent_sq_sum = torch.zeros((),device=device, dtype=torch.float64)

    latent_count = 0

    for batch_idx, batch in enumerate(
        tqdm(
            loader,
            desc="Estimating latent scale",
            leave=False
        )
    ):

        if batch_idx >= num_batches:
            break

        mu = batch["mu"].to(device, non_blocking=True, dtype=torch.float32)
        logvar = batch["logvar"].to(device,non_blocking=True, dtype=torch.float32)
        variance = torch.exp(logvar)

        latent_sum += mu.double().sum()
        latent_sq_sum += (mu.double().square() + variance.double()).sum()
        latent_count += mu.numel()

    mean = (latent_sum / latent_count).item()
    second_moment = (latent_sq_sum / latent_count).item()
    variance = (second_moment - mean ** 2)
    std = math.sqrt(max(variance, 1e-12))

    scale = 1.0 / std

    print(f"Latent mean:  {mean:.6f}")
    print(f"Latent std:   {std:.6f}")
    print(f"Latent scale: {scale:.6f}")

    return scale

@torch.no_grad()
def load_vae(
    config,
    checkpoint_path,
    device,
):
    vae = build_vae(config).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    load_vae_state_dict(
        vae,
        checkpoint
    )

    vae.eval()
    vae.requires_grad_(False)

    return vae


# Helpers for CFG
def get_raw_model(model):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model

def get_null_class(model):
    return get_raw_model(model).null_class

def prepare_modality_condition(
    batch,
    model,
    device,
    condition_dropout=0.0
):
    modality = batch["modality"].to(
        device,
        non_blocking=True,
        dtype=torch.long
    )

    if condition_dropout > 0:
        drop_mask = torch.rand(modality.shape, device=device) < condition_dropout

        modality = modality.clone()
        modality[drop_mask] = get_null_class(model)

    return modality


@torch.no_grad()
def update_ema(
    ema_model,
    model,
    decay=0.9999
):
    model_to_update = get_raw_model(model)

    ema_params = dict(ema_model.named_parameters())

    model_params = dict(model_to_update.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())

    for name, buffer in model_to_update.named_buffers():
        ema_buffers[name].copy_(buffer)

# ---------------------------- Utility Functions ----------------------------
def append_history(
    history_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    learning_rate: float
):

    file_exists = history_path.exists()

    with open(history_path, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "learning_rate"
                ]
            )

        writer.writerow(
            [
                epoch,
                train_loss,
                val_loss,
                learning_rate
            ]
        )

def save_checkpoint(
    path: Path,
    model,
    ema_model,
    optimiser,
    scaler,
    epoch: int,
    best_val_loss: float,
    latent_scale: float,
    config: dict
):

    model_to_save = (
        model._orig_mod
        if hasattr(model, "_orig_mod")
        else model
    )

    torch.save(
        {
            "model": model_to_save.state_dict(),
            "ema_model": (
                ema_model.state_dict()
                if ema_model is not None
                else None
            ),
            "optimizer": optimiser.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "latent_scale": latent_scale,
            "config": config
        },
        path
    )

def load_diffusion_checkpoint(
    path,
    model,
    ema_model,
    optimiser,
    scaler,
    device
):

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])

    if ema_model is not None:
        if "ema_model" in checkpoint and checkpoint["ema_model"] is not None:
            ema_model.load_state_dict(checkpoint["ema_model"])
        else:
            ema_model.load_state_dict(checkpoint["model"])
            print("Checkpoint has no EMA weights: initialising EMA from model weights.")

    optimiser.load_state_dict(checkpoint["optimizer"])

    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    latent_scale = checkpoint["latent_scale"]

    return (start_epoch, best_val_loss, latent_scale)

def images_to_unit_rgb(images):
    images = images.float().clamp(-1, 1).add(1.0).div(2.0)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return images


@torch.no_grad()
def generate_sample_grid(
    model,
    vae,
    schedule,
    latent_shape,
    latent_scale,
    modality_name,
    guidance_scale,
    device,
    seed,
    output_path,
):
    latents = sample_ddpm(
        model=model,
        schedule=schedule,
        shape=(16, *latent_shape),
        device=device,
        modality_id=MODALITY_TO_ID[modality_name],
        guidance_scale=guidance_scale,
        seed=seed,
    )
    images = vae.decode(latents / latent_scale)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        images.float().clamp(-1, 1).add(1.0).div(2.0),
        output_path,
        nrow=4
    )


@torch.no_grad()
def compute_validation_fid(
    model,
    vae,
    schedule,
    raw_val_loader,
    latent_shape,
    latent_scale,
    modality_name,
    guidance_scale,
    device,
    num_samples,
    generation_batch_size,
    fid_feature,
    seed,
):
    """Compute modality-specific FID against raw validation images."""
    fid = FrechetInceptionDistance(
        feature=fid_feature,
        normalize=True,
    ).to(device)

    real_count = 0
    for batch in tqdm(raw_val_loader, desc=f"FID real {modality_name}", leave=False):
        keep = [
            i
            for i, value in enumerate(batch["modality"])
            if value == modality_name
        ]
        if not keep:
            continue

        remaining = num_samples - real_count
        if remaining <= 0:
            break

        images = batch["image"][keep][:remaining].to(
            device, non_blocking=True
        )
        fid.update(images_to_unit_rgb(images), real=True)
        real_count += images.shape[0]

    if real_count < 2:
        raise RuntimeError(
            f"Need at least two validation images for modality "
            f"'{modality_name}', found {real_count}."
        )

    fake_count = 0
    while fake_count < real_count:
        current_batch_size = min(
            generation_batch_size,
            real_count - fake_count,
        )
        latents = sample_ddpm(
            model=model,
            schedule=schedule,
            shape=(current_batch_size, *latent_shape),
            device=device,
            modality_id=MODALITY_TO_ID[modality_name],
            guidance_scale=guidance_scale,
            seed=seed + fake_count,
        )
        images = vae.decode(latents / latent_scale)
        fid.update(images_to_unit_rgb(images), real=False)
        fake_count += current_batch_size

    return fid.compute().item(), real_count

def append_fid_history(path, rows):
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        fieldnames = [
            "epoch",
            "weights",
            "modality",
            "guidance_scale",
            "num_samples",
            "fid",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

# ---------------------------- Training ----------------------------

def train_one_epoch(
    model,
    ema_model,
    loader,
    optimiser,
    schedule,
    device,
    latent_scale,
    scaler,
    use_amp,
    amp_dtype,
    decay,
    condition_dropout
):

    model.train()

    total_loss = torch.zeros((), device=device)
    num_samples = 0

    progress = tqdm(loader, desc="Train", leave=False)

    for step, batch in enumerate(progress):

        mu = batch["mu"].to(device, non_blocking=True, dtype=torch.float32)
        logvar = batch["logvar"].to(device, non_blocking=True, dtype=torch.float32)

        batch_size = mu.shape[0]
        std_dev = torch.exp(0.5*logvar)

        posterior_noise = torch.randn_like(std_dev)

        z_0 = (mu + std_dev * posterior_noise) * latent_scale
        modality = prepare_modality_condition(batch, model, device, condition_dropout)

        timesteps = torch.randint(
            low=0,
            high=schedule.num_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long
        )

        diffusion_noise = torch.randn_like(z_0)
        z_t = schedule.add_noise(z_0, diffusion_noise, timesteps)

        # Speed Up: channels Last latent
        z_t = z_t.contiguous(memory_format=torch.channels_last) 

        optimiser.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp
        ):
            noise_pred = model(z_t, timesteps, modality)

            loss = F.mse_loss(noise_pred, diffusion_noise)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        if ema_model is not None:
            update_ema(ema_model, model, decay)

        total_loss += loss.detach() * batch_size
        num_samples += batch_size

        if step % 20 == 0:
            progress.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss.item() / num_samples

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
    validation_generator = torch.Generator(device=device)
    validation_generator.manual_seed(seed)

    total_loss = torch.zeros((), device=device)
    num_samples = 0

    progress = tqdm(loader, desc="Validation", leave=False)

    for step, batch in enumerate(progress):

        mu = batch["mu"].to(device, non_blocking=True, dtype=torch.float32)
        logvar = batch["logvar"].to(device, non_blocking=True, dtype=torch.float32)
        
        batch_size = mu.shape[0]
        std_dev = torch.exp(0.5*logvar)
        
        posterior_noise = torch.randn(
            std_dev.shape, 
            generator=validation_generator,
            device=device,
            dtype=std_dev.dtype
        )
        
        z_0 = (mu + std_dev * posterior_noise) * latent_scale

        modality = batch["modality"].to(device, non_blocking=True, dtype=torch.long)

        timesteps = torch.randint(
            0,
            schedule.num_timesteps,
            (batch_size,),
            generator=validation_generator,
            device=device,
            dtype=torch.long
        )

        diffusion_noise = torch.randn(
            z_0.shape,
            generator=validation_generator,
            device=device,
            dtype=z_0.dtype
        )

        z_t = schedule.add_noise(z_0, diffusion_noise, timesteps)
        # Speed Up: channels Last latent
        z_t = z_t.contiguous(memory_format=torch.channels_last) 

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp
        ):
            noise_pred = model(z_t, timesteps, modality)

            loss = F.mse_loss(noise_pred, diffusion_noise)

        total_loss += loss.detach() * batch_size
        num_samples += batch_size

    return total_loss.item() / num_samples


def build_overfit_subset(
    config,
    cached_dataset,
    patient=None,
):
    raw_dataset = buildMRIDataset(
        config,
        split="train",
        stage="backbone"
    )

    print(f"Raw multimodal samples: {len(raw_dataset)}")
    print(f"Cached samples:         {len(cached_dataset)}")
    print(
        f"Ratio:                  "
        f"{len(raw_dataset) / len(cached_dataset):.2f}x"
    )

    if len(raw_dataset) != len(cached_dataset):
        raise RuntimeError(
            "Raw training dataset and cached latent dataset "
            "have different lengths. Cannot safely align indices."
        )

    patient_to_indices = {}

    for sample_idx, (
        volume_idx,
        modality,
        slice_idx
    ) in enumerate(raw_dataset.samples):

        row = raw_dataset._get_volume_row(
            volume_idx
        )

        subject = str(
            row["subject"]
        )

        if subject not in patient_to_indices:
            patient_to_indices[subject] = {
                "indices": [],
                "modalities": set()
            }

        patient_to_indices[
            subject
        ]["indices"].append(
            sample_idx
        )

        patient_to_indices[
            subject
        ]["modalities"].add(
            modality
        )

    expected_modalities = {
        "t1",
        "t1gd",
        "t2",
        "flair"
    }

    if patient is None:

        valid_patients = [
            subject
            for subject, info
            in patient_to_indices.items()
            if expected_modalities.issubset(
                info["modalities"]
            )
        ]

        if not valid_patients:
            raise RuntimeError(
                "No training patient with all four modalities found."
            )

        patient = valid_patients[0]

    if patient not in patient_to_indices:
        raise ValueError(
            f"Patient '{patient}' not found "
            "in the backbone training split."
        )

    info = patient_to_indices[
        patient
    ]

    missing = (
        expected_modalities
        - info["modalities"]
    )

    if missing:
        raise RuntimeError(
            f"Patient '{patient}' is missing "
            f"modalities: {sorted(missing)}"
        )

    indices = info["indices"]

    modality_counts = {}

    for idx in indices:
        _, modality, _ = raw_dataset.samples[idx]
        
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    print()
    print("=" * 50)
    print("OVERFIT MODE")
    print("=" * 50)
    print(
        f"Patient: {patient}"
    )
    print(
        f"Total samples: {len(indices)}"
    )

    for modality in sorted(
        modality_counts
    ):
        print(
            f"{modality}: "
            f"{modality_counts[modality]} slices"
        )

    print("=" * 50)
    print()

    return Subset(
        cached_dataset,
        indices
    )


def main(args):

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError("Diffusion training requires CUDA for this training script.")

    # Speed Ups

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(config["training"].get("seed", 42))

    torch.cuda.manual_seed_all(config["training"].get("seed", 42))

    train_dataset = CachedBackboneDataset(config["data"]["cached_train_dir"])

    raw_val_dataset = buildMRIDataset(
        config=config,
        split="val",
        stage="backbone",
    )
    val_dataset = CachedBackboneDataset(config["data"]["cached_val_dir"])

    train_sampler = ShardShuffleSampler(train_dataset, config["training"].get("seed", 42))

    if args.overfit:
        train_dataset = build_overfit_subset(
            config,
            train_dataset,
            args.overfit_patient
        )
        val_dataset = train_dataset
        # FID remains a true validation-set metric even during an overfit run.

    batch_size = config["training"]["batch_size"]
    drop_last = False
    if args.overfit:
        batch_size = 1

    num_workers = config["training"].get("num_workers", 4)
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=drop_last
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False
    )

    evaluation_config = config.get("evaluation", {})
    evaluation_batch_size = evaluation_config.get(
        "batch_size",
        batch_size
    )
    raw_val_loader = DataLoader(
        raw_val_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    diffusion_config = config["diffusion"]

    model = build_diffusion(config).to(device)

    condition_dropout = diffusion_config.get("condition_dropout", 0.1)

    # Speed Up: Diff model Channels last
    model = model.to(memory_format=torch.channels_last)

    if config.get("ema", {}).get("enabled", True):
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        ema_model.requires_grad_(False)
    else:
        ema_model = None
    decay = config.get("ema", {}).get("decay", 0.9999)
    if ema_model is not None:
        print(f"EMA enabled: decay={decay}")
    else:
        print("EMA disabled")

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Diffusion parameters: "
        f"{num_parameters / 1e6:.2f}M"
    )

    schedule = DDPMSchedule(
        num_timesteps=diffusion_config.get("num_timesteps", 1000),
        beta_start=diffusion_config.get("beta_start", 1e-4),
        beta_end=diffusion_config.get("beta_end", 2e-2),
        device=device
    )

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].get("learning_rate", 1e-4),
        weight_decay=config["training"].get("weight_decay", 1e-4),
        fused=True
    )

    # Speed up: Mixed Precision and BF16
    use_amp = bool(config["training"].get("mixed_precision", True))

    amp_dtype_name = config["training"].get("amp_dtype", "bf16")

    if amp_dtype_name == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_dtype_name == "fp16":
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unknown amp_dtype: {amp_dtype_name}")

    if (amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported()):
        print("BF16 unsupported; falling back to FP16")
        amp_dtype = torch.float16

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    print(
        f"AMP: {use_amp}, "
        f"dtype: {amp_dtype}"
    )

    output_dir = Path(
        config["training"].get("output_dir", "outputs/diffusion")
    )

    checkpoint_dir = (output_dir / "checkpoints")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_path = (output_dir / "training_history.csv")

    start_epoch = 0
    best_val_loss = float("inf")
    latent_scale = None

    if args.resume is not None:
        start_epoch, best_val_loss, latent_scale = load_diffusion_checkpoint(
                                                    args.resume,
                                                    model,
                                                    ema_model,
                                                    optimiser,
                                                    scaler,
                                                    device
                                                )
        fine_tuning_lr = config["training"].get(
            "fine_tuning_learning_rate",
            5e-6,
        )
        for param_group in optimiser.param_groups:
            param_group["lr"] = fine_tuning_lr
        print(f"Fine-tuning learning rate: {fine_tuning_lr:.2e}")
        print(
            f"Resumed from epoch "
            f"{start_epoch}"
        )

        print(
            f"Latent scale: "
            f"{latent_scale:.6f}"
        )

    if config["training"].get("compile", True):
        model = torch.compile(
            model,
            mode=config["training"].get("compile_mode", "reduce-overhead")
        )

    if latent_scale is None:

        latent_scale = estimate_latent_scale(
            train_loader,
            device,
            num_batches=config["training"].get("latent_scale_batches", 100)
        )

    vae = load_vae(
        config=config,
        checkpoint_path=args.vae_checkpoint,
        device=device
    )

    latent_shape = tuple(train_dataset[0]["mu"].shape)
    sample_modalities = evaluation_config.get(
        "modalities",
        config["data"].get("modalities", list(MODALITY_TO_ID)),
    )
    unknown_modalities = [
        name for name in sample_modalities if name not in MODALITY_TO_ID
    ]
    if unknown_modalities:
        raise ValueError(
            f"Unknown evaluation modalities: {unknown_modalities}"
        )

    guidance_scale = float(
        evaluation_config.get(
            "guidance_scale",
            diffusion_config.get("guidance_scale", 1.0),
        )
    )
    sample_every = int(evaluation_config.get("sample_every", 1))
    fid_every = int(evaluation_config.get("fid_every", 1))
    fid_samples = int(evaluation_config.get("fid_samples", 256))
    fid_batch_size = int(
        evaluation_config.get("generation_batch_size", 16)
    )
    fid_feature = int(evaluation_config.get("fid_feature", 2048))
    evaluation_seed = int(evaluation_config.get("seed", 1234))
    validation_loss_seed = int(evaluation_config.get("validation_loss_seed", 4321))

    if sample_every < 1 or fid_every < 1:
        raise ValueError("sample_every and fid_every must both be at least 1.")
    if fid_samples < 2:
        raise ValueError("evaluation.fid_samples must be at least 2.")
    if fid_feature not in {64, 192, 768, 2048}:
        raise ValueError(
            "evaluation.fid_feature must be one of 64, 192, 768, or 2048."
        )
    if guidance_scale != 1.0 and condition_dropout <= 0.0:
        raise RuntimeError(
            "Classifier-free guidance above/below 1 requires unconditional "
            "training examples. Set diffusion.condition_dropout to a value "
            "such as 0.1."
        )
    if ema_model is None:
        raise RuntimeError(
            "EMA must be enabled because this run is configured to save and "
            "evaluate both raw and EMA samples. Set ema.enabled: true."
        )

    samples_dir = output_dir / "samples"
    raw_samples_dir = samples_dir / "raw"
    ema_samples_dir = samples_dir / "ema"
    raw_samples_dir.mkdir(parents=True, exist_ok=True)
    ema_samples_dir.mkdir(parents=True, exist_ok=True)
    fid_history_path = output_dir / "validation_fid.csv"

    num_epochs = config["training"]["num_epochs"]

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        print(
            f"\nEpoch "
            f"{epoch + 1}/{num_epochs}"
        )

        train_loss = train_one_epoch(
            model=model,
            ema_model=ema_model,
            loader=train_loader,
            optimiser=optimiser,
            schedule=schedule,
            device=device,
            latent_scale=latent_scale,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            decay=decay,
            condition_dropout=condition_dropout
        )

        val_loss = validate(
            model=model,
            loader=val_loader,
            schedule=schedule,
            device=device,
            latent_scale=latent_scale,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            seed=validation_loss_seed
        )

        learning_rate = optimiser.param_groups[0]["lr"]

        print(f"Train loss: {train_loss:.6f}")

        print(f"Val loss:   {val_loss:.6f}")

        append_history(
            history_path,
            epoch + 1,
            train_loss,
            val_loss,
            learning_rate
        )
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        latest_path = checkpoint_dir / "latest.pt"

        save_checkpoint(
            latest_path,
            model,
            ema_model,
            optimiser,
            scaler,
            epoch,
            best_val_loss,
            latent_scale,
            config
        )

        if is_best:

            best_path = checkpoint_dir/ "best.pt"

            save_checkpoint(
                best_path,
                model,
                ema_model,
                optimiser,
                scaler,
                epoch,
                best_val_loss,
                latent_scale,
                config
            )

            print(
                f"New best validation loss: "
                f"{best_val_loss:.6f}"
            )

        epoch_number = epoch + 1

        if epoch_number % sample_every == 0:
            for modality_offset, modality_name in enumerate(sample_modalities):
                sample_seed = evaluation_seed + modality_offset * 100_000

                generate_sample_grid(
                    model=model,
                    vae=vae,
                    schedule=schedule,
                    latent_shape=latent_shape,
                    latent_scale=latent_scale,
                    modality_name=modality_name,
                    guidance_scale=guidance_scale,
                    device=device,
                    seed=sample_seed,
                    output_path=(
                        raw_samples_dir
                        / f"epoch_{epoch_number:04d}_{modality_name}.png"
                    ),
                )
                generate_sample_grid(
                    model=ema_model,
                    vae=vae,
                    schedule=schedule,
                    latent_shape=latent_shape,
                    latent_scale=latent_scale,
                    modality_name=modality_name,
                    guidance_scale=guidance_scale,
                    device=device,
                    seed=sample_seed,
                    output_path=(
                        ema_samples_dir
                        / f"epoch_{epoch_number:04d}_{modality_name}.png"
                    ),
                )

            print(
                f"Saved raw and EMA 4x4 sample grids for epoch "
                f"{epoch_number}."
            )

        if epoch_number % fid_every == 0 or epoch_number == num_epochs:
            fid_rows = []
            for modality_offset, modality_name in enumerate(sample_modalities):
                fid_seed = evaluation_seed + modality_offset * 100_000
                for weights_name, evaluation_model in (
                    ("raw", model),
                    ("ema", ema_model),
                ):
                    fid_score, real_count = compute_validation_fid(
                        model=evaluation_model,
                        vae=vae,
                        schedule=schedule,
                        raw_val_loader=raw_val_loader,
                        latent_shape=latent_shape,
                        latent_scale=latent_scale,
                        modality_name=modality_name,
                        guidance_scale=guidance_scale,
                        device=device,
                        num_samples=fid_samples,
                        generation_batch_size=fid_batch_size,
                        fid_feature=fid_feature,
                        seed=fid_seed,
                    )
                    fid_rows.append(
                        {
                            "epoch": epoch_number,
                            "weights": weights_name,
                            "modality": modality_name,
                            "guidance_scale": guidance_scale,
                            "num_samples": real_count,
                            "fid": fid_score,
                        }
                    )
                    print(
                        f"Validation FID [{weights_name}, {modality_name}] "
                        f"at epoch {epoch_number}: {fid_score:.4f} "
                        f"({real_count} samples)"
                    )

            append_fid_history(fid_history_path, fid_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--overfit-patient", type=str, default=None)
    args = parser.parse_args()
    main(args)
