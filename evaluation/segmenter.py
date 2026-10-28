from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .utils import (
    MODALITIES,
    check_mask,
    digest,
    load_cases,
    region_metrics,
    summarise,
    write_csv,
    write_json
)


# --------------------- Segmentation Blocks ---------------------

class ConvBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))

        return x


# --------------------- UNet Architecture ---------------------

class UNet2D(nn.Module):
    def __init__(
        self,
        width: int = 32
    ):
        super().__init__()

        self.width = width
        ch1 = width
        ch2 = width * 2
        ch3 = width * 4
        ch4 = width * 8
        ch5 = width * 16

        self.ch1 = ch1
        self.ch2 = ch2
        self.ch3 = ch3
        self.ch4 = ch4
        self.ch5 = ch5

        # --------------------- Encoder Blocks ---------------------
        self.enc1_block = ConvBlock2D(1, ch1)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2_block = ConvBlock2D(ch1, ch2)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3_block = ConvBlock2D(ch2, ch3)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4_block = ConvBlock2D(ch3, ch4)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --------------------- Mid Block ---------------------
        self.mid_block = ConvBlock2D(ch4, ch5)

        # --------------------- Decoder Blocks ---------------------
        self.dec1_block = ConvBlock2D(ch5 + ch4, ch4)
        self.dec2_block = ConvBlock2D(ch4 + ch3, ch3)
        self.dec3_block = ConvBlock2D(ch3 + ch2, ch2)
        self.dec4_block = ConvBlock2D(ch2 + ch1, ch1)

        self.out_conv = nn.Conv2d(ch1, 4, kernel_size=1)

    def forward(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        x = self.enc1_block(image)
        skip1 = x
        x = self.down1(x)

        x = self.enc2_block(x)
        skip2 = x
        x = self.down2(x)

        x = self.enc3_block(x)
        skip3 = x
        x = self.down3(x)

        x = self.enc4_block(x)
        skip4 = x
        x = self.down4(x)

        x = self.mid_block(x)

        # Match each skip grid exactly, including odd input dimensions.
        x = F.interpolate(x, size=skip4.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip4], dim=1)
        x = self.dec1_block(x)

        x = F.interpolate(x, size=skip3.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec2_block(x)

        x = F.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec3_block(x)

        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec4_block(x)

        return self.out_conv(x)


# --------------------- Data Loading / Checking ---------------------

def normalize(image):
    """Apply identical image-only normalization to real and generated MRI."""
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2 or min(image.shape) < 32 or not np.isfinite(image).all():
        raise ValueError("Expected finite 2D MRI, at least 32 pixels per side")

    # Minimum-intensity pixels define the background; no target mask is used.
    foreground = image > image.min()
    values = image[foreground]

    if values.size < 2:
        return np.zeros_like(image)

    lower, upper = np.percentile(values, [1, 99])
    clipped = np.clip(image, lower, upper)
    mean = clipped[foreground].mean()
    std = max(float(clipped[foreground].std()), 1e-6)

    normalized = (clipped - mean) / std
    normalized[~foreground] = 0

    return normalized.astype(np.float32)


class RealSlices(Dataset):
    def __init__(
        self,
        manifest,
        modality,
        split,
        augment=False
    ):
        self.cases = load_cases(manifest)
        self.modality = modality
        self.augment = augment

        root = Path(manifest).resolve().parent

        for case in self.cases:
            if case["source"] != "real" or case["split"] != split:
                raise ValueError(f"Expected real {split} cases only")

            case["image_path"] = root / case["images"][modality]
            image = np.load(case["image_path"], allow_pickle=False)

            if image.shape != case["array"].shape:
                raise ValueError("MRI and mask must share a grid")

            normalize(image)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        image = normalize(np.load(case["image_path"], allow_pickle=False))
        target = case["array"].copy().astype(np.int64)

        # Convert raw mask labels 0/1/2/4 to network class indices 0/1/2/3.
        target[target == 4] = 3

        if self.augment:
            for axis in (0, 1):
                if torch.rand(()) < 0.5:
                    image = np.flip(image, axis)
                    target = np.flip(target, axis)

        return (
            torch.from_numpy(image.copy()).unsqueeze(0),
            torch.from_numpy(target.copy())
        )


def segmenter_stages(config):
    settings = config.get("segmenter", {})
    if "stages" in settings and "stage" in settings:
        raise ValueError("Use segmenter.stages, not both stages and legacy stage")
    stages = settings.get("stages", [settings["stage"]] if "stage" in settings else ["backbone", "controlnet"])
    if (
        not isinstance(stages, list) or not stages
        or any(stage not in ("backbone", "controlnet") for stage in stages)
        or len(set(stages)) != len(stages)
    ):
        raise ValueError("segmenter.stages must list backbone and/or controlnet without duplicates")
    return stages


def tumour_slice_flags(mask_path, slice_axis, image_size):
    """Read a mask once and preserve the loader's post-resize tumour criterion."""
    from src.data.load_masks import load_mask_volume

    volume = load_mask_volume(str(mask_path))
    slices = np.moveaxis(volume, slice_axis, 0)
    flags = np.zeros(len(slices), dtype=bool)
    # Bound temporary memory; use the exact nearest-neighbour resize used by
    # _load_segmentation, including small lesions that disappear when resized.
    for start in range(0, len(slices), 32):
        batch = torch.from_numpy(slices[start:start + 32].copy()).float()[:, None]
        resized = F.interpolate(batch, size=(image_size, image_size), mode="nearest")
        flags[start:start + len(batch)] = (resized > 0).flatten(1).any(1).numpy()
    return flags


class SplitFileSlices(Dataset):
    """Wrap the same buildMRIDataset factory used by train_diff_backbone."""
    def __init__(self, config, modality, split, augment=False):
        from src.data.load_brain import buildMRIDataset
        from src.data.load_masks import patient_id

        config = copy.deepcopy(config)
        data = config["data"]
        split_path = Path(data["split_file"])
        if not split_path.is_file():
            # The shared factory can create splits; evaluation must never do so.
            raise FileNotFoundError(split_path)

        with split_path.open() as handle:
            partitions = json.load(handle)
        patient_partitions = {}
        for stage in ("backbone", "controlnet"):
            for partition in ("train", "val", "test"):
                for scan_key in partitions[stage][partition]:
                    patient = patient_id(scan_key)
                    previous = patient_partitions.setdefault(patient, partition)
                    if previous != partition:
                        raise ValueError(f"Patient {patient} appears in both {previous} and {partition}")

        if "root" not in data:
            data["root"] = str(Path(data["mask_manifest"]).parent.parent)
        if data.get("mask_manifest"):
            expected = Path(data["root"]) / "tumor_masks_upenn_ucsf" / "tumor_masks_manifest.csv"
            if Path(data["mask_manifest"]).resolve() != expected.resolve():
                raise ValueError("buildMRIDataset expects mask_manifest under data.root/tumor_masks_upenn_ucsf")

        data["modalities"] = [modality]
        self.modality = modality
        self.augment = augment
        self.stages = segmenter_stages(config)
        self.datasets = []
        self.indices = []
        self.cases = []
        seen = set()
        tumour_only = bool(data.get("tumour_only", True))
        # Retain only small Boolean vectors here, not every patient's volume.
        mask_flags = {}

        for dataset_index, stage in enumerate(self.stages):
            print(f"\nBuilding {stage}/{split} MRI dataset ({modality})...", flush=True)
            dataset = buildMRIDataset(config=config, split=split, stage=stage)
            # Enable paired targets for backbone data as well as ControlNet.
            dataset.include_segmentation = True
            self.datasets.append(dataset)
            stage_start = len(self.indices)
            previous_volume = None
            description = "Filtering tumour slices" if tumour_only else "Registering slices"
            with tqdm(
                dataset.samples,
                total=len(dataset.samples),
                desc=f"{stage}/{split}: {description}",
                unit="slice",
                mininterval=0.5,
                dynamic_ncols=True
            ) as progress:
                for index, (volume_index, _, slice_index) in enumerate(progress):
                    row = dataset._get_volume_row(volume_index)
                    # Refresh before loading a new patient's mask so a slow
                    # read can be traced to the displayed scan identifier.
                    progress.set_postfix(
                        scan=str(row["subject_prefix"]),
                        kept=len(self.indices) - stage_start,
                        refresh=volume_index != previous_volume
                    )
                    previous_volume = volume_index
                    scan_key = f"{row['cohort']}::{row['subject_prefix']}"
                    case_id = f"{scan_key}::z{slice_index}"
                    if case_id in seen:
                        continue
                    seen.add(case_id)
                    if tumour_only:
                        mask_path = str(row["mask_path"])
                        if mask_path not in mask_flags:
                            mask_flags[mask_path] = tumour_slice_flags(
                                mask_path,
                                int(data.get("slice_axis", 2)),
                                int(data.get("image_size", 256))
                            )
                        if not mask_flags[mask_path][slice_index]:
                            continue
                    self.indices.append((dataset_index, index))
                    self.cases.append({
                        "case_id": case_id, "stage": stage,
                        "subject": patient_id(scan_key), "source": "real", "split": split
                    })
                progress.set_postfix(kept=len(self.indices) - stage_start)
            print(
                f"{stage}/{split}: kept {len(self.indices) - stage_start:,} "
                f"of {len(dataset.samples):,} candidate slices.",
                flush=True
            )
        if not self.indices:
            raise ValueError(f"No eligible {self.stages}/{split} MRI-mask slices")
        self.cache_refs = None
        if data.get("segmenter_cache_dir"):
            self.prepare_slice_cache(data["segmenter_cache_dir"], split)

    def prepare_slice_cache(self, cache_root, split):
        from .slice_cache import prepare_volume_cache, SliceCacheReader

        groups = defaultdict(list)
        for index, (dataset_index, sample_index) in enumerate(self.indices):
            volume_index, _, slice_index = self.datasets[dataset_index].samples[sample_index]
            groups[(dataset_index, volume_index)].append((index, slice_index))
        self.cache_refs = [None] * len(self.indices)
        print(f"Preparing/reusing {split} slice cache: {cache_root}", flush=True)
        with tqdm(groups.items(), total=len(groups), desc=f"{split}: slice cache", unit="volume") as progress:
            for (dataset_index, volume_index), entries in progress:
                dataset = self.datasets[dataset_index]
                row = dataset._get_volume_row(volume_index)
                progress.set_postfix(scan=str(row["subject_prefix"]))
                path = prepare_volume_cache(dataset, row, self.modality, [s for _, s in entries], cache_root)
                for offset, (index, _) in enumerate(entries):
                    self.cache_refs[index] = (str(path), offset)
        self.cache_reader = SliceCacheReader()

    def __len__(self):
        return len(self.indices)

    def image_and_target(self, index):
        if self.cache_refs is not None:
            return self.cache_reader.read(*self.cache_refs[index])
        dataset_index, sample_index = self.indices[index]
        sample = self.datasets[dataset_index][sample_index]
        image = sample["image"].squeeze(0).numpy()
        target = check_mask(sample["seg"].squeeze(0).numpy())
        if image.shape != target.shape:
            raise ValueError("MRI and target must share a grid")
        return image, target

    def __getitem__(self, index):
        image, target = self.image_and_target(index)
        image = normalize(image)
        target = target.astype(np.int64)
        target[target == 4] = 3
        if self.augment:
            for axis in (0, 1):
                if torch.rand(()) < 0.5:
                    image = np.flip(image, axis)
                    target = np.flip(target, axis)
        return torch.from_numpy(image.copy())[None], torch.from_numpy(target.copy()).long()


def build_loaders(args):
    if getattr(args, "data_config", None) is not None:
        train_dataset = SplitFileSlices(args.data_config, args.modality, "train", augment=True)
        val_dataset = SplitFileSlices(args.data_config, args.modality, "val")
    else:
        train_dataset = RealSlices(
            manifest=args.train_manifest,
            modality=args.modality,
            split="train",
            augment=True
        )
        val_dataset = RealSlices(
            manifest=args.val_manifest,
            modality=args.modality,
            split="val"
        )

    train_subjects = {case["subject"] for case in train_dataset.cases}
    val_subjects = {case["subject"] for case in val_dataset.cases}

    if train_subjects & val_subjects:
        raise ValueError("Patient leakage between segmenter train and validation")

    workers = getattr(args, "num_workers", 0)
    options = {
        "num_workers": workers,
        "pin_memory": torch.device(args.device).type == "cuda",
        "persistent_workers": workers > 0 and getattr(args, "persistent_workers", True)
    }
    if workers > 0:
        options["prefetch_factor"] = getattr(args, "prefetch_factor", 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **options
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=getattr(args, "validation_batch_size", args.batch_size),
        shuffle=False,
        **options,
        generator=torch.Generator().manual_seed(0)
    )

    return train_loader, val_loader


# --------------------- Training / Validation Logic ---------------------

def build_segmentation_components(args, device):
    model = UNet2D(width=32).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", .01)
    )

    print(
        "Segmentation parameters:",
        f"{sum(parameter.numel() for parameter in model.parameters()):,}",
        flush=True
    )

    return model, optimizer


def segmentation_loss(logits, target):
    ce_loss = F.cross_entropy(logits, target)
    probabilities = logits.softmax(dim=1)
    truth = F.one_hot(target, num_classes=4).permute(0, 3, 1, 2)

    axes = (2, 3)
    intersection = (probabilities * truth).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + truth.sum(dim=axes)
    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)

    # CE supervises absent classes; foreground Dice uses target-present classes.
    present = truth.sum(dim=axes)[:, 1:] > 0
    if present.any():
        dice_loss = (1 - dice[:, 1:])[present].mean()
    else:
        dice_loss = logits.sum() * 0

    return ce_loss + dice_loss


def train_epoch(
    model,
    loader,
    optimizer,
    device
):
    model.train()
    losses = []
    data_wait = 0.0
    step_time = 0.0
    previous_end = perf_counter()

    progress = tqdm(loader, desc="Train", leave=False)

    for step, (image, target) in enumerate(progress):
        started = perf_counter()
        data_wait += started - previous_end
        image = image.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(image)
        loss = segmentation_loss(logits, target)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach()))
        step_time += perf_counter() - started

        if step % 20 == 0:
            progress.set_postfix(loss=f"{losses[-1]:.5f}",
                                 data_s=f"{data_wait / (step + 1):.3f}",
                                 step_s=f"{step_time / (step + 1):.3f}")
        previous_end = perf_counter()

    if not losses:
        raise ValueError("Training loader produced no samples")

    return {"loss": float(np.mean(losses)), "data_wait_seconds": data_wait,
            "step_seconds": step_time, "batches": len(losses)}


@torch.inference_mode()
def validate(
    model,
    loader,
    device
):
    model.eval()
    dataset = loader.dataset
    records = []
    index = 0

    # Validation loaders are sequential so predictions map to their case IDs.
    progress = tqdm(loader, desc="Validation", leave=False)

    for images, targets in progress:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        predictions = model(images).argmax(dim=1).cpu().numpy().astype(np.uint8)
        predictions[predictions == 3] = 4

        for prediction, target in zip(predictions, targets.numpy()):
            target = target.copy()
            target[target == 3] = 4
            records.append({
                "plan_index": index,
                "modality": dataset.modality,
                "metrics": region_metrics(target, prediction)
            })
            index += 1

    plan = [case | {"variant": "real_reference"} for case in dataset.cases]
    rows = summarise(records, plan)

    return records, rows


def validate_model(model, dataset, device):
    """Keep the earlier dataset-based validation entry point available."""
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        generator=torch.Generator().manual_seed(0)
    )
    return validate(model, loader, device)


# --------------------- Checkpointing ---------------------

def load_segmenter_state_dict(model, state):
    """Load explicit block names or remap checkpoints from the earlier layout."""
    if any(key.startswith("down.0.") for key in state):
        block_names = {
            "down.0": "enc1_block",
            "down.1": "enc2_block",
            "down.2": "enc3_block",
            "down.3": "enc4_block",
            "down.4": "mid_block",
            "up.0": "dec1_block",
            "up.1": "dec2_block",
            "up.2": "dec3_block",
            "up.3": "dec4_block"
        }
        layer_names = {"0": "conv1", "1": "norm1", "3": "conv2", "4": "norm2"}
        renamed = {}

        for key, value in state.items():
            parts = key.split(".")
            if parts[0] == "head":
                new_key = ".".join(["out_conv", *parts[1:]])
            else:
                block = block_names[".".join(parts[:2])]
                layer = layer_names[parts[2]]
                new_key = ".".join([block, layer, *parts[3:]])

            renamed[new_key] = value

        state = renamed

    model.load_state_dict(state, strict=True)


def save_checkpoint(
    path,
    model,
    epoch,
    validation_dice,
    args,
    train_subjects,
    val_subjects
):
    state = {
        "model": model.state_dict(),
        "model_layout": "explicit_blocks_v1",
        "width": model.width,
        "modality": args.modality,
        "normalization": "foreground_robust_z_v1",
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "epoch": epoch,
        "validation_dice": validation_dice,
        "seed": args.seed,
    }
    if getattr(args, "data_config", None) is not None:
        data = args.data_config["data"]
        state["split_file_sha256"] = digest(data["split_file"])
        manifest = data.get("mask_manifest") or str(
            Path(data["root"]) / "tumor_masks_upenn_ucsf" / "tumor_masks_manifest.csv"
        )
        state["mask_manifest_sha256"] = digest(manifest)
        state["data_config"] = data
        state["split_stages"] = segmenter_stages(args.data_config)
    else:
        state["train_manifest_sha256"] = digest(args.train_manifest)
        state["val_manifest_sha256"] = digest(args.val_manifest)

    path = Path(path)
    temporary_path = path.with_suffix(".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


# --------------------- Inference / Real-Image Evaluation ---------------------

def make_segmenter(config):
    device = torch.device(config.get("device", "cuda"))
    models = {}
    metadata = {}

    for modality, path in config["checkpoints"].items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        if (
            checkpoint["modality"] != modality
            or checkpoint["normalization"] != "foreground_robust_z_v1"
        ):
            raise ValueError("Checkpoint modality or preprocessing mismatch")

        model = UNet2D(width=checkpoint["width"]).to(device)
        load_segmenter_state_dict(model, checkpoint["model"])
        model.requires_grad_(False)
        model.eval()

        models[modality] = model
        metadata[modality] = {
            "sha256": digest(path),
            "train_subjects": checkpoint["train_subjects"],
            "val_subjects": checkpoint["val_subjects"],
            "split_file_sha256": checkpoint.get("split_file_sha256"),
            "split_stages": checkpoint.get("split_stages"),
            "normalization": checkpoint["normalization"]
        }

    @torch.inference_mode()
    def predict(image, modality):
        if modality not in models:
            raise ValueError(f"No independently trained {modality} segmenter checkpoint")

        image_tensor = torch.from_numpy(normalize(image))[None, None].to(device)
        logits = models[modality](image_tensor)
        labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        labels[labels == 3] = 4

        return check_mask(labels)

    predict.metadata = metadata
    return predict


def score_real(args):
    using_splits = getattr(args, "data_config", None) is not None
    dataset = (
        SplitFileSlices(args.data_config, args.modality, "test")
        if using_splits else RealSlices(args.manifest, args.modality, "test")
    )
    predictor = make_segmenter({
        "device": args.device,
        "checkpoints": {args.modality: args.checkpoint}
    })
    metadata = predictor.metadata[args.modality]

    test_subjects = {case["subject"] for case in dataset.cases}
    used_subjects = set(metadata["train_subjects"] + metadata["val_subjects"])

    if test_subjects & used_subjects:
        raise ValueError("Real test patients overlap segmenter train/validation")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []

    for index, case in enumerate(dataset.cases):
        if using_splits:
            image, target = dataset.image_and_target(index)
        else:
            image = np.load(case["image_path"], allow_pickle=False)
            target = case["array"]
        prediction = predictor(image, args.modality)
        records.append({
            "plan_index": index,
            "modality": args.modality,
            "metrics": region_metrics(target, prediction)
        })

    plan = [case | {"variant": "real_reference"} for case in dataset.cases]

    write_json(output_dir / "metrics.json", records)
    write_csv(output_dir / "summary.csv", summarise(records, plan))
    write_json(output_dir / "provenance.json", {
        "checkpoint": metadata,
        "manifest_sha256": digest(
            args.data_config["data"]["split_file"] if using_splits else args.manifest
        )
    })


# --------------------- Argument Loading ---------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--train-manifest")
    train_parser.add_argument("--val-manifest")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--lr", type=float)
    train_parser.add_argument("--seed", type=int)
    train_parser.set_defaults(func=train)

    score_parser = commands.add_parser("score-real")
    score_parser.add_argument("--manifest")
    score_parser.add_argument("--checkpoint", required=True)
    score_parser.set_defaults(func=score_real)

    for command in (train_parser, score_parser):
        command.add_argument("--config")
        command.add_argument("--modality", choices=MODALITIES)
        command.add_argument("--device", default="cuda")
        command.add_argument("--output")

    args = parser.parse_args()
    config = {}
    if args.config:
        import yaml
        with open(args.config) as handle:
            config = yaml.safe_load(handle)
        args.data_config = config
        if any(getattr(args, name, None) for name in ("manifest", "train_manifest", "val_manifest")):
            parser.error("Use --config with the existing split file, or legacy manifests, not both")
    settings = config.get("segmenter", {})
    args.modality = args.modality or settings.get("modality")
    if args.modality not in MODALITIES:
        parser.error("Choose --modality or segmenter.modality")
    configured_output = settings.get("output_dir")
    if configured_output and args.func == score_real:
        configured_output = str(configured_output) + "_real_test"
    args.output = args.output or configured_output
    if args.output is None:
        args.output = f"outputs/segmenter_{args.modality}" + ("_real_test" if args.func == score_real else "")
    if args.func == train:
        if not args.config and not (args.train_manifest and args.val_manifest):
            parser.error("Supply --config (data.split_file), or both legacy manifests")
        training = config.get("training", {}) | settings.get("training", {})
        for name, key, default in (("epochs", "epochs", 100), ("batch_size", "batch_size", 8),
                                   ("lr", "learning_rate", 1e-4), ("seed", "seed", config.get("seed", 17))):
            if getattr(args, name) is None:
                setattr(args, name, training.get(key, default))
        args.num_workers = int(training.get("num_workers", 0))
        args.weight_decay = float(training.get("weight_decay", .01))
        args.prefetch_factor = int(training.get("prefetch_factor", 2))
        args.persistent_workers = bool(training.get("persistent_workers", True))
        args.validation_batch_size = int(training.get("validation_batch_size", args.batch_size))
    elif not args.config and not args.manifest:
        parser.error("Supply --config (data.split_file), or a legacy --manifest")
    return args


# --------------------- Main Loop ---------------------

def train(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    train_loader, val_loader = build_loaders(args)
    train_subjects = sorted({case["subject"] for case in train_loader.dataset.cases})
    val_subjects = sorted({case["subject"] for case in val_loader.dataset.cases})

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"Device: {device}", flush=True)
    print("Training slices:", len(train_loader.dataset), flush=True)
    print("Validation slices:", len(val_loader.dataset), flush=True)

    model, optimizer = build_segmentation_components(args, device)
    best_val_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)

        train_results = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device
        )

        _, val_rows = validate(
            model=model,
            loader=val_loader,
            device=device
        )

        scores = [
            row["subject_macro_dice"]
            for row in val_rows
            if row["subject_macro_dice"] is not None
        ]
        if not scores:
            raise ValueError("Validation contains no present tumour regions")

        score = float(np.mean(scores))
        history.append({
            "epoch": epoch,
            "train_loss": train_results["loss"],
            "val_macro_dice": score,
            "data_wait_seconds": train_results["data_wait_seconds"],
            "step_seconds": train_results["step_seconds"]
        })
        write_json(output_dir / "history.json", history)

        if score > best_val_dice:
            best_val_dice = score
            save_checkpoint(
                path=output_dir / "best.pt",
                model=model,
                epoch=epoch,
                validation_dice=score,
                args=args,
                train_subjects=train_subjects,
                val_subjects=val_subjects
            )
            write_csv(output_dir / "best_validation.csv", val_rows)

        print(
            f"Train loss: {train_results['loss']:.6f} | "
            f"Val macro Dice: {score:.4f}",
            flush=True
        )


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()