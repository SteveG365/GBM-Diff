import bisect
import csv
import json
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F
from functools import lru_cache
from torch.utils.data import Dataset, Sampler
import nibabel as nib


def prepare_mask_slice(
    mask_slice: np.ndarray,
    image_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a 2D mask into one-hot input and integer target."""

    if mask_slice.ndim != 2:
        raise ValueError(
            f"Expected a 2D slice, got shape {mask_slice.shape}"
        )

    if not np.isin(mask_slice, [0, 1, 2, 4]).all():
        raise ValueError(
            f"Unexpected mask labels: {np.unique(mask_slice)}"
        )

    target = torch.from_numpy(mask_slice.copy()).long()
    target[target == 4] = 3     # deviation from BraTs convention, need 0,1,2,3 for VAE

    target = F.interpolate(
        target[None, None].float(),
        size=(image_size, image_size),
        mode="nearest"
    )[0, 0].long()

    encoder_input = F.one_hot(
        target,
        num_classes=4
    ).permute(2, 0, 1).float()

    return encoder_input, target

def restore_mask_labels(class_indices: torch.Tensor) -> torch.Tensor:
    """Convert internal class indices 0/1/2/3 back to labels 0/1/2/4."""

    label_values = torch.tensor(
        [0, 1, 2, 4],
        device=class_indices.device,
        dtype=torch.long,
    )

    return label_values[class_indices]



@lru_cache(maxsize=2)
def load_mask_volume(mask_path: str) -> np.ndarray:
    """Loads a Volume"""
    volume = np.asarray(nib.load(mask_path).dataobj)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D mask, got {volume.shape}: {mask_path}"
        )

    if not np.isin(volume, [0, 1, 2, 4]).all():
        raise ValueError(
            f"Unexpected labels in {mask_path}: {np.unique(volume)}"
        )

    return volume.astype(np.uint8)

class TumourMaskSliceDataset(Dataset):
    def __init__(
        self,
        mask_paths: list[str | Path],
        image_size: int = 256,
        slice_axis: int = 2,
        tumour_only: bool = True
    ):
        self.mask_paths = [
            str(Path(path).expanduser().resolve())
            for path in mask_paths
        ]

        if not self.mask_paths:
            raise ValueError("No mask paths supplied")

        if len(set(self.mask_paths)) != len(self.mask_paths):
            raise ValueError("Duplicate mask paths supplied")

        if slice_axis not in (0, 1, 2):
            raise ValueError("slice_axis must be 0, 1, or 2")

        self.image_size = image_size
        self.slice_axis = slice_axis
        self.tumour_only = tumour_only

        # Each entry is (volume_index, slice_index).
        self.samples = self._build_slice_index()

        if not self.samples:
            raise ValueError("No eligible slices found")

    def _build_slice_index(self):
        samples = []

        for volume_idx, mask_path in enumerate(self.mask_paths):
            volume = load_mask_volume(mask_path)

            # Reduce over the two axes within each slice.
            spatial_axes = tuple(
                axis
                for axis in range(3)
                if axis != self.slice_axis
            )

            foreground_counts = np.count_nonzero(volume, axis=spatial_axes)

            if self.tumour_only:
                slice_indices = np.flatnonzero(foreground_counts > 0)
            else:
                slice_indices = range(volume.shape[self.slice_axis])

            for slice_idx in slice_indices:
                samples.append((volume_idx, int(slice_idx)))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        volume_idx, slice_idx = self.samples[index]

        mask_path = self.mask_paths[volume_idx]
        volume = load_mask_volume(mask_path)

        mask_slice = np.take(volume, slice_idx, axis=self.slice_axis)

        encoder_input, target = prepare_mask_slice(mask_slice, image_size=self.image_size)

        depth = volume.shape[self.slice_axis]

        return {
            "image": encoder_input,
            "target": target,
            "mask_path": mask_path,
            "slice_idx": slice_idx,
            "slice_position": slice_idx / max(depth - 1, 1)
        }


# ------------------------ Dataset Splitting ------------------------

def build_mask_lookup(manifest_path):
    manifest_path = Path(manifest_path)
    mask_lookup = {}

    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            mask_path = Path(row["mask_file"])

            if not mask_path.is_absolute():
                mask_path = manifest_path.parent / mask_path

            suffix = "_segm_preprocessed.nii.gz"

            if not mask_path.name.endswith(suffix):
                raise ValueError(
                    f"Unexpected filename: {mask_path.name}"
                )

            scan_id = mask_path.name.removesuffix(suffix)
            scan_key = f"{row['cohort'].lower()}::{scan_id}"

            if scan_key in mask_lookup:
                raise ValueError(f"Duplicate scan: {scan_key}")

            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)

            mask_lookup[scan_key] = mask_path

    return mask_lookup

def get_mask_paths(splits, mask_lookup, partition):
    if partition not in ("train", "val", "test"):
        raise ValueError(f"Unknown partition: {partition}")

    scan_keys = sorted({
        scan_key
        for stage in ("backbone", "controlnet")
        for scan_key in splits[stage][partition]
    })

    missing = [
        key for key in scan_keys
        if key not in mask_lookup
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} scans missing from manifest: {missing[:5]}"
        )

    return [mask_lookup[key] for key in scan_keys]


def patient_id(scan_key):
    cohort, scan_id = scan_key.split("::", maxsplit=1)

    if cohort == "upenn":
        match = re.fullmatch(
            r"UPenn_(UPENN-GBM-\d+)(?:_\d+)?",
            scan_id
        )
    elif cohort == "ucsf":
        match = re.fullmatch(
            r"(UCSF-PDGM-\d+)",
            scan_id
        )
    else:
        raise ValueError(f"Unknown cohort: {cohort}")

    if match is None:
        raise ValueError(f"Unrecognised scan identifier: {scan_key}")

    return f"{cohort}::{match.group(1)}"


def load_mask_path_splits(split_path, manifest_path):
    with open(split_path) as f:
        splits = json.load(f)

    patient_partitions = {}

    for stage in ("backbone", "controlnet"):
        for partition in ("train", "val", "test"):
            for scan_key in splits[stage][partition]:
                patient = patient_id(scan_key)

                previous = patient_partitions.setdefault(
                    patient,
                    partition,
                )

                if previous != partition:
                    raise ValueError(
                        f"Patient {patient} appears in both "
                        f"{previous} and {partition}"
                    )

    mask_lookup = build_mask_lookup(manifest_path)

    return {
        partition: get_mask_paths(
            splits,
            mask_lookup,
            partition,
        )
        for partition in ("train", "val", "test")
    }

# ------------------------ Condition and Circle Loading Logic ------------------------

def build_mask_conditions(
    target: torch.Tensor,
    slice_position: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    target:
        [H, W], integer class indices 0/1/2/3.
    Returns:
        circle: [1, H, W] float32
        conditions: [4], ordered as [slice_position, p1, p2, p4].
    """
    if target.ndim != 2:
        raise ValueError(f"Expected target shape [H, W], got {target.shape}")
    if target.dtype != torch.long:
        raise ValueError("target must have dtype torch.long")
    if ((target < 0) | (target > 3)).any():
        raise ValueError("Expected internal labels 0/1/2/3")
    if not 0.0 <= slice_position <= 1.0:
        raise ValueError("slice_position must be between 0 and 1")

    foreground = target > 0
    if not foreground.any():
        raise ValueError("Cannot construct tumour conditions from an empty mask")

    # Coordinates are (y, x)
    coordinates = foreground.nonzero(as_tuple=False).float()
    centre = coordinates.mean(dim=0)

    radius_squared = ((coordinates - centre).square().sum(dim=1)).max()
    height, width = target.shape

    rows, columns = torch.meshgrid(
        torch.arange(height, device=target.device, dtype=torch.float32),
        torch.arange(width, device=target.device, dtype=torch.float32),
        indexing="ij"
    )
    distance_squared = (rows - centre[0]).square() + (columns - centre[1]).square()
    
    # Filled circle
    circle = (distance_squared <= radius_squared + 1e-4).float().unsqueeze(0)

    counts = torch.bincount(target.flatten(), minlength=4).float()
    proportions = counts[1:] / counts[1:].sum()

    conditions = torch.cat([
        proportions.new_tensor([slice_position]),
        proportions
    ])

    return circle, conditions

# ------------------------ Cached Latent Mask Dataset ------------------------

class CachedMaskLatentDataset(Dataset):
    def __init__(self, cache_dir, split):
        self.cache_dir = Path(cache_dir)

        with (self.cache_dir / "metadata.json").open() as f:
            self.metadata = json.load(f)
        if self.metadata["status"] != "complete":
            raise ValueError("Latent cache is incomplete")
        if self.metadata["cache_version"] != 1:
            raise ValueError("Unsupported latent cache version")
        if self.metadata["posterior"]["scaled"]:
            raise ValueError("Expected unscaled cached latents")
        expected_order = ["slice_position", "p1", "p2", "p4"]
        if self.metadata["conditioning"]["order"] != expected_order:
            raise ValueError("Unexpected conditioning order")
        if split not in self.metadata["splits"]:
            raise ValueError(f"Split {split} is absent from cache")

        self.split = split
        self.shards = self.metadata["splits"][split]["shards"]
        self.image_size = self.metadata["preprocessing"]["image_size"]

        self.tolerance = self.metadata["conditioning"]["circle_boundary_tolerance"]

        self.shard_sizes = [
            int(shard["samples"])
            for shard in self.shards
        ]

        if not self.shard_sizes or min(self.shard_sizes) < 1:
            raise ValueError("Cache contains no samples or an empty shard")

        # Global sample offsets for each shard.
        self.offsets = [0]

        for size in self.shard_sizes:
            self.offsets.append(self.offsets[-1] + size)

        expected_samples = self.metadata["splits"][split]["samples"]

        if self.offsets[-1] != expected_samples:
            raise ValueError("Cache sample counts are inconsistent")

        self._cached_shard_index = None
        self._cached_shard = None

        # Reuse the coordinate grid for every circle.
        self.rows, self.columns = torch.meshgrid(
            torch.arange(self.image_size, dtype=torch.float32),
            torch.arange(self.image_size, dtype=torch.float32),
            indexing="ij"
        )

    def __len__(self):
        return self.offsets[-1]

    def _load_shard(self, shard_index):
        if shard_index != self._cached_shard_index:
            path = self.cache_dir / self.shards[shard_index]["file"]

            shard = torch.load(
                path,
                map_location="cpu",
                weights_only=True
            )

            expected_count = self.shard_sizes[shard_index]
            expected_shape = tuple(self.metadata["vae"]["latent_shape"])

            if shard["mu"].shape != (expected_count, *expected_shape):
                raise ValueError(f"Unexpected latent shape: {path}")

            if shard["logvar"].shape != shard["mu"].shape:
                raise ValueError(f"Posterior shape mismatch: {path}")

            self._cached_shard = shard
            self._cached_shard_index = shard_index

        return self._cached_shard

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)

        shard_index = bisect.bisect_right(self.offsets, index) - 1

        local_index = index - self.offsets[shard_index]
        shard = self._load_shard(shard_index)

        centre = shard["circle_centre_rc"][local_index]
        radius_squared = shard["circle_radius_squared"][local_index]

        circle = (
            (self.rows - centre[0]).square()
            + (self.columns - centre[1]).square()
            <= radius_squared + self.tolerance
        ).float().unsqueeze(0)

        return {
            "mu": shard["mu"][local_index],
            "logvar": shard["logvar"][local_index],
            "circle": circle,
            "conditions": shard["conditions"][local_index],
            "volume_index": shard["volume_index"][local_index],
            "slice_idx": shard["slice_idx"][local_index],
            "circle_centre": centre,
            "circle_radius_sqr": radius_squared
        }

class ShardBatchSampler(Sampler):
    """
    Keep each batch within one shard.

    Training shuffles shard order and sample order within shards.
    Validation preserves both orders.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        seed=42
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)

        number_of_shards = len(self.dataset.shard_sizes)

        if self.shuffle:
            shard_order = torch.randperm(number_of_shards, generator=generator).tolist()
        else:
            shard_order = range(number_of_shards)

        for shard_index in shard_order:
            size = self.dataset.shard_sizes[shard_index]
            offset = self.dataset.offsets[shard_index]

            if self.shuffle:
                local_indices = torch.randperm(size, generator=generator).tolist()
            else:
                local_indices = list(range(size))

            for start in range(0, size, self.batch_size):
                yield [
                    offset + local_index
                    for local_index in local_indices[start:start + self.batch_size]
                ]

    def __len__(self):
        return sum(
            (size + self.batch_size - 1) // self.batch_size
            for size in self.dataset.shard_sizes
        )