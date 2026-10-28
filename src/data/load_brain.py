import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path
import nibabel as nib
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import bisect
import json
from functools import lru_cache

DEFAULT_MODALITIES = ("t1", "t1gd", "t2", "flair")

@lru_cache(maxsize=4)
def _load_volume_cached(path):
    nii = nib.load(path)

    return np.asarray(
        nii.dataobj,
        dtype=np.float32
    )

def _strip_nii_gz(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return Path(filename).stem


def _subject_prefix_from_mask(mask_path: Path) -> str:
    name = _strip_nii_gz(mask_path.name)

    suffix = "_segm_preprocessed"

    if not name.endswith(suffix):
        raise ValueError(
            f"Unexpected mask filename: {mask_path.name}"
        )

    return name[:-len(suffix)]


def _find_modalities(
    image_dir: Path,
    subject_prefix: str,
    cohort: str
):
    modality_aliases = {
        "upenn": {
            "t1": "t1",
            "t1gd": "t1gd",
            "t2": "t2",
            "flair": "flair"
        },
        "ucsf": {
            "t1": "t1",
            "t1gd": "t1c",
            "t2": "t2",
            "flair": "flair"
        },
    }

    aliases = modality_aliases[cohort]

    paths = {}

    for canonical_name, file_suffix in aliases.items():
        path = image_dir / (
            f"{subject_prefix}_{file_suffix}_preprocessed.nii.gz"
        )

        if path.exists():
            paths[canonical_name] = path

    return paths


def build_gbm_index(
    data_root: str | Path,
    modalities=DEFAULT_MODALITIES,
    require_all_modalities: bool = True
) -> pd.DataFrame:

    data_root = Path(data_root)

    mask_root = data_root / "tumor_masks_upenn_ucsf"
    manifest_path = mask_root / "tumor_masks_manifest.csv"

    manifest = pd.read_csv(manifest_path)

    records = []

    for _, row in manifest.iterrows():

        cohort = str(row["cohort"]).lower()

        if cohort not in {"upenn", "ucsf"}:
            continue

        mask_file = Path(row["mask_file"])

        if mask_file.is_absolute():
            mask_path = mask_file
        else:
            mask_path = mask_root / mask_file

        if not mask_path.exists():
            print(f"Missing mask: {mask_path}")
            continue

        subject_prefix = _subject_prefix_from_mask(mask_path)

        image_dir = data_root / cohort

        image_paths = _find_modalities(
            image_dir=image_dir,
            subject_prefix=subject_prefix,
            cohort=cohort
        )

        missing_modalities = [
            modality
            for modality in modalities
            if modality not in image_paths
        ]

        if require_all_modalities and missing_modalities:
            print(
                f"Skipping {subject_prefix}: "
                f"missing {missing_modalities}"
            )
            continue

        record = {
            "subject": row["subject"],
            "cohort": cohort,
            "subject_prefix": subject_prefix,
            "mask_path": str(mask_path),
        }

        for modality in modalities:
            path = image_paths.get(modality)
            record[f"{modality}_path"] = (
                str(path) if path is not None else None
            )

        for column in manifest.columns:
            if column not in record:
                record[column] = row[column]

        records.append(record)

    return pd.DataFrame(records)


class GBMMultimodalDataset(Dataset):

    def __init__(
        self,
        data_root: str | Path,
        modalities=DEFAULT_MODALITIES,
        require_all_modalities: bool = True,
        dtype=np.float32
    ):
        self.modalities = tuple(modalities)
        self.dtype = dtype

        self.index = build_gbm_index(
            data_root=data_root,
            modalities=self.modalities,
            require_all_modalities=require_all_modalities
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        row = self.index.iloc[idx]

        images = {}

        reference_shape = None
        reference_affine = None

        for modality in self.modalities:

            path = row[f"{modality}_path"]

            if path is None or pd.isna(path):
                continue

            nii = nib.load(path)

            volume = np.asarray(nii.dataobj, dtype=self.dtype)

            if reference_shape is None:
                reference_shape = volume.shape
                reference_affine = nii.affine.copy()

            elif volume.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch for {row['subject_prefix']}: "
                    f"{modality} has shape {volume.shape}, "
                    f"expected {reference_shape}"
                )

            images[modality] = torch.from_numpy(volume.copy())

        mask_nii = nib.load(row["mask_path"])

        mask = np.asarray(mask_nii.dataobj, dtype=np.float32)

        if reference_shape is not None and mask.shape != reference_shape:
            raise ValueError(
                f"Mask shape mismatch for "
                f"{row['subject_prefix']}: "
                f"{mask.shape} != {reference_shape}"
            )

        segmentation = torch.from_numpy(mask.copy()).long()

        metadata = {
            key: value
            for key, value in row.to_dict().items()
            if not key.endswith("_path")
        }

        metadata["affine"] = reference_affine
        metadata["image_paths"] = {
            modality: row[f"{modality}_path"]
            for modality in self.modalities
            if row[f"{modality}_path"] is not None
            and not pd.isna(row[f"{modality}_path"])
        }

        metadata["mask_path"] = row["mask_path"]

        return {
            "images": images,
            "seg": segmentation,
            "metadata": metadata
        }

class BrainMRISliceDataset(Dataset):
    """
    Only loads the slice indeces into RAM - 
    instead of the entire slices
    """
    def __init__(
        self,
        volume_dataset: Dataset,
        modality: str = "t1gd",
        image_size: int = 256,
        slice_axis: int = 2,
        min_foreground_fraction: float = 0.01,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5
    ):
        self.volume_dataset = volume_dataset
        self.modality = modality
        self.image_size = image_size
        self.slice_axis = slice_axis

        self.min_foreground_fraction = min_foreground_fraction
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

        self.samples = self._build_slice_index()    # stores (patient_idx, slice_idx)


    def _build_slice_index(self):

        samples = []

        for volume_idx in tqdm(
            range(len(self.volume_dataset)),
            desc="indexing MRI volumes"
        ):

            sample = self.volume_dataset[volume_idx]

            if self.modality not in sample["images"]:
                continue

            volume = sample["images"][self.modality]

            num_slices = volume.shape[self.slice_axis]

            for slice_idx in range(num_slices):

                image_slice = torch.select(
                    volume,
                    dim=self.slice_axis,
                    index=slice_idx
                )

                foreground_fraction = torch.count_nonzero(image_slice).item() / image_slice.numel()
                

                if foreground_fraction >= self.min_foreground_fraction:
                    samples.append((volume_idx, slice_idx))

        return samples

    def __len__(self):
        return len(self.samples)

    def _get_volume_row(self, volume_idx):
        if isinstance(self.volume_dataset, Subset):
            base_idx = self.volume_dataset.indices[volume_idx]
            return self.volume_dataset.dataset.index.iloc[base_idx]

        return self.volume_dataset.index.iloc[volume_idx]
    
    def _normalise(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:

        image = image.float()

        foreground = image[image != 0]

        if foreground.numel() == 0:
            return torch.zeros_like(image)

        lower = torch.quantile(
            foreground,
            self.lower_percentile / 100.0
        )

        upper = torch.quantile(
            foreground,
            self.upper_percentile / 100.0
        )

        if upper <= lower:
            return torch.zeros_like(image)

        image = torch.clamp(
            image,
            min=lower.item(),
            max=upper.item()
        )

        image = (image - lower) / (upper - lower)

        image = image * 2.0 - 1.0

        return image

    def __getitem__(self, idx):

        volume_idx, slice_idx = self.samples[idx]
        row = self._get_volume_row(volume_idx)

        image_path = row[f"{self.modality}_path"]
        nii = nib.load(image_path)
        volume = np.asarray(nii.dataobj, dtype=np.float32)

        image = torch.from_numpy(np.take(volume, slice_idx, axis=self.slice_axis).copy())

        image = self._normalise(image)

        image = image.unsqueeze(0).unsqueeze(0)

        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        )

        image = image.squeeze(0)

        return {
            "image": image,
            "subject": str(row["subject"]),
            "cohort": str(row["cohort"]),
            "slice_idx": slice_idx,
            "modality": self.modality,
        }


class BrainMRIMultimodalSliceDataset(Dataset):

    def __init__(
        self,
        volume_dataset: Dataset,
        modalities=DEFAULT_MODALITIES,
        image_size: int = 256,
        slice_axis: int = 2,
        min_foreground_fraction: float = 0.01,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        include_segmentation: bool = False
    ):
        self.volume_dataset = volume_dataset
        self.modalities = tuple(modalities)

        self.image_size = image_size
        self.slice_axis = slice_axis

        self.min_foreground_fraction = min_foreground_fraction
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

        self.include_segmentation = include_segmentation

        self.samples = self._build_slice_index()

    def _get_volume_row(self, volume_idx):

        if isinstance(self.volume_dataset, Subset):
            base_idx = self.volume_dataset.indices[volume_idx]
            return self.volume_dataset.dataset.index.iloc[base_idx]

        return self.volume_dataset.index.iloc[volume_idx]

    def _build_slice_index(self):

        samples = []

        for volume_idx in tqdm(
            range(len(self.volume_dataset)),
            desc="Indexing multimodal MRI slices"
        ):
            row = self._get_volume_row(volume_idx)

            for modality in self.modalities:

                image_path = row[f"{modality}_path"]

                if image_path is None or pd.isna(image_path):
                    continue
                try:
                    nii = nib.load(image_path)
                    volume = np.asarray(nii.dataobj, dtype=np.float32)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to read MRI volume while building slice index: {image_path}"
                    ) from exc

                spatial_axes = tuple(
                    axis
                    for axis in range(volume.ndim)
                    if axis != self.slice_axis
                )

                foreground_counts = np.count_nonzero(volume, axis=spatial_axes)

                slice_size = np.prod([
                    volume.shape[axis]
                    for axis in spatial_axes
                ])

                foreground_fractions = foreground_counts / slice_size

                valid_slices = np.flatnonzero(foreground_fractions >= self.min_foreground_fraction)

                samples.extend([
                    (
                        volume_idx,
                        modality,
                        int(slice_idx)
                    )
                    for slice_idx in valid_slices
                ])

        return samples

    def __len__(self):
        return len(self.samples)

    def _normalise(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:

        image = image.float()
        foreground = image[image != 0]
        if foreground.numel() == 0:
            return torch.zeros_like(image)

        lower = torch.quantile(foreground, self.lower_percentile / 100.0)

        upper = torch.quantile(foreground, self.upper_percentile / 100.0)

        if upper <= lower:
            return torch.zeros_like(image)

        image = torch.clamp(
            image,
            min=lower.item(),
            max=upper.item()
        )

        image = (image - lower) / (upper - lower)
        image = image * 2.0 - 1.0
        return image

    def _load_image(
        self,
        path,
        slice_idx
    ):
        volume = _load_volume_cached(
            str(path)
        )

        image = torch.from_numpy(
            np.take(
                volume,
                slice_idx,
                axis=self.slice_axis
            ).copy()
        )

        image = self._normalise(image)

        image = (
            image
            .unsqueeze(0)
            .unsqueeze(0)
        )

        image = F.interpolate(
            image,
            size=(
                self.image_size,
                self.image_size
            ),
            mode="bilinear",
            align_corners=False
        )

        return image.squeeze(0)

    def _load_segmentation(
        self,
        path,
        slice_idx
    ):

        nii = nib.load(path)

        volume = np.asarray(
            nii.dataobj,
            dtype=np.float32
        )

        segmentation = torch.from_numpy(
            np.take(
                volume,
                slice_idx,
                axis=self.slice_axis
            ).copy()
        )

        segmentation = (
            segmentation
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )

        segmentation = F.interpolate(
            segmentation,
            size=(
                self.image_size,
                self.image_size
            ),
            mode="nearest"
        )

        return (
            segmentation
            .squeeze(0)
            .long()
        )

    def __getitem__(self, idx):

        volume_idx, modality, slice_idx = self.samples[idx]

        row = self._get_volume_row(volume_idx)

        image = self._load_image(row[f"{modality}_path"], slice_idx)

        sample = {
            "image": image,
            "subject": str(row["subject"]),
            "subject_prefix": str(
                row["subject_prefix"]
            ),
            "cohort": str(row["cohort"]),
            "slice_idx": slice_idx,
            "modality": modality
        }

        if self.include_segmentation:
            sample["seg"] = self._load_segmentation(row["mask_path"], slice_idx)
            

        return sample


# ------------------------------ Data Splitting ------------------------------
def _patient_key(row):
    return (
        f"{row['cohort']}::"
        f"{row['subject_prefix']}"
    )


def _split_stage_indices(
    indices,
    cohorts,
    val_fraction,
    test_fraction,
    seed
):
    indices = np.asarray(indices)
    stage_cohorts = cohorts[indices]

    holdout_fraction = val_fraction + test_fraction

    train_indices, holdout_indices = (
        train_test_split(
            indices,
            test_size=holdout_fraction,
            random_state=seed,
            stratify=stage_cohorts
        )
    )

    holdout_cohorts = cohorts[holdout_indices]

    test_relative_fraction = test_fraction / holdout_fraction
    

    val_indices, test_indices = (
        train_test_split(
            holdout_indices,
            test_size=test_relative_fraction,
            random_state=seed,
            stratify=holdout_cohorts
        )
    )

    return {
        "train": train_indices.tolist(),
        "val": val_indices.tolist(),
        "test": test_indices.tolist(),
    }


def _create_patient_split_manifest(
    dataset,
    control_fraction,
    val_fraction,
    test_fraction,
    seed,
):
    indices = np.arange(len(dataset))

    cohorts = dataset.index["cohort"].to_numpy()

    backbone_indices, control_indices = (
        train_test_split(
            indices,
            test_size=control_fraction,
            random_state=seed,
            stratify=cohorts
        )
    )

    backbone = _split_stage_indices(
        indices=backbone_indices,
        cohorts=cohorts,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed
    )

    controlnet = _split_stage_indices(
        indices=control_indices,
        cohorts=cohorts,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed
    )

    def indices_to_keys(split_indices):

        return [
            _patient_key(
                dataset.index.iloc[idx]
            )
            for idx in split_indices
        ]

    manifest = {
        "seed": seed,
        "control_fraction":
            control_fraction,
        "val_fraction":
            val_fraction,
        "test_fraction":
            test_fraction,

        "backbone": {
            split: indices_to_keys(
                split_indices
            )
            for split, split_indices
            in backbone.items()
        },

        "controlnet": {
            split: indices_to_keys(
                split_indices
            )
            for split, split_indices
            in controlnet.items()
        }
    }
    return manifest

def _load_or_create_split_manifest(
    dataset,
    split_file,
    control_fraction,
    val_fraction,
    test_fraction,
    seed
):
    split_file = Path(split_file)

    if split_file.exists():

        with open(split_file, "r") as f:
            manifest = json.load(f)

        print(
            f"Using existing patient split: "
            f"{split_file}"
        )

        return manifest

    manifest = (
        _create_patient_split_manifest(
            dataset=dataset,
            control_fraction=control_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed
        )
    )

    split_file.parent.mkdir(parents=True, exist_ok=True)

    with open(split_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Created patient split: "
        f"{split_file}"
    )

    return manifest


def _subset_from_manifest(
    dataset,
    manifest,
    stage,
    split,
):
    if stage not in {"backbone","controlnet"}:
        raise ValueError("stage must be 'backbone' or 'controlnet'")

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val' or 'test'")

    selected_keys = set(manifest[stage][split])
    indices = []

    for idx, row in (dataset.index.iterrows()):

        if _patient_key(row) in selected_keys:
            indices.append(idx)

    if len(indices) != len(selected_keys):
        raise RuntimeError(
            "Some patients in the saved "
            "split could not be found in "
            "the current dataset index."
        )

    return Subset(dataset, indices)

def buildMRIDataset(
    config: dict,
    split: str,
    stage: str = "backbone"
):

    data_config = config["data"]
    modalities = tuple(
        data_config.get("modalities", DEFAULT_MODALITIES)
    )

    volume_dataset = (
        GBMMultimodalDataset(
            data_root=data_config["root"],
            modalities=modalities,
            require_all_modalities=(
                data_config.get(
                    "require_all_modalities",
                    True
                )
            )
        )
    )

    manifest = (
        _load_or_create_split_manifest(
            dataset=volume_dataset,
            split_file=data_config["split_file"],
            control_fraction=data_config.get("control_fraction", 0.25),
            val_fraction=data_config.get("val_fraction", 0.1),
            test_fraction= data_config.get("test_fraction", 0.1),
            seed=config.get("seed", 42)
        )
    )

    selected_volumes = (
        _subset_from_manifest(
            dataset=volume_dataset,
            manifest=manifest,
            stage=stage,
            split=split
        )
    )

    return (
        BrainMRIMultimodalSliceDataset(
            volume_dataset=selected_volumes,
            modalities=modalities,
            image_size=data_config.get("image_size", 256),
            slice_axis=data_config.get("slice_axis", 2),
            min_foreground_fraction=data_config.get("min_foreground_fraction", 0.01),
            lower_percentile= data_config.get("lower_percentile", 0.5),
            upper_percentile=data_config.get("upper_percentile", 99.5),
            include_segmentation = (stage == "controlnet")
        )
    )


class CachedVAELatentDataset(Dataset):

    def __init__(self, latent_dir):
        self.latent_dir = Path(latent_dir)

        self.shard_paths = sorted(
            self.latent_dir.glob("latents_*.pt")
        )

        self.shard_sizes = []

        for path in self.shard_paths:
            shard = torch.load(
                path,
                map_location="cpu",
                weights_only=True
            )

            self.shard_sizes.append(
                shard["mu"].shape[0]
            )

        self.cumulative_sizes = []

        total = 0

        for size in self.shard_sizes:
            total += size
            self.cumulative_sizes.append(total)

        self._cached_shard_idx = None
        self._cached_shard = None

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _load_shard(self, shard_idx):

        if shard_idx != self._cached_shard_idx:

            self._cached_shard = torch.load(
                self.shard_paths[shard_idx],
                map_location="cpu",
                weights_only=True
            )

            self._cached_shard_idx = shard_idx

        return self._cached_shard

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

        shard = self._load_shard(shard_idx)

        return {
            "mu": shard["mu"][local_idx],
            "logvar": shard["logvar"][local_idx]
        }


MODALITY_TO_ID = {
    "t1": 0,
    "t1gd": 1,
    "t2": 2,
    "flair": 3,
}


class ModalityConditionedCachedDataset(Dataset):

    def __init__(self, cached_dataset, raw_dataset):
        if len(cached_dataset) != len(raw_dataset):
            raise ValueError(
                f"Cached/raw length mismatch: "
                f"{len(cached_dataset)} vs {len(raw_dataset)}"
            )

        self.cached_dataset = cached_dataset
        self.raw_dataset = raw_dataset

    def __len__(self):
        return len(self.cached_dataset)

    def __getitem__(self, idx):

        item = self.cached_dataset[idx]

        _, modality, _ = self.raw_dataset.samples[idx]

        return {
            **item,
            "modality": MODALITY_TO_ID[modality]
        }