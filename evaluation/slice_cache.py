from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare_volume_cache(dataset, row, modality, slice_indices, cache_root):
    """Match BrainMRIMultimodalSliceDataset preprocessing, with one volume read."""
    image_path = row[f"{modality}_path"]
    mask_path = row["mask_path"]
    identity = {
        "version": 1,
        "image": file_identity(image_path), "mask": file_identity(mask_path),
        "slice_axis": dataset.slice_axis, "image_size": dataset.image_size,
        "lower_percentile": dataset.lower_percentile,
        "upper_percentile": dataset.upper_percentile,
        "slice_indices": [int(i) for i in slice_indices],
        "image_preprocessing": "loader_percentile_to_minus1_plus1_then_bilinear_v1",
        "mask_preprocessing": "nearest_raw_labels_v1"
    }
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / cache_key
    if destination.exists():
        if json.loads((destination / "metadata.json").read_text()) != identity:
            raise ValueError(f"Cache metadata mismatch: {destination}")
        return destination

    import nibabel as nib
    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    if image_nii.shape != mask_nii.shape or not np.allclose(image_nii.affine, mask_nii.affine):
        raise ValueError(f"MRI/mask geometry differs: {image_path}")
    image_volume = np.asarray(image_nii.dataobj, dtype=np.float32)
    mask_volume = np.asarray(mask_nii.dataobj, dtype=np.float32)
    if not np.isfinite(image_volume).all() or not np.isin(mask_volume, (0, 1, 2, 4)).all():
        raise ValueError(f"Nonfinite MRI or invalid mask labels: {image_path}")

    shape = (len(slice_indices), dataset.image_size, dataset.image_size)
    # Publish only complete caches. Interrupted builds cannot be mistaken for
    # usable data, and existing valid caches are never overwritten.
    with tempfile.TemporaryDirectory(prefix="building-", dir=cache_root) as temporary:
        temporary = Path(temporary)
        images = np.lib.format.open_memmap(temporary / "images.npy", mode="w+", dtype=np.float32, shape=shape)
        masks = np.lib.format.open_memmap(temporary / "masks.npy", mode="w+", dtype=np.uint8, shape=shape)
        progress = tqdm(slice_indices, desc="Resizing cached slices", unit="slice", leave=False)
        for offset, index in enumerate(progress):
            image = torch.from_numpy(np.take(image_volume, index, axis=dataset.slice_axis).copy())
            image = dataset._normalise(image)
            image = F.interpolate(image[None, None], size=shape[1:], mode="bilinear", align_corners=False)[0, 0]
            mask = torch.from_numpy(np.take(mask_volume, index, axis=dataset.slice_axis).copy())
            mask = F.interpolate(mask[None, None], size=shape[1:], mode="nearest")[0, 0]
            images[offset] = image.numpy()
            masks[offset] = mask.numpy().astype(np.uint8)
        images.flush()
        masks.flush()
        del images, masks
        (temporary / "metadata.json").write_text(json.dumps(identity, sort_keys=True))
        try:
            temporary.rename(destination)
        except OSError:
            # Another process may have published this exact cache meanwhile.
            metadata = destination / "metadata.json"
            if not metadata.exists() or json.loads(metadata.read_text()) != identity:
                raise
    return destination


class SliceCacheReader:
    """Small per-worker LRU of file mappings; volumes are not copied into RAM."""
    def __init__(self, max_open=8):
        self.max_open = max_open
        self.pid = os.getpid()
        self.maps = OrderedDict()

    def __getstate__(self):
        return {"max_open": self.max_open, "pid": None, "maps": OrderedDict()}

    def read(self, directory, index):
        if self.pid != os.getpid():
            self.maps.clear()
            self.pid = os.getpid()
        directory = str(directory)
        if directory not in self.maps:
            root = Path(directory)
            self.maps[directory] = (
                np.load(root / "images.npy", mmap_mode="r", allow_pickle=False),
                np.load(root / "masks.npy", mmap_mode="r", allow_pickle=False)
            )
            while len(self.maps) > self.max_open:
                self.maps.popitem(last=False)
        self.maps.move_to_end(directory)
        images, masks = self.maps[directory]
        return images[index].copy(), masks[index].copy()