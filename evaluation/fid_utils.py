from __future__ import annotations
import hashlib
import json

import torch
from torchmetrics.image.fid import FrechetInceptionDistance


# --------------------- Reproducible Sample Seeds ---------------------

def paired_sample_seed(base_seed, case_id):
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")

    payload = json.dumps(
        [base_seed, str(case_id)],
        separators=(",", ":")
    ).encode("utf-8")

    digest = hashlib.sha256(payload).digest()

    return int.from_bytes(
        digest[:8], byteorder="big"
    ) % (2**63 - 1)


# --------------------- FID Image Preparation ---------------------

def prepare_fid_images(images, device):
    images = torch.as_tensor(
        images,
        dtype=torch.float32,
        device=device
    )

    if images.ndim == 2:
        images = images[None, None]
    elif images.ndim == 3:
        images = images[:, None]

    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(
            "Expected grayscale images with shape "
            "H×W, N×H×W or N×1×H×W"
        )

    if images.numel() == 0 or not torch.isfinite(images).all():
        raise ValueError("FID images must be nonempty and finite")

    if images.min().item() < 0 or images.max().item() > 1:
        raise ValueError("FID images must have intensities in [0, 1]")

    # Apply identical quantization to real and generated images
    images = (
        images.mul(255)
        .round()
        .to(torch.uint8)
    )

    # Standard Inception FID expects three channels
    return images.repeat(1, 3, 1, 1)

# --------------------- Paired FID Comparison ---------------------

class FIDComparison:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        self.metrics = {
            mode: FrechetInceptionDistance(
                feature=2048,
                normalize=False,
                reset_real_features=False,
            ).to(self.device)
            for mode in ("control_on", "control_off")
        }

        self.real_count = 0
        self.generated_counts = {
            "control_on": 0,
            "control_off": 0,
        }

    @torch.inference_mode()
    def add_real(self, images):
        batch = prepare_fid_images(images, device=self.device)

        for metric in self.metrics.values():
            metric.update(batch, real=True)

        self.real_count += int(batch.shape[0])

    @torch.inference_mode()
    def add_generated(self, images, control_enabled):
        mode = "control_on" if control_enabled else "control_off"

        batch = prepare_fid_images(images, device=self.device)

        self.metrics[mode].update(batch, real=False)
        self.generated_counts[mode] += int(batch.shape[0])

    @torch.inference_mode()
    def merge(self, other):
        if not isinstance(other, FIDComparison):
            raise TypeError("Can only merge another FIDComparison")

        if other is self:
            raise ValueError("Cannot merge a FIDComparison into itself")

        state_names = (
            "real_features_sum",
            "real_features_cov_sum",
            "real_features_num_samples",
            "fake_features_sum",
            "fake_features_cov_sum",
            "fake_features_num_samples"
        )

        for mode in ("control_on", "control_off"):
            target = self.metrics[mode]
            source = other.metrics[mode]

            for name in state_names:
                target_state = getattr(target, name)
                source_state = getattr(source, name)

                target_state.add_(
                    source_state.to(
                        device=target_state.device,
                        dtype=target_state.dtype
                    )
                )

            # compute() in TorchMetrics uses orig_dtype for its output.
            if hasattr(source, "orig_dtype"):
                target.orig_dtype = source.orig_dtype
            elif not hasattr(target, "orig_dtype"):
                target.orig_dtype = torch.float32

        self.real_count += other.real_count

        for mode in ("control_on", "control_off"):
            self.generated_counts[mode] += other.generated_counts[mode]

    @torch.inference_mode()
    def compute(self, require_equal_counts=True):
        if self.real_count < 2:
            raise ValueError("FID requires at least two real images")

        for mode, count in self.generated_counts.items():
            if require_equal_counts and count != self.real_count:
                raise ValueError(
                    f"{mode}: expected {self.real_count} generated "
                    f"images to match the real set, received {count}"
                )

            if count < 2:
                raise ValueError(
                    f"FID requires at least two generated images for {mode}"
                )

        scores = {}

        for mode, metric in self.metrics.items():
            score = metric.compute()

            if not torch.isfinite(score):
                raise RuntimeError(f"Non-finite FID for {mode}")

            scores[f"fid_{mode}"] = float(score.cpu())

        return {
            "real_count": self.real_count,
            "generated_count_per_arm": self.real_count,
            **scores,
            "fid_on_minus_off": (
                scores["fid_control_on"]
                - scores["fid_control_off"]
            ),
        }

    def reset_generated(self):
        # reset_real_features=False preserves real statistics.
        for metric in self.metrics.values():
            metric.reset()

        self.generated_counts = {
            "control_on": 0,
            "control_off": 0,
        }