from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from importlib import import_module
import yaml
from collections.abc import Mapping
from src.data.load_masks import load_mask_path_splits, TumourMaskSliceDataset, restore_mask_labels
from src.data.load_brain import build_gbm_index
from tqdm.auto import tqdm
from src.data.load_masks import patient_id
from scipy import ndimage as ndi
import nibabel as nib
import torch
import torch.nn.functional as F
import copy
from types import SimpleNamespace
import importlib.metadata

from .fid_utils import FIDComparison, paired_sample_seed
from .utils import digest, write_csv, write_json
from .segmenter import SplitFileSlices

# --------------------- Saved Visual Examples ---------------------


# --------------------- FID Configuration ---------------------

def load_fid_config(path):
    with Path(path).expanduser().open() as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Expected a YAML configuration mapping")

    for name in ("data", "generator", "evaluation"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Missing configuration section: {name}")

    settings = config["evaluation"]

    if settings.get("split") not in ("val", "test"):
        raise ValueError("evaluation.split must be val or test")

    modalities = settings.get("modalities", [])

    if (
        not isinstance(modalities, list)
        or not modalities
        or any(
            modality not in ("t1", "t1gd", "t2", "flair")
            for modality in modalities
        )
        or len(set(modalities)) != len(modalities)
    ):
        raise ValueError("Provide unique supported modalities")

    seeds = settings.get("seeds", [])

    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Provide unique nonnegative integer repeat seeds")

    limit = settings.get("max_samples")

    if limit is not None and (type(limit) is not int or limit < 2):
        raise ValueError("max_samples must be null or at least two")

    if not settings.get("output_dir"):
        raise ValueError("Specify evaluation.output_dir")

    selection_seed = config.get("seed", 42)

    if type(selection_seed) is not int or selection_seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    for key, default, minimum in (
        ("fid_batch_size", 16, 1),
        ("generation_batch_size", 1, 1),
        ("num_visual_examples", 3, 0)
    ):
        value = settings.get(key, default)

        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid evaluation.{key}")

    # Remove any legacy segmenter configuration.
    # The dataset defaults to both backbone and ControlNet stages.
    config.pop("segmenter", None)

    model_config_path = Path(
        config["generator"]["config"]["config_path"]
    ).expanduser()

    with model_config_path.open() as handle:
        training_config = yaml.safe_load(handle)

    defaults = {
        "image_size": 256,
        "slice_axis": 2,
        "lower_percentile": 0.5,
        "upper_percentile": 99.5,
        "min_foreground_fraction": 0.01,
    }

    for key, default in defaults.items():
        expected = training_config.get("data", {}).get(key, default)

        if key in config["data"] and config["data"][key] != expected:
            raise ValueError(
                f"data.{key} differs from MRI training preprocessing"
            )

        config["data"][key] = expected

    check_intervention_config(config)

    return config


# --------------------- Paired Real Images / Masks ---------------------

def case_description(dataset, index):
    dataset_index, sample_index = dataset.indices[index]
    raw_dataset = dataset.datasets[dataset_index]

    volume_index, _, slice_index = raw_dataset.samples[sample_index]
    row = raw_dataset._get_volume_row(volume_index)

    return dict(
        dataset.cases[index],
        mask_path=str(row["mask_path"]),
        slice_idx=int(slice_index)
    )


def load_real_pair(dataset, index):
    # This bypasses the segmenter's __getitem__ normalisation
    image, mask = dataset.image_and_target(index)

    image = np.asarray(image, dtype=np.float32)

    if (
        not np.isfinite(image).all()
        or image.min() < -1.00001
        or image.max() > 1.00001
    ):
        raise ValueError("Expected MRI-loader intensities in [-1, 1]")

    whole_tumour(mask)

    if image.shape != mask.shape:
        raise ValueError("Real MRI and mask grids differ")

    # Match the generated MRI intensity range
    image = (np.clip(image, -1, 1) + 1) / 2

    return image, mask

# --------------------- Qualitative Figures ---------------------

def save_fid_figure(path, entries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(entries),
        3,
        figsize=(11, 3 * len(entries)),
        squeeze=False
    )

    for row, (title, mask, image) in enumerate(entries):
        target = mask > 0

        axes[row, 0].imshow(
            target,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row, 0].set_title(f"{title}: WT reference")

        for column in (1, 2):
            axes[row, column].imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=1
            )

        axes[row, 1].set_title("MRI")
        axes[row, 2].set_title("Reference WT outline")

        if target.any() and not target.all():
            axes[row, 2].contour(
                target.astype(float),
                levels=[0.5],
                colors=["lime"],
                linewidths=1
            )

        for axis in axes[row]:
            axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(figure)


def build_fid_datasets(config):
    datasets = {}
    lookups = {}

    for modality in config["evaluation"]["modalities"]:
        dataset = SplitFileSlices(
            copy.deepcopy(config),
            modality,
            config["evaluation"]["split"],
            augment=False
        )

        datasets[modality] = dataset

        lookups[modality] = {
            case["case_id"]: index
            for index, case in enumerate(dataset.cases)
        }

    common = sorted(
        set.intersection(
            *(set(lookup) for lookup in lookups.values())
        )
    )

    print("\n----- Held-out sample selection -----", flush=True)
    print(
        f"Partition: {config['evaluation']['split']}",
        flush=True
    )
    print(
        f"Tumour-only: {config['data'].get('tumour_only', True)}",
        flush=True
    )

    for modality, dataset in datasets.items():
        print(
            f"{modality}: {len(dataset):,} eligible slices; "
            f"{len(lookups[modality]) - len(common):,} "
            "excluded by cross-modality intersection",
            flush=True
        )

    print(
        f"Common eligible slices before subsampling: {len(common):,}",
        flush=True,
    )

    limit = config["evaluation"].get("max_samples")

    if limit is not None and limit < len(common):
        rng = np.random.default_rng(config.get("seed", 42))

        common = sorted(
            rng.choice(
                common,
                size=limit,
                replace=False
            ).tolist()
        )

    if len(common) < 2:
        raise ValueError(
            "Need at least two common eligible slices across modalities"
        )
    print(
        f"Selected slices per modality: {len(common):,}",
        flush=True
    )

    return datasets, lookups, common


def run_fid_evaluation(config):
    import importlib.metadata

    settings = config["evaluation"]
    out = Path(settings["output_dir"]).expanduser()

    resume = settings.get("resume", False)
    running_fid_interval = settings.get("running_fid_interval", 5)
    fid_aggregation = settings.get("fid_aggregation", "per_modality")

    if fid_aggregation not in ("per_modality", "pooled", "both"):
        raise ValueError(
            "evaluation.fid_aggregation must be "
            "'per_modality', 'pooled', or 'both'"
        )

    if type(running_fid_interval) is not int or running_fid_interval < 1:
        raise ValueError(
            "evaluation.running_fid_interval must be a positive integer"
        )

    use_per_modality = fid_aggregation in ("per_modality", "both")
    use_pooled = fid_aggregation in ("pooled", "both")

    generation_batch_size = settings.get("generation_batch_size", 1)
    save_sample_interval = settings.get("save_sample_interval", 1)
    fid_device = settings.get("fid_device", "cpu")

    # ---------------------------------------------------------
    # Output directories
    # ---------------------------------------------------------

    out.mkdir(parents=True, exist_ok=resume)

    for directory in ("samples", "plots", "resume"):
        (out / directory).mkdir(exist_ok=resume)

    with (out / "config.yml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    # ---------------------------------------------------------
    # Dataset / selection
    # ---------------------------------------------------------

    datasets, lookups, selected = build_fid_datasets(config)

    first_modality = settings["modalities"][0]

    descriptions = {
        key: case_description(
            datasets[first_modality],
            lookups[first_modality][key],
        )
        for key in selected
    }

    visual_count = min(
        settings.get("num_visual_examples", 3),
        len(selected)
    )

    visual_ids = set(
        np.random.default_rng(config.get("seed", 42)).choice(
            selected,
            size=visual_count,
            replace=False,
        ).tolist()
    )

    n = len(selected)
    modality_count = len(settings["modalities"])
    repeat_count = len(settings["seeds"])

    patient_count = len({
        description["subject"]
        for description in descriptions.values()
    })

    baseline_generations = n * modality_count * repeat_count * 2

    size_settings = config.get("interventions", {}).get("size", {})

    max_intervention_generations = (
        len(visual_ids)
        * modality_count
        * repeat_count
        * len(size_settings.get("area_factors", [0.8, 1.2]))
        if size_settings.get("enabled", False)
        else 0
    )

    print("\n----- Evaluation budget -----", flush=True)
    print(f"Selected patients: {patient_count:,}", flush=True)
    print(f"Selected slice identities: {n:,}", flush=True)
    print(f"Modalities: {settings['modalities']}", flush=True)
    print(f"Repeat seeds: {settings['seeds']}", flush=True)
    print(f"FID aggregation: {fid_aggregation}", flush=True)
    print(
        f"Each individual per-modality FID: "
        f"{n:,} real versus {n:,} generated images",
        flush=True
    )

    if use_pooled:
        pooled_n = n * modality_count
        print(
            f"Final pooled FID: {pooled_n:,} real versus "
            f"{pooled_n:,} generated images",
            flush=True
        )

    print(
        f"Total baseline MRI generations: {baseline_generations:,}",
        flush=True
    )
    print(
        f"Additional intervention generations: "
        f"up to {max_intervention_generations:,}",
        flush=True
    )
    print("Intervention images are excluded from FID.", flush=True)
    print(f"Resume enabled: {resume}", flush=True)

    write_json(
        out / "selection.json",
        {
            "cases": [descriptions[key] for key in selected],
            "visual_case_ids": sorted(visual_ids),
            "eligible_slices_per_modality": {
                modality: len(dataset)
                for modality, dataset in datasets.items()
            },
        }
    )

    # ---------------------------------------------------------
    # Intervention setup
    # ---------------------------------------------------------

    reference_lookup = None

    if size_settings.get("enabled", False) and visual_ids:
        reference_lookup = build_reference_lookup(
            config,
            SimpleNamespace(
                mask_paths=[
                    descriptions[key]["mask_path"]
                    for key in visual_ids
                ]
            ),
            reference_modality=size_settings.get(
                "reference_modality",
                "t1"
            )
        )

    # ---------------------------------------------------------
    # Generator
    # ---------------------------------------------------------

    generator = load_adapter(config["generator"])

    write_json(
        out / "generation_metadata.json",
        getattr(generator, "metadata", {}),
    )

    write_json(
        out / "fid_protocol.json",
        {
            "feature": "Inception-v3 2048",
            "grayscale": "replicated to RGB",
            "input": (
                "[0,1] rounded to uint8; "
                "TorchMetrics internal resizing"
            ),
            "aggregation": fid_aggregation,
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "torchmetrics",
                    "torch-fidelity",
                )
            },
            "split_sha256": digest(config["data"]["split_file"]),
        }
    )

    # ---------------------------------------------------------
    # Helper for restoring checkpointed FID objects onto the
    # current FID device.
    # ---------------------------------------------------------

    def load_checkpoint(path):
        checkpoint = torch.load(
            path,
            map_location=fid_device,
            weights_only=False
        )

        comparison = checkpoint["comparison"]
        comparison.device = torch.device(fid_device)

        for metric in comparison.metrics.values():
            metric.to(comparison.device)

        checkpoint["comparison"] = comparison
        return checkpoint

    def validate_checkpoint_selection(checkpoint, checkpoint_path):
        old_selected = list(checkpoint["selected"])

        if list(selected[:len(old_selected)]) != old_selected:
            raise ValueError(
                "Checkpoint selection is not a prefix of the "
                f"current evaluation selection: {checkpoint_path}"
            )

    # ---------------------------------------------------------
    # Build pooled FID state from ALL existing modality
    # ---------------------------------------------------------

    pooled_comparisons = {}

    if use_pooled:
        for repeat_seed in settings["seeds"]:
            pooled = FIDComparison(fid_device)

            restored_modalities = []

            for modality in settings["modalities"]:
                checkpoint_path = (
                    out / "resume" / f"{modality}_{repeat_seed}.pt"
                )

                if not (resume and checkpoint_path.is_file()):
                    continue

                checkpoint = load_checkpoint(checkpoint_path)
                validate_checkpoint_selection(
                    checkpoint,
                    checkpoint_path
                )

                stored = checkpoint["comparison"]

                # Rolling checkpoints should always have equal
                # real/on/off counts.
                if not (
                    stored.real_count
                    == stored.generated_counts["control_on"]
                    == stored.generated_counts["control_off"]
                ):
                    raise ValueError(
                        "Cannot construct pooled FID from an "
                        "unbalanced checkpoint: "
                        f"{checkpoint_path}\n"
                        f"real={stored.real_count}, "
                        f"on={stored.generated_counts['control_on']}, "
                        f"off={stored.generated_counts['control_off']}"
                    )

                pooled.merge(stored)

                restored_modalities.append(
                    f"{modality}:{stored.real_count}"
                )

            pooled_comparisons[repeat_seed] = pooled

            if restored_modalities:
                print(
                    f"\nRestored pooled FID state for seed "
                    f"{repeat_seed}: "
                    + ", ".join(restored_modalities),
                    flush=True
                )

                print(
                    f"Pooled samples restored: "
                    f"{pooled.real_count:,}",
                    flush=True
                )

                if pooled.real_count >= 2:
                    restored_pooled = pooled.compute()

                    print(
                        f"Restored pooled FID: "
                        f"on={restored_pooled['fid_control_on']:.4f}, "
                        f"off={restored_pooled['fid_control_off']:.4f}, "
                        f"on-off="
                        f"{restored_pooled['fid_on_minus_off']:+.4f}",
                        flush=True
                    )

    scores = []
    log_mode = "a" if resume else "w"

    with (
        (out / "generation.jsonl").open(log_mode) as log,
        (out / "interventions.jsonl").open(log_mode) as audit_log
    ):
        # =====================================================
        # MODALITY LOOP
        # =====================================================

        for modality in settings["modalities"]:
            dataset = datasets[modality]

            # =================================================
            # REPEAT LOOP
            # =================================================

            for repeat_seed in settings["seeds"]:
                checkpoint_path = (
                    out / "resume" / f"{modality}_{repeat_seed}.pt"
                )

                comparison = None
                resume_start = 0

                # ---------------------------------------------
                # Restore this modality's FID state
                # ---------------------------------------------

                if resume and checkpoint_path.is_file():
                    checkpoint = load_checkpoint(checkpoint_path)

                    validate_checkpoint_selection(
                        checkpoint,
                        checkpoint_path
                    )

                    comparison = checkpoint["comparison"]
                    resume_start = int(checkpoint["next_start"])

                    print(
                        f"\nRestoring checkpoint: "
                        f"{modality}, repeat {repeat_seed}",
                        flush=True
                    )

                    print(
                        f"Restored {comparison.real_count:,} "
                        f"matched real/on/off samples.",
                        flush=True
                    )

                    # A checkpoint marked complete is only truly
                    # complete relative to the CURRENT selected set.
                    if resume_start >= len(selected):
                        if use_per_modality:
                            result = comparison.compute()

                            scores.append({
                                "modality": modality,
                                "repeat_seed": repeat_seed,
                                **result
                            })

                            print(
                                f"{modality}, repeat {repeat_seed} "
                                f"already complete: "
                                f"FID on={result['fid_control_on']:.4f}, "
                                f"off={result['fid_control_off']:.4f}",
                                flush=True
                            )
                        else:
                            print(
                                f"{modality}, repeat {repeat_seed} "
                                f"already complete. Skipping.",
                                flush=True
                            )

                        continue

                if comparison is None:
                    comparison = FIDComparison(fid_device)

                # ---------------------------------------------
                # Make sure restored rolling state is balanced
                # ---------------------------------------------

                if not (
                    comparison.real_count
                    == comparison.generated_counts["control_on"]
                    == comparison.generated_counts["control_off"]
                ):
                    raise RuntimeError(
                        f"Restored {modality} FID state is unbalanced: "
                        f"real={comparison.real_count}, "
                        f"on={comparison.generated_counts['control_on']}, "
                        f"off={comparison.generated_counts['control_off']}"
                    )

                print(
                    f"\n{modality} | repeat {repeat_seed}: "
                    f"{len(selected):,} matched real/on/off samples.",
                    flush=True
                )
                print(
                    f"Generation batch size: {generation_batch_size}",
                    flush=True
                )

                total_batches = (
                    len(selected) + generation_batch_size - 1
                ) // generation_batch_size

                completed_batches = (
                    resume_start + generation_batch_size - 1
                ) // generation_batch_size

                progress = tqdm(
                    range(
                        resume_start,
                        len(selected),
                        generation_batch_size
                    ),
                    total=total_batches,
                    initial=completed_batches,
                    desc=f"{modality}: repeat {repeat_seed}"
                )

                # =============================================
                # ROLLING FID LOOP
                # =============================================

                for start in progress:
                    batch_keys = selected[
                        start:start + generation_batch_size
                    ]

                    batch_reals = []
                    batch_masks = []

                    for key in batch_keys:
                        real, mask = load_real_pair(
                            dataset,
                            lookups[modality][key]
                        )

                        batch_reals.append(real)
                        batch_masks.append(mask)

                    real_batch = np.stack(batch_reals, axis=0)
                    mask_batch = np.stack(batch_masks, axis=0)

                    batch_seeds = [
                        paired_sample_seed(repeat_seed, key)
                        for key in batch_keys
                    ]

                    # -----------------------------------------
                    # ControlNet ON
                    # -----------------------------------------

                    control_on_images = np.asarray(
                        generator(
                            mask_batch.copy(),
                            batch_seeds,
                            (modality,),
                            control_enabled=True
                        )[modality]
                    )

                    expected_shape = (
                        len(batch_keys),
                        *batch_masks[0].shape
                    )

                    if control_on_images.shape != expected_shape:
                        raise ValueError(
                            "Generated ControlNet-on batch grid "
                            f"differs from target: expected "
                            f"{expected_shape}, got "
                            f"{control_on_images.shape}"
                        )

                    # -----------------------------------------
                    # ControlNet OFF
                    # -----------------------------------------

                    control_off_images = np.asarray(
                        generator(
                            mask_batch.copy(),
                            batch_seeds,
                            (modality,),
                            control_enabled=False
                        )[modality]
                    )

                    if control_off_images.shape != expected_shape:
                        raise ValueError(
                            "Generated ControlNet-off batch grid "
                            f"differs from target: expected "
                            f"{expected_shape}, got "
                            f"{control_off_images.shape}"
                        )

                    # -----------------------------------------
                    # Modality FID accumulator
                    # -----------------------------------------

                    comparison.add_real(real_batch)

                    comparison.add_generated(
                        control_on_images,
                        control_enabled=True
                    )

                    comparison.add_generated(
                        control_off_images,
                        control_enabled=False
                    )

                    if not (
                        comparison.real_count
                        == comparison.generated_counts["control_on"]
                        == comparison.generated_counts["control_off"]
                    ):
                        raise RuntimeError(
                            "Rolling modality FID counts became "
                            "unbalanced"
                        )

                    # -----------------------------------------
                    # Pooled accumulator
                    # -----------------------------------------

                    if use_pooled:
                        pooled = pooled_comparisons[repeat_seed]

                        pooled.add_real(real_batch)

                        pooled.add_generated(
                            control_on_images,
                            control_enabled=True
                        )

                        pooled.add_generated(
                            control_off_images,
                            control_enabled=False
                        )

                        if not (
                            pooled.real_count
                            == pooled.generated_counts["control_on"]
                            == pooled.generated_counts["control_off"]
                        ):
                            raise RuntimeError(
                                "Rolling pooled FID counts became "
                                "unbalanced"
                            )

                    # =========================================
                    # Per-sample bookkeeping
                    # =========================================

                    for local_index, key in enumerate(batch_keys):
                        position = start + local_index

                        real = batch_reals[local_index]
                        mask = batch_masks[local_index]
                        seed = batch_seeds[local_index]

                        outputs = {
                            "control_on": control_on_images[local_index],
                            "control_off": control_off_images[local_index],
                        }

                        # -------------------------------------
                        # Baseline saving / logging
                        # -------------------------------------

                        for mode, image in outputs.items():
                            if np.asarray(image).shape != mask.shape:
                                raise ValueError(
                                    "Generated image grid differs "
                                    "from target"
                                )

                            artifact = None

                            should_save = (
                                settings.get("save_samples", True)
                                and position % save_sample_interval == 0
                            )

                            if should_save or key in visual_ids:
                                artifact = (
                                    f"samples/{position:06d}_{modality}"
                                    f"_{repeat_seed}_{mode}.npz"
                                )

                                np.savez_compressed(
                                    out / artifact,
                                    image=image,
                                    target_mask=mask
                                )

                            record = dict(
                                descriptions[key],
                                modality=modality,
                                repeat_seed=repeat_seed,
                                seed=seed,
                                variant="baseline",
                                mode=mode,
                                artifact=artifact
                            )

                            log.write(
                                json.dumps(
                                    record,
                                    allow_nan=False
                                ) + "\n"
                            )

                            log.flush()

                        # -------------------------------------
                        # Qualitative examples
                        # -------------------------------------

                        if key in visual_ids:
                            stem = (
                                f"{position:06d}_{modality}_{repeat_seed}"
                            )

                            save_fid_figure(
                                out / "plots" / f"{stem}_on_off.png",
                                [
                                    ("Real reference", mask, real),
                                    (
                                        "Control off",
                                        mask,
                                        outputs["control_off"]
                                    ),
                                    (
                                        "Control on",
                                        mask,
                                        outputs["control_on"]
                                    ),
                                ]
                            )

                            # ---------------------------------
                            # Size interventions
                            # ---------------------------------

                            if size_settings.get("enabled", False):
                                target = torch.from_numpy(
                                    mask.astype(np.int64).copy()
                                )
                                target[target == 4] = 3

                                item = dict(
                                    descriptions[key],
                                    target=target
                                )

                                if (mask > 0).any():
                                    accepted, audit = (
                                        prepare_size_interventions(
                                            item,
                                            config,
                                            reference_lookup,
                                            area_factors=(
                                                size_settings.get(
                                                    "area_factors",
                                                    [0.8, 1.2]
                                                )
                                            ),
                                            min_changed_pixels=(
                                                size_settings.get(
                                                    "min_changed_pixels",
                                                    1
                                                )
                                            )
                                        )
                                    )
                                else:
                                    accepted = []

                                    audit = [{
                                        "accepted": False,
                                        "reasons": [
                                            "empty_original_whole_tumour"
                                        ],
                                    }]

                                for row in audit:
                                    audit_log.write(
                                        json.dumps(
                                            dict(
                                                row,
                                                case_id=key,
                                                modality=modality,
                                                repeat_seed=repeat_seed,
                                            ),
                                            allow_nan=False,
                                        ) + "\n"
                                    )

                                audit_log.flush()

                                edited_entries = [
                                    (
                                        "Baseline",
                                        mask,
                                        outputs["control_on"]
                                    )
                                ]

                                for edit in accepted:
                                    image = generator(
                                        edit["mask"].copy(),
                                        seed,
                                        (modality,),
                                        control_enabled=True,
                                    )[modality]

                                    if (
                                        np.asarray(image).shape != mask.shape
                                        or not np.isfinite(image).all()
                                        or np.min(image) < 0
                                        or np.max(image) > 1
                                    ):
                                        raise ValueError(
                                            "Invalid intervention MRI"
                                        )

                                    artifact = (
                                        f"samples/{stem}_"
                                        f"{edit['variant']}.npz"
                                    )

                                    np.savez_compressed(
                                        out / artifact,
                                        image=image,
                                        target_mask=edit["mask"]
                                    )

                                    log.write(
                                        json.dumps(
                                            dict(
                                                descriptions[key],
                                                modality=modality,
                                                repeat_seed=repeat_seed,
                                                seed=seed,
                                                variant=edit["variant"],
                                                mode="control_on",
                                                artifact=artifact,
                                                parameters=edit["parameters"],
                                            ),
                                            allow_nan=False
                                        ) + "\n"
                                    )

                                    log.flush()

                                    edited_entries.append(
                                        (
                                            edit["variant"],
                                            edit["mask"],
                                            image
                                        )
                                    )

                                save_fid_figure(
                                    out / "plots" / f"{stem}_sizes.png",
                                    edited_entries,
                                )

                    # =========================================
                    # Running FID
                    # =========================================

                    batch_number = (
                        start // generation_batch_size
                    ) + 1

                    if (
                        comparison.real_count >= 2
                        and batch_number % running_fid_interval == 0
                    ):
                        messages = []

                        if use_per_modality:
                            running = comparison.compute()

                            messages.append(
                                f"[{modality} FID] "
                                f"n={comparison.real_count:,} | "
                                f"on={running['fid_control_on']:.4f} | "
                                f"off={running['fid_control_off']:.4f} | "
                                f"Δ={running['fid_on_minus_off']:+.4f}"
                            )

                        if use_pooled:
                            pooled = pooled_comparisons[repeat_seed]

                            if pooled.real_count >= 2:
                                running_pooled = pooled.compute()

                                messages.append(
                                    f"[pooled FID] "
                                    f"n={pooled.real_count:,} | "
                                    f"on="
                                    f"{running_pooled['fid_control_on']:.4f} | "
                                    f"off="
                                    f"{running_pooled['fid_control_off']:.4f} | "
                                    f"Δ="
                                    f"{running_pooled['fid_on_minus_off']:+.4f}"
                                )

                        for message in messages:
                            tqdm.write(message)

                    # =========================================
                    # Checkpoint this modality
                    # =========================================

                    next_start = start + len(batch_keys)

                    checkpoint = {
                        "complete": False,
                        "next_start": next_start,
                        "selected": list(selected),
                        "generation_batch_size": generation_batch_size,
                        "comparison": comparison,
                    }

                    temporary = checkpoint_path.with_suffix(".tmp")

                    torch.save(checkpoint, temporary)
                    temporary.replace(checkpoint_path)

                # =============================================
                # Modality finished
                # =============================================

                modality_result = comparison.compute()

                if use_per_modality:
                    print(
                        f"\nFinal {modality} FID, repeat {repeat_seed}:\n"
                        f"  n:       {comparison.real_count:,}\n"
                        f"  FID on:  "
                        f"{modality_result['fid_control_on']:.4f}\n"
                        f"  FID off: "
                        f"{modality_result['fid_control_off']:.4f}\n"
                        f"  On − off: "
                        f"{modality_result['fid_on_minus_off']:+.4f}",
                        flush=True
                    )

                    scores.append({
                        "modality": modality,
                        "repeat_seed": repeat_seed,
                        **modality_result
                    })

                    write_csv(
                        out / "fid_by_repeat.csv",
                        scores
                    )

                completed_checkpoint = {
                    "complete": True,
                    "next_start": len(selected),
                    "selected": list(selected),
                    "generation_batch_size": generation_batch_size,
                    "comparison": comparison,
                    "result": modality_result,
                }

                temporary = checkpoint_path.with_suffix(".tmp")

                torch.save(completed_checkpoint, temporary)
                temporary.replace(checkpoint_path)

    # =========================================================
    # Pooled results
    # =========================================================

    if use_pooled:
        for repeat_seed in settings["seeds"]:
            pooled = pooled_comparisons[repeat_seed]

            expected_pooled_count = (
                len(selected) * len(settings["modalities"])
            )

            if pooled.real_count != expected_pooled_count:
                raise RuntimeError(
                    "Pooled FID did not receive the expected number "
                    f"of samples: expected {expected_pooled_count}, "
                    f"received {pooled.real_count}"
                )

            result = pooled.compute()

            print(
                f"\nFinal pooled FID, repeat {repeat_seed}:\n"
                f"  n:       {pooled.real_count:,}\n"
                f"  FID on:  {result['fid_control_on']:.4f}\n"
                f"  FID off: {result['fid_control_off']:.4f}\n"
                f"  On − off: {result['fid_on_minus_off']:+.4f}",
                flush=True
            )

            scores.append({
                "modality": "pooled",
                "repeat_seed": repeat_seed,
                **result
            })

    # =========================================================
    # Macro result
    # =========================================================

    if use_per_modality:
        for repeat_seed in settings["seeds"]:
            rows = [
                row
                for row in scores
                if (
                    row["repeat_seed"] == repeat_seed
                    and row["modality"] in settings["modalities"]
                )
            ]

            score_names = (
                "fid_control_on",
                "fid_control_off",
                "fid_on_minus_off",
            )

            scores.append({
                "modality": "macro",
                "repeat_seed": repeat_seed,
                **{
                    name: float(
                        np.mean([row[name] for row in rows])
                    )
                    for name in score_names
                },
            })

    write_csv(out / "fid_by_repeat.csv", scores)

    # =========================================================
    # Summary over repeat seeds
    # =========================================================

    summary_modalities = []

    if use_per_modality:
        summary_modalities.extend(settings["modalities"])
        summary_modalities.append("macro")

    if use_pooled:
        summary_modalities.append("pooled")

    score_names = (
        "fid_control_on",
        "fid_control_off",
        "fid_on_minus_off",
    )

    summary = []

    for modality in summary_modalities:
        rows = [
            row
            for row in scores
            if row["modality"] == modality
        ]

        result = {
            "modality": modality,
            "repeat_count": len(rows),
        }

        if modality == "pooled":
            result["slices_per_modality"] = len(selected)
            result["total_images"] = (
                len(selected) * len(settings["modalities"])
            )
        else:
            result["slices_per_modality"] = len(selected)

        for name in score_names:
            values = [row[name] for row in rows]

            result[name + "_mean"] = float(np.mean(values))
            result[name + "_std"] = float(np.std(values))

        summary.append(result)

    write_csv(out / "fid_summary.csv", summary)

    print(f"\nFinished: {out.resolve()}", flush=True)

# --------------------- Whole Tumour Conversion ---------------------

def whole_tumour(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    if not np.isin(mask, [0, 1, 2, 4]).all():
        raise ValueError(f"Expected raw labels 0/1/2/4, got {np.unique(mask)}")

    return np.isin(mask, [1, 2, 4]) # converts to boolean: 0 bg, 1: tumour


# --------------------- Data Loading / Checking ---------------------

def build_evaluation_dataset(config, split="test"):
    if split not in ("val", "test"):
        raise ValueError("Evaluation requires the validation or test partition")

    dconfig = config["data"]
    split_path = Path(dconfig["split_file"]).expanduser()
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    manifest_path = dconfig.get("mask_manifest")
    if manifest_path is None:
        manifest_path = Path(dconfig["root"]) / "tumor_masks_upenn_ucsf" / "tumor_masks_manifest.csv"
    manifest_path = Path(manifest_path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    mask_paths = load_mask_path_splits(split_path=split_path, manifest_path=manifest_path)
    selected_paths = mask_paths[split]

    print(
        f"\nBuilding {split} evaluation dataset: "
        f"{len(selected_paths)} scans from backbone + controlnet",
        flush=True
    )

    dataset = TumourMaskSliceDataset(
        mask_paths=selected_paths,
        image_size=int(dconfig.get("image_size", 256)),
        slice_axis=int(dconfig.get("slice_axis", 2)),
        tumour_only=bool(dconfig.get("tumour_only", True))
    )

    print(
        f"Evaluation dataset contains {len(dataset)} slices",
        flush=True
    )

    return dataset

def get_conditioning_mask(item):
    target = restore_mask_labels(item["target"])
    return (
        target.detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


# --------------------- Model / Adapter Loading ---------------------

def load_adapter(settings):
    factory_path = settings["factory"]

    module_name, function_name = factory_path.split(":", maxsplit=1)

    module = import_module(module_name)
    build_adapter = getattr(module, function_name)

    adapter = build_adapter(settings.get("config", {}))

    if not callable(adapter):
        raise TypeError(f"{factory_path} must return a callable adapter")

    return adapter


# --------------------- Intervention Configuration Checking ---------------------

def check_intervention_config(config):
    interventions = config.get("interventions", {})

    if not isinstance(interventions, Mapping):
        raise ValueError("interventions must be a mapping")

    settings = interventions.get("size", {})

    if not isinstance(settings, Mapping):
        raise ValueError("interventions.size must be a mapping")

    enabled = settings.get("enabled", False)

    if type(enabled) is not bool:
        raise ValueError("interventions.size.enabled must be true or false")

    if not enabled:
        return

    factors = settings.get("area_factors", [0.8, 1.2])

    if (
        not isinstance(factors, list)
        or not factors
        or any(
            type(value) not in (int, float)
            or not np.isfinite(value)
            or value <= 0
            or value == 1
            for value in factors
        )
    ):
        raise ValueError(
            "area_factors must be a nonempty list of finite, "
            "positive numbers different from 1"
        )

    if len(set(factors)) != len(factors):
        raise ValueError("area_factors must not contain duplicates")

    minimum = settings.get("min_changed_pixels", 1)

    if type(minimum) is not int or minimum < 1:
        raise ValueError("min_changed_pixels must be a positive integer")

    reference_modality = settings.get("reference_modality", "t1")

    if reference_modality not in ("t1", "t1gd", "t2", "flair"):
        raise ValueError("Unsupported reference_modality")


# Phase 2: Interventions

# --------------------- Intervention Feasibility ---------------------

def check_intervention(
    original_mask,
    candidate_mask,
    brain_support,
    *,
    was_clipped,
    min_changed_pixels=1
):
    original = whole_tumour(original_mask)
    candidate = whole_tumour(candidate_mask)

    brain_support = np.asarray(brain_support)

    if (
        original.shape != candidate.shape
        or candidate.shape != brain_support.shape
    ):
        raise ValueError(
            "Original, candidate and brain support must share a grid"
        )

    if not np.isin(brain_support, [0, 1]).all():
        raise ValueError("Brain support must be binary")

    if type(min_changed_pixels) is not int or min_changed_pixels < 1:
        raise ValueError("min_changed_pixels must be a positive integer")

    brain_support = brain_support.astype(bool)

    candidate_area = int(candidate.sum())

    outside_pixels = int(
        np.count_nonzero(candidate & ~brain_support)
    )

    changed_pixels = int(
        np.count_nonzero(original ^ candidate)
    )

    reasons = []

    if was_clipped:
        reasons.append("image_boundary_clipping")

    if candidate_area == 0:
        reasons.append("empty_whole_tumour")

    if outside_pixels > 0:
        reasons.append("outside_brain_support")

    if changed_pixels < min_changed_pixels:
        reasons.append("insufficient_whole_tumour_change")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "candidate_area_pixels": candidate_area,
        "changed_pixels": changed_pixels,
        "outside_brain_pixels": outside_pixels,
        "outside_brain_fraction": (
            outside_pixels / candidate_area
            if candidate_area > 0
            else None
        ),
    }

# --------------------- Tumour Size Intervention ---------------------

def resize_tumour_mask(mask, area_factor):
    mask = np.asarray(mask)
    tumour = whole_tumour(mask)

    if not tumour.any():
        raise ValueError("Cannot resize an empty tumour")
    if not np.isfinite(area_factor) or area_factor <= 0:
        raise ValueError("area_factor must be finite and positive")

    linear_scale = float(np.sqrt(area_factor))

    centre = np.argwhere(tumour).mean(axis=0)
    coordinates = np.argwhere(tumour)

    # Include the full spatial extent of the foreground pixels
    extent = np.max(np.abs(coordinates - centre), axis=0) + 0.5

    padding = int(np.ceil(linear_scale * extent.max())) + 2

    padded = np.pad(
        mask,
        pad_width=padding,
        mode="constant",
        constant_values=0
    )

    padded_centre = centre + padding

    # scipy uses an output-to-input coordinate mapping
    matrix = np.eye(2) / linear_scale
    offset = padded_centre - matrix @ padded_centre

    transformed = ndi.affine_transform(
        padded,
        matrix=matrix,
        offset=offset,
        output_shape=padded.shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False
    )

    height, width = mask.shape

    candidate = transformed[padding:padding + height, padding:padding + width].copy()

    was_clipped = (
        np.count_nonzero(transformed)
        > np.count_nonzero(candidate)
    )

    realised_area_factor = float(np.count_nonzero(candidate) / tumour.sum())

    return candidate, {
        "requested_area_factor": float(area_factor),
        "linear_scale": linear_scale,
        "realised_area_factor": realised_area_factor,
        "was_clipped": bool(was_clipped)
    }


# --------------------- Reference Brain Support ---------------------

def load_brain_support_slice(
    mri_path,
    mask_path,
    slice_index,
    image_size=256,
    slice_axis=2
):
    mri_volume = nib.load(str(mri_path))
    mask_volume = nib.load(str(mask_path))

    if len(mri_volume.shape) != 3:
        raise ValueError("Expected a 3D reference MRI")

    if mri_volume.shape != mask_volume.shape:
        raise ValueError("Reference MRI and tumour mask have different volume shapes")

    if not np.allclose(mri_volume.affine, mask_volume.affine, rtol=0, atol=1e-4):
        raise ValueError("Reference MRI and tumour mask are not on the same spatial grid")

    if slice_axis not in (0, 1, 2):
        raise ValueError("slice_axis must be 0, 1 or 2")

    if not 0 <= slice_index < mri_volume.shape[slice_axis]:
        raise ValueError("slice_index is outside the volume")

    selection = [slice(None)] * 3
    selection[slice_axis] = int(slice_index)

    image = np.asarray(mri_volume.dataobj[tuple(selection)], dtype=np.float32)

    if not np.isfinite(image).all():
        raise ValueError("Reference MRI contains non-finite intensities")

    if np.any(image < 0):
        raise ValueError(
            "Negative intensities found: verify the foreground rule for this scan before using intensity > 0"
        )

    # Extract support before any training-time intensity normalisation
    support = torch.from_numpy((image > 0).astype(np.float32))[None, None]

    support = F.interpolate(
        support,
        size=(image_size, image_size),
        mode="nearest"
    )[0, 0]

    brain_support = support.numpy().astype(bool)

    if not brain_support.any():
        raise ValueError("Reference slice has empty brain support")

    return brain_support



# --------------------- Reference MRI Lookup ---------------------

def build_reference_lookup(
    config,
    dataset,
    reference_modality="t1"
):
    """Builds lookup of Mask Path -> MRI Path for boundary checking"""
    dconfig = config["data"]

    manifest_path = Path(dconfig["mask_manifest"]).expanduser().resolve()

    data_root = Path(dconfig.get("root", manifest_path.parent.parent)).expanduser().resolve()

    expected_manifest = data_root / "tumor_masks_upenn_ucsf" / "tumor_masks_manifest.csv"

    if manifest_path != expected_manifest.resolve():
        raise ValueError(
            "build_gbm_index expects the mask manifest under "
            "data.root/tumor_masks_upenn_ucsf"
        )

    index = build_gbm_index(
        data_root=data_root,
        modalities=(reference_modality,),
        require_all_modalities=False
    )

    selected_masks = {
        str(Path(path).resolve())
        for path in dataset.mask_paths
    }

    lookup = {}

    for row in index.to_dict(orient="records"):
        mask_key = str(Path(row["mask_path"]).resolve())

        if mask_key not in selected_masks:
            continue

        mri_path = row.get(f"{reference_modality}_path")

        if not isinstance(mri_path, (str, Path)):
            continue

        mri_path = Path(mri_path).resolve()

        if not mri_path.is_file():
            continue

        if mask_key in lookup:
            raise ValueError(f"Duplicate reference MRI entry for {mask_key}")

        lookup[mask_key] = mri_path

    missing = selected_masks.difference(lookup)

    if missing:
        examples = "\n".join(sorted(missing)[:5])

        raise ValueError(
            f"Missing {reference_modality} reference MRIs for "
            f"{len(missing)} evaluation scans:\n{examples}"
        )

    print(
        f"Matched {len(lookup)} evaluation scans to "
        f"{reference_modality} reference MRIs.",
        flush=True
    )

    return lookup

# --------------------- Prepare Size Interventions ---------------------

def prepare_size_interventions(
    item,
    config,
    reference_lookup,
    area_factors=(0.8, 1.2),
    min_changed_pixels=1
):
    """
    Prepares the Enlargements/Shrinkings and 
    accepts/rejects based on realism checks
    """
    area_factors = tuple(float(value) for value in area_factors)

    if (
        not area_factors
        or any(
            not np.isfinite(value) or value <= 0 or value == 1
            for value in area_factors
        )
        or len(set(area_factors)) != len(area_factors)
    ):
        raise ValueError("Area factors must be unique, finite, positive and different from 1")

    original_mask = get_conditioning_mask(item)

    mask_key = str(Path(item["mask_path"]).resolve())

    if mask_key not in reference_lookup:
        raise ValueError(f"No reference MRI found for {mask_key}")

    dconfig = config["data"]

    brain_support = load_brain_support_slice(
        mri_path=reference_lookup[mask_key],
        mask_path=item["mask_path"],
        slice_index=int(item["slice_idx"]),
        image_size=int(dconfig.get("image_size", 256)),
        slice_axis=int(dconfig.get("slice_axis", 2))
    )

    original_outside_pixels = int(np.count_nonzero(whole_tumour(original_mask) & ~brain_support))

    accepted = []
    audit = []

    for factor in area_factors:
        candidate, parameters = resize_tumour_mask(
            mask=original_mask,
            area_factor=factor,
        )

        checks = check_intervention(
            original_mask=original_mask,
            candidate_mask=candidate,
            brain_support=brain_support,
            was_clipped=parameters["was_clipped"],
            min_changed_pixels=min_changed_pixels
        )

        variant = f"size_{factor}"

        audit.append({
            "variant": variant,
            "mask_path": str(item["mask_path"]),
            "slice_index": int(item["slice_idx"]),
            "original_outside_brain_pixels": original_outside_pixels,
            "parameters": parameters,
            **checks
        })

        if checks["accepted"]:
            accepted.append({
                "variant": variant,
                "mask": candidate,
                "parameters": parameters
            })

    return accepted, audit


# --------------------- Arguments / Main ---------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Whole Tumour adherence in generated MRIs"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the evaluation YAML configuration"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_fid_config(args.config)

    run_fid_evaluation(config)


if __name__ == "__main__":
    main()