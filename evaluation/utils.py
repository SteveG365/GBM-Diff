from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# --------------------- Labels / Modalities ---------------------

MODALITIES = ("t1", "t1gd", "t2", "flair")

REGIONS = {"WT": (1, 2, 4), "TC": (1, 4), "ET": (4,)}


# --------------------- Mask Checking ---------------------

def check_mask(mask):
    mask = np.asarray(mask)

    if mask.ndim != 2 or min(mask.shape) < 2:
        raise ValueError("Expected a 2D mask")

    if not np.isin(mask, [0, 1, 2, 4]).all():
        raise ValueError(
            f"Expected raw labels 0/1/2/4, got {np.unique(mask)}"
        )

    return mask.astype(np.uint8)


# --------------------- File Utilities ---------------------

def digest(path):
    result = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            result.update(chunk)

    return result.hexdigest()


def write_json(path, value):
    with Path(path).open("w") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def write_csv(path, rows):
    rows = list(rows)

    if not rows:
        Path(path).write_text("")
        return

    columns = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )

    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)

# --------------------- Exported Sample Loading ---------------------

def load_cases(path):
    path = Path(path).expanduser().resolve()

    with path.open() as handle:
        records = json.load(handle)

    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty list of sample records")

    seen = set()

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each sample record must be a mapping")

        for key in ("case_id", "subject", "mask", "split", "source"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Missing or invalid sample field: {key}")

        if record["case_id"] in seen:
            raise ValueError(
                f"Duplicate case_id: {record['case_id']}"
            )

        seen.add(record["case_id"])

        if record["source"] not in ("real", "generated"):
            raise ValueError("source must be real or generated")

        for key in ("mask", "allowed_region"):
            if record.get(key):
                record[key] = str(
                    (path.parent / record[key]).resolve()
                )

        record["array"] = check_mask(
            np.load(record["mask"], allow_pickle=False)
        )

        record.setdefault("stratum", "all")

    return records

# --------------------- Segmentation Metrics ---------------------

def region_metrics(target, prediction):
    target = check_mask(target)
    prediction = check_mask(prediction)

    if target.shape != prediction.shape:
        raise ValueError(
            "Target and prediction must share the same grid"
        )

    results = {}

    for name, labels in REGIONS.items():
        target_region = np.isin(target, labels)
        predicted_region = np.isin(prediction, labels)

        target_pixels = int(target_region.sum())
        predicted_pixels = int(predicted_region.sum())

        intersection = int(np.count_nonzero(target_region & predicted_region))

        dice = None
        relative_area_error = None
        centroid_distance = None

        if target_pixels > 0:
            dice = (2.0 * intersection / (target_pixels + predicted_pixels))

            relative_area_error = (
                predicted_pixels - target_pixels
            ) / target_pixels

        if target_pixels > 0 and predicted_pixels > 0:
            target_centroid = np.argwhere(target_region).mean(axis=0)

            predicted_centroid = np.argwhere(predicted_region).mean(axis=0)

            centroid_distance = float(
                np.linalg.norm(predicted_centroid - target_centroid)
            )

        results[name] = {
            "target_pixels": target_pixels,
            "predicted_pixels": predicted_pixels,
            "dice": dice,
            "missing": (
                target_pixels > 0 and predicted_pixels == 0
            ),
            "absent_false_positive_pixels": (
                predicted_pixels if target_pixels == 0 else None
            ),
            "relative_area_error": relative_area_error,
            "centroid_distance_pixels": centroid_distance
        }

    return results

# --------------------- Patient-Level Summaries ---------------------

def summarise(records, plan):
    def average(values):
        values = [
            value
            for value in values
            if value is not None
        ]

        return float(np.mean(values)) if values else None

    groups = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    for record in records:
        case = plan[record["plan_index"]]

        for region, metrics in record["metrics"].items():
            key = (
                case["source"],
                case["variant"],
                record["modality"],
                region
            )

            groups[key][case["subject"]][case["case_id"]].append(metrics)

    rows = []

    metric_names = (
        "dice",
        "absent_false_positive_pixels",
        "relative_area_error",
        "centroid_distance_pixels"
    )

    for key, subjects in groups.items():
        row = dict(
            zip(("source", "variant", "modality", "region"), key)
        )

        row["subject_count"] = len(subjects)

        all_metrics = [
            metrics
            for cases in subjects.values()
            for observations in cases.values()
            for metrics in observations
        ]

        row["present_evaluations"] = sum(
            metrics["target_pixels"] > 0
            for metrics in all_metrics
        )

        row["absent_evaluations"] = sum(
            metrics["target_pixels"] == 0
            for metrics in all_metrics
        )

        for metric_name in metric_names:
            patient_means = []

            for cases in subjects.values():
                case_means = []

                for observations in cases.values():
                    case_means.append(
                        average(
                            metrics[metric_name]
                            for metrics in observations
                        )
                    )

                patient_means.append(
                    average(case_means)
                )

            row[f"subject_macro_{metric_name}"] = average(patient_means)

        rows.append(row)

    return rows