#!/usr/bin/env python3
"""Frozen RT-DETR Additive/FAM error analysis and qualitative figures.

The quantitative unit is the checkpoint/seed.  Predictions are matched to
ground truth one-to-one at IoU 0.50 after applying fixed confidence thresholds.
The qualitative manifest is generated from annotations only, before model
inference, and is shared by Additive and FAM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import load_fusion_model, load_run_config  # noqa: E402
from sarfusion.data import get_dataloaders  # noqa: E402
from sarfusion.data.utils import load_annotations  # noqa: E402
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint  # noqa: E402
from scripts.run_rtdetr_fam_level_ablation import (  # noqa: E402
    jsonable,
    resolve_device,
    tensors_to_cpu,
)


PROTOCOL_ID = "rtdetr_additive_fam_error_analysis_v1"
CONFIG_PATH = "parameters/RTDETR/rtdetr_protocol.yaml"
MANIFEST_PATH = "parameters/RTDETR/rtdetr_error_analysis_manifest.json"
SEEDS = (40, 41, 42, 43, 44)
FIGURE_SEED = 43
CONFIGURATIONS = {
    "additive": {"project": "RTDETR_Protocol", "use_fam": False},
    "fam": {"project": "RTDETR_FAM_Protocol", "use_fam": True},
}
CONFIDENCE_THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50)
PRIMARY_CONFIDENCE = 0.01
FIGURE_CONFIDENCES = (0.01, 0.25)
IOU_THRESHOLD = 0.50
EVALUATION_SIZE = 640
MAX_FIGURE_PREDICTIONS = 20
SIZE_LIMITS = {"small_max": 32**2, "medium_max": 96**2}
SUMMARY_METRICS = (
    "precision",
    "recall",
    "fp_per_image",
    "no_prediction_fraction",
    "empty_frame_fp_fraction",
    "nonempty_miss_frame_fraction",
    "mean_matched_iou",
    "small_recall",
    "medium_recall",
    "large_recall",
)


def stable_json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def size_category_from_normalized_box(box, evaluation_size=EVALUATION_SIZE):
    area = max(float(box[2]), 0.0) * max(float(box[3]), 0.0) * evaluation_size**2
    if area < SIZE_LIMITS["small_max"]:
        return "small"
    if area < SIZE_LIMITS["medium_max"]:
        return "medium"
    return "large"


def xywh_to_xyxy(boxes):
    boxes = torch.as_tensor(boxes, dtype=torch.float64).reshape(-1, 4)
    result = boxes.clone()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def pairwise_iou(first, second):
    first = torch.as_tensor(first, dtype=torch.float64).reshape(-1, 4)
    second = torch.as_tensor(second, dtype=torch.float64).reshape(-1, 4)
    if not len(first) or not len(second):
        return torch.zeros((len(first), len(second)), dtype=torch.float64)
    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
    first_area = (first[:, 2:] - first[:, :2]).clamp(min=0).prod(dim=1)
    second_area = (second[:, 2:] - second[:, :2]).clamp(min=0).prod(dim=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp(min=1e-12)


def greedy_match(gt_xyxy, pred_xyxy, iou_threshold=IOU_THRESHOLD):
    """Match boxes by descending IoU, with at most one match per box."""
    ious = pairwise_iou(gt_xyxy, pred_xyxy)
    candidates = []
    for gt_index in range(ious.shape[0]):
        for pred_index in range(ious.shape[1]):
            iou = float(ious[gt_index, pred_index])
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_gt = set()
    used_pred = set()
    matches = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append(
            {"gt_index": gt_index, "pred_index": pred_index, "iou": iou}
        )
    return matches


def analyze_image(gt_boxes, pred_boxes, pred_scores, confidence_threshold):
    gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float64).reshape(-1, 4)
    pred_boxes = torch.as_tensor(pred_boxes, dtype=torch.float64).reshape(-1, 4)
    pred_scores = torch.as_tensor(pred_scores, dtype=torch.float64).reshape(-1)
    if len(pred_boxes) != len(pred_scores):
        raise ValueError("Prediction boxes and scores have different lengths")

    keep = pred_scores >= confidence_threshold
    filtered_boxes = pred_boxes[keep]
    filtered_scores = pred_scores[keep]
    matches = greedy_match(
        xywh_to_xyxy(gt_boxes),
        xywh_to_xyxy(filtered_boxes),
    )
    matched_gt = {row["gt_index"] for row in matches}
    matched_pred = {row["pred_index"] for row in matches}
    size_counts = {size: 0 for size in ("small", "medium", "large")}
    size_tp = size_counts.copy()
    for gt_index, box in enumerate(gt_boxes):
        size = size_category_from_normalized_box(box)
        size_counts[size] += 1
        if gt_index in matched_gt:
            size_tp[size] += 1

    n_gt = len(gt_boxes)
    n_predictions = len(filtered_boxes)
    tp = len(matches)
    return {
        "n_gt": n_gt,
        "n_predictions": n_predictions,
        "tp": tp,
        "fp": n_predictions - tp,
        "fn": n_gt - tp,
        "no_prediction": n_predictions == 0,
        "has_annotations": n_gt > 0,
        "matched_iou_sum": sum(row["iou"] for row in matches),
        "matched_iou_count": len(matches),
        "size_gt": size_counts,
        "size_tp": size_tp,
        "matches": matches,
        "filtered_boxes": filtered_boxes.tolist(),
        "filtered_scores": filtered_scores.tolist(),
        "matched_gt_indices": sorted(matched_gt),
        "matched_prediction_indices": sorted(matched_pred),
    }


def dataset_item_paths(item):
    item_type, item_data = item
    if item_type != 2:
        raise RuntimeError("Frozen error analysis expects VIS+IR dataset items")
    (vis_path, vis_label), (ir_path, _ir_label) = item_data
    return Path(vis_path), Path(ir_path), Path(vis_label)


def relative_or_absolute(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_annotation_records(dataset, dataset_root):
    records = []
    session_positions = {}
    for sample_index, item in enumerate(dataset.items):
        vis_path, ir_path, label_path = dataset_item_paths(item)
        annotations = load_annotations(label_path)
        boxes = [row[1:] for row in annotations]
        sizes = [size_category_from_normalized_box(box) for box in boxes]
        session = vis_path.parents[1].name
        position = session_positions.get(session, 0)
        session_positions[session] = position + 1
        records.append(
            {
                "sample_index": sample_index,
                "session": session,
                "session_position": position,
                "vis_path": relative_or_absolute(vis_path, dataset_root),
                "ir_path": relative_or_absolute(ir_path, dataset_root),
                "label_path": relative_or_absolute(label_path, dataset_root),
                "label_sha256": file_sha256(label_path),
                "n_gt": len(boxes),
                "size_counts": {
                    size: sizes.count(size) for size in ("small", "medium", "large")
                },
            }
        )
    return records


def select_qualitative_samples(records):
    """Choose one small-target and one empty frame per session from GT only."""
    selected = []
    sessions = sorted({row["session"] for row in records})
    for session in sessions:
        rows = [row for row in records if row["session"] == session]
        length = len(rows)
        rules = (
            (
                "small_target",
                [row for row in rows if row["size_counts"]["small"] > 0],
                (length - 1) / 3,
            ),
            (
                "empty",
                [row for row in rows if row["n_gt"] == 0],
                2 * (length - 1) / 3,
            ),
        )
        for category, candidates, target_position in rules:
            if not candidates:
                raise RuntimeError(f"No {category} candidate in session {session}")
            chosen = min(
                candidates,
                key=lambda row: (
                    abs(row["session_position"] - target_position),
                    row["sample_index"],
                ),
            )
            selected.append(
                {
                    **chosen,
                    "selection_category": category,
                    "target_session_position": target_position,
                }
            )
    return selected


def build_manifest(dataset, dataset_root):
    records = build_annotation_records(dataset, dataset_root)
    size_totals = {
        size: sum(row["size_counts"][size] for row in records)
        for size in ("small", "medium", "large")
    }
    sessions = {}
    for session in sorted({row["session"] for row in records}):
        rows = [row for row in records if row["session"] == session]
        sessions[session] = {
            "n_frames": len(rows),
            "n_empty_frames": sum(row["n_gt"] == 0 for row in rows),
            "n_objects": sum(row["n_gt"] for row in rows),
        }
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "selection_uses_predictions": False,
        "selection_rule": (
            "for each session, closest eligible frame to one-third position "
            "for small targets and two-thirds position for empty frames"
        ),
        "figure_seed": FIGURE_SEED,
        "primary_confidence_threshold": PRIMARY_CONFIDENCE,
        "figure_confidence_thresholds": list(FIGURE_CONFIDENCES),
        "iou_threshold": IOU_THRESHOLD,
        "max_figure_predictions": MAX_FIGURE_PREDICTIONS,
        "size_definition": {
            "evaluation_size": EVALUATION_SIZE,
            "small": "area < 32^2 pixels",
            "medium": "32^2 <= area < 96^2 pixels",
            "large": "area >= 96^2 pixels",
        },
        "dataset_summary": {
            "n_frames": len(records),
            "n_empty_frames": sum(row["n_gt"] == 0 for row in records),
            "n_objects": sum(row["n_gt"] for row in records),
            "size_counts": size_totals,
            "sessions": sessions,
        },
        "selected_samples": select_qualitative_samples(records),
    }


def write_or_validate_manifest(path, dataset, dataset_root):
    expected = build_manifest(dataset, dataset_root)
    if path.is_file():
        with path.open(encoding="utf-8") as input_file:
            actual = json.load(input_file)
        if actual != expected:
            raise RuntimeError(
                f"Existing manifest {path} differs from the GT-only selection. "
                "Inspect dataset/labels before replacing it."
            )
        print(f"[manifest] verified {path}")
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(expected, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(f"[manifest] created {path}")
    return expected


def metadata_by_index(dataset, dataset_root):
    metadata = {}
    session_positions = {}
    for sample_index, item in enumerate(dataset.items):
        vis_path, ir_path, label_path = dataset_item_paths(item)
        session = vis_path.parents[1].name
        position = session_positions.get(session, 0)
        session_positions[session] = position + 1
        metadata[sample_index] = {
            "session": session,
            "session_position": position,
            "vis_path": relative_or_absolute(vis_path, dataset_root),
            "ir_path": relative_or_absolute(ir_path, dataset_root),
            "label_path": relative_or_absolute(label_path, dataset_root),
        }
    return metadata


def evaluate_checkpoint(
    model,
    test_loader,
    device,
    configuration,
    seed,
    metadata,
    selected_indices,
    max_batches=None,
):
    rows = []
    figure_samples = {}
    with torch.inference_mode():
        for batch_index, batch in enumerate(test_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device, non_blocking=True)
            output = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            predictions = tensors_to_cpu(output["predictions"])
            labels = tensors_to_cpu(batch["labels"])
            sample_indices = batch["sample_idx"].tolist()

            for sample_index, label, prediction in zip(
                sample_indices, labels, predictions
            ):
                gt_boxes = label["boxes"]
                pred_boxes = prediction["boxes"]
                pred_scores = prediction["scores"]
                for confidence in CONFIDENCE_THRESHOLDS:
                    analysis = analyze_image(
                        gt_boxes, pred_boxes, pred_scores, confidence
                    )
                    row = {
                        "configuration": configuration,
                        "seed": seed,
                        "sample_index": sample_index,
                        "session": metadata[sample_index]["session"],
                        "confidence_threshold": confidence,
                        "iou_threshold": IOU_THRESHOLD,
                        **{
                            key: value
                            for key, value in analysis.items()
                            if key
                            not in {
                                "matches",
                                "filtered_boxes",
                                "filtered_scores",
                                "matched_gt_indices",
                                "matched_prediction_indices",
                            }
                        },
                    }
                    rows.append(row)
                    if (
                        seed == FIGURE_SEED
                        and sample_index in selected_indices
                        and confidence in FIGURE_CONFIDENCES
                    ):
                        confidence_key = f"{confidence:.2f}"
                        figure_samples.setdefault(confidence_key, {})[
                            str(sample_index)
                        ] = {
                            "gt_boxes": torch.as_tensor(gt_boxes).tolist(),
                            "pred_boxes": analysis["filtered_boxes"],
                            "pred_scores": analysis["filtered_scores"],
                            "matches": analysis["matches"],
                            "matched_gt_indices": analysis["matched_gt_indices"],
                            "matched_prediction_indices": analysis[
                                "matched_prediction_indices"
                            ],
                            "counts": {
                                key: analysis[key]
                                for key in ("n_gt", "n_predictions", "tp", "fp", "fn")
                            },
                        }
    return rows, figure_samples


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_rows(rows):
    n_images = len(rows)
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in ("n_gt", "n_predictions", "tp", "fp", "fn")
    }
    empty_rows = [row for row in rows if not row["has_annotations"]]
    nonempty_rows = [row for row in rows if row["has_annotations"]]
    matched_count = sum(row["matched_iou_count"] for row in rows)
    size_gt = {
        size: sum(row["size_gt"][size] for row in rows)
        for size in ("small", "medium", "large")
    }
    size_tp = {
        size: sum(row["size_tp"][size] for row in rows)
        for size in ("small", "medium", "large")
    }
    return {
        "n_images": n_images,
        **totals,
        "precision": safe_ratio(totals["tp"], totals["tp"] + totals["fp"]),
        "recall": safe_ratio(totals["tp"], totals["tp"] + totals["fn"]),
        "fp_per_image": safe_ratio(totals["fp"], n_images),
        "no_prediction_fraction": safe_ratio(
            sum(bool(row["no_prediction"]) for row in rows), n_images
        ),
        "n_empty_frames": len(empty_rows),
        "empty_frame_fp_fraction": safe_ratio(
            sum(row["fp"] > 0 for row in empty_rows), len(empty_rows)
        ),
        "n_nonempty_frames": len(nonempty_rows),
        "nonempty_miss_frame_fraction": safe_ratio(
            sum(row["fn"] > 0 for row in nonempty_rows), len(nonempty_rows)
        ),
        "mean_matched_iou": safe_ratio(
            sum(row["matched_iou_sum"] for row in rows), matched_count
        ),
        "size_gt": size_gt,
        "size_tp": size_tp,
        **{
            f"{size}_recall": safe_ratio(size_tp[size], size_gt[size])
            for size in ("small", "medium", "large")
        },
    }


def summarize_values(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def build_aggregates(payloads, manifest, output_dir, protocol_complete):
    rows = [row for payload in payloads for row in payload["rows"]]
    checkpoint_summaries = []
    session_summaries = []
    for configuration in CONFIGURATIONS:
        for seed in SEEDS:
            for confidence in CONFIDENCE_THRESHOLDS:
                selected = [
                    row
                    for row in rows
                    if row["configuration"] == configuration
                    and row["seed"] == seed
                    and row["confidence_threshold"] == confidence
                ]
                if not selected:
                    continue
                checkpoint_summaries.append(
                    {
                        "configuration": configuration,
                        "seed": seed,
                        "confidence_threshold": confidence,
                        "metrics": summarize_rows(selected),
                    }
                )
                for session in sorted({row["session"] for row in selected}):
                    session_summaries.append(
                        {
                            "configuration": configuration,
                            "seed": seed,
                            "confidence_threshold": confidence,
                            "session": session,
                            "metrics": summarize_rows(
                                [row for row in selected if row["session"] == session]
                            ),
                        }
                    )

    across_seeds = {}
    paired_deltas = {}
    for confidence in CONFIDENCE_THRESHOLDS:
        confidence_key = f"{confidence:.2f}"
        across_seeds[confidence_key] = {}
        paired_deltas[confidence_key] = {}
        for configuration in CONFIGURATIONS:
            config_rows = [
                row
                for row in checkpoint_summaries
                if row["configuration"] == configuration
                and row["confidence_threshold"] == confidence
            ]
            across_seeds[confidence_key][configuration] = {
                metric: summarize_values([row["metrics"][metric] for row in config_rows])
                for metric in SUMMARY_METRICS
            }
        for metric in SUMMARY_METRICS:
            seed_values = {}
            for seed in SEEDS:
                additive = next(
                    (
                        row
                        for row in checkpoint_summaries
                        if row["configuration"] == "additive"
                        and row["seed"] == seed
                        and row["confidence_threshold"] == confidence
                    ),
                    None,
                )
                fam = next(
                    (
                        row
                        for row in checkpoint_summaries
                        if row["configuration"] == "fam"
                        and row["seed"] == seed
                        and row["confidence_threshold"] == confidence
                    ),
                    None,
                )
                if additive is None or fam is None:
                    continue
                first = additive["metrics"][metric]
                second = fam["metrics"][metric]
                if first is not None and second is not None:
                    seed_values[str(seed)] = second - first
            paired_deltas[confidence_key][metric] = {
                "seed_values": seed_values,
                "summary": summarize_values(seed_values.values()),
            }

    combined = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_complete": protocol_complete,
        "manifest": manifest,
        "manifest_sha256": stable_json_hash(manifest),
        "configurations": CONFIGURATIONS,
        "seeds": list(SEEDS),
        "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
        "primary_confidence_threshold": PRIMARY_CONFIDENCE,
        "iou_threshold": IOU_THRESHOLD,
        "matching": "one-to-one greedy matching by descending IoU",
        "rows": rows,
        "checkpoint_summaries": checkpoint_summaries,
        "session_summaries": session_summaries,
        "across_seed_summaries": across_seeds,
        "paired_deltas_fam_minus_additive": paired_deltas,
    }
    combined_path = output_dir / "rtdetr_error_analysis.json"
    with combined_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(combined), output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    csv_path = output_dir / "rtdetr_error_analysis_checkpoints.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = [
            "configuration",
            "seed",
            "confidence_threshold",
            "n_images",
            "n_gt",
            "n_predictions",
            "tp",
            "fp",
            "fn",
            *SUMMARY_METRICS,
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in checkpoint_summaries:
            writer.writerow(
                {
                    "configuration": row["configuration"],
                    "seed": row["seed"],
                    "confidence_threshold": row["confidence_threshold"],
                    **{key: row["metrics"].get(key) for key in fieldnames[3:]},
                }
            )
    print(f"Saved combined analysis: {combined_path}")
    print(f"Saved checkpoint CSV: {csv_path}")
    return combined


def denormalized_xyxy(box, width, height):
    x, y, w, h = [float(value) for value in box]
    return (
        (x - w / 2) * width,
        (y - h / 2) * height,
        w * width,
        h * height,
    )


def draw_ground_truth(ax, boxes, matched=None):
    from matplotlib.patches import Rectangle

    matched = set(range(len(boxes))) if matched is None else set(matched)
    width, height = ax.images[0].get_array().shape[1], ax.images[0].get_array().shape[0]
    for index, box in enumerate(boxes):
        x, y, w, h = denormalized_xyxy(box, width, height)
        color = "lime" if index in matched else "orange"
        ax.add_patch(
            Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2.0)
        )


def draw_predictions(ax, sample):
    from matplotlib.patches import Rectangle

    width, height = ax.images[0].get_array().shape[1], ax.images[0].get_array().shape[0]
    matched_predictions = set(sample["matched_prediction_indices"])
    order = sorted(
        range(len(sample["pred_scores"])),
        key=lambda index: -sample["pred_scores"][index],
    )[:MAX_FIGURE_PREDICTIONS]
    for index in order:
        x, y, w, h = denormalized_xyxy(sample["pred_boxes"][index], width, height)
        color = "deepskyblue" if index in matched_predictions else "red"
        ax.add_patch(
            Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.8)
        )
        ax.text(
            x,
            max(0, y - 4),
            f"{sample['pred_scores'][index]:.2f}",
            color=color,
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
        )


def render_qualitative_figures(payloads, manifest, dataset_root, output_dir):
    from PIL import Image
    import matplotlib.pyplot as plt

    lookup = {}
    for payload in payloads:
        if payload["seed"] != FIGURE_SEED:
            continue
        for confidence_key, confidence_samples in payload["figure_samples"].items():
            for sample_index, sample in confidence_samples.items():
                lookup[
                    (payload["configuration"], float(confidence_key), int(sample_index))
                ] = sample

    figure_dir = output_dir / "figures" / "qualitative"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for confidence in FIGURE_CONFIDENCES:
        for record in manifest["selected_samples"]:
            sample_index = record["sample_index"]
            additive = lookup.get(("additive", confidence, sample_index))
            fam = lookup.get(("fam", confidence, sample_index))
            if additive is None or fam is None:
                continue
            vis_path = Path(record["vis_path"])
            ir_path = Path(record["ir_path"])
            if not vis_path.is_absolute():
                vis_path = Path(dataset_root) / vis_path
            if not ir_path.is_absolute():
                ir_path = Path(dataset_root) / ir_path
            rgb = Image.open(vis_path).convert("RGB")
            infrared = Image.open(ir_path).convert("L")

            figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
            axes[0, 0].imshow(rgb)
            draw_ground_truth(axes[0, 0], additive["gt_boxes"])
            axes[0, 0].set_title(f"RGB + ground truth (n={len(additive['gt_boxes'])})")
            axes[0, 1].imshow(infrared, cmap="gray")
            axes[0, 1].set_title("IR")
            for axis, label, sample in (
                (axes[1, 0], "Additive", additive),
                (axes[1, 1], "FAM", fam),
            ):
                axis.imshow(rgb)
                draw_ground_truth(axis, sample["gt_boxes"], sample["matched_gt_indices"])
                draw_predictions(axis, sample)
                counts = sample["counts"]
                axis.set_title(
                    f"{label}: TP={counts['tp']} FP={counts['fp']} FN={counts['fn']}"
                )
            for axis in axes.flat:
                axis.axis("off")
            figure.suptitle(
                f"MtErie sample {sample_index} · {record['selection_category']} · "
                f"seed {FIGURE_SEED} · conf≥{confidence:.2f}, IoU≥{IOU_THRESHOLD:.2f}\n"
                "GT matched=green, GT missed=orange, TP prediction=blue, FP=red",
                fontsize=11,
            )
            confidence_label = str(confidence).replace(".", "")
            output_path = figure_dir / (
                f"sample_{sample_index:03d}_{record['selection_category']}_"
                f"conf_{confidence_label}.png"
            )
            figure.savefig(output_path, dpi=180)
            plt.close(figure)
            print(f"Saved qualitative figure: {output_path}")


def render_paired_summary(combined, output_dir):
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in combined["checkpoint_summaries"]
        if row["confidence_threshold"] == PRIMARY_CONFIDENCE
    ]
    if len(rows) != len(CONFIGURATIONS) * len(SEEDS):
        return
    panels = (
        ("recall", "Recall", True),
        ("fp_per_image", "False positives / image", False),
        ("empty_frame_fp_fraction", "Empty frames with ≥1 FP", False),
        ("nonempty_miss_frame_fraction", "Non-empty frames with ≥1 FN", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, (metric, title, higher_is_better) in zip(axes.flat, panels):
        for seed in SEEDS:
            values = []
            for configuration in CONFIGURATIONS:
                row = next(
                    item
                    for item in rows
                    if item["configuration"] == configuration and item["seed"] == seed
                )
                values.append(row["metrics"][metric])
            axis.plot([0, 1], values, marker="o", alpha=0.75, label=f"seed {seed}")
        axis.set_xticks([0, 1], ["Additive", "FAM"])
        axis.set_title(title + (" ↑" if higher_is_better else " ↓"))
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    figure.suptitle(
        f"RT-DETR paired error analysis · conf≥{PRIMARY_CONFIDENCE:.2f}, "
        f"IoU≥{IOU_THRESHOLD:.2f}"
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "rtdetr_additive_fam_error_summary.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Saved paired summary: {output_path}")


def raw_result_path(output_dir, configuration, seed):
    return output_dir / "raw" / f"{configuration}_seed_{seed}.json"


def load_compatible_raw(path, configuration, seed, checkpoint, manifest_hash, max_batches):
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "configuration": configuration,
        "seed": seed,
        "checkpoint": str(Path(checkpoint).resolve()),
        "manifest_sha256": manifest_hash,
        "max_batches": max_batches,
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"Existing result {path} is incompatible; inspect before --force")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--dataset-root", default="dataset/WiSARD")
    parser.add_argument("--output-dir", default="out/rtdetr_error_analysis")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=list(CONFIGURATIONS),
        default=list(CONFIGURATIONS),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    config_path = (REPO_ROOT / args.config).resolve()
    dataset_root = (REPO_ROOT / args.dataset_root).resolve()
    manifest_path = (REPO_ROOT / args.manifest).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    run_config = load_run_config(config_path, run_index=0)
    dataset_params = run_config["dataset"]
    dataset_params["root"] = str(dataset_root)
    dataloader_params = run_config["dataloader"]
    if args.batch_size is not None:
        dataloader_params["batch_size"] = args.batch_size
    if args.num_workers is not None:
        dataloader_params["num_workers"] = args.num_workers
    (_train, _val, test_loader), _denormalize = get_dataloaders(
        dataset_params, dataloader_params, seed=42
    )
    manifest = write_or_validate_manifest(
        manifest_path, test_loader.dataset, dataset_root
    )
    if args.prepare_only:
        print(json.dumps(manifest["dataset_summary"], indent=2, sort_keys=True))
        print("Selected samples:", [row["sample_index"] for row in manifest["selected_samples"]])
        return

    checkpoints = {
        (configuration, seed): resolve_local_wandb_checkpoint(
            settings["project"],
            seed,
            checkpoint="latest",
            wandb_root=REPO_ROOT / "wandb",
        )
        for configuration, settings in CONFIGURATIONS.items()
        if configuration in args.configurations
        for seed in args.seeds
    }
    if args.dry_run:
        for (configuration, seed), checkpoint in checkpoints.items():
            print(f"configuration={configuration} seed={seed} checkpoint={checkpoint}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    manifest_hash = stable_json_hash(manifest)
    metadata = metadata_by_index(test_loader.dataset, dataset_root)
    selected_indices = {
        row["sample_index"] for row in manifest["selected_samples"]
    }
    payloads = []
    for configuration, settings in CONFIGURATIONS.items():
        if configuration not in args.configurations:
            continue
        for seed in args.seeds:
            checkpoint = checkpoints[(configuration, seed)]
            path = raw_result_path(output_dir, configuration, seed)
            if path.is_file() and not args.force:
                payload = load_compatible_raw(
                    path,
                    configuration,
                    seed,
                    checkpoint,
                    manifest_hash,
                    args.max_batches,
                )
                payloads.append(payload)
                print(f"[skip] {path}")
                continue

            current_config = load_run_config(config_path, run_index=seed - 40)
            model_params = current_config["model"]
            model_params["params"].update(
                {
                    "use_fam": settings["use_fam"],
                    "freeze_fam": False,
                    "fam_variant": "current_dcnv2",
                    "ir_dropout_rate": 0.0,
                    "spatial_jitter_std": 0.0,
                }
            )
            print(f"[run] configuration={configuration} seed={seed}")
            model = load_fusion_model(model_params, checkpoint, device)
            rows, figure_samples = evaluate_checkpoint(
                model,
                test_loader,
                device,
                configuration,
                seed,
                metadata,
                selected_indices,
                max_batches=args.max_batches,
            )
            payload = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "protocol_complete": args.max_batches is None,
                "configuration": configuration,
                "project": settings["project"],
                "seed": seed,
                "checkpoint": str(Path(checkpoint).resolve()),
                "manifest_sha256": manifest_hash,
                "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
                "primary_confidence_threshold": PRIMARY_CONFIDENCE,
                "iou_threshold": IOU_THRESHOLD,
                "max_batches": args.max_batches,
                "rows": rows,
                "figure_samples": figure_samples,
            }
            with path.open("w", encoding="utf-8") as output_file:
                json.dump(jsonable(payload), output_file, indent=2, sort_keys=True)
                output_file.write("\n")
            payloads.append(payload)
            primary = summarize_rows(
                [row for row in rows if row["confidence_threshold"] == PRIMARY_CONFIDENCE]
            )
            print(
                f"[done] {configuration} seed={seed} recall={primary['recall']:.4f} "
                f"FP/image={primary['fp_per_image']:.4f} -> {path}"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    combined = build_aggregates(
        payloads,
        manifest,
        output_dir,
        protocol_complete=(
            args.max_batches is None
            and set(args.configurations) == set(CONFIGURATIONS)
            and set(args.seeds) == set(SEEDS)
        ),
    )
    render_qualitative_figures(payloads, manifest, dataset_root, output_dir)
    render_paired_summary(combined, output_dir)


if __name__ == "__main__":
    main()
