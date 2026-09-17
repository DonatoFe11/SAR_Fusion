#!/usr/bin/env python3
"""Cache historical Additive/FAM detections and measure recall vs FP/image.

Inference only. Original checkpoint hashes, inventories and preprocessing are
reused; native model queries are collected without a confidence cutoff. Each
completed acquisition/checkpoint is cached atomically, allowing safe resume.
Smoke tests are isolated from scientific outputs. Plotting never selects a
deployment threshold or a checkpoint using the test data.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import load_fusion_model
from sarfusion.utils.utils import load_yaml
from scripts.rtdetr_froc_metrics import detection_events, empirical_curve
from scripts.run_rtdetr_carnation_stress_test import file_sha256, stable_json_hash
from scripts.run_rtdetr_paired_modality_evaluation import (
    build_paired_loader, build_source_manifest, load_protocol as load_paired_protocol,
)
from scripts.run_rtdetr_unused_acquisition_confirmation import (
    build_loader, build_manifests, build_run, load_payload, resolve_checkpoints,
    resolve_repo_path, set_evaluation_seed,
)

DEFAULT_PROTOCOL = "parameters/RTDETR/rtdetr_recall_fppi.yaml"
CONFIGURATIONS = ["historical_additive", "historical_fam"]
ACQUISITIONS = ["mterie", "carnation_0025_0026", "fhl_0407_0408"]
COUNTS = {"mterie": (708, 1770, 19), "carnation_0025_0026": (1313, 5238, 100),
          "fhl_0407_0408": (1035, 2022, 239)}


def load_protocol(path):
    protocol = load_yaml(path)["protocol"]
    required = {
        "id": "rtdetr_historical_recall_fppi_v1", "checkpoint": "latest",
        "seeds": [40, 41, 42, 43, 44], "configurations": CONFIGURATIONS,
        "acquisitions": ACQUISITIONS, "condition": "vis_ir", "ground_truth": "vis",
        "collection_threshold": 0.0, "iou_threshold": 0.5,
        "postprocessing": "native_single_class_queries_no_added_nms",
        "matching": "confidence_descending_best_unmatched_iou",
        "equal_scores": "atomic_threshold_group_stable_prediction_order",
        "sweep": "all_distinct_prediction_scores_plus_reject_all",
        "fppi_denominator": "all_images_including_empty_frames",
        "fppi_budgets": [0.1, 0.5, 1.0],
        "fppi_grid": {"min": 0.01, "max": 10.0, "points": 301},
        "budget_rule": "maximum_empirical_recall_at_or_below_budget_no_interpolation",
        "aggregation": "paired_seeds_mean_sample_sd_on_common_fppi_grid",
        "interpretation": {
            "model_selection_allowed": False, "checkpoint_selection_allowed": False,
            "deployment_threshold_selection_allowed": False,
            "frame_independence_claim_allowed": False,
            "negative_results_must_be_reported": True,
        },
    }
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Recall-FPPI protocol changed: {key}")
    return protocol


def xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape == (0,):
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Expected boxes in Nx4 center-XYWH format")
    return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2), axis=1)


def verify_checkpoint_keys(model, checkpoint):
    """Fail closed even though the shared loading helper only warns on mismatch."""
    with safe_open(checkpoint, framework="pt") as source:
        keys = {key.removeprefix("model.") for key in source.keys()}
    expected = set(model.state_dict())
    aliases = {}
    for name, value in list(model.named_parameters(remove_duplicate=False)) + list(
        model.named_buffers(remove_duplicate=False)
    ):
        aliases.setdefault(id(value), set()).add(name)
    recovered = set()
    for names in aliases.values():
        if names & keys:
            recovered.update(names)
    missing = expected - keys - recovered
    unexpected = keys - expected
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def validate_arrays(arrays, metadata):
    n_images = metadata["n_images"]
    if type(n_images) is not int or n_images < 1:
        raise ValueError("Invalid cached image count")
    for name in ("n_gt", "n_empty", "native_queries"):
        if type(metadata[name]) is not int or metadata[name] < 0:
            raise ValueError(f"Invalid cached {name}")
    if metadata["native_queries"] == 0:
        raise ValueError("Native query count must be positive")
    for prefix, values in (("gt", arrays["gt_boxes"]), ("pred", arrays["pred_boxes"])):
        offsets = arrays[f"{prefix}_offsets"]
        if (offsets.shape != (n_images + 1,) or offsets.dtype.kind not in "iu"
                or offsets[0] != 0 or offsets[-1] != len(values)
                or np.any(offsets[1:] < offsets[:-1]) or values.shape != (len(values), 4)):
            raise ValueError(f"Invalid cached {prefix} offsets/boxes")
        if not np.isfinite(values).all() or np.any(values[:, 2:] <= values[:, :2]):
            raise ValueError(f"Invalid cached {prefix} box coordinates")
    if arrays["scores"].shape != (len(arrays["pred_boxes"]),):
        raise ValueError("Cached score count differs from boxes")
    if (not np.isfinite(arrays["scores"]).all()
            or np.any((arrays["scores"] < 0) | (arrays["scores"] > 1))):
        raise ValueError("Invalid cached confidence scores")
    if np.any(np.diff(arrays["pred_offsets"]) != metadata["native_queries"]):
        raise ValueError("Cached predictions do not contain every native query")
    if len(arrays["gt_boxes"]) != metadata["n_gt"]:
        raise ValueError("Cached GT count differs from metadata")
    if int((np.diff(arrays["gt_offsets"]) == 0).sum()) != metadata["n_empty"]:
        raise ValueError("Cached empty-frame count differs from metadata")
    if (arrays["sample_indices"].dtype.kind not in "iu"
            or not np.array_equal(arrays["sample_indices"], np.arange(n_images))):
        raise ValueError("Cached frame order is incomplete or shuffled")


def load_cache(path, expected):
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata"].item()))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"Incompatible cache {path}: {key}; use a new output directory")
        arrays = {key: source[key].copy() for key in (
            "gt_boxes", "gt_offsets", "pred_boxes", "pred_offsets", "scores", "sample_indices"
        )}
    validate_arrays(arrays, metadata)
    return arrays, metadata


def collect_predictions(model, loader, device, expected, max_batches=None):
    gt_boxes, pred_boxes, scores, indices = [], [], [], []
    gt_offsets, pred_offsets = [0], [0]
    start = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc=expected["job"], mininterval=10)):
            if max_batches is not None and batch_index >= max_batches:
                break
            mask = batch.get("pixel_mask")
            output = model(pixel_values=batch["pixel_values"].to(device),
                           pixel_mask=None if mask is None else mask.to(device))
            if len(output["predictions"]) != len(batch["labels"]):
                raise RuntimeError("Prediction/target batch size mismatch")
            if len(batch["sample_idx"]) != len(batch["labels"]):
                raise RuntimeError("Sample-index/target batch size mismatch")
            for sample, label, prediction in zip(
                batch["sample_idx"].tolist(), batch["labels"], output["predictions"]
            ):
                if sample != len(indices):
                    raise RuntimeError("Inference loader is shuffled or skips an image")
                gt = xywh_to_xyxy(label["boxes"].cpu().numpy())
                pred = xywh_to_xyxy(prediction["boxes"].cpu().numpy())
                score = prediction["scores"].cpu().numpy()
                # Every native query must survive collection; no threshold/top-k
                # change or class mixing may silently shorten the sweep.
                if len(pred) != expected["native_queries"]:
                    raise RuntimeError("Not all native single-class queries were collected")
                if (prediction["labels"].shape != (len(pred),)
                        or torch.any(prediction["labels"] != 0)):
                    raise RuntimeError("Unexpected non-person prediction class")
                detection_events(gt, pred, score)  # validate before persisting
                gt_boxes.append(gt)
                pred_boxes.append(pred)
                scores.append(score)
                indices.append(sample)
                gt_offsets.append(gt_offsets[-1] + len(gt))
                pred_offsets.append(pred_offsets[-1] + len(pred))
    if not indices:
        raise RuntimeError("No frames evaluated")
    arrays = dict(gt_boxes=np.concatenate(gt_boxes), pred_boxes=np.concatenate(pred_boxes),
                  scores=np.concatenate(scores), gt_offsets=np.asarray(gt_offsets),
                  pred_offsets=np.asarray(pred_offsets), sample_indices=np.asarray(indices))
    metadata = dict(expected, n_images=len(indices), n_gt=gt_offsets[-1],
                    n_empty=int((np.diff(gt_offsets) == 0).sum()),
                    elapsed_seconds=time.monotonic() - start)
    validate_arrays(arrays, metadata)
    return arrays, metadata


def make_curve(arrays, metadata, iou_threshold):
    events = []
    for index in range(metadata["n_images"]):
        gt = arrays["gt_boxes"][arrays["gt_offsets"][index]:arrays["gt_offsets"][index + 1]]
        sl = slice(arrays["pred_offsets"][index], arrays["pred_offsets"][index + 1])
        events.append(detection_events(gt, arrays["pred_boxes"][sl], arrays["scores"][sl],
                                       iou_threshold=iou_threshold))
    return empirical_curve(events, num_images=metadata["n_images"], total_gt=metadata["n_gt"])


def prepare(protocol, acquisitions, batch_size, workers):
    paired = load_paired_protocol(resolve_repo_path(protocol["mterie_source_protocol"]))
    confirmation = load_payload(resolve_repo_path(protocol["confirmation_source_protocol"]))["protocol"]
    # Only the two historical families are in scope. No dependency on new Stage-B runs.
    confirmation = copy.deepcopy(confirmation)
    confirmation["configurations"] = {
        key: confirmation["configurations"][key] for key in CONFIGURATIONS
    }
    checkpoints = resolve_checkpoints(confirmation, {}, verify_hashes=True)
    manifests, loaders = {}, {}
    if "mterie" in acquisitions:
        manifest = build_source_manifest(paired)
        manifests["mterie"] = manifest
        loaders["mterie"] = build_paired_loader(paired, manifest, batch_size, workers)
    extra = copy.deepcopy(confirmation)
    extra["acquisitions"] = {key: value for key, value in extra["acquisitions"].items()
                             if key in acquisitions}
    for acquisition, manifest in build_manifests(extra).items():
        manifests[acquisition] = manifest
        loaders[acquisition] = build_loader(extra, manifest, batch_size, workers)
    # Supplement the historical MtErie size/label inventory with image hashes.
    root = resolve_repo_path(confirmation["dataset_root"])
    for manifest in manifests.values():
        for row in manifest["rows"]:
            for channel in ("vis", "ir"):
                key = f"{channel}_image_sha256"
                if key not in row:
                    row[key] = file_sha256(root / row[f"{channel}_image"])
        manifest["content_sha256"] = stable_json_hash(manifest["rows"])
    return paired, confirmation, checkpoints, manifests, loaders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--seeds", type=int, nargs="+", choices=range(40, 45))
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS)
    parser.add_argument("--acquisitions", nargs="+", choices=ACQUISITIONS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--max-batches", type=int, help="Isolated smoke test; never publishes thesis results")
    args = parser.parse_args()
    protocol = load_protocol(resolve_repo_path(args.protocol))
    batch_size = args.batch_size if args.batch_size is not None else protocol["batch_size"]
    workers = args.workers if args.workers is not None else protocol["workers"]
    if batch_size < 1 or workers < 0 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("batch-size/max-batches must be positive; workers must be nonnegative")
    acquisitions = args.acquisitions or ACQUISITIONS
    configurations = args.configurations or CONFIGURATIONS
    seeds = args.seeds or protocol["seeds"]
    if any(len(items) != len(set(items)) for items in (acquisitions, configurations, seeds)):
        parser.error("Duplicate job selectors")
    paired, confirmation, checkpoints, manifests, loaders = prepare(
        protocol, acquisitions, batch_size, workers)
    source_hash = stable_json_hash({
        "protocol": protocol, "paired": paired, "confirmation": confirmation,
        "preprocessing_configs": {
            path: file_sha256(resolve_repo_path(path)) for path in (
                paired["training_config"], confirmation["preprocessor_training_config"])
        },
    })
    inference_files = [
        Path(__file__).resolve(), REPO_ROOT / "fam_alignment_check.py",
        REPO_ROOT / "scripts/run_rtdetr_unused_acquisition_confirmation.py",
        REPO_ROOT / "scripts/run_rtdetr_paired_modality_evaluation.py",
    ]
    inference_files.extend(
        path for folder in ("sarfusion/models", "sarfusion/data", "sarfusion/utils")
        for path in sorted((REPO_ROOT / folder).rglob("*.py"))
    )
    implementation_hash = stable_json_hash({
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in inference_files
    })
    print(f"Preflight OK: 10 checkpoint hashes; inventories "
          f"{ {key: len(loaders[key].dataset) for key in acquisitions} }", flush=True)
    if args.dry_run:
        return
    output_dir = resolve_repo_path(args.output_dir or protocol["output_dir"])
    if args.max_batches is not None:
        output_dir = output_dir / f"smoke_{args.max_batches}_batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prevent two invocations racing on cache or published result files.
    output_lock = (output_dir / ".run.lock").open("a")
    try:
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another analysis is already using {output_dir}") from error
    set_evaluation_seed(protocol["evaluation_seed"])
    records = []
    for configuration in configurations:
        for seed in seeds:
            checkpoint = checkpoints[(configuration, seed)]
            model = None
            for acquisition in acquisitions:
                job = f"{acquisition}_{configuration}_seed{seed}"
                path = output_dir / "predictions" / f"{job}.npz"
                expected = {
                    "schema": 1, "job": job, "source_sha256": source_hash,
                    "implementation_sha256": implementation_hash,
                    "checkpoint_sha256": checkpoint["sha256"],
                    "inventory_sha256": manifests[acquisition]["inventory_sha256"],
                    "content_sha256": manifests[acquisition]["content_sha256"],
                    "acquisition": acquisition, "configuration": configuration, "seed": seed,
                    "collection_threshold": 0.0, "native_queries": 300,
                    "batch_size": batch_size, "max_batches": args.max_batches,
                    "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                }
                if path.exists():
                    arrays, metadata = load_cache(path, expected)
                    print(f"[cached] {job}", flush=True)
                else:
                    if args.summarize_only:
                        raise FileNotFoundError(f"Incomplete campaign: {path}")
                    if model is None:
                        if args.device == "cuda" and not torch.cuda.is_available():
                            raise RuntimeError("CUDA unavailable: run on the GPU host; no automatic CPU fallback")
                        run = build_run(confirmation, configuration, seed, {})
                        run["model"]["params"]["threshold"] = 0.0
                        model = load_fusion_model(run["model"], checkpoint["path"], torch.device(args.device))
                        verify_checkpoint_keys(model, checkpoint["path"])
                        if model.model.config.num_queries != 300 or model.model.config.num_labels != 1:
                            raise RuntimeError("Unexpected query count or class count")
                    arrays, metadata = collect_predictions(model, loaders[acquisition], args.device,
                                                            expected, args.max_batches)
                    if args.max_batches is None:
                        actual = tuple(metadata[key] for key in ("n_images", "n_gt", "n_empty"))
                        if actual != COUNTS[acquisition]:
                            raise RuntimeError(f"Processed annotations differ from inventory: {actual}")
                    atomic_npz(path, metadata=json.dumps(metadata, sort_keys=True), **arrays)
                    print(f"[saved] {job} ({metadata['elapsed_seconds'] / 60:.1f} min)", flush=True)
                if args.max_batches is None:
                    actual = tuple(metadata[key] for key in ("n_images", "n_gt", "n_empty"))
                    if actual != COUNTS[acquisition]:
                        raise RuntimeError(f"Cached or processed inventory mismatch: {actual}")
                curve = make_curve(arrays, metadata, protocol["iou_threshold"])
                atomic_npz(output_dir / "curves" / f"{job}.npz", **curve)
                records.append(dict(metadata, curve=curve, cache_path=str(path)))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if args.max_batches is None:
        from scripts.rtdetr_froc_report import write_report
        complete = len(records) == 30
        write_report(records, protocol, output_dir, publish=complete)
        print(f"Report saved; full campaign complete={complete}", flush=True)
    else:
        print(f"Smoke test OK ({len(records)} jobs); no scientific results published", flush=True)


if __name__ == "__main__":
    main()
