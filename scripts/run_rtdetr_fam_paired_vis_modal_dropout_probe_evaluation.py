#!/usr/bin/env python3
"""Evaluate the frozen seed-40 paired-VIS Modal Dropout screen on FHL.

The runner compares matched baseline/candidate ``best`` checkpoints on one
paired 896-frame loader under three channel interventions, then on the same IR
counterparts in their native coordinate system. MtErie is never constructed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import load_fusion_model  # noqa: E402
from sarfusion.data.utils import build_preprocessor, get_collate_fn  # noqa: E402
from sarfusion.data.wisard import (  # noqa: E402
    IR_ITEM,
    MULTI_MODALITY_ITEM,
    WiSARDDataset,
)
from sarfusion.models.checkpoints import (  # noqa: E402
    resolve_local_wandb_checkpoint,
)
from sarfusion.utils.grid import make_grid  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402
from scripts.run_rtdetr_carnation_stress_test import (  # noqa: E402
    SCALAR_METRICS,
    file_sha256,
    load_compatible_raw,
    stable_json_hash,
)
from scripts.run_rtdetr_fam_level_ablation import (  # noqa: E402
    jsonable,
    resolve_device,
)
from scripts.run_rtdetr_paired_modality_evaluation import (  # noqa: E402
    evaluate_modalities,
)


DEFAULT_PROTOCOL = (
    "parameters/RTDETR/"
    "rtdetr_fam_paired_vis_modal_dropout_probe_evaluation.yaml"
)
PROTOCOL_ID = "rtdetr_fam_paired_vis_modal_dropout_probe_evaluation_v1"
FROZEN_RULE_COMMIT = "e7ccd39bc9b38507747089306ec43164b13c7e0c"
CONFIGURATIONS = ("baseline", "paired_vis_dropout")
CONDITIONS = {
    "paired_vis_ir": "vis_ir",
    "paired_vis": "vis",
    "paired_ir_vis_gt": "ir",
}
NATIVE_CONDITION = "native_ir_ir_gt"
EXPECTED_TRAIN_STEPS = 7810
FUSION_REPRODUCTION_TOLERANCE = 0.0002


def resolve_repo_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def normalize(value):
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def load_protocol(path):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Probe evaluation YAML must contain a protocol mapping")
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError("Unexpected paired-VIS probe protocol id")
    if protocol.get("status") != "frozen_before_modality_inference":
        raise ValueError("Probe evaluation must remain frozen before inference")
    if protocol.get("frozen_rule_commit") != FROZEN_RULE_COMMIT:
        raise ValueError("Promotion rule does not point to the pre-training commit")
    if protocol.get("seed") != 40 or protocol.get("checkpoint") != "best":
        raise ValueError("Probe evaluation requires seed 40 best checkpoints")
    if float(protocol.get("confidence_threshold", -1.0)) != 0.01:
        raise ValueError("Probe confidence threshold must remain 0.01")
    if tuple(protocol.get("configurations", {})) != CONFIGURATIONS:
        raise ValueError("Probe configurations or their order changed")
    if protocol.get("conditions") != {
        "paired_vis_ir": "keep_all_four_channels_vis_gt",
        "paired_vis": "zero_ir_channel_vis_gt",
        "paired_ir_vis_gt": "zero_three_vis_channels_vis_gt",
        "native_ir_ir_gt": "native_ir_canvas_zero_rgb_ir_gt",
    }:
        raise ValueError("Probe evaluation conditions changed")

    source = protocol.get("source", {})
    if source.get("pairing") != "existing_wisard_sorted_zip":
        raise ValueError("Probe must reproduce the existing sorted-zip pairing")
    if source.get("paired_ground_truth") != "vis":
        raise ValueError("Paired conditions must retain VIS ground truth")
    if source.get("native_ir_ground_truth") != "ir":
        raise ValueError("Native-IR condition must use IR ground truth")

    expected_rule = {
        "rule": "all_conditions_required",
        "minimum_paired_ir_map50_gain": 0.03,
        "minimum_fusion_map50_delta": -0.01,
        "minimum_native_ir_map50_delta": -0.03,
        "if_pass": "expand_unchanged_to_seeds_41_44",
        "if_fail": "close_pure_paired_vis_replacement",
    }
    if protocol.get("promotion_rule") != expected_rule:
        raise ValueError("Probe promotion rule differs from the pre-training freeze")
    interpretation = protocol.get("interpretation", {})
    if not interpretation or any(value is not False for value in interpretation.values()):
        raise ValueError("Probe interpretation constraints were relaxed")
    return protocol


def sorted_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(path for path in directory.iterdir() if path.is_file())


def temporal_index(path):
    return Path(path).stem.rsplit("_", 1)[-1]


def build_inventory(protocol):
    """Reproduce the exact sorted zip used by ``build_wisard_items``."""
    dataset_root = resolve_repo_path(protocol["dataset_root"])
    rows = []
    vis_inventory = 0
    ir_inventory = 0
    vis_boxes = 0
    ir_boxes = 0
    vis_empty = 0
    ir_empty = 0
    unpaired_vis = []

    for vis_folder, ir_folder in protocol["source"]["paired_folders"]:
        vis_images = sorted_files(dataset_root / vis_folder / "images")
        vis_labels = sorted_files(dataset_root / vis_folder / "labels")
        ir_images = sorted_files(dataset_root / ir_folder / "images")
        ir_labels = sorted_files(dataset_root / ir_folder / "labels")
        if len(vis_images) != len(vis_labels):
            raise RuntimeError("VIS image/label inventory is unbalanced")
        if len(ir_images) != len(ir_labels):
            raise RuntimeError("IR image/label inventory is unbalanced")
        vis_inventory += len(vis_images)
        ir_inventory += len(ir_images)
        unpaired_vis.extend(path.name for path in vis_images[len(ir_images) :])

        for vis_image, vis_label, ir_image, ir_label in zip(
            vis_images, vis_labels, ir_images, ir_labels
        ):
            if temporal_index(vis_image) != temporal_index(ir_image):
                raise RuntimeError(
                    f"Sorted-zip temporal shift: {vis_image.name} vs {ir_image.name}"
                )
            vis_bytes = vis_label.read_bytes()
            ir_bytes = ir_label.read_bytes()
            vis_lines = [line for line in vis_bytes.decode().splitlines() if line.strip()]
            ir_lines = [line for line in ir_bytes.decode().splitlines() if line.strip()]
            vis_boxes += len(vis_lines)
            ir_boxes += len(ir_lines)
            vis_empty += int(not vis_lines)
            ir_empty += int(not ir_lines)
            rows.append(
                {
                    "vis_image": str(vis_image.relative_to(dataset_root)),
                    "vis_image_size": vis_image.stat().st_size,
                    "vis_label": str(vis_label.relative_to(dataset_root)),
                    "vis_label_sha256": hashlib.sha256(vis_bytes).hexdigest(),
                    "ir_image": str(ir_image.relative_to(dataset_root)),
                    "ir_image_size": ir_image.stat().st_size,
                    "ir_label": str(ir_label.relative_to(dataset_root)),
                    "ir_label_sha256": hashlib.sha256(ir_bytes).hexdigest(),
                }
            )

    inventory = {
        "dataset_root": str(dataset_root),
        "pairing": "existing_wisard_sorted_zip",
        "paired_ground_truth": "vis",
        "native_ir_ground_truth": "ir",
        "vis_inventory": vis_inventory,
        "ir_inventory": ir_inventory,
        "paired_frames": len(rows),
        "unpaired_vis": unpaired_vis,
        "vis_boxes": vis_boxes,
        "vis_empty_frames": vis_empty,
        "ir_boxes": ir_boxes,
        "ir_empty_frames": ir_empty,
        "inventory_sha256": stable_json_hash(rows),
        "rows": rows,
    }
    source = protocol["source"]
    expected = {
        "vis_inventory": int(source["expected_vis_inventory"]),
        "ir_inventory": int(source["expected_ir_inventory"]),
        "paired_frames": int(source["expected_paired_frames"]),
        "unpaired_vis": [source["expected_unpaired_vis_terminal"]],
        "vis_boxes": int(source["expected_vis_boxes"]),
        "vis_empty_frames": int(source["expected_vis_empty_frames"]),
        "ir_boxes": int(source["expected_ir_boxes"]),
        "ir_empty_frames": int(source["expected_ir_empty_frames"]),
        "inventory_sha256": source["expected_inventory_sha256"],
    }
    actual = {key: inventory[key] for key in expected}
    if actual != expected:
        raise RuntimeError(
            "FHL probe inventory differs from the freeze:\n"
            + json.dumps({"expected": expected, "actual": actual}, indent=2)
        )
    return inventory


def grid_runs(path):
    payload = load_yaml(path)
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Training YAML has no parameters mapping: {path}")
    return payload, make_grid(parameters)


def validate_training_configs(protocol):
    resolved = {}
    for configuration in CONFIGURATIONS:
        settings = protocol["configurations"][configuration]
        path = resolve_repo_path(settings["training_config"])
        payload, runs = grid_runs(path)
        if payload.get("experiment", {}).get("name") != settings["project"]:
            raise ValueError(f"Project mismatch in {path}")
        matching = [run for run in runs if int(run["seed"]) == int(protocol["seed"])]
        if len(matching) != 1:
            raise ValueError(f"Expected exactly one seed-40 run in {path}")
        if configuration == "paired_vis_dropout" and len(runs) != 1:
            raise ValueError("The candidate training config must contain one run")
        run = matching[0]
        if run["run_test"] is not False or run["test_checkpoint"] != "best":
            raise ValueError("Stage-A probe must disable test and select best")
        train = run["train"]
        if train.get("max_epochs") != 10 or train.get("run_validation") is not True:
            raise ValueError("Stage-A probe requires ten epochs with validation")
        if "early_stopping_patience" in train:
            raise ValueError("Stage-A probe forbids early stopping")
        resolved[configuration] = run

    baseline = copy.deepcopy(resolved["baseline"])
    candidate = copy.deepcopy(resolved["paired_vis_dropout"])
    baseline.pop("tracker")
    candidate.pop("tracker")
    contract = candidate["dataset"].pop("modal_dropout_coordinate_contract", None)
    if contract != "paired_vis":
        raise ValueError("Candidate must use the paired_vis dropout contract")
    if "modal_dropout_coordinate_contract" in baseline["dataset"]:
        raise ValueError("Baseline must preserve the historical default contract")
    if normalize(baseline) != normalize(candidate):
        raise ValueError("Baseline and candidate differ beyond the frozen contract")
    return resolved


def wandb_value(config, key):
    value = config.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def audit_completed_run(checkpoint, expected_run, project, seed):
    checkpoint = Path(checkpoint).resolve()
    config_path = checkpoint.parents[1] / "config.yaml"
    summary_path = checkpoint.parents[1] / "wandb-summary.json"
    if not config_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Incomplete local W&B run for {checkpoint}")
    with config_path.open(encoding="utf-8") as input_file:
        actual = yaml.safe_load(input_file) or {}
    experiment = wandb_value(actual, "experiment") or {}
    if experiment.get("name") != project:
        raise RuntimeError(f"Completed run project mismatch: {checkpoint}")
    if int(wandb_value(actual, "seed")) != int(seed):
        raise RuntimeError(f"Completed run seed mismatch: {checkpoint}")
    for key in (
        "run_test",
        "test_checkpoint",
        "reproducibility",
        "train",
        "model",
        "dataset",
        "dataloader",
    ):
        if normalize(wandb_value(actual, key)) != normalize(expected_run[key]):
            raise RuntimeError(f"Completed run differs from freeze at {key}")

    with summary_path.open(encoding="utf-8") as input_file:
        summary = json.load(input_file)
    if int(summary.get("train/start_epoch", -1)) != 9:
        raise RuntimeError("Probe run did not complete epoch 10")
    if int(summary.get("train/step", -1)) != EXPECTED_TRAIN_STEPS - 1:
        raise RuntimeError("Probe run has an unexpected training-step count")
    best_epoch = int(summary.get("best_epoch", -1))
    best_map50 = float(summary.get("best_map_50", -1.0))
    if not 1 <= best_epoch <= 10 or best_map50 < 0.0:
        raise RuntimeError("Probe run has no valid best-checkpoint summary")
    if any(key.startswith("test/") for key in summary):
        raise RuntimeError("Probe run unexpectedly evaluated a test set")
    return {
        "run_id": checkpoint.parents[2].name.rsplit("-", 1)[-1],
        "run_dir": str(checkpoint.parents[2]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "completed_epochs": 10,
        "optimizer_steps": EXPECTED_TRAIN_STEPS,
        "best_epoch": best_epoch,
        "best_map50_logged": best_map50,
        "runtime_seconds": float(summary["_runtime"]),
    }


def item_paths(item, dataset_root):
    item_type, data = item
    if item_type != MULTI_MODALITY_ITEM:
        raise RuntimeError("Paired probe dataset contains a non-paired item")
    (vis_image, vis_label), (ir_image, ir_label) = data
    return {
        "vis_image": str(Path(vis_image).resolve().relative_to(dataset_root)),
        "vis_label": str(Path(vis_label).resolve().relative_to(dataset_root)),
        "ir_image": str(Path(ir_image).resolve().relative_to(dataset_root)),
        "ir_label": str(Path(ir_label).resolve().relative_to(dataset_root)),
    }


def build_loaders(protocol, inventory, dataset_config, batch_size=None, workers=None):
    transform, _denormalize = build_preprocessor(dataset_config)
    root = inventory["dataset_root"]
    paired_folders = [tuple(pair) for pair in protocol["source"]["paired_folders"]]
    paired_dataset = WiSARDDataset(
        root=root,
        folders=paired_folders,
        transform=transform,
        single_class=True,
        modal_dropout=False,
        use_tiling=False,
        test_all_tiles=False,
    )
    dataset_root = Path(root)
    actual_paths = [item_paths(item, dataset_root) for item in paired_dataset.items]
    expected_paths = [
        {
            key: row[key]
            for key in ("vis_image", "vis_label", "ir_image", "ir_label")
        }
        for row in inventory["rows"]
    ]
    if actual_paths != expected_paths:
        raise RuntimeError("Paired WiSARDDataset order differs from frozen inventory")

    ir_folders = [pair[1] for pair in protocol["source"]["paired_folders"]]
    native_dataset = WiSARDDataset(
        root=root,
        folders=ir_folders,
        transform=transform,
        single_class=True,
        modal_dropout=False,
        use_tiling=False,
        test_all_tiles=False,
    )
    native_paths = []
    for item_type, data in native_dataset.items:
        if item_type != IR_ITEM:
            raise RuntimeError("Native-IR probe dataset contains a non-IR item")
        image, label = data
        native_paths.append(
            {
                "ir_image": str(Path(image).resolve().relative_to(dataset_root)),
                "ir_label": str(Path(label).resolve().relative_to(dataset_root)),
            }
        )
    expected_native = [
        {"ir_image": row["ir_image"], "ir_label": row["ir_label"]}
        for row in inventory["rows"]
    ]
    if native_paths != expected_native:
        raise RuntimeError("Native-IR WiSARDDataset differs from paired counterparts")

    loader_kwargs = {
        "batch_size": int(batch_size or protocol["batch_size"]),
        "num_workers": int(protocol["workers"] if workers is None else workers),
        "shuffle": False,
    }
    seed = int(protocol["evaluation_seed"])
    paired_loader = torch.utils.data.DataLoader(
        paired_dataset,
        collate_fn=get_collate_fn(paired_dataset),
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
    native_loader = torch.utils.data.DataLoader(
        native_dataset,
        collate_fn=get_collate_fn(native_dataset),
        generator=torch.Generator().manual_seed(seed + 1),
        **loader_kwargs,
    )
    return paired_loader, native_loader


def raw_path(output_dir, configuration, condition):
    return output_dir / "raw" / f"{configuration}_{condition}.json"


def set_evaluation_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def promotion_decision(rows, rule):
    values = {
        (row["configuration"], row["condition"]): float(row["metrics"]["map_50"])
        for row in rows
    }
    expected = {
        (configuration, condition)
        for configuration in CONFIGURATIONS
        for condition in (*CONDITIONS, NATIVE_CONDITION)
    }
    if set(values) != expected:
        return None
    deltas = {
        condition: values[("paired_vis_dropout", condition)]
        - values[("baseline", condition)]
        for condition in (*CONDITIONS, NATIVE_CONDITION)
    }
    def at_least(value, threshold):
        return value > threshold or math.isclose(
            value,
            threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    checks = {
        "paired_ir_gain": at_least(
            deltas["paired_ir_vis_gt"],
            float(rule["minimum_paired_ir_map50_gain"]),
        ),
        "fusion_non_regression": at_least(
            deltas["paired_vis_ir"],
            float(rule["minimum_fusion_map50_delta"]),
        ),
        "native_ir_non_regression": at_least(
            deltas[NATIVE_CONDITION],
            float(rule["minimum_native_ir_map50_delta"]),
        ),
    }
    passed = all(checks.values())
    return {
        "status": rule["if_pass"] if passed else rule["if_fail"],
        "passed": passed,
        "map50_deltas_candidate_minus_baseline": deltas,
        "checks": checks,
        "thresholds": {
            "paired_ir_gain": rule["minimum_paired_ir_map50_gain"],
            "fusion_non_regression": rule["minimum_fusion_map50_delta"],
            "native_ir_non_regression": rule["minimum_native_ir_map50_delta"],
        },
    }


def build_aggregate(rows, protocol, protocol_hash, inventory, output_dir, complete):
    rows = sorted(rows, key=lambda row: (row["configuration"], row["condition"]))
    decision = promotion_decision(rows, protocol["promotion_rule"])
    aggregate = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "protocol_complete": bool(complete and decision is not None),
        "purpose": protocol["purpose"],
        "experimental_unit": "single_seed_checkpoint",
        "source_inventory": inventory,
        "source_inventory_sha256": stable_json_hash(inventory["rows"]),
        "results": rows,
        "promotion_decision": decision,
        "interpretation_constraints": protocol["interpretation"],
    }
    json_path = output_dir / "rtdetr_fam_paired_vis_modal_dropout_probe_evaluation.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(aggregate), output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    csv_path = output_dir / "rtdetr_fam_paired_vis_modal_dropout_probe_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "configuration",
                "condition",
                "ground_truth",
                "n_samples",
                *SCALAR_METRICS,
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "configuration": row["configuration"],
                    "condition": row["condition"],
                    "ground_truth": row["ground_truth"],
                    "n_samples": row["n_samples"],
                    **{metric: row["metrics"].get(metric) for metric in SCALAR_METRICS},
                }
            )
    print(f"Saved aggregate: {json_path}")
    print(f"Saved table: {csv_path}")
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    protocol_path = resolve_repo_path(args.protocol)
    protocol = load_protocol(protocol_path)
    protocol_hash = stable_json_hash(protocol)
    inventory = build_inventory(protocol)
    runs = validate_training_configs(protocol)
    paired_loader, native_loader = build_loaders(
        protocol,
        inventory,
        runs["paired_vis_dropout"]["dataset"],
        args.batch_size,
        args.workers,
    )
    print(
        json.dumps(
            {
                "paired_frames": inventory["paired_frames"],
                "vis_boxes": inventory["vis_boxes"],
                "ir_boxes": inventory["ir_boxes"],
                "inventory_sha256": inventory["inventory_sha256"],
                "paired_dataset": len(paired_loader.dataset),
                "native_ir_dataset": len(native_loader.dataset),
                "mt_erie_constructed": False,
            },
            indent=2,
        )
    )
    if args.prepare_only:
        print("Prepare-only OK: configs and FHL inventories match the freeze")
        return

    checkpoints = {}
    seed = int(protocol["seed"])
    for configuration in CONFIGURATIONS:
        settings = protocol["configurations"][configuration]
        checkpoint = resolve_local_wandb_checkpoint(
            settings["project"],
            seed,
            checkpoint="best",
            wandb_root=REPO_ROOT / "wandb",
        )
        audit = audit_completed_run(
            checkpoint,
            runs[configuration],
            settings["project"],
            seed,
        )
        checkpoints[configuration] = audit
        print(
            f"configuration={configuration} run={audit['run_id']} "
            f"best_epoch={audit['best_epoch']} "
            f"best_map50={audit['best_map50_logged']:.6f}"
        )
    if args.dry_run:
        print("Dry run OK: both completed best checkpoints match the freeze")
        return

    output_dir = resolve_repo_path(args.output_dir or protocol["output_dir"])
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    manifest_hash = inventory["inventory_sha256"]
    batch_size = int(args.batch_size or protocol["batch_size"])
    device = resolve_device(args.device)
    rows = []

    for configuration in CONFIGURATIONS:
        checkpoint = checkpoints[configuration]
        conditions = (*CONDITIONS, NATIVE_CONDITION)
        expected = {
            condition: {
                "protocol_id": protocol["id"],
                "protocol_sha256": protocol_hash,
                "source_inventory_sha256": manifest_hash,
                "configuration": configuration,
                "seed": seed,
                "condition": condition,
                "checkpoint_kind": "best",
                "checkpoint": checkpoint["checkpoint"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "batch_size": batch_size,
                "confidence_threshold": float(protocol["confidence_threshold"]),
                "max_batches": args.max_batches,
            }
            for condition in conditions
        }
        missing = []
        for condition in conditions:
            path = raw_path(output_dir, configuration, condition)
            if path.is_file() and not args.force:
                rows.append(load_compatible_raw(path, expected[condition]))
                print(f"[skip] {path}")
            else:
                missing.append(condition)
        if not missing:
            continue

        run = copy.deepcopy(runs[configuration])
        run["model"]["params"]["threshold"] = float(
            protocol["confidence_threshold"]
        )
        print(f"[load] configuration={configuration} device={device}")
        model = load_fusion_model(run["model"], checkpoint["checkpoint"], device)
        set_evaluation_seed(int(protocol["evaluation_seed"]))

        missing_paired = [condition for condition in CONDITIONS if condition in missing]
        if missing_paired:
            modes = [CONDITIONS[condition] for condition in missing_paired]
            print(f"[run] configuration={configuration} paired={','.join(modes)}")
            measured = evaluate_modalities(
                model,
                paired_loader,
                device,
                modes,
                max_batches=args.max_batches,
            )
            for condition in missing_paired:
                mode = CONDITIONS[condition]
                result = measured[mode]
                payload = {
                    **expected[condition],
                    "schema_version": 1,
                    "protocol_complete": args.max_batches is None,
                    "ground_truth": "vis",
                    "channel_intervention": protocol["conditions"][condition],
                    "n_dataset_images": len(paired_loader.dataset),
                    "n_samples": result["n_samples"],
                    "training_summary": checkpoint,
                    "metrics": result["metrics"],
                }
                if condition == "paired_vis_ir" and args.max_batches is None:
                    difference = float(result["metrics"]["map_50"]) - float(
                        checkpoint["best_map50_logged"]
                    )
                    if abs(difference) > FUSION_REPRODUCTION_TOLERANCE:
                        raise RuntimeError(
                            f"Fusion validation failed to reproduce for {configuration}: "
                            f"difference={difference:+.6f}"
                        )
                    payload["fusion_reference_check"] = {
                        "logged_best": checkpoint["best_map50_logged"],
                        "difference": difference,
                        "tolerance": FUSION_REPRODUCTION_TOLERANCE,
                    }
                path = raw_path(output_dir, configuration, condition)
                with path.open("w", encoding="utf-8") as output_file:
                    json.dump(jsonable(payload), output_file, indent=2, sort_keys=True)
                    output_file.write("\n")
                rows.append(payload)
                print(
                    f"[done] {configuration}/{condition} "
                    f"map50={result['metrics']['map_50']:.6f}"
                )

        if NATIVE_CONDITION in missing:
            print(f"[run] configuration={configuration} native_ir")
            result = evaluate_modalities(
                model,
                native_loader,
                device,
                ["vis_ir"],
                max_batches=args.max_batches,
            )["vis_ir"]
            payload = {
                **expected[NATIVE_CONDITION],
                "schema_version": 1,
                "protocol_complete": args.max_batches is None,
                "ground_truth": "ir",
                "channel_intervention": protocol["conditions"][NATIVE_CONDITION],
                "n_dataset_images": len(native_loader.dataset),
                "n_samples": result["n_samples"],
                "training_summary": checkpoint,
                "metrics": result["metrics"],
            }
            path = raw_path(output_dir, configuration, NATIVE_CONDITION)
            with path.open("w", encoding="utf-8") as output_file:
                json.dump(jsonable(payload), output_file, indent=2, sort_keys=True)
                output_file.write("\n")
            rows.append(payload)
            print(
                f"[done] {configuration}/{NATIVE_CONDITION} "
                f"map50={result['metrics']['map_50']:.6f}"
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    complete = args.max_batches is None
    aggregate = build_aggregate(
        rows,
        protocol,
        protocol_hash,
        inventory,
        output_dir,
        complete,
    )
    if aggregate["protocol_complete"]:
        decision = aggregate["promotion_decision"]
        print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
