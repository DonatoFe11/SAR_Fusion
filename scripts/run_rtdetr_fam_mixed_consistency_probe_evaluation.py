#!/usr/bin/env python3
"""Run the frozen seed-40 Stage-A gate for mixed-consistency FAM.

The data construction and inference implementation are shared with the earlier
paired-modality probe. This module freezes the new candidate contract, output
names, and promotion decision while retaining the same FHL inventory and four
evaluation conditions.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sarfusion.utils.grid import make_grid
from sarfusion.utils.utils import load_yaml
from scripts import run_rtdetr_fam_paired_vis_modal_dropout_probe_evaluation as shared
from scripts.run_rtdetr_carnation_stress_test import SCALAR_METRICS, file_sha256
from scripts.run_rtdetr_fam_level_ablation import jsonable


DEFAULT_PROTOCOL = (
    "parameters/RTDETR/rtdetr_fam_mixed_consistency_probe_evaluation.yaml"
)
PROTOCOL_ID = "rtdetr_fam_mixed_consistency_probe_evaluation_v1"
CONFIGURATIONS = ("baseline", "mixed_consistency")
CONDITIONS = shared.CONDITIONS
NATIVE_CONDITION = shared.NATIVE_CONDITION

EXPECTED_CONDITIONS = {
    "paired_vis_ir": "keep_all_four_channels_vis_gt",
    "paired_vis": "zero_ir_channel_vis_gt",
    "paired_ir_vis_gt": "zero_three_vis_channels_vis_gt",
    "native_ir_ir_gt": "native_ir_canvas_zero_rgb_ir_gt",
}
EXPECTED_RULE = {
    "rule": "all_conditions_required",
    "minimum_paired_ir_map50_gain": 0.03,
    "minimum_fusion_map50_delta": -0.01,
    "minimum_native_ir_map50_delta": -0.03,
    "if_pass": "expand_unchanged_to_seeds_41_44",
    "if_fail": "close_mixed_consistency_candidate",
}
EXPECTED_CONSISTENCY = {
    "enabled": True,
    "teacher": "online_eval_stop_gradient",
    "start_epoch": 1,
    "warmup_epochs": 2,
    "confidence_threshold": 0.2,
    "max_teacher_queries": 20,
    "matching_class_cost": 2.0,
    "matching_bbox_cost": 5.0,
    "matching_giou_cost": 2.0,
    "classification_weight": 2.0,
    "bbox_weight": 5.0,
    "giou_weight": 2.0,
}


def resolve_repo_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_protocol(path):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Mixed-consistency evaluation YAML needs a protocol mapping")
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError("Unexpected mixed-consistency evaluation protocol id")
    if protocol.get("status") != "frozen_before_scientific_training":
        raise ValueError("The seed-40 gate was not frozen before scientific training")
    if protocol.get("seed") != 40 or protocol.get("checkpoint") != "best":
        raise ValueError("The Stage-A gate requires seed 40 best checkpoints")
    if float(protocol.get("confidence_threshold", -1.0)) != 0.01:
        raise ValueError("The Stage-A confidence threshold must remain 0.01")
    if tuple(protocol.get("configurations", {})) != CONFIGURATIONS:
        raise ValueError("Stage-A gate configurations or order changed")
    if protocol.get("conditions") != EXPECTED_CONDITIONS:
        raise ValueError("Stage-A gate conditions changed")
    if protocol.get("promotion_rule") != EXPECTED_RULE:
        raise ValueError("Stage-A promotion rule changed after the freeze")

    source = protocol.get("source", {})
    if source.get("pairing") != "existing_wisard_sorted_zip":
        raise ValueError("Stage-A must preserve existing sorted-zip pairing")
    if source.get("paired_ground_truth") != "vis":
        raise ValueError("Paired conditions must use VIS ground truth")
    if source.get("native_ir_ground_truth") != "ir":
        raise ValueError("Native IR must use IR ground truth")
    interpretation = protocol.get("interpretation", {})
    if not interpretation or any(value is not False for value in interpretation.values()):
        raise ValueError("Stage-A interpretation constraints were relaxed")
    return protocol


def _grid_runs(path):
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
        if file_sha256(path) != settings["training_config_sha256"]:
            raise ValueError(f"Training YAML hash differs from freeze: {path}")
        payload, runs = _grid_runs(path)
        if payload.get("experiment", {}).get("name") != settings["project"]:
            raise ValueError(f"Project mismatch in {path}")
        matching = [run for run in runs if int(run["seed"]) == 40]
        if len(matching) != 1:
            raise ValueError(f"Expected exactly one seed-40 run in {path}")
        if configuration == "mixed_consistency" and len(runs) != 1:
            raise ValueError("The seed-40 candidate YAML must contain one run")
        run = matching[0]
        train = run["train"]
        if run["run_test"] is not False or run["test_checkpoint"] != "best":
            raise ValueError("Stage A must disable test and select best")
        if train.get("max_epochs") != 10 or train.get("run_validation") is not True:
            raise ValueError("Stage A requires ten epochs with validation")
        if "early_stopping_patience" in train:
            raise ValueError("Stage A forbids early stopping")
        resolved[configuration] = run

    baseline = copy.deepcopy(resolved["baseline"])
    candidate = copy.deepcopy(resolved["mixed_consistency"])
    baseline.pop("tracker")
    candidate.pop("tracker")
    if candidate["train"].pop("modality_consistency", None) != EXPECTED_CONSISTENCY:
        raise ValueError("Candidate consistency settings differ from the freeze")
    dataset = candidate["dataset"]
    if dataset.pop("modal_dropout_coordinate_contract", None) != "native":
        raise ValueError("Candidate must preserve native Modal Dropout supervision")
    if dataset.pop("paired_consistency", None) is not True:
        raise ValueError("Candidate paired consistency must remain enabled")
    if dataset.pop("paired_consistency_student_probs", None) != [0.5, 0.5]:
        raise ValueError("Candidate RGB/IR student probabilities changed")
    if any(
        key in baseline["dataset"]
        for key in (
            "modal_dropout_coordinate_contract",
            "paired_consistency",
            "paired_consistency_student_probs",
        )
    ):
        raise ValueError("Historical baseline unexpectedly contains the intervention")
    if shared.normalize(baseline) != shared.normalize(candidate):
        raise ValueError("Baseline and candidate differ beyond mixed consistency")
    return resolved


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
        condition: values[("mixed_consistency", condition)]
        - values[("baseline", condition)]
        for condition in (*CONDITIONS, NATIVE_CONDITION)
    }

    def at_least(value, threshold):
        return value > threshold or math.isclose(
            value, threshold, rel_tol=0.0, abs_tol=1e-12
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
        "source_inventory_sha256": shared.stable_json_hash(inventory["rows"]),
        "results": rows,
        "promotion_decision": decision,
        "interpretation_constraints": protocol["interpretation"],
    }
    stem = protocol["artifact_stem"]
    json_path = output_dir / f"{stem}.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(aggregate), output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    csv_path = output_dir / f"{stem}.csv"
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
                    **{
                        metric: row["metrics"].get(metric)
                        for metric in SCALAR_METRICS
                    },
                }
            )
    print(f"Saved aggregate: {json_path}")
    print(f"Saved table: {csv_path}")
    return aggregate


def main():
    # The shared main retains the previously audited inventory construction,
    # checkpoint audit, raw-result compatibility checks, and inference path.
    shared.DEFAULT_PROTOCOL = DEFAULT_PROTOCOL
    shared.CONFIGURATIONS = CONFIGURATIONS
    shared.load_protocol = load_protocol
    shared.validate_training_configs = validate_training_configs
    shared.promotion_decision = promotion_decision
    shared.build_aggregate = build_aggregate
    shared.main()


if __name__ == "__main__":
    main()
