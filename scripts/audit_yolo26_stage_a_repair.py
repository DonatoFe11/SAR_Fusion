#!/usr/bin/env python3
"""Augment the standard control audit with frozen warmup/vitality gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import yaml

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
import audit_yolo26_stage_a_control as base


REPOSITORY = Path(__file__).resolve().parents[1]
REPAIR_SCHEMA = "sarfusion.yolo26.stage_a.repair_control_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "parameters/YOLO26/yolo26s_additive_seed40_stage_a_repair_v1.yaml"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/yolo26_stage_a_repair_v1/additive_seed40"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _repair_gates(config: dict, rows: list[dict[str, str]]) -> tuple[dict, dict]:
    if len(rows) != 50:
        raise RuntimeError(f"Repair results contain {len(rows)} rows, expected 50.")
    lr_keys = ("lr/pg0", "lr/pg1", "lr/pg2")
    maximum_bias_lr = 0.0
    maximum_group_difference = 0.0
    for row in rows:
        values = [float(row[key]) for key in lr_keys]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite repair learning rate: {row}")
        maximum_bias_lr = max(maximum_bias_lr, values[2])
        maximum_group_difference = max(
            maximum_group_difference,
            abs(values[2] - values[0]),
            abs(values[2] - values[1]),
        )
    optimizer = {
        "status": "passed",
        "warmup_bias_lr_config": float(config["training"]["warmup_bias_lr"]),
        "max_epoch_end_bias_lr": maximum_bias_lr,
        "max_epoch_end_lr_group_difference": maximum_group_difference,
        "bias_lr_ceiling": 0.001,
        "lr_group_tolerance": 1e-12,
    }
    if (
        optimizer["warmup_bias_lr_config"] != 0.0
        or maximum_bias_lr > optimizer["bias_lr_ceiling"] + 1e-12
        or maximum_group_difference > optimizer["lr_group_tolerance"]
    ):
        optimizer["status"] = "failed"

    epoch_from = int(config["study"]["control_vitality_epoch_from"])
    threshold = float(config["study"]["control_vitality_min_map50"])
    eligible = [
        (int(float(row["epoch"])), float(row["metrics/mAP50(B)"]))
        for row in rows
        if int(float(row["epoch"])) >= epoch_from
    ]
    best_epoch, best_map50 = max(eligible, key=lambda item: item[1])
    vitality = {
        "status": "passed" if best_map50 >= threshold else "failed",
        "epoch_from": epoch_from,
        "threshold_mAP50": threshold,
        "best_eligible_epoch": best_epoch,
        "best_eligible_mAP50": best_map50,
    }
    return optimizer, vitality


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve() if args.output else run_dir / "control_audit.json"

    # Reuse the complete v1 integrity audit with explicit repair paths.  Its
    # temporary status cannot unlock the repair candidate because the repair
    # runner requires REPAIR_SCHEMA below.
    original_argv = sys.argv
    try:
        sys.argv = [
            str(Path(base.__file__).resolve()),
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
        ]
        base.main()
    finally:
        sys.argv = original_argv

    report = json.loads(output.read_text(encoding="utf-8"))
    if report.get("status") != "control_valid_for_candidate":
        raise RuntimeError("Base integrity audit did not validate the repair control.")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    optimizer, vitality = _repair_gates(config, _read_results(run_dir / "results.csv"))

    report["base_audit_schema"] = report["schema"]
    report["schema"] = REPAIR_SCHEMA
    report["protocol_revision"] = config["study"]["protocol_revision"]
    report["optimizer_repair"] = optimizer
    report["vitality_gate"] = vitality
    if optimizer["status"] != "passed":
        report["status"] = "control_integrity_failed_optimizer_gate"
    elif vitality["status"] != "passed":
        report["status"] = "control_integrity_passed_vitality_failed"
    else:
        report["status"] = "control_valid_for_candidate"

    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if optimizer["status"] != "passed":
        raise RuntimeError(f"Repair optimizer gate failed: {optimizer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
