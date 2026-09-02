#!/usr/bin/env python3
"""Run the frozen YOLO26 warmup-bias repair without altering the v1 runner."""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import yaml

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
import run_yolo26_stage_a as base


REVISION = "warmup_bias_repair_v1"
REPAIR_AUDIT_SCHEMA = "sarfusion.yolo26.stage_a.repair_control_audit.v1"


def _normalize_matched_config(config: dict) -> dict:
    value = deepcopy(config)
    study = value["study"]
    study["arm"] = "additive"
    study.pop("control_config", None)
    study.pop("requires_control_audit", None)
    value["model"]["use_fam"] = False
    value["training"]["name"] = "additive_seed40"
    return value


def _assert_repair_contract(config: dict) -> None:
    study = config["study"]
    training = config["training"]
    required = {
        "protocol_revision": (study.get("protocol_revision"), REVISION),
        "warmup_bias_lr": (float(training["warmup_bias_lr"]), 0.0),
        "optimizer": (training["optimizer"], "AdamW"),
        "batch": (int(training["batch"]), 4),
        "nbs": (int(training["nbs"]), 16),
        "epochs": (int(training["epochs"]), 50),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in required.items()
        if actual != expected
    }
    if mismatches:
        raise RuntimeError(f"YOLO26 repair contract mismatch: {mismatches}")


def _assert_repair_candidate_gate(
    config: dict,
    repository: Path,
    *,
    source_manifest_sha256: str,
    split_config_sha256: str,
    dataset_content_sha256: str,
    weights_sha256: str,
) -> None:
    _assert_repair_contract(config)
    study = config["study"]
    if study["arm"] != "fam":
        return

    control_config_path = (repository / study["control_config"]).resolve()
    control_config = yaml.safe_load(control_config_path.read_text(encoding="utf-8"))
    if _normalize_matched_config(config) != _normalize_matched_config(control_config):
        raise RuntimeError("Repair Additive/FAM configurations are not matched.")

    audit_path = (repository / study["requires_control_audit"]).resolve()
    if not audit_path.is_file():
        raise RuntimeError(
            "Repair FAM is locked until the repaired Additive audit exists: "
            f"{audit_path}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required = {
        "schema": REPAIR_AUDIT_SCHEMA,
        "status": "control_valid_for_candidate",
        "arm": "additive",
        "seed": 40,
        "epochs": 50,
        "source_manifest_sha256": source_manifest_sha256,
        "split_config_sha256": split_config_sha256,
        "dataset_content_sha256": dataset_content_sha256,
        "weights_sha256": weights_sha256,
        "control_config_sha256": base.sha256_file(control_config_path),
        "test_evaluated": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": audit.get(key)}
        for key, expected in required.items()
        if audit.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Repair control audit is invalid or stale: {mismatches}")

    optimizer = audit.get("optimizer_repair", {})
    if optimizer.get("status") != "passed":
        raise RuntimeError(f"Repair optimizer audit did not pass: {optimizer}")
    vitality = audit.get("vitality_gate", {})
    if vitality.get("status") != "passed":
        raise RuntimeError(f"Repair control did not pass its vitality gate: {vitality}")

    replay = audit.get("validation_replay", {})
    replay_error = float(replay.get("absolute_error", float("inf")))
    if (
        replay.get("status") != "passed"
        or replay.get("test_evaluated") is not False
        or not math.isfinite(replay_error)
        or replay_error > float(config["selection"]["replay_tolerance"])
    ):
        raise RuntimeError(f"Repair checkpoint replay is invalid: {replay}")

    for checkpoint_name in ("best", "last"):
        checkpoint = audit.get("checkpoints", {}).get(checkpoint_name, {})
        if checkpoint.get("strict_restore") is not True or checkpoint.get("use_fam") is not False:
            raise RuntimeError(f"Repair control {checkpoint_name} audit is incomplete.")
        if (
            checkpoint.get("fam_vs_fp16_initialization", {}).get(
                "changed_tensors_above_tolerance"
            )
            != 0
        ):
            raise RuntimeError(f"Repair control {checkpoint_name} updated bypassed FAM.")


def main() -> int:
    # The v1 runner resolves this global at call time, so replacing only the
    # candidate gate preserves every training/data/checkpoint code path.
    base._assert_candidate_gate = _assert_repair_candidate_gate
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
