from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ultralytics = pytest.importorskip("ultralytics")
if ultralytics.__version__ != "8.4.138":
    pytest.skip("YOLO26 repair tests require 8.4.138", allow_module_level=True)

from sarfusion.yolo26.protocol import verify_source_manifest
from scripts.audit_yolo26_stage_a_repair import _repair_gates
from scripts.run_yolo26_stage_a_repair import (
    _assert_repair_contract,
    _normalize_matched_config,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((REPOSITORY / "parameters/YOLO26" / name).read_text())


def test_repair_configs_are_matched_and_change_only_warmup_bias_from_pilot():
    control = _load("yolo26s_additive_seed40_stage_a_repair_v1.yaml")
    candidate = _load("yolo26s_fam_seed40_stage_a_repair_v1.yaml")
    pilot = _load("yolo26s_additive_seed40_stage_a.yaml")

    assert _normalize_matched_config(control) == _normalize_matched_config(candidate)
    _assert_repair_contract(control)
    _assert_repair_contract(candidate)
    assert pilot["training"]["warmup_bias_lr"] == 0.1
    assert control["training"]["warmup_bias_lr"] == 0.0

    frozen_training_keys = set(pilot["training"]) - {"project", "name", "warmup_bias_lr"}
    assert {key: pilot["training"][key] for key in frozen_training_keys} == {
        key: control["training"][key] for key in frozen_training_keys
    }


def test_repair_optimizer_and_vitality_gates():
    config = _load("yolo26s_additive_seed40_stage_a_repair_v1.yaml")
    rows = []
    for epoch in range(1, 51):
        rows.append(
            {
                "epoch": str(epoch),
                "metrics/mAP50(B)": "0.11" if epoch == 4 else "0.05",
                "lr/pg0": "0.0005",
                "lr/pg1": "0.0005",
                "lr/pg2": "0.0005",
            }
        )
    optimizer, vitality = _repair_gates(config, rows)
    assert optimizer["status"] == "passed"
    assert vitality["status"] == "passed"

    rows[0]["lr/pg2"] = "0.01"
    optimizer, _ = _repair_gates(config, rows)
    assert optimizer["status"] == "failed"

    rows[0]["lr/pg2"] = rows[0]["lr/pg0"]
    for row in rows:
        row["metrics/mAP50(B)"] = "0.09999"
    _, vitality = _repair_gates(config, rows)
    assert vitality["status"] == "failed"


def test_archived_v1_manifest_remains_valid():
    verify_source_manifest(
        REPOSITORY,
        REPOSITORY / "parameters/YOLO26/stage_a_source_manifest.json",
    )
