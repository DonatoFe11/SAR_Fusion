from __future__ import annotations

import hashlib
import json
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


@pytest.mark.parametrize(
    "manifest_name,result_name",
    [
        ("stage_a_source_manifest.json", "yolo26_additive_seed40_stage_a_v1.json"),
        (
            "stage_a_repair_v1_source_manifest.json",
            "yolo26_additive_seed40_stage_a_repair_v1.json",
        ),
    ],
)
def test_archived_manifests_match_published_experiments(manifest_name, result_name):
    manifest = REPOSITORY / "parameters/YOLO26" / manifest_name
    result_path = REPOSITORY / "notes/Thesis/results" / result_name
    if not result_path.is_file():
        pytest.skip("Requires local thesis results in notes/")
    result = json.loads(result_path.read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == result["source_manifest_sha256"]


@pytest.mark.parametrize(
    "config_name",
    [
        "yolo26s_additive_seed40_stage_a.yaml",
        "yolo26s_fam_seed40_stage_a.yaml",
        "yolo26s_additive_seed40_stage_a_repair_v1.yaml",
        "yolo26s_fam_seed40_stage_a_repair_v1.yaml",
    ],
)
def test_operational_configs_use_valid_revised_manifests(config_name):
    manifest_path = REPOSITORY / _load(config_name)["study"]["source_manifest"]
    assert manifest_path.name.endswith("_thesis_paths_v1.json")
    manifest = verify_source_manifest(REPOSITORY, manifest_path)
    assert manifest["source_revision"] == "thesis_paths_v1"
    assert all(not item["path"].startswith("notes/") for item in manifest["files"])
    archive = REPOSITORY / manifest["archived_source_manifest"]
    assert archive != manifest_path
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archived_source_manifest_sha256"]
    if "parent_source_manifest" in manifest:
        parent = REPOSITORY / manifest["parent_source_manifest"]
        verify_source_manifest(REPOSITORY, parent)
        assert hashlib.sha256(parent.read_bytes()).hexdigest() == manifest["parent_source_manifest_sha256"]
