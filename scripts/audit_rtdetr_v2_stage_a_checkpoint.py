#!/usr/bin/env python3
"""Audit a completed RT-DETRv2 Stage-A run without evaluating test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import yaml
from safetensors import safe_open

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sarfusion.experiment.run import Run
from sarfusion.experiment.utils import WrapperModule
from sarfusion.models import build_model
from sarfusion.utils.reproducibility import (
    RTDETR_V2_FAM_TRAINING_SOURCE_MANIFEST_ID,
    build_training_source_manifest,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wandb_value(config: dict, key: str):
    entry = config[key]
    if not isinstance(entry, dict) or "value" not in entry:
        raise RuntimeError(f"Invalid W&B config entry: {key}")
    return entry["value"]


def fam_delta(initial: dict[str, torch.Tensor], model: torch.nn.Module) -> dict:
    current = model.state_dict()
    changed = []
    maximum = 0.0
    for name, before in initial.items():
        after = current[name].detach().cpu()
        delta = float((after - before).abs().max())
        maximum = max(maximum, delta)
        if delta > 0.0:
            changed.append(name)
    return {
        "tensor_count": len(initial),
        "changed_tensor_count": len(changed),
        "max_abs_delta": maximum,
    }


def audit(run_dir: Path, expect_use_fam: bool) -> dict:
    files_dir = run_dir / "files"
    config_path = files_dir / "config.yaml"
    summary_path = files_dir / "wandb-summary.json"
    if not config_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Incomplete W&B run directory: {run_dir}")

    config = yaml.safe_load(config_path.read_text())
    model_params = wandb_value(config, "model")
    use_fam = bool(model_params["params"]["use_fam"])
    if use_fam != expect_use_fam:
        raise RuntimeError(
            f"use_fam={use_fam} does not match expectation {expect_use_fam}"
        )

    reproducibility = wandb_value(config, "reproducibility")
    manifest_id = reproducibility["training_source_manifest_id"]
    if manifest_id != RTDETR_V2_FAM_TRAINING_SOURCE_MANIFEST_ID:
        raise RuntimeError(f"Unexpected source manifest ID: {manifest_id}")
    current_manifest = build_training_source_manifest(manifest_id)
    declared_manifest_sha256 = reproducibility[
        "training_source_manifest_sha256"
    ]
    if current_manifest["sha256"] != declared_manifest_sha256:
        raise RuntimeError(
            "Current source tree no longer matches the training manifest: "
            f"{current_manifest['sha256']} != {declared_manifest_sha256}"
        )

    torch.manual_seed(int(reproducibility["model_seed"]))
    model = WrapperModule(build_model(model_params))
    initial_fam = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if ".fam_modules." in name
    }
    if expect_use_fam and not initial_fam:
        raise RuntimeError("FAM candidate exposes no FAM state tensors")
    if not expect_use_fam and initial_fam:
        raise RuntimeError("Additive control unexpectedly exposes FAM tensors")

    run = object.__new__(Run)
    run.tracker = SimpleNamespace(local_dir=str(files_dir))
    run.params = {"strict_checkpoint_loading": True}
    run.model = model

    checkpoints = {}
    for checkpoint_type in ("best", "latest"):
        checkpoint_dir = files_dir / checkpoint_type
        model_path = checkpoint_dir / "model.safetensors"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        with safe_open(model_path, framework="pt") as stream:
            serialized_tensor_count = len(list(stream.keys()))
        run.restore_model(checkpoint_type)
        checkpoint_report = {
            "model_sha256": sha256(model_path),
            "model_size_bytes": model_path.stat().st_size,
            "serialized_tensor_count": serialized_tensor_count,
            "files": {
                path.name: sha256(path)
                for path in sorted(checkpoint_dir.iterdir())
                if path.is_file()
            },
        }
        if checkpoint_type == "best":
            checkpoint_report["fam_update"] = fam_delta(initial_fam, model)
        checkpoints[checkpoint_type] = checkpoint_report

    summary = json.loads(summary_path.read_text())
    required_summary = {
        "best_epoch": summary.get("best_epoch"),
        "best_map_50": summary.get("best_map_50"),
        "final_map": summary.get("validate/map"),
        "final_map_50": summary.get("validate/map_50"),
        "final_map_75": summary.get("validate/map_75"),
        "final_train_avg_loss": summary.get("train/avg_loss"),
    }
    if required_summary["best_epoch"] is None or required_summary[
        "best_map_50"
    ] is None:
        raise RuntimeError("W&B summary is missing the primary best metric")

    return {
        "status": "passed",
        "run_id": run_dir.name.rsplit("-", 1)[-1],
        "use_fam": use_fam,
        "manifest_id": manifest_id,
        "manifest_sha256": current_manifest["sha256"],
        "manifest_file_count": len(current_manifest["files"]),
        "model_state_tensor_count": len(model.state_dict()),
        "summary": required_summary,
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--expect-use-fam", action=argparse.BooleanOptionalAction, required=True
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit(args.run_dir.resolve(), args.expect_use_fam)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
