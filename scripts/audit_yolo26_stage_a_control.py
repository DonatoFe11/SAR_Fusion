#!/usr/bin/env python3
"""Fail-closed post-training audit for the YOLO26 Additive seed-40 control.

The resulting ``control_audit.json`` is the only artifact that may unlock the
matched FAM candidate.  This audit intentionally does not score any test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/sarfusion-yolo26/ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sarfusion-yolo26/matplotlib")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import yaml

from sarfusion.yolo26.data import build_stage_a_dataset
from sarfusion.yolo26.model import YOLO26FusionDetectionModel
from sarfusion.yolo26.protocol import (
    assert_environment,
    build_fusion_model,
    load_pretrained_model,
    sha256_file,
    verify_source_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("parameters/YOLO26/yolo26s_additive_seed40_stage_a.yaml"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/yolo26_stage_a/additive_seed40"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"Blank JSONL record at {path}:{line_number}.")
        record = json.loads(line)
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid JSONL record at {path}:{line_number}.")
        records.append(record)
    return records


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} is not finite: {value!r}")
    return number


def audit_results_csv(path: Path, epochs: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if len(rows) != epochs:
        raise RuntimeError(f"results.csv has {len(rows)} rows, expected {epochs}.")
    observed_epochs = [int(float(row["epoch"])) for row in rows]
    if observed_epochs != list(range(1, epochs + 1)):
        raise RuntimeError(f"Unexpected results.csv epochs: {observed_epochs}")
    for row_index, row in enumerate(rows, 1):
        for key, value in row.items():
            if value != "":
                _finite(value, f"results.csv row {row_index} column {key}")
    return {
        "rows": len(rows),
        "epochs": observed_epochs,
        "sha256": sha256_file(path),
    }


def audit_selection_trace(
    records: list[dict[str, Any]],
    *,
    epochs: int,
    min_delta: float,
) -> dict[str, Any]:
    if len(records) != epochs:
        raise RuntimeError(
            f"checkpoint selection trace has {len(records)} rows, expected {epochs}."
        )
    best: float | None = None
    best_epoch: int | None = None
    improvements = 0
    for epoch, record in enumerate(records, 1):
        if int(record["epoch"]) != epoch:
            raise RuntimeError(f"Selection trace epoch mismatch at row {epoch}.")
        if float(record["min_delta"]) != min_delta:
            raise RuntimeError(f"Selection min_delta changed at epoch {epoch}.")
        raw = _finite(record["raw_mAP50"], f"selection epoch {epoch} raw_mAP50")
        expected_improved = best is None or raw > best + min_delta
        if bool(record["improved"]) != expected_improved:
            raise RuntimeError(f"Incorrect checkpoint decision at epoch {epoch}.")
        previous = record.get("previous_best_mAP50")
        if best is None:
            if previous is not None:
                raise RuntimeError("First selection row must have no previous best.")
        elif not math.isclose(float(previous), best, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Incorrect previous best at epoch {epoch}.")
        if expected_improved:
            best = raw
            best_epoch = epoch
            improvements += 1
        if not math.isclose(float(record["best_mAP50"]), best, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Incorrect running best at epoch {epoch}.")
        if int(record["best_epoch"]) != best_epoch:
            raise RuntimeError(f"Incorrect best epoch at epoch {epoch}.")
    return {
        "records": len(records),
        "best_mAP50": best,
        "best_epoch": best_epoch,
        "improvements": improvements,
        "last_selection_fitness": (
            raw if expected_improved else min(raw, float(best) - 1e-12)
        ),
    }


def audit_batch_trace(
    records: list[dict[str, Any]],
    *,
    expected_records: int,
    batch_size: int,
    train_manifest_records: list[dict[str, Any]],
    dataset_root: Path,
) -> dict[str, Any]:
    if len(records) != expected_records:
        raise RuntimeError(f"Data trace has {len(records)} rows, expected {expected_records}.")
    expected_masks = {0: [0.0, 1.0], 1: [1.0, 0.0], 2: [1.0, 1.0]}
    seen: list[int] = []
    for batch_index, record in enumerate(records):
        if int(record["epoch"]) != 0 or int(record["batch"]) != batch_index:
            raise RuntimeError(f"Unexpected batch trace coordinates at row {batch_index}.")
        indices = [int(value) for value in record["sample_index"]]
        files = [str(value) for value in record["im_file"]]
        codes = [int(value) for value in record["modality_code"]]
        masks = record["modality_mask"]
        if not all(len(values) == batch_size for values in (indices, files, codes, masks)):
            raise RuntimeError(f"Trace batch {batch_index} does not have batch {batch_size}.")
        for index, path, code, mask in zip(indices, files, codes, masks):
            if index < 0 or index >= len(train_manifest_records):
                raise RuntimeError(f"Out-of-range sample index {index}.")
            expected_path = (dataset_root / train_manifest_records[index]["rgb"]).resolve()
            if Path(path).resolve() != expected_path:
                raise RuntimeError(
                    f"Trace index/path mismatch: {index}, {path}, {expected_path}."
                )
            if code not in expected_masks or [float(x) for x in mask] != expected_masks[code]:
                raise RuntimeError(f"Trace modality code/mask mismatch in batch {batch_index}.")
        seen.extend(indices)
    if len(seen) != len(set(seen)):
        raise RuntimeError("A sample is duplicated inside the frozen first-epoch trace.")
    return {
        "records": len(records),
        "samples": len(seen),
        "unique_samples": len(set(seen)),
    }


def _parameter_delta(
    initial: dict[str, torch.Tensor],
    observed: dict[str, torch.Tensor],
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    if set(initial) != set(observed):
        raise RuntimeError("Parameter names differ between initialization and checkpoint.")
    changed = 0
    maximum = 0.0
    for name in sorted(initial):
        current = observed[name].detach().cpu()
        expected = initial[name].detach().cpu().to(dtype=current.dtype)
        difference = (current.float() - expected.float()).abs().max().item()
        if difference > tolerance:
            changed += 1
        maximum = max(maximum, float(difference))
    return {
        "tensors": len(initial),
        "changed_tensors_above_tolerance": changed,
        "max_abs_delta": maximum,
        "tolerance": tolerance,
    }


def audit_checkpoint(
    path: Path,
    *,
    expected_training_epoch: int,
    expected_train_fitness: float,
    pretrained,
    initial_model: YOLO26FusionDetectionModel,
    seed: int,
    deterministic: bool,
) -> tuple[dict[str, Any], YOLO26FusionDetectionModel]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # BaseTrainer.final_eval() strips both artifacts: EMA is promoted to
    # ``model``, optimizer state and best_fitness are cleared, and epoch is
    # set to -1.  The metric attached when that checkpoint was selected stays
    # in train_metrics and is the auditable checkpoint-to-trace link.
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, YOLO26FusionDetectionModel):
        raise RuntimeError(f"{path} does not contain the custom YOLO26 fusion EMA.")
    if bool(model.use_fam):
        raise RuntimeError(f"{path} is not an Additive use_fam=false checkpoint.")
    if int(checkpoint.get("epoch", 0)) != -1:
        raise RuntimeError(f"Expected a finalized (epoch=-1) checkpoint in {path}.")
    for stripped_key in ("optimizer", "best_fitness", "ema", "updates", "scaler"):
        if checkpoint.get(stripped_key) is not None:
            raise RuntimeError(f"Checkpoint field {stripped_key!r} was not stripped in {path}.")
    train_metrics = checkpoint.get("train_metrics") or {}
    if not math.isclose(
        _finite(train_metrics.get("fitness"), f"{path} train_metrics.fitness"),
        expected_train_fitness,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"Checkpoint fitness disagrees with the selection trace: {path}.")
    state = model.state_dict()
    if not all(torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
        raise RuntimeError(f"Non-finite checkpoint tensor in {path}.")

    restored = build_fusion_model(
        pretrained,
        seed=seed,
        use_fam=False,
        deterministic=deterministic,
    )
    incompatible = restored.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict restore failed for {path}: {incompatible}")

    fam_delta = _parameter_delta(
        dict(initial_model.fam_modules.named_parameters()),
        dict(model.fam_modules.named_parameters()),
        # The live EMA repeatedly averages identical bypassed FAM values in
        # FP32 before Ultralytics serializes them in FP16.  Permit a small
        # rounding envelope, while any optimizer update at lr=1e-3 would be
        # materially larger.
        tolerance=5e-4,
    )
    if fam_delta["changed_tensors_above_tolerance"] != 0:
        raise RuntimeError(
            "Bypassed Additive FAM parameters changed; FP16-quantized initialization "
            f"comparison: {fam_delta}"
        )
    shared_delta = _parameter_delta(
        dict(initial_model.model.named_parameters()),
        dict(model.model.named_parameters()),
    )
    if shared_delta["changed_tensors_above_tolerance"] == 0:
        raise RuntimeError("No shared RGB/neck/head parameter changed during training.")
    ir_delta = _parameter_delta(
        dict(initial_model.ir_backbone.named_parameters()),
        dict(model.ir_backbone.named_parameters()),
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "training_epoch": expected_training_epoch,
        "serialized_epoch": int(checkpoint["epoch"]),
        "selection_fitness": float(train_metrics["fitness"]),
        "use_fam": bool(model.use_fam),
        "serialization_dtype": str(next(model.parameters()).dtype),
        "strict_restore": True,
        "fam_vs_fp16_initialization": fam_delta,
        "shared_vs_fp16_initialization": shared_delta,
        "ir_vs_fp16_initialization": ir_delta,
    }, model


def main() -> int:
    args = parse_args()
    repository = REPOSITORY
    os.chdir(repository)
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve() if args.output else run_dir / "control_audit.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "sarfusion.yolo26.stage_a.run.v1":
        raise RuntimeError("Unsupported control configuration schema.")
    if config["study"]["arm"] != "additive" or config["model"]["use_fam"] is not False:
        raise RuntimeError("The control audit accepts only the Additive use_fam=false arm.")
    seed = int(config["study"]["seed"])
    epochs = int(config["training"]["epochs"])
    if seed != 40 or epochs != 50:
        raise RuntimeError("The frozen control is seed 40 for exactly 50 epochs.")

    environment = assert_environment()
    source_path = (repository / config["study"]["source_manifest"]).resolve()
    source_manifest = verify_source_manifest(repository, source_path)
    source_sha = sha256_file(source_path)
    split_path = (repository / config["study"]["split_config"]).resolve()
    split = yaml.safe_load(split_path.read_text(encoding="utf-8"))

    completion = _read_json(run_dir / "completion.json")
    if completion.get("status") != "completed" or completion.get("arm") != "additive":
        raise RuntimeError(f"Invalid completion artifact: {completion}")
    if int(completion.get("seed", -1)) != seed or int(completion.get("epochs_required", -1)) != epochs:
        raise RuntimeError("Completion seed/epoch contract mismatch.")
    if completion.get("test_evaluated") is not False:
        raise RuntimeError("The control unexpectedly evaluated a test split.")
    replay_tolerance = float(config["selection"]["replay_tolerance"])
    replay = completion.get("validation_replay") or {}
    if replay.get("status") != "passed" or replay.get("test_evaluated") is not False:
        raise RuntimeError(f"Serialized best-checkpoint replay did not pass: {replay}")
    if replay.get("metric") != "metrics/mAP50(B)":
        raise RuntimeError(f"Unexpected replay metric: {replay}")
    if int(replay.get("samples", -1)) != int(split["val"]["expected_pairs"]):
        raise RuntimeError(f"Unexpected replay inventory: {replay}")
    replay_error = _finite(replay.get("absolute_error"), "replay absolute_error")
    replay_selected = _finite(
        replay.get("selected_live_ema_mAP50"), "replay selected_live_ema_mAP50"
    )
    replay_serialized = _finite(
        replay.get("serialized_best_mAP50"), "replay serialized_best_mAP50"
    )
    if (
        float(replay.get("absolute_tolerance", -1.0)) != replay_tolerance
        or replay_error > replay_tolerance
        or not math.isclose(
            replay_error,
            abs(replay_serialized - replay_selected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            replay_selected,
            _finite(completion.get("best_mAP50"), "completion best_mAP50"),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(f"Checkpoint replay exceeded the frozen tolerance: {replay}")

    run_manifest = _read_json(run_dir / "run_manifest.json")
    if run_manifest.get("schema") != "sarfusion.yolo26.stage_a.execution.v1":
        raise RuntimeError("Invalid run manifest schema.")
    if run_manifest.get("arm") != "additive" or int(run_manifest.get("seed", -1)) != seed:
        raise RuntimeError("Run manifest arm/seed mismatch.")
    if run_manifest.get("source_manifest_sha256") != source_sha:
        raise RuntimeError("Run used a different frozen source manifest.")
    if run_manifest.get("source_manifest") != source_manifest:
        raise RuntimeError("Embedded source manifest differs from the current freeze.")
    if run_manifest.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("Run configuration bytes changed after training.")
    if run_manifest.get("weights_sha256") != config["model"]["weights_sha256"]:
        raise RuntimeError("Run pretrained-weight hash mismatch.")

    with tempfile.TemporaryDirectory(prefix="yolo26-control-audit-") as temporary:
        current_dataset = build_stage_a_dataset(split, Path(temporary))
    frozen_dataset = run_manifest.get("dataset_manifest", {})
    for key in ("schema", "counts", "content_sha256", "dropped_unpaired_frames", "records"):
        if frozen_dataset.get(key) != current_dataset.get(key):
            raise RuntimeError(f"Run/current dataset manifest mismatch for {key}.")
    dataset_sha = current_dataset["content_sha256"]

    results = audit_results_csv(run_dir / "results.csv", epochs)
    selection = audit_selection_trace(
        _read_jsonl(run_dir / "checkpoint_selection.jsonl"),
        epochs=epochs,
        min_delta=float(config["selection"]["checkpoint_min_delta"]),
    )
    if not math.isclose(
        float(completion["best_mAP50"]), selection["best_mAP50"], rel_tol=0.0, abs_tol=1e-12
    ) or int(completion["best_epoch"]) != int(selection["best_epoch"]):
        raise RuntimeError("Completion and checkpoint-selection trace disagree.")

    train_records = [record for record in current_dataset["records"] if record["split"] == "train"]
    batch_trace = audit_batch_trace(
        _read_jsonl(run_dir / "data_trace.jsonl"),
        expected_records=int(config["selection"]["trace_batches"]),
        batch_size=int(config["training"]["batch"]),
        train_manifest_records=train_records,
        dataset_root=Path(current_dataset["root"]),
    )

    weights_path = (repository / config["model"]["weights"]).resolve()
    if sha256_file(weights_path) != config["model"]["weights_sha256"]:
        raise RuntimeError("Current official checkpoint hash mismatch.")
    pretrained, _ = load_pretrained_model(weights_path)
    deterministic = bool(config["training"]["deterministic"])
    initial_model = build_fusion_model(
        pretrained, seed=seed, use_fam=False, deterministic=deterministic
    )
    initial_report = initial_model.initialization_report()
    if run_manifest.get("control_initialization") != initial_report:
        raise RuntimeError("Fresh control initialization differs from the run manifest.")
    if run_manifest.get("active_initialization") != initial_report:
        raise RuntimeError("Active initialization in the run manifest is not the control.")

    best_path = run_dir / "weights/best.pt"
    last_path = run_dir / "weights/last.pt"
    best_audit, _ = audit_checkpoint(
        best_path,
        expected_training_epoch=int(selection["best_epoch"]),
        expected_train_fitness=float(selection["best_mAP50"]),
        pretrained=pretrained,
        initial_model=initial_model,
        seed=seed,
        deterministic=deterministic,
    )
    last_audit, _ = audit_checkpoint(
        last_path,
        expected_training_epoch=epochs,
        expected_train_fitness=float(selection["last_selection_fitness"]),
        pretrained=pretrained,
        initial_model=initial_model,
        seed=seed,
        deterministic=deterministic,
    )

    report = {
        "schema": "sarfusion.yolo26.stage_a.control_audit.v1",
        "status": "control_valid_for_candidate",
        "arm": "additive",
        "seed": seed,
        "epochs": epochs,
        "environment": environment,
        "control_config_sha256": sha256_file(config_path),
        "source_manifest_sha256": source_sha,
        "split_config_sha256": sha256_file(split_path),
        "dataset_content_sha256": dataset_sha,
        "weights_sha256": sha256_file(weights_path),
        "completion": completion,
        "results_csv": results,
        "checkpoint_selection": selection,
        "batch_trace": batch_trace,
        "checkpoints": {"best": best_audit, "last": last_audit},
        "validation_replay": replay,
        "test_evaluated": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
