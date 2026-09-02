#!/usr/bin/env python3
"""Run one frozen YOLO26 RGB+IR Stage A arm.

The command performs source/data/weight checks and a candidate-safe GPU
forward/backward probe before starting the scientific run.  It never evaluates
the test split.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

# Configure writable, isolated runtime state before importing Ultralytics.
REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
RUNTIME_ROOT = Path(tempfile.gettempdir()) / "sarfusion-yolo26"
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_ROOT / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_ROOT / "matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
# Stabilize cuBLAS GEMMs where possible.  This must be present before
# CUDA/PyTorch initialization; the DCNv2 backward itself remains explicitly
# nondeterministic and both matched arms therefore use deterministic=false.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.utils import SETTINGS

from sarfusion.yolo26.data import build_stage_a_dataset
from sarfusion.yolo26.protocol import (
    assert_environment,
    build_fusion_model,
    load_pretrained_model,
    sha256_file,
    verify_source_manifest,
)
from sarfusion.yolo26.trainer import YOLO26FusionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--skip-gpu-preflight",
        action="store_true",
        help="For CPU/unit diagnostics only; never use for a scientific run.",
    )
    return parser.parse_args()


def _assert_candidate_gate(
    config: dict,
    repository: Path,
    *,
    source_manifest_sha256: str,
    split_config_sha256: str,
    dataset_content_sha256: str,
    weights_sha256: str,
) -> None:
    study = config["study"]
    if study["arm"] != "fam":
        return
    path = repository / study["requires_control_audit"]
    if not path.is_file():
        raise RuntimeError(
            "FAM is locked until the Additive seed-40 checkpoint audit exists: "
            f"{path}"
        )
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "control_valid_for_candidate":
        raise RuntimeError(f"Control audit does not authorize FAM: {audit}")
    control_config = repository / "parameters/YOLO26/yolo26s_additive_seed40_stage_a.yaml"
    required = {
        "schema": "sarfusion.yolo26.stage_a.control_audit.v1",
        "arm": "additive",
        "seed": 40,
        "epochs": 50,
        "source_manifest_sha256": source_manifest_sha256,
        "split_config_sha256": split_config_sha256,
        "dataset_content_sha256": dataset_content_sha256,
        "weights_sha256": weights_sha256,
        "control_config_sha256": sha256_file(control_config),
        "test_evaluated": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": audit.get(key)}
        for key, expected in required.items()
        if audit.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Control audit is stale or belongs to different source/data/weights: "
            f"{mismatches}"
        )
    replay = audit.get("validation_replay", {})
    replay_error = float(replay.get("absolute_error", float("inf")))
    replay_tolerance = float(config["selection"]["replay_tolerance"])
    if (
        replay.get("status") != "passed"
        or replay.get("test_evaluated") is not False
        or not math.isfinite(replay_error)
        or replay_error > replay_tolerance
        or float(replay.get("absolute_tolerance", -1.0)) != replay_tolerance
    ):
        raise RuntimeError(f"Control best-checkpoint replay is invalid: {replay}")
    for checkpoint_name in ("best", "last"):
        checkpoint = audit.get("checkpoints", {}).get(checkpoint_name, {})
        if checkpoint.get("strict_restore") is not True or checkpoint.get("use_fam") is not False:
            raise RuntimeError(f"Control {checkpoint_name} checkpoint audit is incomplete.")
        if (
            checkpoint.get("fam_vs_fp16_initialization", {}).get(
                "changed_tensors_above_tolerance"
            )
            != 0
        ):
            raise RuntimeError(f"Control {checkpoint_name} unexpectedly updated FAM.")
        if (
            checkpoint.get("shared_vs_fp16_initialization", {}).get(
                "changed_tensors_above_tolerance", 0
            )
            <= 0
        ):
            raise RuntimeError(f"Control {checkpoint_name} did not update shared parameters.")


def _compare_initialization(control: dict, candidate: dict) -> None:
    for key in ("shared_sha256", "ir_sha256", "fam_sha256"):
        if control[key] != candidate[key]:
            raise RuntimeError(
                f"Matched initialization failed for {key}: "
                f"{control[key]} != {candidate[key]}"
            )


def candidate_safe_gpu_preflight(model, config: dict) -> dict:
    """Exercise the more expensive FAM arm with the frozen physical batch."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the preregistered YOLO26 run.")
    device_value = config["training"]["device"]
    device = torch.device(f"cuda:{device_value}")
    batch_size = int(config["preflight"]["batch"])
    image_size = int(config["preflight"]["imgsz"])
    if batch_size != int(config["training"]["batch"]):
        raise RuntimeError("Preflight and training physical batches differ.")

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.args = get_cfg(overrides=config["training"])
    model.to(device).train()
    ema_shadow = deepcopy(model).to(device).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr0"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    images = torch.rand(
        batch_size,
        4,
        image_size,
        image_size,
        device=device,
    )
    synthetic = {
        "img": images,
        "modality_mask": torch.ones(batch_size, 2, device=device),
        "batch_idx": torch.arange(batch_size, device=device),
        "cls": torch.zeros(batch_size, 1, device=device),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.15, 0.20]], device=device
        ).repeat(batch_size, 1),
    }
    before = [
        next(module.parameters()).detach().clone()
        for module in model.fam_modules
    ]
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss, loss_items = model(synthetic)
        scalar_loss = loss.sum()
    if not torch.isfinite(scalar_loss):
        raise RuntimeError(f"Non-finite preflight loss: {scalar_loss}")
    scalar_loss.backward()

    gradient_norms = []
    for index, module in enumerate(model.fam_modules):
        gradients = [
            parameter.grad.detach().float().norm()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value) for value in gradients):
            raise RuntimeError(f"FAM P{index + 3} has missing/non-finite gradients.")
        norm = float(torch.stack(gradients).norm().item())
        if norm <= 0.0:
            raise RuntimeError(f"FAM P{index + 3} gradient norm is zero.")
        gradient_norms.append(norm)
    optimizer.step()
    updates = [
        float((next(module.parameters()).detach() - old).abs().max().item())
        for module, old in zip(model.fam_modules, before)
    ]
    if any(value <= 0.0 for value in updates):
        raise RuntimeError(f"At least one FAM did not update: {updates}")

    # Ultralytics validates detection models at 2x the physical train batch.
    # Exercise that exact FAM inference shape while optimizer and EMA states
    # are still resident, because validation must not trigger a later OOM.
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    val_batch_size = batch_size * 2
    val_images = torch.rand(
        val_batch_size,
        4,
        image_size,
        image_size,
        device=device,
    )
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        val_predictions = model.predict(
            val_images,
            modality_mask=torch.ones(val_batch_size, 2, device=device),
        )
    prediction_tensors = _prediction_tensors(val_predictions)
    if not prediction_tensors:
        raise RuntimeError("Candidate validation preflight returned no prediction tensors.")
    if not all(torch.isfinite(value).all() for value in prediction_tensors):
        raise RuntimeError("Candidate validation preflight produced non-finite predictions.")
    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "batch": batch_size,
        "validation_batch": val_batch_size,
        "imgsz": image_size,
        "amp_dtype": "float16",
        "loss": float(scalar_loss.detach().item()),
        "loss_items": {key: float(value) for key, value in loss_items.items()},
        "fam_gradient_norms": gradient_norms,
        "fam_max_updates": updates,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "ema_shadow_included": True,
    }
    del synthetic, images, val_images, val_predictions, optimizer, ema_shadow
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _prediction_tensors(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        tensors = []
        for key in sorted(value):
            tensors.extend(_prediction_tensors(value[key]))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_prediction_tensors(item))
        return tensors
    return []


def main() -> int:
    args = parse_args()
    repository = REPOSITORY
    os.chdir(repository)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "sarfusion.yolo26.stage_a.run.v1":
        raise RuntimeError("Unsupported YOLO26 run schema.")
    arm = config["study"]["arm"]
    seed = int(config["study"]["seed"])
    if arm not in {"additive", "fam"} or seed != 40:
        raise RuntimeError("Stage A accepts only Additive/FAM seed 40.")
    if bool(config["model"]["use_fam"]) != (arm == "fam"):
        raise RuntimeError("Arm and use_fam disagree.")
    if not config["selection"].get("no_test", False):
        raise RuntimeError("Stage A must not evaluate a test split.")

    environment = assert_environment()
    source_manifest_path = (repository / config["study"]["source_manifest"]).resolve()
    source_manifest = verify_source_manifest(
        repository,
        source_manifest_path,
    )
    training = dict(config["training"])
    run_dir = (repository / training["project"] / training["name"]).resolve()
    if run_dir.exists():
        raise RuntimeError(
            f"Frozen run directory already exists; refusing an implicit rerun: {run_dir}"
        )
    materialized_dir = (
        repository
        / training["project"]
        / "_manifests"
        / f"{arm}_seed{seed}"
    )
    split_config = yaml.safe_load(
        (repository / config["study"]["split_config"]).read_text(encoding="utf-8")
    )
    dataset_manifest = build_stage_a_dataset(split_config, materialized_dir)

    weight_path = (repository / config["model"]["weights"]).resolve()
    if sha256_file(weight_path) != config["model"]["weights_sha256"]:
        raise RuntimeError("Run config and actual yolo26s.pt checksum disagree.")
    _assert_candidate_gate(
        config,
        repository,
        source_manifest_sha256=sha256_file(source_manifest_path),
        split_config_sha256=sha256_file(repository / config["study"]["split_config"]),
        dataset_content_sha256=dataset_manifest["content_sha256"],
        weights_sha256=sha256_file(weight_path),
    )
    pretrained, _ = load_pretrained_model(weight_path)
    deterministic = bool(training["deterministic"])

    control = build_fusion_model(
        pretrained,
        seed=seed,
        use_fam=False,
        deterministic=deterministic,
    )
    control_init = control.initialization_report()
    del control
    gc.collect()
    candidate = build_fusion_model(
        pretrained,
        seed=seed,
        use_fam=True,
        deterministic=deterministic,
    )
    candidate_init = candidate.initialization_report()
    _compare_initialization(control_init, candidate_init)

    if args.skip_gpu_preflight:
        preflight = {"status": "skipped", "scientific_run_allowed": False}
        del candidate
    else:
        preflight = candidate_safe_gpu_preflight(candidate, config)

    active = build_fusion_model(
        pretrained,
        seed=seed,
        use_fam=arm == "fam",
        deterministic=deterministic,
        verbose=True,
    )
    active_init = active.initialization_report()
    expected_init = candidate_init if arm == "fam" else control_init
    _compare_initialization(expected_init, active_init)

    if args.skip_gpu_preflight:
        print(json.dumps({"initialization": active_init, "preflight": preflight}, indent=2))
        return 0

    training["model"] = str(weight_path)
    training["data"] = dataset_manifest["data_yaml"]
    training["project"] = str((repository / config["training"]["project"]).resolve())
    SETTINGS.update({"wandb": False})

    # Our GPU preflight exercises the actual 4-channel FAM model in FP16.
    # The upstream generic AMP check would instead download/test YOLO26n RGB.
    import ultralytics.engine.trainer as trainer_module

    trainer_module.check_amp = lambda _model: True
    trainer = YOLO26FusionTrainer(
        overrides=training,
        dataset_options=config["dataset"],
        expected_batch=int(config["training"]["batch"]),
        checkpoint_min_delta=float(config["selection"]["checkpoint_min_delta"]),
        trace_batches=int(config["selection"]["trace_batches"]),
    )
    trainer.model = active
    run_manifest = {
        "schema": "sarfusion.yolo26.stage_a.execution.v1",
        "arm": arm,
        "seed": seed,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_manifest": source_manifest,
        "environment": environment,
        "weights": str(weight_path),
        "weights_sha256": sha256_file(weight_path),
        "dataset_manifest": dataset_manifest,
        "control_initialization": control_init,
        "candidate_initialization": candidate_init,
        "active_initialization": active_init,
        "candidate_safe_preflight": preflight,
        "checkpoint_selection": config["selection"],
        "promotion_delta": config["study"]["promotion_delta"],
    }
    (trainer.save_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        trainer.train()
    except Exception as error:
        (trainer.save_dir / "completion.json").write_text(
            json.dumps(
                {"status": "failed", "error": repr(error)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise

    required = [trainer.best, trainer.last, trainer.csv]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Training returned without required artifacts: {missing}")
    # BaseTrainer.train() finishes by reloading the serialized FP16 best.pt
    # and validating it on the already-built paired validation loader.  Treat
    # that built-in final evaluation as the preregistered checkpoint replay.
    final_metrics = trainer.metrics or {}
    replay_key = YOLO26FusionTrainer.MAP50_KEY
    replay_map50 = float(final_metrics.get(replay_key, float("nan")))
    selected_map50 = float(trainer.best_fitness)
    replay_error = abs(replay_map50 - selected_map50)
    replay_tolerance = float(config["selection"]["replay_tolerance"])
    validation_replay = {
        "status": "passed",
        "metric": replay_key,
        "selected_live_ema_mAP50": selected_map50,
        "serialized_best_mAP50": replay_map50,
        "absolute_error": replay_error,
        "absolute_tolerance": replay_tolerance,
        "samples": int(dataset_manifest["counts"]["val"]),
        "test_evaluated": False,
    }
    if (
        not math.isfinite(selected_map50)
        or not math.isfinite(replay_map50)
        or not math.isfinite(replay_error)
        or replay_error > replay_tolerance
    ):
        validation_replay["status"] = "failed"
        failure = {
            "status": "integrity_failed",
            "arm": arm,
            "seed": seed,
            "validation_replay": validation_replay,
            "test_evaluated": False,
        }
        (trainer.save_dir / "completion.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "Serialized best-checkpoint replay failed: "
            f"mAP50={replay_map50}, selected={selected_map50}, "
            f"error={replay_error}, tolerance={replay_tolerance}."
        )
    completion = {
        "status": "completed",
        "arm": arm,
        "seed": seed,
        "epochs_required": int(training["epochs"]),
        "best_checkpoint": str(trainer.best),
        "last_checkpoint": str(trainer.last),
        "results_csv": str(trainer.csv),
        "best_mAP50": trainer.best_fitness,
        "best_epoch": trainer.selection_best_epoch,
        "validation_replay": validation_replay,
        "test_evaluated": False,
    }
    (trainer.save_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
