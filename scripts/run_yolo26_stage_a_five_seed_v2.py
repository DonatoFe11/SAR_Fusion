#!/usr/bin/env python3
"""Run one seed of the five-seed YOLO26 Stage A protocol.

Reuse v1 model construction and GPU checks. Each arm completes all five
seeds; per-seed performance does not gate subsequent runs."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

# This import sets writable cache paths before importing Ultralytics.
from scripts.run_yolo26_stage_a import (
    _compare_initialization, candidate_safe_gpu_preflight,
    assert_environment, build_fusion_model, build_stage_a_dataset,
    load_pretrained_model, sha256_file, verify_source_manifest,
    YOLO26FusionTrainer, SETTINGS,
)
import yaml

SEEDS = [40, 41, 42, 43, 44]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    # A scientific campaign never skips the candidate-safe GPU preflight.
    args.skip_gpu_preflight = False
    return args


def validate_campaign_config(config):
    study, train = config["study"], config["training"]
    if config.get("schema") != "sarfusion.yolo26.stage_a.five_seed.v2":
        raise ValueError("Unexpected five-seed schema")
    if study.get("seeds") != SEEDS or study.get("protocol_revision") != "five_seed_stage_a_v2":
        raise ValueError("Stage A requires all five seeds and the v2 protocol")
    if study.get("arm") not in {"additive", "fam"}:
        raise ValueError("Unknown arm")
    forbidden = {"requires_control_audit", "control_config", "future_seeds_on_pass",
                 "control_vitality_min_map50", "control_vitality_epoch_from", "seed"}
    if forbidden.intersection(study):
        raise ValueError("Archived seed-screen fields are not allowed")
    expected = {"epochs": 50, "batch": 4, "nbs": 16, "imgsz": 640,
                "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01,
                "warmup_bias_lr": 0.0, "patience": 0, "resume": False,
                "exist_ok": False, "val": True, "save": True,
                "deterministic": False}
    if any(train.get(k) != v for k, v in expected.items()):
        raise ValueError("The fixed 50-epoch matched recipe was changed")
    if "seed" in train or "name" in train:
        raise ValueError("The runner assigns seed and unique run name")
    if config["model"]["use_fam"] != (study["arm"] == "fam"):
        raise ValueError("Arm and FAM disagree")
    if config["selection"].get("checkpoint") != "best" or config["selection"].get("no_test") is not True:
        raise ValueError("Stage A requires best selection and no test")
    if config["preflight"].get("candidate_safe_gpu_step") is not True:
        raise ValueError("The GPU integrity probe is mandatory")


def validate_completed_epochs(csv_path, selection_path, epochs):
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        rows = [{k.strip(): v.strip() for k, v in row.items()}
                for row in csv.DictReader(stream)]
    records = [json.loads(line) for line in Path(selection_path).read_text().splitlines()]
    expected = list(range(1, epochs + 1))
    if [int(float(row["epoch"])) for row in rows] != expected:
        raise RuntimeError("Training did not complete the full epoch budget")
    if [int(row["epoch"]) for row in records] != expected:
        raise RuntimeError("Incomplete checkpoint-selection history")
    if not all(math.isfinite(float(row["raw_mAP50"])) for row in records):
        raise RuntimeError("Non-finite validation metrics")


def main() -> int:
    args = parse_args()
    repository = REPOSITORY
    os.chdir(repository)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "sarfusion.yolo26.stage_a.five_seed.v2":
        raise RuntimeError("Unsupported YOLO26 run schema.")
    validate_campaign_config(config)
    arm = config["study"]["arm"]
    seed = int(args.seed)
    config["study"]["seed"] = seed
    config["training"]["seed"] = seed
    config["training"]["name"] = f"{arm}_seed{seed}"
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
        del candidate
        gc.collect()

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

    # The GPU preflight uses the 4-channel FAM model in FP16.
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
        "schema": "sarfusion.yolo26.stage_a.execution.v2",
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

    validate_completed_epochs(trainer.csv, trainer.selection_path, int(training["epochs"]))
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
