#!/usr/bin/env python3
"""Run five-seed Stage A with zero-initialized FAM offsets.

Each worker registers the model factory. Training and best-checkpoint replay
run in separate processes. Completed seeds are skipped; pending replays
resume without retraining. Stage B is launched separately."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from sarfusion.models import build_model
from sarfusion.experiment.experiment import ExpSettings
from sarfusion.experiment.run import Run
from sarfusion.utils.grid import make_grid
from sarfusion.utils.utils import load_yaml, nested_dict_update
from sarfusion.utils.reproducibility import (
    configure_reproducibility, verify_training_source_manifest,
)
from scripts.rtdetr_fam_zero_offset import (
    MANIFEST_ID, MODEL_NAME, PROTOCOL_PATH, register_experiment,
)

CONFIG = ROOT / "parameters/RTDETR/rtdetr_fam_zero_offset_stage_a_five_seed.yaml"
BASE_CONFIG = ROOT / "parameters/RTDETR/rtdetr_fam_stage_a_five_seed_v2.yaml"
OUTPUT = ROOT / "out/rtdetr_fam_zero_offset_stage_a"
SEEDS = [40, 41, 42, 43, 44]
PROJECT = "RTDETR_FAM_ZeroOffset_StageA_FiveSeed"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def normalized(value):
    return json.loads(json.dumps(value))


def expand_config(settings):
    base = settings["parameters"]
    grids = [base] + [nested_dict_update(deepcopy(base), override)
                      for override in settings.get("other_grids", [])]
    return [run for grid in grids for run in make_grid(grid)]


def validate_config(settings):
    runs = expand_config(settings)
    baseline = expand_config(load_yaml(BASE_CONFIG))
    experiment = settings["experiment"]
    if (experiment.get("name") != PROJECT or not experiment.get("isolate_runs")
            or experiment.get("continue_with_errors") is not False
            or experiment.get("resume", False) or experiment.get("resume_last", False)
            or experiment.get("start_from_grid", 0) != 0
            or experiment.get("start_from_run", 0) != 0
            or experiment.get("search") != "grid"):
        raise ValueError("Expected a fresh isolated five-seed Stage-A experiment")
    if [run["seed"] for run in runs] != SEEDS:
        raise ValueError("Exactly five coupled seeds 40--44 are required")
    for candidate, reference in zip(runs, baseline):
        if candidate["model"]["name"] != MODEL_NAME:
            raise ValueError("Use the offset-only factory, not identity_dcnv2 or Base")
        clone = deepcopy(candidate)
        clone["model"]["name"] = "fusion_rtdetr"
        for run in (clone, reference):
            run.pop("tracker", None)
            run["reproducibility"].pop("training_source_manifest_id", None)
            run["reproducibility"].pop("training_source_manifest_sha256", None)
        if normalized(clone) != normalized(reference):
            raise ValueError("Only offset initialization may differ from the fixed Stage-A recipe")
        repro = candidate["reproducibility"]
        if repro.get("training_source_manifest_id") != MANIFEST_ID:
            raise ValueError("The offset-only source manifest is required")
        if any(repro[key] != candidate["seed"] for key in ("model_seed", "data_seed", "training_seed")):
            raise ValueError("Model, data and training seeds must be coupled")
    return runs


def audit_controls(protocol, runs):
    if (protocol["seeds"] != SEEDS or [row["seed"] for row in protocol["controls"]] != SEEDS
            or protocol["promotion"] != {"mean_delta_min": 0.01, "min_positive_seeds": 4}
            or protocol["expected_train_frames"] != 3123
            or protocol["expected_val_frames"] != 896
            or protocol["replay_tolerance"] != 0.0002):
        raise ValueError("Frozen five-seed protocol has changed")
    for reference, run in zip(protocol["controls"], runs):
        files = ROOT / reference["files"]
        if file_sha256(files / "config.yaml") != reference["config_sha256"]:
            raise RuntimeError(f"Control config changed: seed {reference['seed']}")
        raw = load_yaml(files / "config.yaml")
        actual = {key: value.get("value") if isinstance(value, dict) and "value" in value else value
                  for key, value in raw.items()}
        if (actual["experiment"]["name"] != protocol["control_project"]
                or actual["seed"] != reference["seed"]):
            raise RuntimeError("Wrong control project or seed")
        expected = deepcopy(run)
        expected["model"]["name"] = "fusion_rtdetr"
        for key in ("model", "train", "dataloader", "dataset", "run_test", "test_checkpoint"):
            old = deepcopy(actual[key])
            if key == "dataset":
                old.setdefault("modal_dropout_coordinate_contract", "native")
            if normalized(old) != normalized(expected[key]):
                raise RuntimeError(f"Control seed {reference['seed']} recipe differs at {key}")
        summary = read_json(files / "wandb-summary.json")
        if summary.get("train/start_epoch") != 9:
            raise RuntimeError("Control did not finish the ten-epoch budget")
        for key, expected_value in (("best_epoch", reference["best_epoch"]),
                                    ("best_map_50", reference["best_map50"]),
                                    ("validate/map_50", reference["latest_map50"])):
            if summary.get(key) != expected_value:
                raise RuntimeError(f"Control selection metadata changed: {key}")
        with (files / "reproducibility_trace.jsonl").open() as stream:
            initialized = next((json.loads(line) for line in stream
                                if '"model_initialized"' in line), None)
        if not initialized or initialized["model_sha256"] != reference["initial_model_sha256"]:
            raise RuntimeError("Control initialization provenance differs")
        for kind in ("best", "latest"):
            if not (files / kind / "model.safetensors").is_file():
                raise FileNotFoundError(files / kind / "model.safetensors")


def campaign_inputs():
    register_experiment()
    settings = load_yaml(CONFIG)
    runs = validate_config(settings)
    protocol = read_json(ROOT / PROTOCOL_PATH)
    audit_controls(protocol, runs)
    repro = runs[0]["reproducibility"]
    manifest = verify_training_source_manifest(MANIFEST_ID, repro["training_source_manifest_sha256"], required=True)
    return settings, runs, protocol, manifest


def check_initialization(report, reference):
    if report["reference_initialization"]["model_sha256"] != reference["initial_model_sha256"]:
        raise RuntimeError("Fresh standard-FAM initialization does not match the reused control; stop before training")
    if (not report["rng_preserved"] or not report["parameters_preserved"]
            or not report["offsets_initially_zero"] or report["masks_reset"] or report["dcnv2_reset"]):
        raise RuntimeError("Invalid offset-only intervention")


class RecordedRun(Run):
    """The ordinary training loop, with read-only per-epoch validation recording."""
    def __init__(self, output):
        super().__init__()
        self.output = Path(output)
        self.validation_history = []

    def validate_epoch(self, epoch):
        metrics = super().validate_epoch(epoch)
        score = float(metrics["map_50"])
        if not math.isfinite(score):
            raise RuntimeError("Non-finite validation mAP50")
        self.validation_history.append({"epoch": epoch + 1, "map50": score})
        write_json(self.output / "validation.json", self.validation_history)
        return metrics


def selection_from_history(history):
    if [row["epoch"] for row in history] != list(range(1, 11)):
        raise RuntimeError("All ten validation epochs must be present")
    selected = None
    for row in history:
        if not math.isfinite(row["map50"]) or not 0 <= row["map50"] <= 1:
            raise RuntimeError("Invalid validation mAP50")
        if selected is None or row["map50"] > selected["map50"] + 0.001:
            selected = row
    return selected


def train_worker(seed, output, settings, params, protocol, manifest):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; --check-init is CPU-safe")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "started.json", {"seed": seed, "config_sha256": file_sha256(CONFIG),
                                        "source_sha256": manifest["sha256"]})
    run = RecordedRun(output)
    try:
        run.init({"experiment": dict(ExpSettings(settings["experiment"])), **params})
        model = run.accelerator.unwrap_model(run.model).model
        report = model.zero_offset_initialization_report
        reference = protocol["controls"][SEEDS.index(seed)]
        check_initialization(report, reference)
        if (len(run.train_loader.dataset) != protocol["expected_train_frames"]
                or len(run.val_loader.dataset) != protocol["expected_val_frames"]):
            raise RuntimeError("Unexpected Stage-A dataset sizes")
        optimizer_ids = {id(p) for group in run.optimizer.param_groups for p in group["params"]}
        for name, parameter in model.named_parameters():
            if "fam_modules" in name and (not parameter.requires_grad or id(parameter) not in optimizer_ids):
                raise RuntimeError(f"FAM parameter is not trainable/in the optimizer: {name}")
        write_json(output / "initialization.json", report)
        files = Path(run.tracker.local_dir).resolve()
        write_json(files / "zero_offset_initialization.json", report)
        # Do not hold a second model reference past Run.end().
        del model
        run.launch()
        selected = selection_from_history(run.validation_history)
        if selected["epoch"] != run.best_epoch + 1 or selected["map50"] != run.best_metric:
            raise RuntimeError("Recorded history and best-checkpoint selection disagree")
        checkpoints = {kind: files / kind / "model.safetensors" for kind in ("best", "latest")}
        if not all(path.is_file() for path in checkpoints.values()):
            raise RuntimeError("Missing best/latest checkpoint")
        write_json(output / "training_complete.json", {
            "seed": seed, "files": str(files.relative_to(ROOT)),
            "best_epoch": selected["epoch"], "best_map50": selected["map50"],
            "latest_map50": run.validation_history[-1]["map50"], "epochs": 10,
            "config_sha256": file_sha256(CONFIG), "source_sha256": manifest["sha256"],
            "checkpoint_sha256": {kind: file_sha256(path) for kind, path in checkpoints.items()},
            "test_evaluated": False,
        })
    finally:
        run.end()


def verify_completed_artifacts(output, manifest, *, replay_required):
    result = read_json(output / "training_complete.json")
    if (result["config_sha256"] != file_sha256(CONFIG) or result["source_sha256"] != manifest["sha256"]
            or result["epochs"] != 10 or result["test_evaluated"]):
        raise RuntimeError("Existing run has a different configuration/source/budget")
    files = ROOT / result["files"]
    for kind in ("best", "latest"):
        if file_sha256(files / kind / "model.safetensors") != result["checkpoint_sha256"][kind]:
            raise RuntimeError(f"Serialized {kind} checkpoint has changed")
    selected = selection_from_history(read_json(output / "validation.json"))
    if selected != {"epoch": result["best_epoch"], "map50": result["best_map50"]}:
        raise RuntimeError("Completion metadata disagrees with validation history")
    if replay_required:
        replay = read_json(output / "replay.json")
        if (not replay["replay_within_tolerance"] or replay["replay_tolerance"] != 0.0002
                or replay["checkpoint_sha256"] != result["checkpoint_sha256"]["best"]
                or replay["validation_size"] != 896
                or replay["summary_best_map_50"] != result["best_map50"]):
            raise RuntimeError("Best checkpoint has no matching successful validation replay")
    return result


def replay_worker(output, protocol, manifest):
    from scripts.replay_rtdetr_v2_stage_a_validation import replay_single
    completed = verify_completed_artifacts(output, manifest, replay_required=False)
    report = replay_single((ROOT / completed["files"]).parent, expect_use_fam=True,
                           tolerance=protocol["replay_tolerance"],
                           expected_validation_size=protocol["expected_val_frames"])
    write_json(output / "replay.json", report)
    if not report["replay_within_tolerance"]:
        raise RuntimeError("Best-checkpoint replay mismatch; do not rerun training or promote")


def statistics_summary(values):
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half_width = 2.7764451051977987 * sd / math.sqrt(5)
    return {"mean": mean, "sample_sd": sd, "median": statistics.median(values),
            "ci95_t": [mean - half_width, mean + half_width]}


def aggregate(results, protocol):
    if [row["seed"] for row in results] != SEEDS:
        raise RuntimeError("The decision requires all five seeds, including low-scoring ones")
    rows = []
    for candidate, control in zip(results, protocol["controls"]):
        rows.append({"seed": candidate["seed"], "control_run_id": control["run_id"],
                     "candidate_files": candidate["files"],
                     "control_best": control["best_map50"], "candidate_best": candidate["best_map50"],
                     "best_delta": candidate["best_map50"] - control["best_map50"],
                     "control_latest": control["latest_map50"], "candidate_latest": candidate["latest_map50"],
                     "latest_delta": candidate["latest_map50"] - control["latest_map50"]})
    stats = {key: statistics_summary([row[key] for row in rows])
             for key in ("control_best", "candidate_best", "best_delta", "control_latest", "candidate_latest", "latest_delta")}
    wins = sum(row["best_delta"] > 0 for row in rows)
    passes = stats["best_delta"]["mean"] >= 0.01 and wins >= 4
    return {"protocol_id": protocol["protocol_id"], "rows": rows, "statistics": stats,
            "best_positive_seeds": wins, "stage_a_performance_gate_passed": passes,
            "stage_b_launched": False, "test_evaluated": False,
            "control_reuse_limit": "Matched recipe and initial state, not identical historical runtime/source provenance.",
            "next_step": "Review audits and prepare matched full-data Stage B" if passes else "No Stage B under the frozen rule"}


def child_command(seed, mode):
    return [sys.executable, str(Path(__file__).resolve()), "--seed", str(seed), "--worker", mode]


def supervise(settings, runs, protocol, manifest, *, restart_incomplete=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another zero-offset campaign is already running") from error
        for seed in SEEDS:
            output = OUTPUT / f"seed{seed}"
            if output.is_symlink():
                raise RuntimeError("Refusing a symlinked output directory")
            if not (output / "training_complete.json").is_file():
                if output.exists():
                    if not restart_incomplete:
                        raise RuntimeError(f"Incomplete seed {seed}: use --restart-incomplete to archive it and restart this seed from pretrained weights")
                    archive = OUTPUT / "incomplete" / f"seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    output.rename(archive)
                    print(f"Archived incomplete run to {archive}; original W&B checkpoints are preserved", flush=True)
                env = {**os.environ, "PYTHONHASHSEED": str(seed)}
                subprocess.run(child_command(seed, "train"), env=env, check=True)
            verify_completed_artifacts(output, manifest, replay_required=False)
            if not (output / "replay.json").is_file():
                subprocess.run(child_command(seed, "replay"), env={**os.environ, "PYTHONHASHSEED": str(seed)}, check=True)
            verify_completed_artifacts(output, manifest, replay_required=True)
            print(f"Seed {seed} complete (10 epochs + best replay).", flush=True)
        results = [verify_completed_artifacts(OUTPUT / f"seed{seed}", manifest, replay_required=True) for seed in SEEDS]
        report = aggregate(results, protocol)
        write_json(OUTPUT / "decision.json", report)
        print(json.dumps(report, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Check frozen inputs and print the five training commands")
    modes.add_argument("--check-init", action="store_true", help="CPU initialization audit for every seed; no training or W&B run")
    modes.add_argument("--aggregate-only", action="store_true", help="Validate completed artifacts and regenerate the decision")
    modes.add_argument("--worker", choices=("train", "replay"), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, choices=SEEDS, help=argparse.SUPPRESS)
    parser.add_argument("--restart-incomplete", action="store_true", help="Archive incomplete attempt records and restart only unfinished seeds from scratch")
    args = parser.parse_args(argv)
    if (args.worker is None) != (args.seed is None):
        parser.error("--seed is reserved for a single internal worker")
    os.chdir(ROOT)
    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
        os.environ.setdefault(key, value)
    settings, runs, protocol, manifest = campaign_inputs()
    if args.dry_run:
        print("5 new RT-DETR FAM trainings; reuse 5 pinned controls. 10 epochs, best primary, latest diagnostic. No Stage B/test.")
        for seed in SEEDS:
            print(" ".join(child_command(seed, "train")))
        return 0
    if args.check_init:
        torch.set_num_threads(4)
        for params, reference in zip(runs, protocol["controls"]):
            configure_reproducibility(params["seed"])
            model = build_model(params["model"])
            report = model.zero_offset_initialization_report
            check_initialization(report, reference)
            print(f"Seed {params['seed']}: reference hash matched; only offsets reset; masks/DCNv2/RNG preserved.", flush=True)
            del model
        return 0
    if args.worker:
        output = OUTPUT / f"seed{args.seed}"
        if args.worker == "train":
            train_worker(args.seed, output, settings, runs[SEEDS.index(args.seed)], protocol, manifest)
        else:
            replay_worker(output, protocol, manifest)
        return 0
    if args.aggregate_only:
        report = aggregate([verify_completed_artifacts(OUTPUT / f"seed{s}", manifest, replay_required=True) for s in SEEDS], protocol)
        write_json(OUTPUT / "decision.json", report)
        print(json.dumps(report, indent=2))
        return 0
    supervise(settings, runs, protocol, manifest, restart_incomplete=args.restart_incomplete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
