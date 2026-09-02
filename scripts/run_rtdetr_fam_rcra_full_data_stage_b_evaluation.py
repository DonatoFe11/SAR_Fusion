#!/usr/bin/env python3
"""Evaluate matched full-data FAM/RCRA epoch-10 checkpoints on MtErie.

The protocol, training recipes and engineering decision rule are frozen before
Stage-B training.  This runner rejects mismatched local runs, evaluates the ten
`latest` checkpoints on one fixed 708-pair VIS+IR loader, caches raw metrics and
writes the paired RCRA-minus-FAM result.  MtErie is an already-used internal
benchmark and is not described as a newly blind test set.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import load_fusion_model  # noqa: E402
from sarfusion.data.temporal_split import (  # noqa: E402
    load_temporal_split_manifest,
    manifest_folder_pairs,
)
from sarfusion.models.checkpoints import (  # noqa: E402
    resolve_local_wandb_checkpoint,
)
from sarfusion.utils.grid import make_grid  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402
from scripts.run_rtdetr_carnation_stress_test import (  # noqa: E402
    SCALAR_METRICS,
    file_sha256,
    load_compatible_raw,
    paired_tests,
    stable_json_hash,
    summarize_values,
)
from scripts.run_rtdetr_fam_level_ablation import (  # noqa: E402
    jsonable,
    resolve_device,
)
from scripts.run_rtdetr_paired_modality_evaluation import (  # noqa: E402
    build_paired_loader,
    build_source_manifest,
    evaluate_modalities,
    summarize_signed_values,
)


DEFAULT_PROTOCOL = (
    "parameters/RTDETR/rtdetr_fam_rcra_full_data_stage_b_evaluation.yaml"
)
EXPECTED_PROTOCOL_ID = "rtdetr_fam_rcra_full_data_stage_b_evaluation_v1"
EXPECTED_SEEDS = [40, 41, 42, 43, 44]
EXPECTED_CONFIGURATIONS = ("fam", "rcra")
EXPECTED_TRAIN_STEPS = 10050


def _resolve_repo_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_protocol(path):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Stage-B evaluation YAML must contain a protocol mapping")
    if protocol.get("id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("Unexpected Stage-B evaluation protocol id")
    if protocol.get("status") != "frozen_before_training_and_inference":
        raise ValueError("Stage-B protocol must remain frozen before training")
    if protocol.get("checkpoint") != "latest":
        raise ValueError("Stage B must evaluate only epoch-10 latest checkpoints")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("Stage-B protocol must contain seeds 40--44 in order")
    if tuple(protocol.get("configurations", {})) != EXPECTED_CONFIGURATIONS:
        raise ValueError("Stage B must contain FAM followed by RCRA")
    if float(protocol.get("confidence_threshold", -1.0)) != 0.01:
        raise ValueError("Stage-B confidence threshold must remain 0.01")

    source = protocol.get("source", {})
    if source.get("ground_truth") != "vis":
        raise ValueError("Stage-B evaluation must use VIS ground truth")
    if source.get("pairing") != "existing_wisard_sorted_zip":
        raise ValueError("Stage-B evaluation must reproduce WiSARD pairing")

    rule = protocol.get("primary_comparison", {})
    expected_rule = {
        "metric": "map_50",
        "delta": "rcra_minus_fam",
        "minimum_mean_gain": 0.01,
        "minimum_positive_seed_wins": 4,
        "rule": "both_conditions_required",
        "confidence_interval_role": "uncertainty_report_not_decision_override",
        "if_pass": "retain_rcra_as_final_architecture",
        "if_fail": "retain_fam_as_final_performance_baseline",
    }
    if rule != expected_rule:
        raise ValueError("Stage-B primary decision rule differs from the freeze")

    interpretation = protocol.get("interpretation", {})
    if interpretation.get("final_architecture_decision_allowed") is not True:
        raise ValueError("The predeclared Stage-B architecture decision must be allowed")
    forbidden = (
        "checkpoint_selection_allowed",
        "threshold_tuning_allowed",
        "seed_selection_allowed",
        "additional_architecture_tuning_allowed",
        "historical_fam_as_primary_comparator_allowed",
        "claim_blind_test_allowed",
    )
    if any(interpretation.get(key) is not False for key in forbidden):
        raise ValueError("Stage-B interpretation constraints were relaxed")
    return protocol


def _grid_from_training_config(path):
    raw = load_yaml(path)
    parameters = raw.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Training config has no parameters mapping: {path}")
    return raw, make_grid(parameters)


def _normalize(value):
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def validate_training_configs(protocol):
    """Validate the two grids and return flattened configs by seed."""
    flattened = {}
    expected_folders = _normalize(protocol["training_source"]["paired_folders"])
    for configuration in EXPECTED_CONFIGURATIONS:
        settings = protocol["configurations"][configuration]
        path = _resolve_repo_path(settings["training_config"])
        raw, grid = _grid_from_training_config(path)
        if raw.get("experiment", {}).get("name") != settings["project"]:
            raise ValueError(f"Project mismatch in {path}")
        if len(grid) != len(EXPECTED_SEEDS):
            raise ValueError(f"{path} must expand to exactly five runs")
        if [int(run["seed"]) for run in grid] != EXPECTED_SEEDS:
            raise ValueError(f"{path} does not map run indices to seeds 40--44")

        for run in grid:
            seed = int(run["seed"])
            train = run["train"]
            model = run["model"]["params"]
            dataset = run["dataset"]
            dataloader = run["dataloader"]
            if run["run_test"] is not False or run["test_checkpoint"] != "latest":
                raise ValueError("Stage-B training must disable test and target latest")
            if train.get("max_epochs") != 10 or train.get("run_validation") is not False:
                raise ValueError("Stage B requires ten epochs without validation")
            if train.get("save_checkpoints") is not True:
                raise ValueError("Stage B must save a checkpoint")
            if train.get("save_final_checkpoint_only") is not True:
                raise ValueError("Stage B must save only the final latest checkpoint")
            for forbidden in (
                "early_stopping_patience",
                "max_steps_per_epoch",
                "gradient_accumulation_steps",
                "checkpoint_min_delta",
            ):
                if forbidden in train:
                    raise ValueError(f"Stage B forbids train.{forbidden}")
            if float(train.get("initial_lr", -1.0)) != 2e-5:
                raise ValueError("Stage-B detector LR must remain 2e-5")
            if train.get("optimizer") != "AdamW":
                raise ValueError("Stage-B optimizer must remain AdamW")
            if dataloader.get("batch_size") != 4:
                raise ValueError("Stage B requires direct batch four")
            if _normalize(dataset.get("train_folders")) != expected_folders:
                raise ValueError("Stage-B training folders differ from the freeze")
            if dataset.get("modal_dropout") is not True:
                raise ValueError("Stage B must retain training modal dropout")
            if dataset.get("modal_dropout_probs") != [0.2, 0.2, 0.6]:
                raise ValueError("Stage-B modal-dropout probabilities changed")
            if model.get("use_fam") is not True:
                raise ValueError("Both Stage-B configurations require FAM")
            if model.get("fam_variant") != "current_dcnv2":
                raise ValueError("Stage B must use current_dcnv2 FAM")
            if model.get("use_p2") is not False:
                raise ValueError("P2 must remain disabled in Stage B")
            if model.get("use_reliability_gating") is not False:
                raise ValueError("Post-fusion reliability gating must remain disabled")
            if model.get("use_scalar_residual_alignment") is not False:
                raise ValueError("The scalar attribution control must remain disabled")

            expected_gate = configuration == "rcra"
            if model.get("use_residual_alignment_gating") is not expected_gate:
                raise ValueError(f"Unexpected RCRA state for {configuration}")
            configured_lr = train.get("alignment_gate_lr")
            expected_lr = settings["alignment_gate_lr"]
            if configured_lr != expected_lr:
                raise ValueError(f"Unexpected alignment LR for {configuration}")
            flattened[(configuration, seed)] = run

    # After neutralizing the declared RCRA recipe, every scientific parameter
    # must match the FAM baseline exactly for every paired seed.
    for seed in EXPECTED_SEEDS:
        fam = copy.deepcopy(flattened[("fam", seed)])
        rcra = copy.deepcopy(flattened[("rcra", seed)])
        fam.pop("tracker")
        rcra.pop("tracker")
        rcra["train"].pop("alignment_gate_lr")
        rcra["model"]["params"]["use_residual_alignment_gating"] = False
        if _normalize(fam) != _normalize(rcra):
            raise ValueError(f"FAM/RCRA Stage-B configs are not matched for seed {seed}")
    return flattened


def verify_training_inventory(protocol):
    source = protocol["training_source"]
    manifest_path = _resolve_repo_path(source["inventory_manifest"])
    dataset_root = _resolve_repo_path(protocol["dataset_root"])
    manifest, inventory = load_temporal_split_manifest(
        manifest_path, dataset_root, verify=True
    )
    if _normalize(manifest_folder_pairs(manifest)) != _normalize(
        source["paired_folders"]
    ):
        raise RuntimeError("Stage-B folders differ from the frozen inventory")
    if inventory["n_source_frames"] != int(source["expected_frames"]):
        raise RuntimeError("Stage-B full-data frame count changed")
    if inventory["source_inventory_sha256"] != source["expected_inventory_sha256"]:
        raise RuntimeError("Stage-B full-data inventory hash changed")
    return {
        "n_frames": inventory["n_source_frames"],
        "inventory_sha256": inventory["source_inventory_sha256"],
        "sequence_frames": {
            entry["id"]: entry["n_frames"] for entry in inventory["sequences"]
        },
    }


def _wandb_value(config, key):
    value = config.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def audit_completed_run(checkpoint_path, expected_run, project, seed, configuration):
    config_path = checkpoint_path.parents[1] / "config.yaml"
    summary_path = checkpoint_path.parents[1] / "wandb-summary.json"
    if not config_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Incomplete W&B files for {checkpoint_path}")
    with config_path.open(encoding="utf-8") as input_file:
        actual = yaml.safe_load(input_file) or {}
    experiment = _wandb_value(actual, "experiment") or {}
    if experiment.get("name") != project:
        raise RuntimeError(f"Completed run project mismatch for seed {seed}")
    if int(_wandb_value(actual, "seed")) != int(seed):
        raise RuntimeError(f"Completed run seed mismatch for {checkpoint_path}")
    for key in (
        "run_test",
        "test_checkpoint",
        "reproducibility",
        "train",
        "model",
        "dataset",
        "dataloader",
    ):
        if _normalize(_wandb_value(actual, key)) != _normalize(expected_run[key]):
            raise RuntimeError(
                f"Completed {configuration}/seed {seed} differs at {key}"
            )

    with summary_path.open(encoding="utf-8") as input_file:
        summary = json.load(input_file)
    if int(summary.get("train/start_epoch", -1)) != 9:
        raise RuntimeError(f"{configuration}/seed {seed} did not reach epoch 10")
    if int(summary.get("train/step", -1)) != EXPECTED_TRAIN_STEPS - 1:
        raise RuntimeError(f"{configuration}/seed {seed} has wrong step count")
    for key in ("train/lr_new_modules", "train/lr_backbone", "train/lr_dino"):
        if float(summary.get(key, -1.0)) != 2e-5:
            raise RuntimeError(f"{configuration}/seed {seed} has wrong {key}")
    if configuration == "rcra":
        if float(summary.get("train/lr_alignment_gate", -1.0)) != 2e-4:
            raise RuntimeError(f"RCRA/seed {seed} has wrong alignment LR")
    elif "train/lr_alignment_gate" in summary:
        raise RuntimeError(f"FAM/seed {seed} unexpectedly has an alignment LR")
    if any(key.startswith("test/") for key in summary):
        raise RuntimeError(f"{configuration}/seed {seed} ran an automatic test")
    return {
        "run_id": checkpoint_path.parents[2].name.rsplit("-", 1)[-1],
        "run_dir": str(checkpoint_path.parents[2].resolve()),
        "final_epoch": 10,
        "optimizer_steps": EXPECTED_TRAIN_STEPS,
        "runtime_seconds": float(summary["_runtime"]),
    }


def stage_b_decision(deltas, rule):
    summary = summarize_signed_values(deltas)
    if summary is None:
        return None
    wins = sum(float(delta) > 0.0 for delta in deltas)
    passes_mean = summary["mean"] >= float(rule["minimum_mean_gain"])
    passes_wins = wins >= int(rule["minimum_positive_seed_wins"])
    passed = passes_mean and passes_wins
    return {
        "status": "pass_rcra" if passed else "fail_retain_fam",
        "passes_mean_gain": passes_mean,
        "passes_win_count": passes_wins,
        "candidate_wins": wins,
        "ties": sum(float(delta) == 0.0 for delta in deltas),
        "summary": summary,
        "selected_architecture": "rcra" if passed else "fam",
    }


def raw_result_path(output_dir, configuration, seed):
    return output_dir / "raw" / f"{configuration}_seed_{seed}_latest.json"


def build_aggregate(payloads, protocol, protocol_hash, manifest, training_inventory, output_dir):
    rows = sorted(payloads, key=lambda row: (row["configuration"], row["seed"]))
    summaries = {}
    for configuration in EXPECTED_CONFIGURATIONS:
        selected = [row for row in rows if row["configuration"] == configuration]
        summaries[configuration] = {
            metric: summarize_values([row["metrics"][metric] for row in selected])
            for metric in SCALAR_METRICS
            if selected and all(row["metrics"].get(metric) is not None for row in selected)
        }

    paired_deltas = {}
    baseline_values = []
    candidate_values = []
    for seed in EXPECTED_SEEDS:
        values = {
            row["configuration"]: float(row["metrics"]["map_50"])
            for row in rows
            if row["seed"] == seed
        }
        if set(values) == set(EXPECTED_CONFIGURATIONS):
            paired_deltas[str(seed)] = values["rcra"] - values["fam"]
            baseline_values.append(values["fam"])
            candidate_values.append(values["rcra"])

    expected_keys = {
        (configuration, seed)
        for configuration in EXPECTED_CONFIGURATIONS
        for seed in EXPECTED_SEEDS
    }
    actual_keys = {(row["configuration"], row["seed"]) for row in rows}
    decision = stage_b_decision(
        list(paired_deltas.values()), protocol["primary_comparison"]
    )
    if actual_keys != expected_keys:
        decision = None
    aggregate = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "protocol_complete": actual_keys == expected_keys,
        "checkpoint": "latest_after_fixed_epoch_10",
        "source_manifest": manifest,
        "source_manifest_sha256": stable_json_hash(manifest),
        "training_inventory": training_inventory,
        "interpretation_constraints": protocol["interpretation"],
        "historical_reference": protocol["historical_reference"],
        "results": rows,
        "across_seed_summaries": summaries,
        "rcra_minus_fam_map50": {
            "seed_values": paired_deltas,
            "decision": decision,
            "tests_exploratory": (
                paired_tests(baseline_values, candidate_values)
                if len(baseline_values) == len(EXPECTED_SEEDS)
                else None
            ),
        },
    }
    json_path = output_dir / "rtdetr_fam_rcra_full_data_stage_b_evaluation.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(aggregate), output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    csv_path = output_dir / "rtdetr_fam_rcra_full_data_stage_b_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        fields = [
            "configuration",
            "seed",
            "run_id",
            "checkpoint_sha256",
            "n_samples",
            *SCALAR_METRICS,
        ]
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "configuration": row["configuration"],
                    "seed": row["seed"],
                    "run_id": row["training_summary"]["run_id"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "n_samples": row["n_samples"],
                    **{metric: row["metrics"].get(metric) for metric in SCALAR_METRICS},
                }
            )
    print(f"Saved aggregate: {json_path}")
    print(f"Saved table: {csv_path}")
    return aggregate


def _set_evaluation_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--configurations", nargs="+", choices=EXPECTED_CONFIGURATIONS
    )
    parser.add_argument("--seeds", nargs="+", type=int, choices=EXPECTED_SEEDS)
    args = parser.parse_args()

    protocol_path = _resolve_repo_path(args.protocol)
    protocol = load_protocol(protocol_path)
    flattened = validate_training_configs(protocol)
    training_inventory = verify_training_inventory(protocol)
    manifest = build_source_manifest(protocol)
    loader = build_paired_loader(protocol, manifest, args.batch_size, args.workers)
    print(
        json.dumps(
            {
                "training_frames": training_inventory["n_frames"],
                "training_inventory_sha256": training_inventory["inventory_sha256"],
                "evaluation_frames": manifest["n_frames"],
                "evaluation_vis_boxes": manifest["n_vis_boxes"],
                "evaluation_inventory_sha256": manifest["inventory_sha256"],
                "checkpoint": protocol["checkpoint"],
            },
            indent=2,
        )
    )
    if args.prepare_only:
        print("Prepare-only OK: matched grids and frozen train/test inventories")
        return

    configurations = args.configurations or list(EXPECTED_CONFIGURATIONS)
    seeds = args.seeds or list(EXPECTED_SEEDS)
    checkpoints = {}
    for configuration in configurations:
        settings = protocol["configurations"][configuration]
        for seed in seeds:
            checkpoint = Path(
                resolve_local_wandb_checkpoint(
                    settings["project"],
                    seed,
                    checkpoint="latest",
                    wandb_root=REPO_ROOT / "wandb",
                )
            ).resolve()
            summary = audit_completed_run(
                checkpoint,
                flattened[(configuration, seed)],
                settings["project"],
                seed,
                configuration,
            )
            checkpoints[(configuration, seed)] = {
                "path": checkpoint,
                "sha256": file_sha256(checkpoint),
                "training_summary": summary,
            }
            print(f"configuration={configuration} seed={seed} checkpoint={checkpoint}")
    if args.dry_run:
        print(f"Dry run OK: {len(checkpoints)} matched latest checkpoints resolved")
        return

    output_dir = _resolve_repo_path(args.output_dir or protocol["output_dir"])
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    batch_size = int(args.batch_size or protocol["batch_size"])
    protocol_hash = stable_json_hash(protocol)
    manifest_hash = stable_json_hash(manifest)
    device = resolve_device(args.device)
    payloads = []
    _set_evaluation_seed(int(protocol["evaluation_seed"]))

    for configuration in configurations:
        settings = protocol["configurations"][configuration]
        for seed in seeds:
            checkpoint = checkpoints[(configuration, seed)]
            expected = {
                "protocol_id": protocol["id"],
                "protocol_sha256": protocol_hash,
                "source_manifest_sha256": manifest_hash,
                "configuration": configuration,
                "project": settings["project"],
                "seed": seed,
                "checkpoint_kind": "latest",
                "checkpoint": str(checkpoint["path"]),
                "checkpoint_sha256": checkpoint["sha256"],
                "batch_size": batch_size,
                "confidence_threshold": float(protocol["confidence_threshold"]),
            }
            path = raw_result_path(output_dir, configuration, seed)
            if path.is_file() and not args.force:
                payloads.append(load_compatible_raw(path, expected))
                print(f"[skip] {path}")
                continue

            run = copy.deepcopy(flattened[(configuration, seed)])
            run["model"]["params"]["threshold"] = float(
                protocol["confidence_threshold"]
            )
            print(f"[load] configuration={configuration} seed={seed} device={device}")
            model = load_fusion_model(run["model"], checkpoint["path"], device)
            measured = evaluate_modalities(model, loader, device, ["vis_ir"])["vis_ir"]
            payload = {
                **expected,
                "schema_version": 1,
                "protocol_complete": True,
                "ground_truth": "vis",
                "n_dataset_images": len(loader.dataset),
                "n_samples": measured["n_samples"],
                "training_summary": checkpoint["training_summary"],
                "metrics": measured["metrics"],
            }
            with path.open("w", encoding="utf-8") as output_file:
                json.dump(jsonable(payload), output_file, indent=2, sort_keys=True)
                output_file.write("\n")
            payloads.append(payload)
            print(
                f"[done] {configuration}/seed={seed} "
                f"map50={payload['metrics']['map_50']:.6f} -> {path}"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = build_aggregate(
        payloads,
        protocol,
        protocol_hash,
        manifest,
        training_inventory,
        output_dir,
    )
    if aggregate["protocol_complete"]:
        print("Protocol complete: all ten matched Stage-B evaluations are present")


if __name__ == "__main__":
    main()
