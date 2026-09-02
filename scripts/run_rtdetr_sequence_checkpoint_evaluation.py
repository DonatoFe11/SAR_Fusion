#!/usr/bin/env python3
"""Evaluate fixed-10 RT-DETR+FAM best/latest checkpoints on paired MtErie.

The runner resolves the five completed training runs from their exact W&B
project and seed, evaluates both checkpoints on the same frozen 708-frame
VIS+IR loader, persists checkpoint hashes and raw metrics, then writes paired
best-minus-latest summaries.  `best` remains the predeclared primary result;
MtErie is never used to revise the selected epoch.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import load_fusion_model, load_run_config  # noqa: E402
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402
from scripts.run_rtdetr_carnation_stress_test import (  # noqa: E402
    SCALAR_METRICS,
    file_sha256,
    stable_json_hash,
)
from scripts.run_rtdetr_fam_level_ablation import jsonable, resolve_device  # noqa: E402
from scripts.run_rtdetr_paired_modality_evaluation import (  # noqa: E402
    build_paired_loader,
    build_source_manifest,
    evaluate_modalities,
    summarize_signed_values,
)


DEFAULT_PROTOCOL = (
    "parameters/RTDETR/rtdetr_fam_sequence_validation_checkpoint_evaluation.yaml"
)
EXPECTED_SEEDS = [40, 41, 42, 43, 44]
EXPECTED_CHECKPOINTS = ["best", "latest"]


def load_protocol(path):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Checkpoint-evaluation YAML must contain a protocol mapping")
    if protocol.get("id") != "rtdetr_fam_sequence_validation_checkpoint_evaluation_v1":
        raise ValueError("Unexpected checkpoint-evaluation protocol id")
    if protocol.get("status") != "frozen_before_inference":
        raise ValueError("Protocol must remain frozen_before_inference")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("Protocol must contain seeds 40--44 in order")
    if protocol.get("checkpoints") != EXPECTED_CHECKPOINTS:
        raise ValueError("Protocol must evaluate best and latest in that order")
    if protocol.get("primary_checkpoint") != "best":
        raise ValueError("The predeclared primary checkpoint must be best")
    if protocol.get("diagnostic_checkpoint") != "latest":
        raise ValueError("The diagnostic checkpoint must be latest")
    source = protocol.get("source", {})
    if source.get("ground_truth") != "vis":
        raise ValueError("Paired MtErie evaluation must use VIS ground truth")
    interpretation = protocol.get("interpretation", {})
    if not interpretation.get("best_is_predeclared_primary"):
        raise ValueError("Protocol must declare best as the primary checkpoint")
    for forbidden in (
        "model_selection_from_mterie_allowed",
        "threshold_tuning_allowed",
        "seed_selection_allowed",
    ):
        if interpretation.get(forbidden) is not False:
            raise ValueError(f"{forbidden} must be false")
    return protocol


def raw_result_path(output_dir, seed, checkpoint):
    return output_dir / "raw" / f"seed_{seed}_{checkpoint}.json"


def _set_evaluation_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summary_path(checkpoint_path):
    return checkpoint_path.parents[1] / "wandb-summary.json"


def load_training_summary(checkpoint_path, seed):
    path = _summary_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing W&B summary for seed {seed}: {path}")
    with path.open(encoding="utf-8") as input_file:
        summary = json.load(input_file)
    required = ("best_epoch", "best_map_50", "validate/map_50", "_runtime")
    missing = [key for key in required if key not in summary]
    if missing:
        raise RuntimeError(f"Training summary for seed {seed} lacks {missing}")
    return {
        "run_id": checkpoint_path.parents[2].name.rsplit("-", 1)[-1],
        "run_dir": str(checkpoint_path.parents[2].resolve()),
        "best_epoch": int(summary["best_epoch"]),
        "best_validation_map50": float(summary["best_map_50"]),
        "latest_epoch": 10,
        "latest_validation_map50": float(summary["validate/map_50"]),
        "runtime_seconds": float(summary["_runtime"]),
    }


def expected_raw(protocol, protocol_hash, manifest_hash, seed, checkpoint, path, batch_size):
    return {
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "source_manifest_sha256": manifest_hash,
        "training_project": protocol["training_project"],
        "seed": int(seed),
        "checkpoint_kind": checkpoint,
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "batch_size": int(batch_size),
        "confidence_threshold": float(protocol["confidence_threshold"]),
    }


def load_compatible_raw(path, expected):
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Existing raw result is incompatible with the frozen protocol: {path}\n"
            + json.dumps(mismatches, indent=2)
        )
    return payload


def summarize_values(values):
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def build_aggregate(payloads, protocol, protocol_hash, manifest, output_dir):
    rows = sorted(payloads, key=lambda row: (row["seed"], row["checkpoint_kind"]))
    summaries = {}
    for checkpoint in EXPECTED_CHECKPOINTS:
        selected = [row for row in rows if row["checkpoint_kind"] == checkpoint]
        summaries[checkpoint] = {
            metric: summarize_values([row["metrics"][metric] for row in selected])
            for metric in SCALAR_METRICS
            if all(row["metrics"].get(metric) is not None for row in selected)
        }

    deltas = {}
    for seed in EXPECTED_SEEDS:
        values = {
            row["checkpoint_kind"]: float(row["metrics"]["map_50"])
            for row in rows
            if row["seed"] == seed
        }
        if set(values) == set(EXPECTED_CHECKPOINTS):
            deltas[str(seed)] = values["best"] - values["latest"]

    training = {str(row["seed"]): row["training_summary"] for row in rows if row["checkpoint_kind"] == "best"}
    expected_keys = {(seed, checkpoint) for seed in EXPECTED_SEEDS for checkpoint in EXPECTED_CHECKPOINTS}
    actual_keys = {(row["seed"], row["checkpoint_kind"]) for row in rows}
    aggregate = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "protocol_complete": actual_keys == expected_keys,
        "primary_checkpoint": "best",
        "diagnostic_checkpoint": "latest",
        "source_manifest": manifest,
        "source_manifest_sha256": stable_json_hash(manifest),
        "interpretation_constraints": protocol["interpretation"],
        "training_runs": training,
        "runtime_seconds": summarize_values(
            [entry["runtime_seconds"] for entry in training.values()]
        ),
        "results": rows,
        "checkpoint_summaries": summaries,
        "best_minus_latest_map50": {
            "seed_values": deltas,
            "summary": summarize_signed_values(deltas.values()),
            "best_wins": sum(delta > 0 for delta in deltas.values()),
            "ties": sum(delta == 0 for delta in deltas.values()),
        },
    }
    json_path = output_dir / "rtdetr_fam_sequence_checkpoint_evaluation.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(aggregate), output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    csv_path = output_dir / "rtdetr_fam_sequence_checkpoint_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        fields = [
            "seed", "checkpoint_kind", "selected_epoch", "validation_map50",
            "map", "map_50", "map_75", "mar_100", "checkpoint_sha256",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            summary = row["training_summary"]
            checkpoint = row["checkpoint_kind"]
            writer.writerow(
                {
                    "seed": row["seed"],
                    "checkpoint_kind": checkpoint,
                    "selected_epoch": summary[f"{checkpoint}_epoch"],
                    "validation_map50": summary[f"{checkpoint}_validation_map50"],
                    "map": row["metrics"].get("map"),
                    "map_50": row["metrics"].get("map_50"),
                    "map_75": row["metrics"].get("map_75"),
                    "mar_100": row["metrics"].get("mar_100"),
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
            )
    print(f"Saved aggregate: {json_path}")
    print(f"Saved table: {csv_path}")
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, choices=EXPECTED_SEEDS)
    parser.add_argument("--checkpoints", nargs="+", choices=EXPECTED_CHECKPOINTS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = REPO_ROOT / protocol_path
    protocol = load_protocol(protocol_path.resolve())
    protocol_hash = stable_json_hash(protocol)
    manifest = build_source_manifest(protocol)
    loader = build_paired_loader(protocol, manifest, args.batch_size, args.workers)
    print(
        json.dumps(
            {
                "paired_frames": manifest["n_frames"],
                "vis_boxes": manifest["n_vis_boxes"],
                "vis_empty_frames": manifest["n_vis_empty_frames"],
                "inventory_sha256": manifest["inventory_sha256"],
                "ground_truth": manifest["ground_truth"],
            },
            indent=2,
        )
    )
    if args.prepare_only:
        return

    seeds = args.seeds or list(protocol["seeds"])
    checkpoints = args.checkpoints or list(protocol["checkpoints"])
    resolved = {}
    for seed in seeds:
        for checkpoint in checkpoints:
            path = Path(
                resolve_local_wandb_checkpoint(
                    protocol["training_project"],
                    seed,
                    checkpoint=checkpoint,
                    wandb_root=REPO_ROOT / "wandb",
                )
            ).resolve()
            resolved[(seed, checkpoint)] = path
            print(f"seed={seed} checkpoint={checkpoint} path={path}")
    if args.dry_run:
        print(f"Dry run OK: {len(resolved)} checkpoints resolved")
        return

    output_dir = Path(args.output_dir or protocol["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    batch_size = int(args.batch_size or protocol["batch_size"])
    device = resolve_device(args.device)
    manifest_hash = stable_json_hash(manifest)
    _set_evaluation_seed(int(protocol["evaluation_seed"]))
    payloads = []

    for seed in seeds:
        run_config = load_run_config(
            REPO_ROOT / protocol["training_config"], run_index=seed - EXPECTED_SEEDS[0]
        )
        if int(run_config["seed"]) != seed:
            raise RuntimeError(f"Training run_index does not map to seed {seed}")
        model_params = run_config["model"]
        model_params["params"]["threshold"] = float(protocol["confidence_threshold"])
        training_summary = load_training_summary(resolved[(seed, checkpoints[0])], seed)
        for checkpoint in checkpoints:
            checkpoint_path = resolved[(seed, checkpoint)]
            expected = expected_raw(
                protocol, protocol_hash, manifest_hash, seed, checkpoint,
                checkpoint_path, batch_size,
            )
            path = raw_result_path(output_dir, seed, checkpoint)
            if path.is_file() and not args.force:
                payloads.append(load_compatible_raw(path, expected))
                print(f"[skip] {path}")
                continue
            print(f"[load] seed={seed} checkpoint={checkpoint} device={device}")
            model = load_fusion_model(model_params, checkpoint_path, device)
            measured = evaluate_modalities(model, loader, device, ["vis_ir"])["vis_ir"]
            payload = {
                **expected,
                "schema_version": 1,
                "protocol_complete": True,
                "n_dataset_images": len(loader.dataset),
                "n_samples": measured["n_samples"],
                "training_summary": training_summary,
                "metrics": measured["metrics"],
            }
            with path.open("w", encoding="utf-8") as output_file:
                json.dump(jsonable(payload), output_file, indent=2, sort_keys=True)
                output_file.write("\n")
            payloads.append(payload)
            print(
                f"[done] seed={seed} checkpoint={checkpoint} "
                f"map50={payload['metrics']['map_50']:.6f} -> {path}"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = build_aggregate(
        payloads, protocol, protocol_hash, manifest, output_dir
    )
    if aggregate["protocol_complete"]:
        print("Protocol complete: all 10 best/latest evaluations are present")


if __name__ == "__main__":
    main()
