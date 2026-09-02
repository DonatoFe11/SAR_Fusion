#!/usr/bin/env python3
"""Characterize the selected current-code FAM on paired sensor interventions.

Stage B has already closed architecture selection. This runner audits the five
matched FAM runs, reuses one frozen 708-frame VIS+IR loader and measures VIS+IR,
VIS-only and IR-only by channel masking with unchanged VIS ground truth.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import load_fusion_model  # noqa: E402
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402
from scripts.run_rtdetr_carnation_stress_test import (  # noqa: E402
    SCALAR_METRICS,
    file_sha256,
    load_compatible_raw,
    stable_json_hash,
    summarize_values,
)
from scripts.run_rtdetr_fam_level_ablation import (  # noqa: E402
    jsonable,
    resolve_device,
)
from scripts.run_rtdetr_fam_rcra_full_data_stage_b_evaluation import (  # noqa: E402
    audit_completed_run,
    load_protocol as load_stage_b_protocol,
    stage_b_decision,
    validate_training_configs,
)
from scripts.run_rtdetr_paired_modality_evaluation import (  # noqa: E402
    EXPECTED_MODALITIES,
    build_paired_loader,
    build_source_manifest,
    evaluate_modalities,
    summarize_signed_values,
)


DEFAULT_PROTOCOL = (
    "parameters/RTDETR/rtdetr_fam_full_data_paired_modality_evaluation.yaml"
)
PROTOCOL_ID = "rtdetr_fam_full_data_paired_modality_evaluation_v1"
EXPECTED_SEEDS = [40, 41, 42, 43, 44]
EXPECTED_PROJECT = "RTDETR_FAM_FullData_StageB_FiveSeed"


def resolve_repo_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_protocol(path):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("FAM modality YAML must contain a protocol mapping")
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError("Unexpected current-code FAM modality protocol id")
    if protocol.get("status") != "frozen_before_descriptive_inference":
        raise ValueError("FAM modality protocol must remain frozen before inference")
    if protocol.get("checkpoint") != "latest":
        raise ValueError("FAM modality evaluation must use latest checkpoints")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("FAM modality protocol must contain seeds 40--44")
    if protocol.get("project") != EXPECTED_PROJECT:
        raise ValueError("FAM modality protocol points to an unexpected project")
    if float(protocol.get("confidence_threshold", -1.0)) != 0.01:
        raise ValueError("FAM modality confidence threshold must remain 0.01")
    if protocol.get("modalities") != EXPECTED_MODALITIES:
        raise ValueError("Unexpected paired-modality channel interventions")

    source = protocol.get("source", {})
    if source.get("ground_truth") != "vis":
        raise ValueError("All modality interventions must use VIS ground truth")
    if source.get("pairing") != "existing_wisard_sorted_zip":
        raise ValueError("The modality source must reproduce WiSARD pairing")

    selection = protocol.get("selection_source", {})
    if selection.get("required_decision") != "fail_retain_fam":
        raise ValueError("The modality protocol must preserve the Stage-B decision")
    if selection.get("required_selected_architecture") != "fam":
        raise ValueError("The modality protocol must characterize selected FAM")

    interpretation = protocol.get("interpretation", {})
    expected_constraints = {
        "model_selection_allowed",
        "checkpoint_selection_allowed",
        "threshold_tuning_allowed",
        "seed_selection_allowed",
        "architecture_tuning_allowed",
        "claim_blind_test_allowed",
    }
    if set(interpretation) != expected_constraints:
        raise ValueError("Unexpected modality interpretation constraints")
    if any(value is not False for value in interpretation.values()):
        raise ValueError("Descriptive modality results cannot reopen selection")

    references = protocol.get("expected_fusion_map50", {})
    if float(references.get("tolerance", -1.0)) != 0.0002:
        raise ValueError("Unexpected VIS+IR reconstruction tolerance")
    values = {int(seed): float(value) for seed, value in references.get("values", {}).items()}
    if set(values) != set(EXPECTED_SEEDS):
        raise ValueError("VIS+IR references must cover all five seeds")
    return protocol


def verify_closed_selection(protocol):
    """Recompute the Stage-B decision from the versioned paired result."""
    selection = protocol["selection_source"]
    stage_b_path = resolve_repo_path(selection["protocol"])
    stage_b_protocol = load_stage_b_protocol(stage_b_path)
    result_path = resolve_repo_path(selection["result_csv"])
    with result_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if len(rows) != len(EXPECTED_SEEDS):
        raise RuntimeError("Stage-B result must contain five paired rows")
    if [int(row["seed"]) for row in rows] != EXPECTED_SEEDS:
        raise RuntimeError("Stage-B result seeds differ from the freeze")

    deltas = [float(row["rcra_minus_fam_map50"]) for row in rows]
    decision = stage_b_decision(deltas, stage_b_protocol["primary_comparison"])
    if decision["status"] != selection["required_decision"]:
        raise RuntimeError("Stage-B decision no longer matches the modality freeze")
    if decision["selected_architecture"] != selection["required_selected_architecture"]:
        raise RuntimeError("Stage-B selected architecture no longer matches FAM")

    references = {
        int(seed): float(value)
        for seed, value in protocol["expected_fusion_map50"]["values"].items()
    }
    measured = {int(row["seed"]): float(row["fam_map50"]) for row in rows}
    if any(abs(measured[seed] - references[seed]) > 5e-10 for seed in EXPECTED_SEEDS):
        raise RuntimeError("Frozen VIS+IR references differ from the Stage-B CSV")
    return {
        "stage_b_protocol_sha256": stable_json_hash(stage_b_protocol),
        "stage_b_result": str(result_path),
        "stage_b_result_sha256": file_sha256(result_path),
        "decision": decision,
    }, stage_b_protocol


def check_fusion_reference(protocol, seed, measured):
    references = protocol["expected_fusion_map50"]
    expected = float(references["values"][seed])
    tolerance = float(references["tolerance"])
    difference = float(measured) - expected
    if abs(difference) > tolerance:
        raise RuntimeError(
            f"VIS+IR reconstruction failed for FAM/seed {seed}: "
            f"measured={measured:.6f}, expected={expected:.6f}, "
            f"difference={difference:+.6f}, tolerance={tolerance:.6f}"
        )
    return {"stage_b": expected, "difference": difference, "tolerance": tolerance}


def raw_result_path(output_dir, seed, modality):
    return output_dir / "raw" / f"fam_seed_{seed}_{modality}.json"


def build_aggregate(
    payloads,
    protocol,
    protocol_hash,
    manifest,
    selection_audit,
    output_dir,
    complete,
):
    rows = sorted(payloads, key=lambda row: (row["seed"], row["modality"]))
    summaries = {}
    for modality in protocol["modalities"]:
        selected = [row for row in rows if row["modality"] == modality]
        summaries[modality] = {
            metric: summarize_values([row["metrics"].get(metric) for row in selected])
            for metric in SCALAR_METRICS
        }

    paired = {
        "fusion_minus_vis": {},
        "fusion_minus_ir": {},
        "fusion_minus_best_single": {},
        "vis_minus_ir": {},
    }
    for seed in protocol["seeds"]:
        values = {
            row["modality"]: float(row["metrics"]["map_50"])
            for row in rows
            if row["seed"] == seed
        }
        if set(values) != set(protocol["modalities"]):
            continue
        paired["fusion_minus_vis"][str(seed)] = values["vis_ir"] - values["vis"]
        paired["fusion_minus_ir"][str(seed)] = values["vis_ir"] - values["ir"]
        paired["fusion_minus_best_single"][str(seed)] = values["vis_ir"] - max(
            values["vis"], values["ir"]
        )
        paired["vis_minus_ir"][str(seed)] = values["vis"] - values["ir"]

    paired_summaries = {
        name: {
            "seed_values": seed_values,
            "summary": summarize_signed_values(seed_values.values()),
            "positive_seed_count": sum(value > 0 for value in seed_values.values()),
            "ties": sum(value == 0 for value in seed_values.values()),
        }
        for name, seed_values in paired.items()
    }
    expected_keys = {
        (seed, modality)
        for seed in protocol["seeds"]
        for modality in protocol["modalities"]
    }
    actual_keys = {(row["seed"], row["modality"]) for row in rows}
    aggregate = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "protocol_complete": complete and actual_keys == expected_keys,
        "purpose": protocol["purpose"],
        "experimental_unit": "checkpoint/seed",
        "common_sample_rule": (
            "same paired batch, same VIS ground truth, channel masking only"
        ),
        "selection_audit": selection_audit,
        "source_manifest": manifest,
        "source_manifest_sha256": stable_json_hash(manifest),
        "interpretation_constraints": protocol["interpretation"],
        "results": rows,
        "across_seed_summaries": summaries,
        "paired_map50_deltas": paired_summaries,
    }
    json_path = output_dir / "rtdetr_fam_full_data_paired_modality_evaluation.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(aggregate), output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    csv_path = output_dir / "rtdetr_fam_full_data_paired_modality_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["seed", "run_id", "modality", "n_samples", *SCALAR_METRICS],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": row["seed"],
                    "run_id": row["training_summary"]["run_id"],
                    "modality": row["modality"],
                    "n_samples": row["n_samples"],
                    **{metric: row["metrics"].get(metric) for metric in SCALAR_METRICS},
                }
            )
    print(f"Saved aggregate: {json_path}")
    print(f"Saved table: {csv_path}")
    return aggregate


def set_evaluation_seed(seed):
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
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, choices=EXPECTED_SEEDS)
    parser.add_argument(
        "--modalities", nargs="+", choices=tuple(EXPECTED_MODALITIES)
    )
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    protocol_path = resolve_repo_path(args.protocol)
    protocol = load_protocol(protocol_path)
    protocol_hash = stable_json_hash(protocol)
    selection_audit, stage_b_protocol = verify_closed_selection(protocol)
    flattened = validate_training_configs(stage_b_protocol)
    if resolve_repo_path(protocol["training_config"]) != resolve_repo_path(
        stage_b_protocol["configurations"]["fam"]["training_config"]
    ):
        raise RuntimeError("Modality training config differs from selected Stage-B FAM")
    if protocol["project"] != stage_b_protocol["configurations"]["fam"]["project"]:
        raise RuntimeError("Modality project differs from selected Stage-B FAM")

    manifest = build_source_manifest(protocol)
    loader = build_paired_loader(protocol, manifest, args.batch_size, args.workers)
    print(
        json.dumps(
            {
                "selected_architecture": selection_audit["decision"][
                    "selected_architecture"
                ],
                "paired_frames": manifest["n_frames"],
                "vis_boxes": manifest["n_vis_boxes"],
                "vis_empty_frames": manifest["n_vis_empty_frames"],
                "inventory_sha256": manifest["inventory_sha256"],
                "ground_truth_for_every_condition": manifest["ground_truth"],
            },
            indent=2,
        )
    )
    if args.prepare_only:
        print("Prepare-only OK: closed selection and frozen paired inventory")
        return

    seeds = args.seeds or list(protocol["seeds"])
    modalities = args.modalities or list(protocol["modalities"])
    checkpoints = {}
    for seed in seeds:
        checkpoint = Path(
            resolve_local_wandb_checkpoint(
                protocol["project"],
                seed,
                checkpoint="latest",
                wandb_root=REPO_ROOT / "wandb",
            )
        ).resolve()
        summary = audit_completed_run(
            checkpoint,
            flattened[("fam", seed)],
            protocol["project"],
            seed,
            "fam",
        )
        checkpoints[seed] = {
            "path": checkpoint,
            "sha256": file_sha256(checkpoint),
            "training_summary": summary,
        }
        print(f"configuration=fam seed={seed} checkpoint={checkpoint}")
    if args.dry_run:
        print(f"Dry run OK: {len(seeds) * len(modalities)} evaluations resolved")
        return

    output_dir = resolve_repo_path(args.output_dir or protocol["output_dir"])
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    batch_size = int(args.batch_size or protocol["batch_size"])
    manifest_hash = stable_json_hash(manifest)
    device = resolve_device(args.device)
    payloads = []
    set_evaluation_seed(int(protocol["evaluation_seed"]))

    for seed in seeds:
        checkpoint = checkpoints[seed]
        expected_by_modality = {}
        missing_modalities = []
        for modality in modalities:
            expected = {
                "protocol_id": protocol["id"],
                "protocol_sha256": protocol_hash,
                "source_manifest_sha256": manifest_hash,
                "configuration": "fam",
                "project": protocol["project"],
                "seed": seed,
                "modality": modality,
                "checkpoint_kind": "latest",
                "checkpoint": str(checkpoint["path"]),
                "checkpoint_sha256": checkpoint["sha256"],
                "batch_size": batch_size,
                "confidence_threshold": float(protocol["confidence_threshold"]),
                "max_batches": args.max_batches,
            }
            expected_by_modality[modality] = expected
            path = raw_result_path(output_dir, seed, modality)
            if path.is_file() and not args.force:
                payloads.append(load_compatible_raw(path, expected))
                print(f"[skip] {path}")
            else:
                missing_modalities.append(modality)
        if not missing_modalities:
            continue

        run = copy.deepcopy(flattened[("fam", seed)])
        run["model"]["params"]["threshold"] = float(
            protocol["confidence_threshold"]
        )
        print(f"[load] configuration=fam seed={seed} device={device}")
        model = load_fusion_model(run["model"], checkpoint["path"], device)
        print(f"[run] seed={seed} modalities={','.join(missing_modalities)}")
        measured = evaluate_modalities(
            model,
            loader,
            device,
            missing_modalities,
            max_batches=args.max_batches,
        )
        for modality in missing_modalities:
            result = measured[modality]
            payload = {
                **expected_by_modality[modality],
                "schema_version": 1,
                "protocol_complete": args.max_batches is None,
                "channel_intervention": protocol["modalities"][modality],
                "ground_truth": "vis",
                "n_dataset_images": len(loader.dataset),
                "n_samples": result["n_samples"],
                "training_summary": checkpoint["training_summary"],
                "metrics": result["metrics"],
            }
            if modality == "vis_ir" and args.max_batches is None:
                payload["fusion_reference_check"] = check_fusion_reference(
                    protocol, seed, result["metrics"]["map_50"]
                )
            path = raw_result_path(output_dir, seed, modality)
            with path.open("w", encoding="utf-8") as output_file:
                json.dump(jsonable(payload), output_file, indent=2, sort_keys=True)
                output_file.write("\n")
            payloads.append(payload)
            print(
                f"[done] seed={seed} modality={modality} "
                f"map50={result['metrics']['map_50']:.6f} n={result['n_samples']}"
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    complete = (
        args.max_batches is None
        and set(seeds) == set(protocol["seeds"])
        and set(modalities) == set(protocol["modalities"])
    )
    aggregate = build_aggregate(
        payloads,
        protocol,
        protocol_hash,
        manifest,
        selection_audit,
        output_dir,
        complete,
    )
    if aggregate["protocol_complete"]:
        print("Protocol complete: all 15 selected-FAM modality evaluations are present")


if __name__ == "__main__":
    main()
