#!/usr/bin/env python3
"""Post-hoc native-IR coordinate diagnostic for selected Stage-B FAM.

The paired sensor ablation masks RGB after constructing a VIS-coordinate
four-channel sample and retains VIS ground truth. This runner instead evaluates
the exact 708 IR counterparts with native IR preprocessing and native IR labels
to determine whether a low paired masked-IR score reflects IR-branch collapse.
The two metrics answer different questions and must not be ranked directly.
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
from sarfusion.data.utils import build_preprocessor, get_collate_fn  # noqa: E402
from sarfusion.data.wisard import IR_ITEM, WiSARDDataset  # noqa: E402
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402
from scripts.run_rtdetr_carnation_stress_test import (  # noqa: E402
    SCALAR_METRICS,
    file_sha256,
    load_compatible_raw,
    stable_json_hash,
    summarize_values,
)
from scripts.run_rtdetr_fam_full_data_paired_modality_evaluation import (  # noqa: E402
    load_protocol as load_paired_protocol,
    verify_closed_selection,
)
from scripts.run_rtdetr_fam_level_ablation import (  # noqa: E402
    jsonable,
    resolve_device,
)
from scripts.run_rtdetr_fam_rcra_full_data_stage_b_evaluation import (  # noqa: E402
    audit_completed_run,
    validate_training_configs,
)
from scripts.run_rtdetr_paired_modality_evaluation import (  # noqa: E402
    build_source_manifest,
    evaluate_modalities,
)


DEFAULT_PROTOCOL = (
    "parameters/RTDETR/rtdetr_fam_full_data_native_ir_coordinate_diagnostic.yaml"
)
PROTOCOL_ID = "rtdetr_fam_full_data_native_ir_coordinate_diagnostic_v1"
EXPECTED_SEEDS = [40, 41, 42, 43, 44]
EXPECTED_PROJECT = "RTDETR_FAM_FullData_StageB_FiveSeed"


def resolve_repo_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_protocol(path):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Native-IR diagnostic YAML must contain a protocol mapping")
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError("Unexpected native-IR diagnostic protocol id")
    if protocol.get("status") != "post_hoc_diagnostic":
        raise ValueError("Native-IR analysis must remain labelled post-hoc")
    if protocol.get("checkpoint") != "latest":
        raise ValueError("Native-IR diagnostic must use latest checkpoints")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("Native-IR diagnostic must contain seeds 40--44")
    if protocol.get("project") != EXPECTED_PROJECT:
        raise ValueError("Native-IR diagnostic points to an unexpected project")
    if float(protocol.get("confidence_threshold", -1.0)) != 0.01:
        raise ValueError("Native-IR threshold must remain 0.01")

    source = protocol.get("source", {})
    if source.get("ground_truth") != "ir":
        raise ValueError("Native-IR diagnostic must use IR ground truth")
    if source.get("preprocessing") != "native_ir_without_adapt_ir2rgb":
        raise ValueError("Native-IR diagnostic preprocessing changed")
    if source.get("input_contract") != "zero_rgb_channels_plus_native_ir_channel":
        raise ValueError("Native-IR diagnostic input contract changed")
    if int(source.get("expected_frames", -1)) != 708:
        raise ValueError("Native-IR diagnostic must contain 708 counterparts")
    if int(source.get("expected_ir_boxes", -1)) != 1824:
        raise ValueError("Unexpected native-IR box count")
    if int(source.get("expected_ir_empty_frames", -1)) != 3:
        raise ValueError("Unexpected native-IR empty-frame count")

    interpretation = protocol.get("interpretation", {})
    if not interpretation or any(value is not False for value in interpretation.values()):
        raise ValueError("Post-hoc native-IR results cannot alter selection")
    return protocol


def build_native_ir_loader(protocol, paired_protocol, paired_manifest, run, workers=None):
    source = protocol["source"]
    ir_folders = list(source["native_ir_folders"])
    paired_ir_folders = [pair[1] for pair in paired_protocol["source"]["paired_folders"]]
    paired_vis_folders = [pair[0] for pair in paired_protocol["source"]["paired_folders"]]
    if ir_folders != paired_ir_folders:
        raise RuntimeError("Native-IR folders differ from the paired inventory")
    if list(source["corresponding_vis_folders"]) != paired_vis_folders:
        raise RuntimeError("VIS counterparts differ from the paired inventory")
    if paired_manifest["inventory_sha256"] != source["paired_inventory_sha256"]:
        raise RuntimeError("Paired source inventory changed")

    dataset_params = copy.deepcopy(run["dataset"])
    dataset_root = resolve_repo_path(protocol["dataset_root"])
    dataset_params["root"] = str(dataset_root)
    transform, _ = build_preprocessor(dataset_params)
    dataset = WiSARDDataset(
        root=dataset_root,
        folders=ir_folders,
        transform=transform,
        single_class=True,
        modal_dropout=False,
        use_tiling=False,
        test_all_tiles=False,
    )

    actual_paths = []
    n_boxes = 0
    n_empty_frames = 0
    for item_type, item in dataset.items:
        if item_type != IR_ITEM:
            raise RuntimeError("Native-IR diagnostic contains a non-IR item")
        image_path, label_path = map(Path, item)
        lines = [line for line in label_path.read_text().splitlines() if line.strip()]
        n_boxes += len(lines)
        n_empty_frames += int(not lines)
        actual_paths.append(
            {
                "ir_image": str(image_path.resolve().relative_to(dataset_root)),
                "ir_label": str(label_path.resolve().relative_to(dataset_root)),
            }
        )
    expected_paths = [
        {"ir_image": row["ir_image"], "ir_label": row["ir_label"]}
        for row in paired_manifest["rows"]
    ]
    if actual_paths != expected_paths:
        raise RuntimeError("Native-IR dataset order differs from paired counterparts")

    summary = {
        "n_frames": len(dataset),
        "n_ir_boxes": n_boxes,
        "n_ir_empty_frames": n_empty_frames,
        "native_ir_paths_sha256": stable_json_hash(actual_paths),
    }
    expected = {
        "n_frames": int(source["expected_frames"]),
        "n_ir_boxes": int(source["expected_ir_boxes"]),
        "n_ir_empty_frames": int(source["expected_ir_empty_frames"]),
    }
    if {key: summary[key] for key in expected} != expected:
        raise RuntimeError(
            "Native-IR inventory differs from expected: "
            + json.dumps({"expected": expected, "actual": summary}, indent=2)
        )

    generator = torch.Generator().manual_seed(int(protocol["evaluation_seed"]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(protocol["batch_size"]),
        num_workers=int(protocol["workers"] if workers is None else workers),
        shuffle=False,
        collate_fn=get_collate_fn(dataset),
        generator=generator,
    )
    return loader, summary


def raw_result_path(output_dir, seed):
    return output_dir / "raw" / f"fam_seed_{seed}_native_ir.json"


def build_aggregate(payloads, protocol, protocol_hash, paired_manifest, native_summary, output_dir):
    rows = sorted(payloads, key=lambda row: row["seed"])
    expected_seeds = set(protocol["seeds"])
    actual_seeds = {row["seed"] for row in rows}
    aggregate = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "protocol_status": protocol["status"],
        "protocol_complete": actual_seeds == expected_seeds,
        "purpose": protocol["purpose"],
        "experimental_unit": "checkpoint/seed",
        "coordinate_contract": {
            "input": protocol["source"]["input_contract"],
            "preprocessing": protocol["source"]["preprocessing"],
            "ground_truth": protocol["source"]["ground_truth"],
        },
        "paired_source_manifest_sha256": stable_json_hash(paired_manifest),
        "native_ir_inventory": native_summary,
        "interpretation_constraints": protocol["interpretation"],
        "results": rows,
        "across_seed_summary": {
            metric: summarize_values([row["metrics"].get(metric) for row in rows])
            for metric in SCALAR_METRICS
        },
    }
    json_path = output_dir / "rtdetr_fam_full_data_native_ir_coordinate_diagnostic.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(aggregate), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    csv_path = output_dir / "rtdetr_fam_full_data_native_ir_coordinate_diagnostic.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["seed", "run_id", "n_samples", *SCALAR_METRICS],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": row["seed"],
                    "run_id": row["training_summary"]["run_id"],
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
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, choices=EXPECTED_SEEDS)
    args = parser.parse_args()

    protocol = load_protocol(resolve_repo_path(args.protocol))
    paired_protocol = load_paired_protocol(
        resolve_repo_path(protocol["paired_ablation_protocol"])
    )
    selection_audit, stage_b_protocol = verify_closed_selection(paired_protocol)
    if selection_audit["decision"]["selected_architecture"] != "fam":
        raise RuntimeError("Native-IR diagnostic requires selected FAM")
    runs = validate_training_configs(stage_b_protocol)
    paired_manifest = build_source_manifest(paired_protocol)
    loader, native_summary = build_native_ir_loader(
        protocol,
        paired_protocol,
        paired_manifest,
        runs[("fam", 40)],
        workers=args.workers,
    )
    print(json.dumps(native_summary, indent=2))
    if args.prepare_only:
        print("Prepare-only OK: exact 708 native IR counterparts and labels")
        return

    seeds = args.seeds or list(protocol["seeds"])
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
        training_summary = audit_completed_run(
            checkpoint,
            runs[("fam", seed)],
            protocol["project"],
            seed,
            "fam",
        )
        checkpoints[seed] = {
            "path": checkpoint,
            "sha256": file_sha256(checkpoint),
            "training_summary": training_summary,
        }
        print(f"seed={seed} checkpoint={checkpoint}")
    if args.dry_run:
        print(f"Dry run OK: {len(checkpoints)} native-IR evaluations resolved")
        return

    output_dir = resolve_repo_path(args.output_dir or protocol["output_dir"])
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    protocol_hash = stable_json_hash(protocol)
    paired_manifest_hash = stable_json_hash(paired_manifest)
    native_inventory_hash = stable_json_hash(native_summary)
    device = resolve_device(args.device)
    payloads = []
    set_evaluation_seed(int(protocol["evaluation_seed"]))

    for seed in seeds:
        checkpoint = checkpoints[seed]
        expected = {
            "protocol_id": protocol["id"],
            "protocol_sha256": protocol_hash,
            "paired_source_manifest_sha256": paired_manifest_hash,
            "native_ir_inventory_sha256": native_inventory_hash,
            "project": protocol["project"],
            "seed": seed,
            "checkpoint_kind": "latest",
            "checkpoint": str(checkpoint["path"]),
            "checkpoint_sha256": checkpoint["sha256"],
            "confidence_threshold": float(protocol["confidence_threshold"]),
        }
        path = raw_result_path(output_dir, seed)
        if path.is_file() and not args.force:
            payloads.append(load_compatible_raw(path, expected))
            print(f"[skip] {path}")
            continue

        run = copy.deepcopy(runs[("fam", seed)])
        run["model"]["params"]["threshold"] = float(
            protocol["confidence_threshold"]
        )
        print(f"[load] seed={seed} device={device}")
        model = load_fusion_model(run["model"], checkpoint["path"], device)
        measured = evaluate_modalities(model, loader, device, ["vis_ir"])["vis_ir"]
        payload = {
            **expected,
            "schema_version": 1,
            "protocol_status": protocol["status"],
            "protocol_complete": True,
            "input_contract": protocol["source"]["input_contract"],
            "preprocessing": protocol["source"]["preprocessing"],
            "ground_truth": "ir",
            "n_samples": measured["n_samples"],
            "training_summary": checkpoint["training_summary"],
            "metrics": measured["metrics"],
        }
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(jsonable(payload), output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        payloads.append(payload)
        print(f"[done] seed={seed} native_ir_map50={payload['metrics']['map_50']:.6f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate = build_aggregate(
        payloads,
        protocol,
        protocol_hash,
        paired_manifest,
        native_summary,
        output_dir,
    )
    if aggregate["protocol_complete"]:
        print("Diagnostic complete: all five native-IR coordinate evaluations present")


if __name__ == "__main__":
    main()
