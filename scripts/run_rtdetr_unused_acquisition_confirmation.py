#!/usr/bin/env python3
"""Run the frozen RT-DETR confirmation on two unused WiSARD acquisitions.

The runner is deliberately fail-closed. It verifies content-addressed source
inventories, exact checkpoint hashes, the already-closed Stage-B decision and
the author's consultation attestation before loading a model. All conditions
reuse paired inputs and VIS ground truth; VIS-only is a channel intervention.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import load_fusion_model, load_run_config  # noqa: E402
from sarfusion.data.utils import build_preprocessor, get_collate_fn  # noqa: E402
from sarfusion.data.wisard import MULTI_MODALITY_ITEM, WiSARDDataset  # noqa: E402
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402
from scripts.run_rtdetr_carnation_stress_test import (  # noqa: E402
    SCALAR_METRICS,
    file_sha256,
    load_compatible_raw,
    numeric_frame_id,
    paired_tests,
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
    evaluate_modalities,
    summarize_signed_values,
)


DEFAULT_PROTOCOL = (
    "parameters/RTDETR/rtdetr_unused_acquisition_confirmation.yaml"
)
PROTOCOL_ID = "rtdetr_unused_acquisition_confirmation_v1"
EXPECTED_SEEDS = [40, 41, 42, 43, 44]
EXPECTED_ACQUISITIONS = ("carnation_0025_0026", "fhl_0407_0408")
EXPECTED_CONFIGURATIONS = (
    "historical_additive",
    "historical_fam",
    "stage_b_fam",
    "stage_b_rcra",
)
EXPECTED_CONDITIONS = {
    "vis_ir": "keep_all_four_channels",
    "vis": "zero_ir_channel",
}
ATTESTATION_STATUSES = {
    "pending",
    "no_prior_model_or_manual_experimental_use",
    "prior_or_uncertain_use",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def resolve_repo_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_sha256(value, description):
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} is not a frozen SHA-256")


def load_payload(path, require_frozen_values=True):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    attestation = payload.get("attestation")
    if not isinstance(protocol, dict) or not isinstance(attestation, dict):
        raise ValueError("Confirmation YAML requires protocol and attestation mappings")
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError("Unexpected unused-acquisition protocol id")
    if protocol.get("status") != "frozen_before_inference":
        raise ValueError("Confirmation protocol must remain frozen before inference")
    if protocol.get("checkpoint") != "latest":
        raise ValueError("Confirmation must use only latest checkpoints")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("Confirmation must contain seeds 40--44 in order")
    if float(protocol.get("confidence_threshold", -1.0)) != 0.01:
        raise ValueError("Confirmation confidence threshold must remain 0.01")
    if protocol.get("ground_truth") != "vis":
        raise ValueError("Confirmation must use VIS ground truth")
    if protocol.get("pairing") != "shared_numeric_frame_suffix":
        raise ValueError("Confirmation pairing rule changed")
    selection = protocol.get("selection_source", {})
    if selection.get("required_decision") != "fail_retain_fam":
        raise ValueError("Confirmation must preserve the closed Stage-B decision")
    if not selection.get("protocol") or not selection.get("result_csv"):
        raise ValueError("Confirmation must identify the frozen Stage-B evidence")
    if tuple(protocol.get("acquisitions", {})) != EXPECTED_ACQUISITIONS:
        raise ValueError("Unexpected acquisition set or order")
    if protocol.get("conditions") != EXPECTED_CONDITIONS:
        raise ValueError("Unexpected confirmation channel conditions")
    if tuple(protocol.get("configurations", {})) != EXPECTED_CONFIGURATIONS:
        raise ValueError("Unexpected confirmation configuration set or order")

    expected_jobs = {
        "historical_additive": ["vis_ir"],
        "historical_fam": ["vis_ir"],
        "stage_b_fam": ["vis_ir", "vis"],
        "stage_b_rcra": ["vis_ir"],
    }
    for configuration, conditions in expected_jobs.items():
        settings = protocol["configurations"][configuration]
        if settings.get("conditions") != conditions:
            raise ValueError(f"Conditions changed for {configuration}")
        if settings.get("family") not in {"historical_locked", "current_stage_b"}:
            raise ValueError(f"Unexpected evidence family for {configuration}")
        if require_frozen_values:
            hashes = settings.get("expected_checkpoint_sha256", {})
            if set(int(seed) for seed in hashes) != set(EXPECTED_SEEDS):
                raise ValueError(f"Checkpoint hashes incomplete for {configuration}")
            for seed in EXPECTED_SEEDS:
                _require_sha256(
                    hashes.get(seed), f"{configuration}/seed {seed} checkpoint"
                )

    expected_comparisons = {
        "historical_fam_minus_additive": (
            "historical_fam",
            "vis_ir",
            "historical_additive",
            "vis_ir",
        ),
        "stage_b_rcra_minus_fam": (
            "stage_b_rcra",
            "vis_ir",
            "stage_b_fam",
            "vis_ir",
        ),
        "stage_b_fam_fusion_minus_vis": (
            "stage_b_fam",
            "vis_ir",
            "stage_b_fam",
            "vis",
        ),
    }
    if set(protocol.get("comparisons", {})) != set(expected_comparisons):
        raise ValueError("Unexpected confirmation comparisons")
    for name, expected in expected_comparisons.items():
        comparison = protocol["comparisons"][name]
        actual = (
            comparison.get("candidate_configuration"),
            comparison.get("candidate_condition"),
            comparison.get("reference_configuration"),
            comparison.get("reference_condition"),
        )
        if actual != expected or comparison.get("metric") != "map_50":
            raise ValueError(f"Comparison {name} differs from the freeze")

    interpretation = protocol.get("interpretation", {})
    if interpretation.get("negative_results_must_be_reported") is not True:
        raise ValueError("Negative confirmation results must remain reportable")
    if any(
        interpretation.get(key) is not False
        for key in interpretation
        if key != "negative_results_must_be_reported"
    ):
        raise ValueError("A forbidden confirmation interpretation was enabled")

    status = attestation.get("status")
    if status not in ATTESTATION_STATUSES:
        raise ValueError("Unexpected author-attestation status")
    if status == "pending":
        if attestation.get("statement") is not None or attestation.get("recorded_on") is not None:
            raise ValueError("Pending attestation must not contain a statement/date")
    elif not attestation.get("statement") or not attestation.get("recorded_on"):
        raise ValueError("Completed attestation requires statement and recorded_on")

    if require_frozen_values:
        for acquisition, settings in protocol["acquisitions"].items():
            _require_sha256(
                settings.get("expected_common_frame_ids_sha256"),
                f"{acquisition} frame-id inventory",
            )
            _require_sha256(
                settings.get("expected_inventory_sha256"),
                f"{acquisition} content inventory",
            )
    return payload


def _sorted_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(path for path in directory.iterdir() if path.is_file())


def _index_numeric(paths, description):
    indexed = {}
    for path in paths:
        frame_id = numeric_frame_id(path)
        if frame_id in indexed:
            raise RuntimeError(f"Duplicate frame {frame_id} in {description}")
        indexed[frame_id] = path
    return indexed


def build_acquisition_manifest(
    acquisition, settings, dataset_root, verify_expected=True
):
    """Build a content-addressed VIS-GT paired inventory."""
    vis_root = dataset_root / settings["vis_folder"]
    ir_root = dataset_root / settings["ir_folder"]
    vis_images = _index_numeric(
        _sorted_files(vis_root / "images"), f"{acquisition} VIS images"
    )
    vis_labels = _index_numeric(
        _sorted_files(vis_root / "labels"), f"{acquisition} VIS labels"
    )
    ir_images = _index_numeric(
        _sorted_files(ir_root / "images"), f"{acquisition} IR images"
    )
    ir_labels = _sorted_files(ir_root / "labels")
    if set(vis_images) != set(vis_labels):
        raise RuntimeError(f"{acquisition} VIS image/label frame IDs differ")

    common_ids = sorted(set(vis_images) & set(ir_images))
    excluded_vis = sorted(set(vis_images) - set(common_ids))
    excluded_ir = sorted(set(ir_images) - set(common_ids))
    n_vis_boxes = 0
    n_vis_empty_frames = 0
    rows = []
    for frame_id in common_ids:
        vis_image = vis_images[frame_id]
        vis_label = vis_labels[frame_id]
        ir_image = ir_images[frame_id]
        label_bytes = vis_label.read_bytes()
        labels = [line for line in label_bytes.decode().splitlines() if line.strip()]
        n_vis_boxes += len(labels)
        n_vis_empty_frames += int(not labels)
        rows.append(
            {
                "frame_id": frame_id,
                "vis_image": str(vis_image.relative_to(dataset_root)),
                "vis_image_size": vis_image.stat().st_size,
                "vis_image_sha256": file_sha256(vis_image),
                "vis_label": str(vis_label.relative_to(dataset_root)),
                "vis_label_sha256": hashlib.sha256(label_bytes).hexdigest(),
                "ir_image": str(ir_image.relative_to(dataset_root)),
                "ir_image_size": ir_image.stat().st_size,
                "ir_image_sha256": file_sha256(ir_image),
            }
        )

    frame_ids_sha256 = hashlib.sha256(
        ",".join(str(frame_id) for frame_id in common_ids).encode()
    ).hexdigest()
    manifest = {
        "acquisition": acquisition,
        "vis_folder": settings["vis_folder"],
        "ir_folder": settings["ir_folder"],
        "pairing": "shared_numeric_frame_suffix",
        "ground_truth": "vis",
        "vis_images": len(vis_images),
        "vis_labels": len(vis_labels),
        "ir_images": len(ir_images),
        "ir_labels": len(ir_labels),
        "common_frames": len(common_ids),
        "vis_boxes": n_vis_boxes,
        "vis_empty_frames": n_vis_empty_frames,
        "excluded_vis_frame_ids": excluded_vis,
        "excluded_ir_frame_ids": excluded_ir,
        "common_frame_ids_sha256": frame_ids_sha256,
        "inventory_sha256": stable_json_hash(rows),
        "image_content_hashes_included": True,
        "rows": rows,
    }
    if verify_expected:
        expected = {
            "vis_images": int(settings["expected_vis_images"]),
            "vis_labels": int(settings["expected_vis_labels"]),
            "ir_images": int(settings["expected_ir_images"]),
            "ir_labels": int(settings["expected_ir_labels"]),
            "common_frames": int(settings["expected_common_frames"]),
            "vis_boxes": int(settings["expected_vis_boxes"]),
            "vis_empty_frames": int(settings["expected_vis_empty_frames"]),
            "excluded_vis_frame_ids": settings["expected_excluded_vis_frame_ids"],
            "excluded_ir_frame_ids": settings["expected_excluded_ir_frame_ids"],
            "common_frame_ids_sha256": settings[
                "expected_common_frame_ids_sha256"
            ],
            "inventory_sha256": settings["expected_inventory_sha256"],
        }
        actual = {key: manifest[key] for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"{acquisition} differs from the frozen inventory:\n"
                + json.dumps({"expected": expected, "actual": actual}, indent=2)
            )
    return manifest


def compact_manifest(manifest):
    return {key: value for key, value in manifest.items() if key != "rows"}


def build_manifests(protocol, verify_expected=True):
    dataset_root = resolve_repo_path(protocol["dataset_root"])
    return {
        acquisition: build_acquisition_manifest(
            acquisition, settings, dataset_root, verify_expected=verify_expected
        )
        for acquisition, settings in protocol["acquisitions"].items()
    }


def _paired_item_paths(item, dataset_root):
    item_type, data = item
    if item_type != MULTI_MODALITY_ITEM:
        raise RuntimeError("Confirmation loader contains a non-paired item")
    (vis_image, vis_label), (ir_image, ir_label) = data
    if ir_label:
        raise RuntimeError("Confirmation IR stream unexpectedly has annotations")
    return {
        "frame_id": numeric_frame_id(vis_image),
        "vis_image": str(Path(vis_image).resolve().relative_to(dataset_root)),
        "vis_label": str(Path(vis_label).resolve().relative_to(dataset_root)),
        "ir_image": str(Path(ir_image).resolve().relative_to(dataset_root)),
    }


def build_loader(protocol, manifest, batch_size=None, workers=None):
    run = load_run_config(
        resolve_repo_path(protocol["preprocessor_training_config"]), run_index=0
    )
    dataset_params = copy.deepcopy(run["dataset"])
    dataset_params["root"] = str(resolve_repo_path(protocol["dataset_root"]))
    transform, _denormalize = build_preprocessor(dataset_params)
    dataset_root = Path(dataset_params["root"])
    dataset = WiSARDDataset(
        root=dataset_params["root"],
        folders=[(manifest["vis_folder"], manifest["ir_folder"])],
        transform=transform,
        single_class=True,
        modal_dropout=False,
        use_tiling=False,
        test_all_tiles=False,
    )
    # The generic WiSARD loader pairs equal-length streams by sorted position.
    # Carnation contains four asymmetric frame-ID gaps, so positional pairing
    # would shift 661 rows. Rebuild the items from the frozen common-ID rows.
    dataset.items = [
        (
            MULTI_MODALITY_ITEM,
            (
                (
                    str(dataset_root / row["vis_image"]),
                    str(dataset_root / row["vis_label"]),
                ),
                (str(dataset_root / row["ir_image"]), ""),
            ),
        )
        for row in manifest["rows"]
    ]
    actual = [_paired_item_paths(item, dataset_root) for item in dataset.items]
    expected = [
        {
            "frame_id": row["frame_id"],
            "vis_image": row["vis_image"],
            "vis_label": row["vis_label"],
            "ir_image": row["ir_image"],
        }
        for row in manifest["rows"]
    ]
    if actual != expected:
        raise RuntimeError("WiSARDDataset order differs from frozen acquisition")
    # Exercise the previously unused ONLY_VIS_LABELS code path without loading
    # a model or producing a prediction.
    first_item = dataset[0]
    if tuple(first_item["pixel_values"].shape) != (4, 640, 640):
        raise RuntimeError("Unexpected confirmation preprocessor output shape")
    generator = torch.Generator().manual_seed(int(protocol["evaluation_seed"]))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size or protocol["batch_size"]),
        num_workers=int(protocol["workers"] if workers is None else workers),
        shuffle=False,
        collate_fn=get_collate_fn(dataset),
        generator=generator,
    )


def verify_closed_stage_b(protocol):
    selection = protocol["selection_source"]
    stage_b_protocol = load_stage_b_protocol(resolve_repo_path(selection["protocol"]))
    with resolve_repo_path(selection["result_csv"]).open(
        newline="", encoding="utf-8"
    ) as input_file:
        rows = list(csv.DictReader(input_file))
    if len(rows) != len(EXPECTED_SEEDS):
        raise RuntimeError("Closed Stage-B result must contain five rows")
    if [int(row["seed"]) for row in rows] != EXPECTED_SEEDS:
        raise RuntimeError("Closed Stage-B seed order changed")
    deltas = [float(row["rcra_minus_fam_map50"]) for row in rows]
    decision = stage_b_decision(deltas, stage_b_protocol["primary_comparison"])
    if decision["status"] != "fail_retain_fam":
        raise RuntimeError("Stage-B decision no longer retains FAM")
    return stage_b_protocol, validate_training_configs(stage_b_protocol), decision


def build_run(protocol, configuration, seed, stage_b_runs):
    settings = protocol["configurations"][configuration]
    if settings["family"] == "current_stage_b":
        key = "fam" if configuration == "stage_b_fam" else "rcra"
        run = copy.deepcopy(stage_b_runs[(key, seed)])
    else:
        run = load_run_config(
            resolve_repo_path(settings["training_config"]), run_index=seed - 40
        )
        if int(run["seed"]) != seed:
            raise RuntimeError(f"Historical grid does not map to seed {seed}")
        model = run["model"]["params"]
        model.update(
            {
                "use_fam": bool(settings["expected_use_fam"]),
                "fam_variant": "current_dcnv2",
                "freeze_fam": False,
                "ir_dropout_rate": 0.0,
                "spatial_jitter_std": 0.0,
                "use_p2": False,
                "use_reliability_gating": False,
                "use_residual_alignment_gating": False,
                "use_scalar_residual_alignment": False,
            }
        )
    model = run["model"]["params"]
    if model.get("use_fam") is not bool(settings["expected_use_fam"]):
        raise RuntimeError(f"Unexpected FAM state for {configuration}/seed {seed}")
    if model.get("use_residual_alignment_gating", False) is not bool(
        settings["expected_use_residual_alignment_gating"]
    ):
        raise RuntimeError(f"Unexpected RCRA state for {configuration}/seed {seed}")
    run["model"]["params"]["threshold"] = float(protocol["confidence_threshold"])
    return run


def resolve_checkpoints(protocol, stage_b_runs, verify_hashes=True):
    checkpoints = {}
    for configuration, settings in protocol["configurations"].items():
        for seed in EXPECTED_SEEDS:
            checkpoint = Path(
                resolve_local_wandb_checkpoint(
                    settings["project"],
                    seed,
                    checkpoint="latest",
                    wandb_root=REPO_ROOT / "wandb",
                )
            ).resolve()
            digest = file_sha256(checkpoint)
            if verify_hashes:
                expected = settings["expected_checkpoint_sha256"][seed]
                if digest != expected:
                    raise RuntimeError(
                        f"Checkpoint hash changed for {configuration}/seed {seed}"
                    )
            training_summary = None
            if settings["family"] == "current_stage_b":
                key = "fam" if configuration == "stage_b_fam" else "rcra"
                training_summary = audit_completed_run(
                    checkpoint,
                    stage_b_runs[(key, seed)],
                    settings["project"],
                    seed,
                    key,
                )
            checkpoints[(configuration, seed)] = {
                "path": checkpoint,
                "sha256": digest,
                "training_summary": training_summary,
            }
    return checkpoints


def ensure_attestation_ready(attestation):
    if attestation["status"] == "pending":
        raise RuntimeError(
            "Author attestation is pending. Inventory and dry-run are allowed, "
            "but scientific inference is blocked."
        )
    return (
        "previously_unused_internal_acquisition_confirmation"
        if attestation["status"] == "no_prior_model_or_manual_experimental_use"
        else "additional_internal_acquisitions"
    )


def raw_result_path(output_dir, acquisition, configuration, seed, condition):
    return (
        output_dir
        / "raw"
        / f"{acquisition}_{configuration}_seed_{seed}_{condition}.json"
    )


def _comparison_values(rows, acquisition, comparison):
    reference = []
    candidate = []
    deltas = {}
    metric = comparison["metric"]
    for seed in EXPECTED_SEEDS:
        reference_value = next(
            float(row["metrics"][metric])
            for row in rows
            if row["acquisition"] == acquisition
            and row["configuration"] == comparison["reference_configuration"]
            and row["condition"] == comparison["reference_condition"]
            and row["seed"] == seed
        )
        candidate_value = next(
            float(row["metrics"][metric])
            for row in rows
            if row["acquisition"] == acquisition
            and row["configuration"] == comparison["candidate_configuration"]
            and row["condition"] == comparison["candidate_condition"]
            and row["seed"] == seed
        )
        reference.append(reference_value)
        candidate.append(candidate_value)
        deltas[str(seed)] = candidate_value - reference_value
    return reference, candidate, deltas


def build_aggregate(
    payloads,
    protocol,
    attestation,
    protocol_hash,
    manifests,
    selection_decision,
    output_dir,
):
    rows = sorted(
        payloads,
        key=lambda row: (
            row["acquisition"],
            row["configuration"],
            row["seed"],
            row["condition"],
        ),
    )
    expected_jobs = {
        (acquisition, configuration, seed, condition)
        for acquisition in EXPECTED_ACQUISITIONS
        for configuration, settings in protocol["configurations"].items()
        for seed in EXPECTED_SEEDS
        for condition in settings["conditions"]
    }
    actual_jobs = {
        (row["acquisition"], row["configuration"], row["seed"], row["condition"])
        for row in rows
    }

    summaries = {}
    for acquisition in EXPECTED_ACQUISITIONS:
        summaries[acquisition] = {}
        for configuration, settings in protocol["configurations"].items():
            summaries[acquisition][configuration] = {}
            for condition in settings["conditions"]:
                selected = [
                    row
                    for row in rows
                    if row["acquisition"] == acquisition
                    and row["configuration"] == configuration
                    and row["condition"] == condition
                ]
                summaries[acquisition][configuration][condition] = {
                    metric: summarize_values(
                        [row["metrics"].get(metric) for row in selected]
                    )
                    for metric in SCALAR_METRICS
                }

    comparisons = {}
    macro_by_comparison = {}
    for name, comparison in protocol["comparisons"].items():
        comparisons[name] = {}
        acquisition_deltas = {}
        for acquisition in EXPECTED_ACQUISITIONS:
            reference, candidate, deltas = _comparison_values(
                rows, acquisition, comparison
            )
            acquisition_deltas[acquisition] = deltas
            comparisons[name][acquisition] = {
                "seed_values": deltas,
                "summary": summarize_signed_values(deltas.values()),
                "positive_seed_count": sum(value > 0.0 for value in deltas.values()),
                "ties": sum(value == 0.0 for value in deltas.values()),
                "tests_exploratory": paired_tests(reference, candidate),
            }
        macro = {
            str(seed): statistics.fmean(
                acquisition_deltas[acquisition][str(seed)]
                for acquisition in EXPECTED_ACQUISITIONS
            )
            for seed in EXPECTED_SEEDS
        }
        macro_by_comparison[name] = {
            "seed_values_equal_weight_acquisition_macro": macro,
            "summary": summarize_signed_values(macro.values()),
            "positive_seed_count": sum(value > 0.0 for value in macro.values()),
            "role": "descriptive_only",
        }

    aggregate = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "protocol_complete": actual_jobs == expected_jobs,
        "purpose": protocol["purpose"],
        "attestation": attestation,
        "dataset_description": ensure_attestation_ready(attestation),
        "experimental_unit": "checkpoint/seed within acquisition",
        "frame_level_independence_claim_allowed": False,
        "source_manifests": {
            name: compact_manifest(manifest) for name, manifest in manifests.items()
        },
        "closed_stage_b_decision": selection_decision,
        "interpretation_constraints": protocol["interpretation"],
        "results": rows,
        "across_seed_summaries": summaries,
        "paired_map50_comparisons": comparisons,
        "equal_weight_acquisition_macro": macro_by_comparison,
    }
    json_path = output_dir / "rtdetr_unused_acquisition_confirmation.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(aggregate), output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    csv_path = output_dir / "rtdetr_unused_acquisition_confirmation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "acquisition",
                "configuration",
                "seed",
                "condition",
                "checkpoint_sha256",
                "n_samples",
                *SCALAR_METRICS,
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "acquisition": row["acquisition"],
                    "configuration": row["configuration"],
                    "seed": row["seed"],
                    "condition": row["condition"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--checkpoint-inventory-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol_path = resolve_repo_path(args.protocol)
    if args.inventory_only or args.checkpoint_inventory_only:
        payload = load_payload(protocol_path, require_frozen_values=False)
    else:
        payload = load_payload(protocol_path, require_frozen_values=True)
    protocol = payload["protocol"]
    attestation = payload["attestation"]

    if args.inventory_only:
        manifests = build_manifests(protocol, verify_expected=False)
        print(
            json.dumps(
                {name: compact_manifest(value) for name, value in manifests.items()},
                indent=2,
                sort_keys=True,
            )
        )
        return

    stage_b_protocol, stage_b_runs, selection_decision = verify_closed_stage_b(
        protocol
    )
    del stage_b_protocol

    if args.checkpoint_inventory_only:
        checkpoints = resolve_checkpoints(
            protocol, stage_b_runs, verify_hashes=False
        )
        print(
            json.dumps(
                {
                    configuration: {
                        str(seed): {
                            "path": str(checkpoints[(configuration, seed)]["path"]),
                            "sha256": checkpoints[(configuration, seed)]["sha256"],
                        }
                        for seed in EXPECTED_SEEDS
                    }
                    for configuration in EXPECTED_CONFIGURATIONS
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    manifests = build_manifests(protocol, verify_expected=True)
    loaders = {
        acquisition: build_loader(
            protocol,
            manifest,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        for acquisition, manifest in manifests.items()
    }
    print(
        json.dumps(
            {
                acquisition: compact_manifest(manifest)
                for acquisition, manifest in manifests.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.prepare_only:
        print("Prepare-only OK: frozen inventories and paired loaders match")
        return

    checkpoints = resolve_checkpoints(protocol, stage_b_runs, verify_hashes=True)
    if args.dry_run:
        print(
            f"Dry run OK: {len(checkpoints)} checkpoint hashes and 50 frozen "
            "acquisition/configuration/seed/condition jobs"
        )
        return

    dataset_description = ensure_attestation_ready(attestation)
    print(f"Dataset description: {dataset_description}")
    output_dir = resolve_repo_path(args.output_dir or protocol["output_dir"])
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    protocol_hash = stable_json_hash(protocol)
    batch_size = int(args.batch_size or protocol["batch_size"])
    device = resolve_device(args.device)
    set_evaluation_seed(int(protocol["evaluation_seed"]))
    payloads = []

    for configuration, settings in protocol["configurations"].items():
        for seed in EXPECTED_SEEDS:
            checkpoint = checkpoints[(configuration, seed)]
            missing = {}
            expected_jobs = {}
            for acquisition, manifest in manifests.items():
                for condition in settings["conditions"]:
                    expected = {
                        "protocol_id": protocol["id"],
                        "protocol_sha256": protocol_hash,
                        "attestation_status": attestation["status"],
                        "acquisition": acquisition,
                        "source_inventory_sha256": manifest["inventory_sha256"],
                        "configuration": configuration,
                        "project": settings["project"],
                        "seed": seed,
                        "condition": condition,
                        "checkpoint_kind": "latest",
                        "checkpoint": str(checkpoint["path"]),
                        "checkpoint_sha256": checkpoint["sha256"],
                        "batch_size": batch_size,
                        "confidence_threshold": float(protocol["confidence_threshold"]),
                    }
                    expected_jobs[(acquisition, condition)] = expected
                    path = raw_result_path(
                        output_dir, acquisition, configuration, seed, condition
                    )
                    if path.is_file() and not args.force:
                        payloads.append(load_compatible_raw(path, expected))
                        print(f"[skip] {path}")
                    else:
                        missing.setdefault(acquisition, []).append(condition)
            if not missing:
                continue

            run = build_run(protocol, configuration, seed, stage_b_runs)
            print(f"[load] configuration={configuration} seed={seed} device={device}")
            model = load_fusion_model(run["model"], checkpoint["path"], device)
            for acquisition, conditions in missing.items():
                print(
                    f"[run] acquisition={acquisition} configuration={configuration} "
                    f"seed={seed} conditions={conditions}"
                )
                measured = evaluate_modalities(
                    model, loaders[acquisition], device, conditions
                )
                for condition in conditions:
                    expected = expected_jobs[(acquisition, condition)]
                    result = measured[condition]
                    payload_row = {
                        **expected,
                        "schema_version": 1,
                        "protocol_complete": True,
                        "ground_truth": "vis",
                        "n_dataset_images": len(loaders[acquisition].dataset),
                        "n_samples": result["n_samples"],
                        "training_summary": checkpoint["training_summary"],
                        "metrics": result["metrics"],
                    }
                    path = raw_result_path(
                        output_dir, acquisition, configuration, seed, condition
                    )
                    with path.open("w", encoding="utf-8") as output_file:
                        json.dump(
                            jsonable(payload_row),
                            output_file,
                            indent=2,
                            sort_keys=True,
                        )
                        output_file.write("\n")
                    payloads.append(payload_row)
                    print(
                        f"[done] {acquisition}/{configuration}/seed={seed}/"
                        f"{condition} map50={result['metrics']['map_50']:.6f}"
                    )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = build_aggregate(
        payloads,
        protocol,
        attestation,
        protocol_hash,
        manifests,
        selection_decision,
        output_dir,
    )
    if aggregate["protocol_complete"]:
        print("Protocol complete: all 50 frozen confirmation results are present")


if __name__ == "__main__":
    main()
