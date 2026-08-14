#!/usr/bin/env python3
"""Evaluate RT-DETR sensor ablations on the same 708 paired MtErie frames.

Unlike the historical native-sensor evaluation, this runner creates exactly one
VIS+IR dataset and reuses each batch and its VIS ground truth for all conditions.
VIS-only and IR-only are interventions on the four-channel input tensor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from sarfusion.data.utils import build_preprocessor, get_collate_fn  # noqa: E402
from sarfusion.data.wisard import MULTI_MODALITY_ITEM, WiSARDDataset  # noqa: E402
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint  # noqa: E402
from sarfusion.utils.metrics import DetectionEvaluator, MetricCollection  # noqa: E402
from sarfusion.utils.structures import DataDict, WrapperModelOutput  # noqa: E402
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
    tensors_to_cpu,
)


DEFAULT_PROTOCOL = "parameters/RTDETR/rtdetr_paired_modality_evaluation.yaml"
EXPECTED_SEEDS = [40, 41, 42, 43, 44]
EXPECTED_MODALITIES = {
    "vis_ir": "keep_all_four_channels",
    "vis": "zero_ir_channel",
    "ir": "zero_three_vis_channels",
}
EXPECTED_CONFIGURATIONS = {
    "additive",
    "fam",
    "fam_ir_dropout",
    "fam_ssj",
    "identity_dcnv2",
    "grid_sample",
}


def load_protocol(path):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Paired-modality YAML must contain a protocol mapping")
    if protocol.get("id") != "rtdetr_paired_modality_evaluation_v1":
        raise ValueError("Unexpected paired-modality protocol id")
    if protocol.get("status") != "frozen_before_inference":
        raise ValueError("Paired-modality protocol must remain frozen_before_inference")
    if protocol.get("checkpoint") != "latest":
        raise ValueError("Paired-modality evaluation must use latest checkpoints")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("Paired-modality protocol must contain seeds 40--44")
    if protocol.get("modalities") != EXPECTED_MODALITIES:
        raise ValueError("Unexpected paired-modality channel interventions")
    if set(protocol.get("configurations", {})) != EXPECTED_CONFIGURATIONS:
        raise ValueError("Paired-modality protocol must contain all six configurations")
    source = protocol.get("source", {})
    if source.get("ground_truth") != "vis":
        raise ValueError("All paired-modality conditions must use VIS ground truth")
    if source.get("pairing") != "existing_wisard_sorted_zip":
        raise ValueError("The source must reproduce the existing WiSARD pairing")
    interpretation = protocol.get("interpretation", {})
    if not interpretation or any(bool(value) for value in interpretation.values()):
        raise ValueError("Paired evaluation cannot be used for tuning or selection")
    return protocol


def _sorted_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(path for path in directory.iterdir() if path.is_file())


def build_source_manifest(protocol, repo_root=REPO_ROOT):
    """Freeze the exact sorted-zip pairs consumed by WiSARDDataset."""
    dataset_root = Path(protocol["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = Path(repo_root) / dataset_root
    dataset_root = dataset_root.resolve()
    rows = []
    n_vis_boxes = 0
    n_vis_empty_frames = 0
    for vis_folder, ir_folder in protocol["source"]["paired_folders"]:
        vis_images = _sorted_files(dataset_root / vis_folder / "images")
        vis_labels = _sorted_files(dataset_root / vis_folder / "labels")
        ir_images = _sorted_files(dataset_root / ir_folder / "images")
        ir_labels = _sorted_files(dataset_root / ir_folder / "labels")
        lengths = {len(vis_images), len(vis_labels), len(ir_images), len(ir_labels)}
        if len(lengths) != 1:
            raise RuntimeError(
                f"Unbalanced paired folders {vis_folder}/{ir_folder}: "
                f"{len(vis_images)}, {len(vis_labels)}, {len(ir_images)}, {len(ir_labels)}"
            )
        for vis_image, vis_label, ir_image, ir_label in zip(
            vis_images, vis_labels, ir_images, ir_labels
        ):
            vis_label_bytes = vis_label.read_bytes()
            lines = [line for line in vis_label_bytes.decode().splitlines() if line.strip()]
            n_vis_boxes += len(lines)
            n_vis_empty_frames += int(not lines)
            rows.append(
                {
                    "vis_image": str(vis_image.relative_to(dataset_root)),
                    "vis_image_size": vis_image.stat().st_size,
                    "vis_label": str(vis_label.relative_to(dataset_root)),
                    "vis_label_sha256": hashlib.sha256(vis_label_bytes).hexdigest(),
                    "ir_image": str(ir_image.relative_to(dataset_root)),
                    "ir_image_size": ir_image.stat().st_size,
                    "ir_label": str(ir_label.relative_to(dataset_root)),
                    "ir_label_sha256": hashlib.sha256(ir_label.read_bytes()).hexdigest(),
                }
            )
    inventory_sha256 = stable_json_hash(rows)
    manifest = {
        "dataset_root": str(dataset_root),
        "pairing": "existing_wisard_sorted_zip",
        "ground_truth": "vis",
        "paired_folders": protocol["source"]["paired_folders"],
        "n_frames": len(rows),
        "n_vis_boxes": n_vis_boxes,
        "n_vis_empty_frames": n_vis_empty_frames,
        "inventory_sha256": inventory_sha256,
        "rows": rows,
    }
    expected = {
        "n_frames": int(protocol["source"]["expected_frames"]),
        "n_vis_boxes": int(protocol["source"]["expected_vis_boxes"]),
        "n_vis_empty_frames": int(protocol["source"]["expected_vis_empty_frames"]),
        "inventory_sha256": protocol["source"]["expected_inventory_sha256"],
    }
    actual = {key: manifest[key] for key in expected}
    if actual != expected:
        raise RuntimeError(
            "Paired MtErie source differs from frozen inventory:\n"
            + json.dumps({"expected": expected, "actual": actual}, indent=2)
        )
    return manifest


def _item_paths(item, dataset_root):
    item_type, data = item
    if item_type != MULTI_MODALITY_ITEM:
        raise RuntimeError("Paired evaluation dataset contains a non-paired item")
    (vis_image, vis_label), (ir_image, ir_label) = data
    return {
        "vis_image": str(Path(vis_image).resolve().relative_to(dataset_root)),
        "vis_label": str(Path(vis_label).resolve().relative_to(dataset_root)),
        "ir_image": str(Path(ir_image).resolve().relative_to(dataset_root)),
        "ir_label": str(Path(ir_label).resolve().relative_to(dataset_root)),
    }


def build_paired_loader(protocol, manifest, batch_size=None, workers=None):
    dataset_params = load_run_config(
        REPO_ROOT / protocol["training_config"], run_index=0
    )["dataset"]
    dataset_params["root"] = manifest["dataset_root"]
    transform, _denormalize = build_preprocessor(dataset_params)
    dataset = WiSARDDataset(
        root=manifest["dataset_root"],
        folders=[tuple(pair) for pair in protocol["source"]["paired_folders"]],
        transform=transform,
        single_class=True,
        modal_dropout=False,
        use_tiling=False,
        test_all_tiles=False,
    )
    dataset_paths = [_item_paths(item, Path(manifest["dataset_root"])) for item in dataset.items]
    manifest_paths = [
        {key: row[key] for key in ("vis_image", "vis_label", "ir_image", "ir_label")}
        for row in manifest["rows"]
    ]
    if dataset_paths != manifest_paths:
        raise RuntimeError("WiSARDDataset order differs from the frozen paired inventory")
    generator = torch.Generator().manual_seed(int(protocol["evaluation_seed"]))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size or protocol["batch_size"]),
        num_workers=int(protocol["workers"] if workers is None else workers),
        shuffle=False,
        collate_fn=get_collate_fn(dataset),
        generator=generator,
    )


def mask_modalities(pixel_values, modality):
    if pixel_values.ndim != 4 or pixel_values.shape[1] != 4:
        raise ValueError(f"Expected Bx4xHxW input, received {tuple(pixel_values.shape)}")
    if modality not in EXPECTED_MODALITIES:
        raise ValueError(f"Unknown modality {modality}")
    if modality == "vis_ir":
        return pixel_values
    masked = pixel_values.clone()
    if modality == "vis":
        masked[:, 3:] = 0
    else:
        masked[:, :3] = 0
    return masked


def evaluate_modalities(model, loader, device, modalities, max_batches=None):
    """Evaluate requested interventions in one traversal with identical labels."""
    evaluators = {
        modality: DetectionEvaluator(MetricCollection({}), id2class=loader.dataset.id2class)
        for modality in modalities
    }
    n_samples = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            base_pixels = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device, non_blocking=True)
            labels = tensors_to_cpu(batch["labels"])
            for modality in modalities:
                output = model(
                    pixel_values=mask_modalities(base_pixels, modality),
                    pixel_mask=pixel_mask,
                )
                evaluators[modality].update(
                    DataDict(labels=labels),
                    WrapperModelOutput(predictions=tensors_to_cpu(output["predictions"])),
                )
            n_samples += len(labels)
    return {
        modality: {"metrics": jsonable(evaluator.compute()), "n_samples": n_samples}
        for modality, evaluator in evaluators.items()
    }


def raw_result_path(output_dir, configuration, seed, modality):
    return output_dir / "raw" / f"{configuration}_seed_{seed}_{modality}.json"


def check_fusion_reference(protocol, configuration, seed, measured):
    references = protocol["expected_fusion_map50"]
    expected = float(references[configuration][seed])
    tolerance = float(references["tolerance"])
    difference = float(measured) - expected
    if abs(difference) > tolerance:
        raise RuntimeError(
            f"VIS+IR sanity check failed for {configuration}/seed {seed}: "
            f"measured={measured:.6f}, historical={expected:.6f}, "
            f"difference={difference:+.6f}, tolerance={tolerance:.6f}"
        )
    return {"historical": expected, "difference": difference, "tolerance": tolerance}


def summarize_signed_values(values):
    """Summarize deltas, where negative values are meaningful observations."""
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    result = {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_std": sample_std,
        "min": min(values),
        "max": max(values),
    }
    if len(values) > 1:
        from scipy.stats import t

        half_width = float(t.ppf(0.975, len(values) - 1)) * sample_std / len(values) ** 0.5
        result["ci95_t"] = [result["mean"] - half_width, result["mean"] + half_width]
    else:
        result["ci95_t"] = None
    return result


def build_aggregates(payloads, protocol, protocol_hash, manifest, output_dir, complete):
    rows = sorted(payloads, key=lambda row: (row["configuration"], row["seed"], row["modality"]))
    summaries = {}
    for configuration in protocol["configurations"]:
        summaries[configuration] = {}
        for modality in protocol["modalities"]:
            selected = [
                row for row in rows
                if row["configuration"] == configuration and row["modality"] == modality
            ]
            summaries[configuration][modality] = {
                metric: summarize_values([row["metrics"].get(metric) for row in selected])
                for metric in SCALAR_METRICS
            }

    fusion_deltas = {}
    for configuration in protocol["configurations"]:
        seed_values = {}
        for seed in protocol["seeds"]:
            values = {
                row["modality"]: float(row["metrics"]["map_50"])
                for row in rows
                if row["configuration"] == configuration and row["seed"] == seed
            }
            if set(values) == set(protocol["modalities"]):
                seed_values[str(seed)] = values["vis_ir"] - max(values["vis"], values["ir"])
        fusion_deltas[configuration] = {
            "seed_values": seed_values,
            "summary": summarize_signed_values(seed_values.values()),
            "fusion_wins": sum(value > 0 for value in seed_values.values()),
            "ties": sum(value == 0 for value in seed_values.values()),
        }

    configuration_deltas = {}
    comparisons = {
        "fam_minus_additive": ("additive", "fam"),
        "fam_ir_dropout_minus_fam": ("fam", "fam_ir_dropout"),
        "fam_ssj_minus_fam": ("fam", "fam_ssj"),
        "identity_dcnv2_minus_fam": ("fam", "identity_dcnv2"),
        "grid_sample_minus_fam": ("fam", "grid_sample"),
    }
    for comparison, (baseline, candidate) in comparisons.items():
        configuration_deltas[comparison] = {}
        for modality in protocol["modalities"]:
            baseline_values = []
            candidate_values = []
            for seed in protocol["seeds"]:
                values = {
                    row["configuration"]: float(row["metrics"]["map_50"])
                    for row in rows
                    if row["modality"] == modality and row["seed"] == seed
                }
                if baseline in values and candidate in values:
                    baseline_values.append(values[baseline])
                    candidate_values.append(values[candidate])
            deltas = [b - a for a, b in zip(baseline_values, candidate_values)]
            entry = {
                "seed_values": dict(zip(map(str, protocol["seeds"][: len(deltas)]), deltas)),
                "summary": summarize_signed_values(deltas),
                "candidate_wins": sum(value > 0 for value in deltas),
            }
            if len(deltas) == len(protocol["seeds"]):
                entry["tests_exploratory"] = paired_tests(baseline_values, candidate_values)
            configuration_deltas[comparison][modality] = entry

    expected_keys = {
        (configuration, seed, modality)
        for configuration in protocol["configurations"]
        for seed in protocol["seeds"]
        for modality in protocol["modalities"]
    }
    actual_keys = {(row["configuration"], row["seed"], row["modality"]) for row in rows}
    combined = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "protocol_complete": complete and actual_keys == expected_keys,
        "purpose": protocol["purpose"],
        "experimental_unit": "checkpoint/seed",
        "common_sample_rule": "same paired batch, same VIS ground truth, channel masking only",
        "source_manifest": manifest,
        "source_manifest_sha256": stable_json_hash(manifest),
        "interpretation_constraints": protocol["interpretation"],
        "results": rows,
        "across_seed_summaries": summaries,
        "fusion_minus_best_single_map50": fusion_deltas,
        "paired_configuration_map50_deltas": configuration_deltas,
    }
    json_path = output_dir / "rtdetr_paired_modality_evaluation.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(combined), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    csv_path = output_dir / "rtdetr_paired_modality_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["configuration", "seed", "modality", "n_samples", *SCALAR_METRICS],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "configuration": row["configuration"],
                    "seed": row["seed"],
                    "modality": row["modality"],
                    "n_samples": row["n_samples"],
                    **{metric: row["metrics"].get(metric) for metric in SCALAR_METRICS},
                }
            )
    print(f"Saved aggregate: {json_path}")
    print(f"Saved checkpoint table: {csv_path}")
    return combined


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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--configurations", nargs="+", choices=sorted(EXPECTED_CONFIGURATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=EXPECTED_SEEDS)
    parser.add_argument("--modalities", nargs="+", choices=tuple(EXPECTED_MODALITIES))
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

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
                "dataset_size": len(loader.dataset),
                "ground_truth_for_every_condition": manifest["ground_truth"],
            },
            indent=2,
        )
    )
    if args.prepare_only:
        print("Prepare-only OK: one paired loader matches the frozen 708-frame inventory")
        return

    configurations = args.configurations or list(protocol["configurations"])
    seeds = args.seeds or list(protocol["seeds"])
    modalities = args.modalities or list(protocol["modalities"])
    checkpoints = {}
    for configuration in configurations:
        settings = protocol["configurations"][configuration]
        for seed in seeds:
            checkpoint = Path(
                resolve_local_wandb_checkpoint(
                    settings["project"], seed, checkpoint="latest", wandb_root=REPO_ROOT / "wandb"
                )
            ).resolve()
            checkpoints[(configuration, seed)] = {
                "path": checkpoint,
                "sha256": file_sha256(checkpoint),
            }
            print(f"configuration={configuration} seed={seed} checkpoint={checkpoint}")
    if args.dry_run:
        print(f"Dry run OK: {len(configurations) * len(seeds) * len(modalities)} evaluations")
        return

    output_dir = Path(args.output_dir or protocol["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    batch_size = int(args.batch_size or protocol["batch_size"])
    manifest_hash = stable_json_hash(manifest)
    payloads = []
    _set_evaluation_seed(int(protocol["evaluation_seed"]))

    for configuration in configurations:
        settings = protocol["configurations"][configuration]
        for seed in seeds:
            checkpoint_info = checkpoints[(configuration, seed)]
            expected_by_modality = {}
            missing_modalities = []
            for modality in modalities:
                expected = {
                    "protocol_id": protocol["id"],
                    "protocol_sha256": protocol_hash,
                    "source_manifest_sha256": manifest_hash,
                    "configuration": configuration,
                    "seed": seed,
                    "modality": modality,
                    "checkpoint": str(checkpoint_info["path"]),
                    "checkpoint_sha256": checkpoint_info["sha256"],
                    "max_batches": args.max_batches,
                    "batch_size": batch_size,
                }
                expected_by_modality[modality] = expected
                path = raw_result_path(output_dir, configuration, seed, modality)
                if path.is_file() and not args.force:
                    payloads.append(load_compatible_raw(path, expected))
                    print(f"[skip] {path}")
                else:
                    missing_modalities.append(modality)
            if not missing_modalities:
                continue

            current_config = load_run_config(
                REPO_ROOT / protocol["training_config"], run_index=seed - 40
            )
            if int(current_config["seed"]) != seed:
                raise RuntimeError(f"Training config run_index does not map to seed {seed}")
            model_params = current_config["model"]
            model_params["params"].update(
                {
                    "threshold": float(protocol["confidence_threshold"]),
                    "use_fam": bool(settings["use_fam"]),
                    "fam_variant": settings["fam_variant"],
                    "freeze_fam": False,
                    "ir_dropout_rate": 0.0,
                    "spatial_jitter_std": 0.0,
                }
            )
            print(f"[load] configuration={configuration} seed={seed} device={device}")
            model = load_fusion_model(model_params, checkpoint_info["path"], device)
            print(
                f"[run] configuration={configuration} seed={seed} "
                f"modalities={','.join(missing_modalities)}"
            )
            measured = evaluate_modalities(
                model, loader, device, missing_modalities, max_batches=args.max_batches
            )
            for modality in missing_modalities:
                result = measured[modality]
                payload = {
                    **expected_by_modality[modality],
                    "schema_version": 1,
                    "protocol_complete": args.max_batches is None,
                    "project": settings["project"],
                    "channel_intervention": protocol["modalities"][modality],
                    "ground_truth": "vis",
                    "n_dataset_images": len(loader.dataset),
                    "n_samples": result["n_samples"],
                    "metrics": result["metrics"],
                }
                if modality == "vis_ir" and args.max_batches is None:
                    payload["fusion_reference_check"] = check_fusion_reference(
                        protocol, configuration, seed, result["metrics"]["map_50"]
                    )
                path = raw_result_path(output_dir, configuration, seed, modality)
                with path.open("w", encoding="utf-8") as output_file:
                    json.dump(payload, output_file, indent=2, sort_keys=True)
                    output_file.write("\n")
                payloads.append(payload)
                print(
                    f"[done] modality={modality} map50={result['metrics']['map_50']:.6f} "
                    f"n={result['n_samples']} -> {path}"
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    complete = (
        args.max_batches is None
        and set(configurations) == set(protocol["configurations"])
        and set(seeds) == set(protocol["seeds"])
        and set(modalities) == set(protocol["modalities"])
    )
    combined = build_aggregates(
        payloads, protocol, protocol_hash, manifest, output_dir, complete=complete
    )
    if combined["protocol_complete"]:
        print("Protocol complete: all 90 paired RT-DETR evaluations are present")


if __name__ == "__main__":
    main()
