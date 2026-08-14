#!/usr/bin/env python3
"""Run the frozen one-shot RT-DETR Carnation stress test.

The runner evaluates the five final Additive and five final FAM checkpoints in
VIS+IR, VIS-only and IR-only mode.  All modes are restricted to the same 739
numeric frame identifiers.  Results are resumable per checkpoint/modality and
are aggregated with the checkpoint/seed as the experimental unit.
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
from sarfusion.data.wisard import (  # noqa: E402
    IR_ITEM,
    MULTI_MODALITY_ITEM,
    RGB_ITEM,
    WiSARDDataset,
)
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint  # noqa: E402
from sarfusion.utils.metrics import DetectionEvaluator, MetricCollection  # noqa: E402
from sarfusion.utils.structures import DataDict, WrapperModelOutput  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402
from scripts.run_rtdetr_fam_level_ablation import (  # noqa: E402
    jsonable,
    resolve_device,
    tensors_to_cpu,
)


DEFAULT_PROTOCOL = "parameters/RTDETR/rtdetr_carnation_stress_test.yaml"
EXPECTED_SEEDS = [40, 41, 42, 43, 44]
EXPECTED_MODALITIES = {"vis_ir": "fusion", "vis": "rgb", "ir": "ir"}
SCALAR_METRICS = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def numeric_frame_id(path):
    suffix = Path(path).stem.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"Cannot extract a numeric frame suffix from {path}")
    return int(suffix)


def load_protocol(path):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Carnation YAML must contain a protocol mapping")
    if protocol.get("id") != "rtdetr_carnation_stress_test_v1":
        raise ValueError("Unexpected Carnation protocol id")
    if protocol.get("status") != "frozen_before_inference":
        raise ValueError("Carnation protocol must remain frozen_before_inference")
    if protocol.get("checkpoint") != "latest":
        raise ValueError("Carnation must use the final RT-DETR latest checkpoint")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("Carnation protocol must contain seeds 40--44")
    if protocol.get("modalities") != EXPECTED_MODALITIES:
        raise ValueError("Carnation modalities must be VIS+IR, VIS and IR")
    if float(protocol.get("confidence_threshold", -1)) != 0.01:
        raise ValueError("Carnation confidence threshold must remain 0.01")
    if protocol.get("fam_variant") != "current_dcnv2":
        raise ValueError("Carnation must evaluate the final current_dcnv2 FAM")
    if set(protocol.get("configurations", {})) != {"additive", "fam"}:
        raise ValueError("Carnation protocol must contain Additive and FAM")
    source = protocol.get("source", {})
    if source.get("pairing") != "shared_numeric_frame_suffix":
        raise ValueError("Carnation must use shared numeric frame identifiers")
    interpretation = protocol.get("interpretation", {})
    if not interpretation or any(bool(value) for value in interpretation.values()):
        raise ValueError("Carnation stress results cannot be used for tuning or selection")
    return protocol


def _index_stream(dataset_root, folder):
    folder_root = Path(dataset_root) / folder
    streams = {}
    for kind in ("images", "labels"):
        directory = folder_root / kind
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        indexed = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            frame_id = numeric_frame_id(path)
            if frame_id in indexed:
                raise RuntimeError(f"Duplicate {kind} frame id {frame_id} in {directory}")
            indexed[frame_id] = path
        streams[kind] = indexed
    if set(streams["images"]) != set(streams["labels"]):
        raise RuntimeError(f"Image/label frame ids differ in {folder_root}")
    return {
        frame_id: {
            "image": streams["images"][frame_id],
            "label": streams["labels"][frame_id],
        }
        for frame_id in streams["images"]
    }


def _stream_inventory(dataset_root, indexed, common_ids):
    digest = hashlib.sha256()
    n_boxes = 0
    n_empty_frames = 0
    for frame_id in common_ids:
        row = indexed[frame_id]
        label_bytes = row["label"].read_bytes()
        labels = [line for line in label_bytes.decode().splitlines() if line.strip()]
        n_boxes += len(labels)
        n_empty_frames += int(not labels)
        image_relative = row["image"].relative_to(dataset_root)
        label_relative = row["label"].relative_to(dataset_root)
        digest.update(
            (
                f"{frame_id}|{image_relative}|{row['image'].stat().st_size}|"
                f"{label_relative}|"
            ).encode()
        )
        digest.update(hashlib.sha256(label_bytes).hexdigest().encode())
        digest.update(b"\n")
    return {
        "n_frames": len(common_ids),
        "n_boxes": n_boxes,
        "n_empty_frames": n_empty_frames,
        "inventory_sha256": digest.hexdigest(),
    }


def build_source_manifest(protocol, repo_root=REPO_ROOT):
    dataset_root = Path(protocol["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = Path(repo_root) / dataset_root
    dataset_root = dataset_root.resolve()
    source = protocol["source"]
    vis_index = _index_stream(dataset_root, source["vis_folder"])
    ir_index = _index_stream(dataset_root, source["ir_folder"])
    common_ids = sorted(set(vis_index) & set(ir_index))
    ids_sha256 = hashlib.sha256(
        ",".join(str(frame_id) for frame_id in common_ids).encode()
    ).hexdigest()
    vis_summary = _stream_inventory(dataset_root, vis_index, common_ids)
    ir_summary = _stream_inventory(dataset_root, ir_index, common_ids)
    manifest = {
        "dataset_root": str(dataset_root),
        "vis_folder": source["vis_folder"],
        "ir_folder": source["ir_folder"],
        "pairing": source["pairing"],
        "common_frame_count": len(common_ids),
        "common_frame_ids": common_ids,
        "common_frame_ids_sha256": ids_sha256,
        "vis_only_source_frame_count": len(vis_index),
        "ir_only_source_frame_count": len(ir_index),
        "excluded_vis_frame_ids": sorted(set(vis_index) - set(common_ids)),
        "excluded_ir_frame_ids": sorted(set(ir_index) - set(common_ids)),
        "vis": vis_summary,
        "ir": ir_summary,
    }
    expected = {
        "common_frame_count": int(source["expected_common_frames"]),
        "common_frame_ids_sha256": source["expected_common_frame_ids_sha256"],
        "vis.n_boxes": int(source["expected_vis_boxes"]),
        "ir.n_boxes": int(source["expected_ir_boxes"]),
        "vis.n_empty_frames": int(source["expected_vis_empty_frames"]),
        "ir.n_empty_frames": int(source["expected_ir_empty_frames"]),
        "vis.inventory_sha256": source["expected_vis_inventory_sha256"],
        "ir.inventory_sha256": source["expected_ir_inventory_sha256"],
    }
    actual = {
        "common_frame_count": manifest["common_frame_count"],
        "common_frame_ids_sha256": manifest["common_frame_ids_sha256"],
        "vis.n_boxes": manifest["vis"]["n_boxes"],
        "ir.n_boxes": manifest["ir"]["n_boxes"],
        "vis.n_empty_frames": manifest["vis"]["n_empty_frames"],
        "ir.n_empty_frames": manifest["ir"]["n_empty_frames"],
        "vis.inventory_sha256": manifest["vis"]["inventory_sha256"],
        "ir.inventory_sha256": manifest["ir"]["inventory_sha256"],
    }
    if actual != expected:
        raise RuntimeError(
            "Carnation source differs from the frozen inventory:\n"
            + json.dumps({"expected": expected, "actual": actual}, indent=2)
        )
    return manifest


def _dataset_item_frame_id(item):
    item_type, data = item
    if item_type in {RGB_ITEM, IR_ITEM}:
        image_path, _label_path = data
        return numeric_frame_id(image_path)
    if item_type == MULTI_MODALITY_ITEM:
        (vis_path, _vis_label), (ir_path, _ir_label) = data
        vis_id = numeric_frame_id(vis_path)
        ir_id = numeric_frame_id(ir_path)
        if vis_id != ir_id:
            raise RuntimeError(f"Mismatched Carnation pair: VIS {vis_id}, IR {ir_id}")
        return vis_id
    raise RuntimeError(f"Unexpected WiSARD item type: {item_type}")


def build_modality_loaders(protocol, source_manifest, batch_size=None, workers=None):
    dataset_params = load_run_config(
        REPO_ROOT / protocol["training_config"], run_index=0
    )["dataset"]
    dataset_params["root"] = source_manifest["dataset_root"]
    transform, _denormalize = build_preprocessor(dataset_params)
    common_ids = set(source_manifest["common_frame_ids"])
    folders = {
        "vis_ir": [
            (source_manifest["vis_folder"], source_manifest["ir_folder"])
        ],
        "vis": [source_manifest["vis_folder"]],
        "ir": [source_manifest["ir_folder"]],
    }
    loaders = {}
    for offset, modality in enumerate(protocol["modalities"]):
        dataset = WiSARDDataset(
            root=source_manifest["dataset_root"],
            folders=folders[modality],
            transform=transform,
            single_class=True,
            modal_dropout=False,
            use_tiling=False,
            test_all_tiles=False,
        )
        dataset.items = [
            item for item in dataset.items if _dataset_item_frame_id(item) in common_ids
        ]
        actual_ids = [_dataset_item_frame_id(item) for item in dataset.items]
        if actual_ids != source_manifest["common_frame_ids"]:
            raise RuntimeError(
                f"{modality} dataset order/content differs from frozen common ids"
            )
        generator = torch.Generator().manual_seed(
            int(protocol["evaluation_seed"]) + offset
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(batch_size or protocol["batch_size"]),
            num_workers=int(protocol["workers"] if workers is None else workers),
            shuffle=False,
            collate_fn=get_collate_fn(dataset),
            generator=generator,
        )
        loaders[modality] = loader
    return loaders


def evaluate_model(model, loader, device, max_batches=None):
    evaluator = DetectionEvaluator(MetricCollection({}), id2class=loader.dataset.id2class)
    n_samples = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device, non_blocking=True)
            output = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            predictions = tensors_to_cpu(output["predictions"])
            labels = tensors_to_cpu(batch["labels"])
            evaluator.update(
                DataDict(labels=labels),
                WrapperModelOutput(predictions=predictions),
            )
            n_samples += len(labels)
    return jsonable(evaluator.compute()), n_samples


def summarize_values(values):
    # TorchMetrics uses -1 for an AP/AR slice with no eligible ground-truth
    # instances.  Every scalar metric in this protocol is otherwise
    # non-negative, so the sentinel must be treated as unavailable rather than
    # averaged as a real score.
    values = [
        float(value)
        for value in values
        if value is not None and float(value) >= 0.0
    ]
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


def paired_tests(additive, fam):
    from scipy.stats import ttest_rel, wilcoxon

    additive = [float(value) for value in additive]
    fam = [float(value) for value in fam]
    deltas = [second - first for first, second in zip(additive, fam)]
    t_result = ttest_rel(fam, additive)
    if all(delta == 0 for delta in deltas):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
        wilcoxon_method = "all paired deltas are zero"
    else:
        method = "exact" if all(delta != 0 for delta in deltas) else "auto"
        w_result = wilcoxon(fam, additive, alternative="two-sided", method=method)
        wilcoxon_statistic = float(w_result.statistic)
        wilcoxon_p = float(w_result.pvalue)
        wilcoxon_method = method
    return {
        "paired_t": {"statistic": float(t_result.statistic), "pvalue": float(t_result.pvalue)},
        "wilcoxon_two_sided": {
            "statistic": wilcoxon_statistic,
            "pvalue": wilcoxon_p,
            "method": wilcoxon_method,
        },
    }


def raw_result_path(output_dir, configuration, seed, modality):
    return output_dir / "raw" / f"{configuration}_seed_{seed}_{modality}.json"


def load_compatible_raw(path, expected):
    with Path(path).open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"Existing result {path} is incompatible; inspect it before --force"
        )
    return payload


def build_aggregates(payloads, protocol, protocol_hash, source_manifest, output_dir, complete):
    rows = sorted(
        payloads,
        key=lambda row: (row["modality"], row["configuration"], row["seed"]),
    )
    across_seeds = {}
    paired_deltas = {}
    paired_map50_tests = {}
    for modality in protocol["modalities"]:
        across_seeds[modality] = {}
        paired_deltas[modality] = {}
        for configuration in protocol["configurations"]:
            selected = [
                row
                for row in rows
                if row["modality"] == modality
                and row["configuration"] == configuration
            ]
            across_seeds[modality][configuration] = {
                metric: summarize_values([row["metrics"].get(metric) for row in selected])
                for metric in SCALAR_METRICS
            }
        for metric in SCALAR_METRICS:
            seed_values = {}
            for seed in protocol["seeds"]:
                values = {
                    row["configuration"]: row["metrics"].get(metric)
                    for row in rows
                    if row["modality"] == modality and row["seed"] == seed
                }
                if (
                    set(values) == {"additive", "fam"}
                    and None not in values.values()
                    and all(float(value) >= 0.0 for value in values.values())
                ):
                    seed_values[str(seed)] = values["fam"] - values["additive"]
            paired_deltas[modality][metric] = {
                "seed_values": seed_values,
                "summary": summarize_values(seed_values.values()),
                "fam_wins": sum(value > 0 for value in seed_values.values()),
                "ties": sum(value == 0 for value in seed_values.values()),
            }
        additive_map50 = []
        fam_map50 = []
        for seed in protocol["seeds"]:
            matches = {
                row["configuration"]: row["metrics"]["map_50"]
                for row in rows
                if row["modality"] == modality and row["seed"] == seed
            }
            if set(matches) == {"additive", "fam"}:
                additive_map50.append(matches["additive"])
                fam_map50.append(matches["fam"])
        if len(additive_map50) == len(protocol["seeds"]):
            paired_map50_tests[modality] = paired_tests(additive_map50, fam_map50)

    expected_keys = {
        (configuration, seed, modality)
        for configuration in protocol["configurations"]
        for seed in protocol["seeds"]
        for modality in protocol["modalities"]
    }
    actual_keys = {
        (row["configuration"], row["seed"], row["modality"]) for row in rows
    }
    combined = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "protocol_complete": complete and actual_keys == expected_keys,
        "purpose": protocol["purpose"],
        "experimental_unit": "checkpoint/seed",
        "checkpoint": "latest after the final fixed 10-epoch RT-DETR protocol",
        "confidence_threshold": protocol["confidence_threshold"],
        "source_manifest": source_manifest,
        "source_manifest_sha256": stable_json_hash(source_manifest),
        "interpretation_constraints": protocol["interpretation"],
        "results": rows,
        "across_seed_summaries": across_seeds,
        "paired_deltas_fam_minus_additive": paired_deltas,
        "paired_map50_tests_exploratory": paired_map50_tests,
    }
    output_path = output_dir / "rtdetr_carnation_stress_test.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(combined), output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    csv_path = output_dir / "rtdetr_carnation_stress_test.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = ["configuration", "seed", "modality", "n_samples", *SCALAR_METRICS]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
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
    print(f"Saved aggregate: {output_path}")
    print(f"Saved checkpoint table: {csv_path}")
    return combined


def render_paired_map50(combined, output_dir):
    import matplotlib.pyplot as plt

    rows = combined["results"]
    modalities = (("vis_ir", "VIS+IR"), ("vis", "VIS only"), ("ir", "IR only"))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (modality, title) in zip(axes, modalities):
        for seed in EXPECTED_SEEDS:
            values = [
                next(
                    row["metrics"]["map_50"]
                    for row in rows
                    if row["modality"] == modality
                    and row["configuration"] == configuration
                    and row["seed"] == seed
                )
                for configuration in ("additive", "fam")
            ]
            axis.plot([0, 1], values, marker="o", alpha=0.75, label=f"seed {seed}")
        axis.set_xticks([0, 1], ["Additive", "FAM"])
        axis.set_title(title)
        axis.set_ylabel("mAP@50")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("RT-DETR · one-shot Carnation stress test · five paired seeds")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "rtdetr_carnation_paired_map50.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Saved paired figure: {output_path}")


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
    parser.add_argument("--configurations", nargs="+", choices=("additive", "fam"))
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
    source_manifest = build_source_manifest(protocol)
    loaders = build_modality_loaders(
        protocol,
        source_manifest,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "common_frames": source_manifest["common_frame_count"],
                "vis_boxes": source_manifest["vis"]["n_boxes"],
                "ir_boxes": source_manifest["ir"]["n_boxes"],
                "excluded_ir_frame_ids": source_manifest["excluded_ir_frame_ids"],
                "modality_dataset_sizes": {
                    modality: len(loader.dataset) for modality, loader in loaders.items()
                },
            },
            indent=2,
        )
    )
    if args.prepare_only:
        print("Prepare-only OK: source inventory and all modality datasets match the frozen protocol")
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
                    settings["project"],
                    seed,
                    checkpoint=protocol["checkpoint"],
                    wandb_root=REPO_ROOT / "wandb",
                )
            ).resolve()
            checkpoints[(configuration, seed)] = {
                "path": checkpoint,
                "sha256": file_sha256(checkpoint),
            }
            print(f"configuration={configuration} seed={seed} checkpoint={checkpoint}")
    if args.dry_run:
        print(
            f"Dry run OK: {len(configurations) * len(seeds) * len(modalities)} "
            "checkpoint/modality evaluations"
        )
        return

    output_dir = Path(args.output_dir or protocol["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    batch_size = int(args.batch_size or protocol["batch_size"])
    source_manifest_hash = stable_json_hash(source_manifest)
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
                    "source_manifest_sha256": source_manifest_hash,
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
                    "use_fam": bool(settings["expected_use_fam"]),
                    "freeze_fam": False,
                    "fam_variant": protocol["fam_variant"],
                    "ir_dropout_rate": 0.0,
                    "spatial_jitter_std": 0.0,
                }
            )
            print(f"[load] configuration={configuration} seed={seed} device={device}")
            model = load_fusion_model(model_params, checkpoint_info["path"], device)
            for modality in missing_modalities:
                print(f"[run] configuration={configuration} seed={seed} modality={modality}")
                metrics, n_samples = evaluate_model(
                    model,
                    loaders[modality],
                    device,
                    max_batches=args.max_batches,
                )
                payload = {
                    **expected_by_modality[modality],
                    "schema_version": 1,
                    "protocol_complete": args.max_batches is None,
                    "project": settings["project"],
                    "mode": protocol["modalities"][modality],
                    "n_dataset_images": len(loaders[modality].dataset),
                    "n_samples": n_samples,
                    "metrics": metrics,
                }
                path = raw_result_path(output_dir, configuration, seed, modality)
                with path.open("w", encoding="utf-8") as output_file:
                    json.dump(payload, output_file, indent=2, sort_keys=True)
                    output_file.write("\n")
                payloads.append(payload)
                print(f"[done] map50={metrics['map_50']:.6f} n={n_samples} -> {path}")
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
        payloads,
        protocol,
        protocol_hash,
        source_manifest,
        output_dir,
        complete=complete,
    )
    if combined["protocol_complete"]:
        render_paired_map50(combined, output_dir)
        print("Protocol complete: all 30 frozen Carnation evaluations are present")


if __name__ == "__main__":
    main()
