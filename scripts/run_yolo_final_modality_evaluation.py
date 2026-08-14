#!/usr/bin/env python3
"""Evaluate all final YOLO Additive/FAM checkpoints in three sensor modes.

The same paired WiSARD test frames are used for VIS+IR, VIS-only and IR-only.
Only the explicit feature mask changes, so the unavailable backbone is not run
and FAM is bypassed for IR-only inference. Results are saved after every
checkpoint/modality pair and can be resumed without repeating compatible work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch
from ultralytics.data.utils import check_det_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_yolo_modalities import (  # noqa: E402
    build_dataloader,
    evaluate,
    load_yolo_model,
    load_yolo_run_config,
)
from sarfusion.utils.utils import load_yaml  # noqa: E402


DEFAULT_PROTOCOL = "parameters/YOLO/yolov10_final_modality_evaluation.yaml"
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
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_protocol(path):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Evaluation YAML must contain a protocol mapping")
    if protocol.get("checkpoint") != "last":
        raise ValueError("The frozen YOLO protocol must evaluate last.pt")
    if protocol.get("split") != "test":
        raise ValueError("The frozen YOLO protocol must evaluate the test split")
    if list(protocol.get("seeds", [])) != [40, 41, 42, 43, 44]:
        raise ValueError("The frozen YOLO protocol must contain seeds 40--44")
    expected_modalities = {"vis_ir": "fusion", "vis": "rgb", "ir": "ir"}
    if protocol.get("modalities") != expected_modalities:
        raise ValueError(
            "Frozen modality mapping must be vis_ir=fusion, vis=rgb, ir=ir"
        )
    if set(protocol.get("configurations", {})) != {"additive", "fam"}:
        raise ValueError("Frozen protocol must contain additive and fam")
    return protocol


def resolve_run_checkpoint(run_root, seed, checkpoint_name="last"):
    """Resolve a YOLO run by the seed recorded in its args.yaml."""
    run_root = Path(run_root)
    matches = []
    for args_path in sorted(run_root.glob("*/args.yaml")):
        args = load_yaml(args_path)
        if int(args.get("seed", -1)) == int(seed):
            matches.append((args_path.parent, args))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one run with seed={seed} below {run_root}, found {len(matches)}"
        )
    run_dir, args = matches[0]
    if args.get("test_checkpoint") != checkpoint_name:
        raise RuntimeError(
            f"{run_dir} declares test_checkpoint={args.get('test_checkpoint')!r}, "
            f"expected {checkpoint_name!r}"
        )
    if not args.get("modal_dropout") or args.get("modal_dropout_strategy") != "feature":
        raise RuntimeError(f"{run_dir} is not a feature-gated Modal Dropout run")
    if [float(value) for value in args.get("modal_dropout_probs", [])] != [0.2, 0.2, 0.6]:
        raise RuntimeError(f"{run_dir} has unexpected Modal Dropout probabilities")

    results_path = run_dir / "results.csv"
    with results_path.open(encoding="utf-8", newline="") as input_file:
        result_rows = list(csv.DictReader(input_file))
    if len(result_rows) != 200:
        raise RuntimeError(f"{results_path} contains {len(result_rows)} epochs, expected 200")

    checkpoint = run_dir / "weights" / f"{checkpoint_name}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint.resolve(), run_dir.resolve()


def data_source_fingerprint(data_yaml, split):
    data = check_det_dataset(str(data_yaml))
    source = data[split]
    sources = source if isinstance(source, list) else [source]
    rows = []
    for item in sources:
        path = Path(item).resolve()
        rows.append(
            {
                "path": str(path),
                "sha256": file_sha256(path) if path.is_file() else None,
            }
        )
    return rows


def summarize_values(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def raw_result_path(output_dir, configuration, seed, modality):
    return output_dir / "raw" / f"{configuration}_seed_{seed}_{modality}.json"


def load_compatible_raw(path, expected):
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"Existing result {path} is incompatible; inspect it before using --force"
        )
    return payload


def build_aggregates(payloads, protocol, protocol_hash, output_dir, complete):
    rows = sorted(
        payloads,
        key=lambda row: (row["modality"], row["configuration"], row["seed"]),
    )
    across_seeds = {}
    paired_deltas = {}
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
                values = {}
                for configuration in protocol["configurations"]:
                    match = next(
                        (
                            row
                            for row in rows
                            if row["modality"] == modality
                            and row["configuration"] == configuration
                            and row["seed"] == seed
                        ),
                        None,
                    )
                    if match is not None:
                        values[configuration] = match["metrics"].get(metric)
                if set(values) == {"additive", "fam"}:
                    seed_values[str(seed)] = values["fam"] - values["additive"]
            paired_deltas[modality][metric] = {
                "seed_values": seed_values,
                "summary": summarize_values(seed_values.values()),
            }

    fusion_sanity = []
    for row in rows:
        if row["modality"] != "vis_ir":
            continue
        reference = protocol["configurations"][row["configuration"]][
            "reference_vis_ir_map50"
        ][row["seed"]]
        fusion_sanity.append(
            {
                "configuration": row["configuration"],
                "seed": row["seed"],
                "standalone_map50": row["metrics"]["map_50"],
                "automatic_test_map50_rounded": reference,
                "absolute_difference": abs(row["metrics"]["map_50"] - reference),
            }
        )

    expected_keys = {
        (configuration, seed, modality)
        for configuration in protocol["configurations"]
        for seed in protocol["seeds"]
        for modality in protocol["modalities"]
    }
    actual_keys = {
        (row["configuration"], row["seed"], row["modality"])
        for row in rows
    }
    maximum_sanity_difference = (
        max(row["absolute_difference"] for row in fusion_sanity)
        if fusion_sanity
        else None
    )
    sanity_passed = (
        maximum_sanity_difference is not None
        and maximum_sanity_difference <= float(protocol["vis_ir_sanity_tolerance"])
    )
    # Completion records whether every frozen inference unit exists. The
    # cross-evaluator sanity check is deliberately reported separately: a
    # failed tolerance must remain visible, but it does not erase completed
    # raw evaluations or invite them to be repeated until they agree.
    protocol_complete = complete and actual_keys == expected_keys

    combined = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "protocol_complete": protocol_complete,
        "experimental_unit": "checkpoint/seed",
        "checkpoint": "last.pt after 200 fixed epochs",
        "split": protocol["split"],
        "feature_masks": protocol["modalities"],
        "results": rows,
        "across_seed_summaries": across_seeds,
        "paired_deltas_fam_minus_additive": paired_deltas,
        "vis_ir_sanity_against_automatic_test": fusion_sanity,
        "vis_ir_sanity_maximum_difference": maximum_sanity_difference,
        "vis_ir_sanity_passed": sanity_passed,
        "vis_ir_sanity_review_required": protocol_complete and not sanity_passed,
    }
    output_path = output_dir / "yolo_final_modality_evaluation.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(combined, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    csv_path = output_dir / "yolo_final_modality_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = ["configuration", "seed", "modality", *SCALAR_METRICS]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "configuration": row["configuration"],
                    "seed": row["seed"],
                    "modality": row["modality"],
                    **{metric: row["metrics"].get(metric) for metric in SCALAR_METRICS},
                }
            )
    print(f"Saved aggregate: {output_path}")
    print(f"Saved checkpoint table: {csv_path}")
    return combined


def render_paired_map50(combined, output_dir):
    """Render the paired five-seed mAP@50 comparison for all sensor modes."""
    import matplotlib.pyplot as plt

    rows = combined["results"]
    modalities = (
        ("vis_ir", "VIS+IR"),
        ("vis", "VIS only"),
        ("ir", "IR only"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (modality, title) in zip(axes, modalities):
        for seed in range(40, 45):
            values = []
            for configuration in ("additive", "fam"):
                row = next(
                    item
                    for item in rows
                    if item["modality"] == modality
                    and item["configuration"] == configuration
                    and item["seed"] == seed
                )
                values.append(row["metrics"]["map_50"])
            axis.plot([0, 1], values, marker="o", alpha=0.75, label=f"seed {seed}")
        axis.set_xticks([0, 1], ["Additive", "FAM"])
        axis.set_title(title)
        axis.set_ylabel("mAP@50")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "YOLOv10 final last.pt · paired modality evaluation across five seeds"
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "yolov10_final_modality_paired_map50.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Saved paired modality figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--configurations", nargs="+", choices=("additive", "fam"))
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(40, 45))
    parser.add_argument("--modalities", nargs="+", choices=("vis_ir", "vis", "ir"))
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    protocol_path = (REPO_ROOT / args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    protocol_hash = stable_json_hash(protocol)
    configurations = args.configurations or list(protocol["configurations"])
    seeds = args.seeds or list(protocol["seeds"])
    modalities = args.modalities or list(protocol["modalities"])
    batch_size = args.batch_size or int(protocol["batch_size"])
    workers = int(protocol["workers"]) if args.workers is None else args.workers
    output_dir = Path(args.output_dir or protocol["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    data_yaml = (REPO_ROOT / protocol["data_yaml"]).resolve()
    data_sources = data_source_fingerprint(data_yaml, protocol["split"])
    checkpoints = {}
    for configuration in configurations:
        settings = protocol["configurations"][configuration]
        run_root = (REPO_ROOT / settings["run_root"]).resolve()
        for seed in seeds:
            checkpoint, run_dir = resolve_run_checkpoint(
                run_root, seed, checkpoint_name=protocol["checkpoint"]
            )
            checkpoints[(configuration, seed)] = {
                "path": checkpoint,
                "sha256": file_sha256(checkpoint),
                "run_dir": run_dir,
            }
            print(
                f"configuration={configuration} seed={seed} "
                f"checkpoint={checkpoint}"
            )
    if args.dry_run:
        print(
            f"Dry run OK: {len(configurations) * len(seeds) * len(modalities)} "
            "checkpoint/modality evaluations"
        )
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    payloads = []
    for configuration in configurations:
        settings = protocol["configurations"][configuration]
        training_config = (REPO_ROOT / settings["training_config"]).resolve()
        for seed in seeds:
            checkpoint_info = checkpoints[(configuration, seed)]
            expected_by_modality = {}
            missing_modalities = []
            for modality in modalities:
                expected = {
                    "protocol_id": protocol["id"],
                    "protocol_sha256": protocol_hash,
                    "configuration": configuration,
                    "seed": seed,
                    "modality": modality,
                    "checkpoint": str(checkpoint_info["path"]),
                    "checkpoint_sha256": checkpoint_info["sha256"],
                    "data_sources": data_sources,
                    "max_batches": args.max_batches,
                    "batch_size": batch_size,
                    "max_det": int(protocol["max_det"]),
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

            run_config = load_yolo_run_config(training_config, run_index=seed - 40)
            if int(run_config["seed"]) != seed:
                raise RuntimeError(f"Training config run_index does not map to seed {seed}")
            actual_use_fam = bool(run_config["model"]["params"]["use_fam"])
            if actual_use_fam != bool(settings["expected_use_fam"]):
                raise RuntimeError(f"Unexpected use_fam for {configuration} seed {seed}")

            print(f"[load] configuration={configuration} seed={seed} device={device}")
            model = load_yolo_model(checkpoint_info["path"], device)
            loader, dataset = build_dataloader(
                str(data_yaml),
                run_config,
                protocol["split"],
                batch_size,
                workers,
            )
            for modality in missing_modalities:
                model_mode = protocol["modalities"][modality]
                print(
                    f"[run] configuration={configuration} seed={seed} "
                    f"modality={modality} mask_mode={model_mode}"
                )
                evaluation_loader = (
                    itertools.islice(loader, args.max_batches)
                    if args.max_batches is not None
                    else loader
                )
                metrics = evaluate(
                    model,
                    evaluation_loader,
                    device,
                    modality=model_mode,
                    max_det=int(protocol["max_det"]),
                )
                payload = {
                    **expected_by_modality[modality],
                    "schema_version": 1,
                    "protocol_complete": args.max_batches is None,
                    "run_dir": str(checkpoint_info["run_dir"]),
                    "training_config": str(training_config),
                    "data_yaml": str(data_yaml),
                    "split": protocol["split"],
                    "feature_mask_mode": model_mode,
                    "n_dataset_images": len(dataset),
                    "metrics": jsonable(metrics),
                }
                path = raw_result_path(output_dir, configuration, seed, modality)
                with path.open("w", encoding="utf-8") as output_file:
                    json.dump(payload, output_file, indent=2, sort_keys=True)
                    output_file.write("\n")
                payloads.append(payload)
                print(
                    f"[done] map50={payload['metrics']['map_50']:.6f} -> {path}"
                )
            del loader, dataset, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    complete = (
        args.max_batches is None
        and set(configurations) == set(protocol["configurations"])
        and set(seeds) == set(protocol["seeds"])
        and set(modalities) == set(protocol["modalities"])
    )
    combined = build_aggregates(
        payloads, protocol, protocol_hash, output_dir, complete=complete
    )
    if combined["protocol_complete"]:
        render_paired_map50(combined, output_dir)
    maximum_difference = combined["vis_ir_sanity_maximum_difference"]
    if maximum_difference is not None:
        print(f"Maximum VIS+IR sanity difference: {maximum_difference:.6f}")
        if complete and not combined["vis_ir_sanity_passed"]:
            print(
                "WARNING: standalone VIS+IR evaluation exceeds the frozen "
                "sanity tolerance. The 30 inferences are complete, but this "
                "cross-evaluator difference must be documented before "
                "interpretation."
            )


if __name__ == "__main__":
    main()
