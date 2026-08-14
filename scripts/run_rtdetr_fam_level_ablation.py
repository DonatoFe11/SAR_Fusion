#!/usr/bin/env python3
"""Inference-only factorial ablation of RT-DETR FAM levels P3/P4/P5.

Each checkpoint is evaluated under all requested subsets of active FAM levels.
When a level is inactive, a forward hook replaces FAM(IR) with the unmodified
IR feature. No parameter or checkpoint is changed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import load_fusion_model, load_run_config  # noqa: E402
from sarfusion.data import get_dataloaders  # noqa: E402
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint  # noqa: E402
from sarfusion.utils.metrics import DetectionEvaluator, MetricCollection  # noqa: E402
from sarfusion.utils.structures import DataDict, WrapperModelOutput  # noqa: E402


PROTOCOL_ID = "rtdetr_fam_level_ablation_v1"
PROJECT = "RTDETR_FAM_Protocol"
CONFIG_PATH = "parameters/RTDETR/rtdetr_protocol.yaml"
LEVEL_LABELS = {0: "P3", 1: "P4", 2: "P5"}
CONDITIONS = {
    "none": (),
    "p3": (0,),
    "p4": (1,),
    "p5": (2,),
    "p3_p4": (0, 1),
    "p3_p5": (0, 2),
    "p4_p5": (1, 2),
    "p3_p4_p5": (0, 1, 2),
}
FULL_FAM_CONDITION = "p3_p4_p5"
SUPPORTED_FAM_CLASSES = {
    "FeatureAlignmentModule",
    "BoundedFeatureAlignmentModule",
    "IdentityInitializedFeatureAlignmentModule",
    "GridSampleFeatureAlignmentModule",
}
PRIMARY_METRICS = ("map", "map_50", "map_75", "mar_100")


def find_fam_modules(model):
    """Return FAM modules in feature-pyramid order."""
    modules = [
        module
        for module in model.modules()
        if type(module).__name__ in SUPPORTED_FAM_CLASSES
    ]
    if len(modules) != 3:
        raise RuntimeError(
            f"Expected exactly three FAM modules (P3/P4/P5), found {len(modules)}"
        )
    return modules


@contextlib.contextmanager
def active_fam_levels(model, active_levels):
    """Temporarily bypass every FAM whose index is not in ``active_levels``."""
    active_levels = frozenset(active_levels)
    invalid = active_levels - set(LEVEL_LABELS)
    if invalid:
        raise ValueError(f"Invalid FAM level indices: {sorted(invalid)}")

    hooks = []
    for level, module in enumerate(find_fam_modules(model)):
        if level in active_levels:
            continue

        def bypass_ir(_module, inputs, _output):
            if len(inputs) < 2:
                raise RuntimeError("FAM bypass requires RGB and IR inputs")
            return inputs[1]

        hooks.append(module.register_forward_hook(bypass_ir))

    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def tensors_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensors_to_cpu(item) for item in value]
    return value


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


def evaluate(model, test_loader, device, active_levels, max_batches=None):
    id2class = test_loader.dataset.id2class
    evaluator = DetectionEvaluator(MetricCollection({}), id2class=id2class)
    n_samples = 0

    with active_fam_levels(model, active_levels), torch.inference_mode():
        progress = tqdm(
            test_loader,
            desc="test",
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        for batch_index, batch in enumerate(progress):
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


def result_path(output_dir, seed, condition):
    return output_dir / "raw" / f"seed_{seed}_{condition}.json"


def load_compatible_result(path, seed, condition, checkpoint):
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    expected = {
        "protocol_id": PROTOCOL_ID,
        "seed": seed,
        "condition": condition,
        "checkpoint": str(Path(checkpoint).resolve()),
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"Existing result {path} does not match the frozen job. "
            "Use --force only after auditing the file."
        )
    return payload


def summarize(values):
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def factorial_effects(condition_values):
    """Return standard 2^3 main effects and interactions for one response."""
    if set(condition_values) != set(CONDITIONS):
        missing = sorted(set(CONDITIONS) - set(condition_values))
        extra = sorted(set(condition_values) - set(CONDITIONS))
        raise ValueError(f"Factorial contrast requires all conditions; missing={missing} extra={extra}")

    effects = {}
    for order in range(1, 4):
        for factors in itertools.combinations(LEVEL_LABELS, order):
            contrast = 0.0
            for condition, active_levels in CONDITIONS.items():
                sign = 1
                for factor in factors:
                    sign *= 1 if factor in active_levels else -1
                contrast += sign * float(condition_values[condition])
            label = ":".join(LEVEL_LABELS[factor] for factor in factors)
            effects[label] = contrast / 4.0
    return effects


def build_aggregates(results, output_dir, protocol_complete):
    aggregates = {}
    paired_deltas = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in results if row["condition"] == condition]
        if not condition_rows:
            continue
        aggregates[condition] = {}
        for metric in PRIMARY_METRICS:
            values = [float(row["metrics"][metric]) for row in condition_rows]
            aggregates[condition][metric] = summarize(values)

    by_seed_condition = {
        (row["seed"], row["condition"]): row for row in results
    }
    for condition in CONDITIONS:
        deltas = {}
        for metric in PRIMARY_METRICS:
            metric_deltas = []
            seed_values = {}
            for seed in sorted({row["seed"] for row in results}):
                current = by_seed_condition.get((seed, condition))
                reference = by_seed_condition.get((seed, FULL_FAM_CONDITION))
                if current is None or reference is None:
                    continue
                delta = float(current["metrics"][metric]) - float(
                    reference["metrics"][metric]
                )
                seed_values[str(seed)] = delta
                metric_deltas.append(delta)
            if metric_deltas:
                deltas[metric] = {
                    "seed_values": seed_values,
                    "summary": summarize(metric_deltas),
                }
        if deltas:
            paired_deltas[condition] = deltas

    factorial_by_seed = []
    for seed in sorted({row["seed"] for row in results}):
        seed_rows = {
            row["condition"]: row for row in results if row["seed"] == seed
        }
        if set(seed_rows) != set(CONDITIONS):
            continue
        factorial_by_seed.append(
            {
                "seed": seed,
                "effects": {
                    metric: factorial_effects(
                        {
                            condition: row["metrics"][metric]
                            for condition, row in seed_rows.items()
                        }
                    )
                    for metric in PRIMARY_METRICS
                },
            }
        )

    factorial_across_seed = {}
    for metric in PRIMARY_METRICS:
        factorial_across_seed[metric] = {}
        for effect in (
            "P3", "P4", "P5", "P3:P4", "P3:P5", "P4:P5", "P3:P4:P5"
        ):
            values = [row["effects"][metric][effect] for row in factorial_by_seed]
            if values:
                factorial_across_seed[metric][effect] = summarize(values)

    combined = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_complete": protocol_complete,
        "level_index": LEVEL_LABELS,
        "conditions": CONDITIONS,
        "reference_condition": FULL_FAM_CONDITION,
        "intervention": "inactive level returns the unmodified IR feature instead of FAM(IR)",
        "results": sorted(results, key=lambda row: (row["seed"], row["condition"])),
        "aggregates": aggregates,
        "paired_deltas_vs_full_fam": paired_deltas,
        "factorial_effects_by_seed": factorial_by_seed,
        "factorial_effects_across_seed": factorial_across_seed,
    }
    combined_path = output_dir / "rtdetr_fam_level_ablation.json"
    with combined_path.open("w", encoding="utf-8") as output_file:
        json.dump(combined, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    csv_path = output_dir / "rtdetr_fam_level_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = [
            "seed",
            "condition",
            "active_levels",
            "n_samples",
            *PRIMARY_METRICS,
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined["results"]:
            writer.writerow(
                {
                    "seed": row["seed"],
                    "condition": row["condition"],
                    "active_levels": "+".join(row["active_level_labels"]) or "none",
                    "n_samples": row["n_samples"],
                    **{metric: row["metrics"][metric] for metric in PRIMARY_METRICS},
                }
            )
    print(f"Saved combined JSON: {combined_path}")
    print(f"Saved raw-metrics CSV: {csv_path}")


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        default=list(CONDITIONS),
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--dataset-root", default="dataset/WiSARD")
    parser.add_argument("--output-dir", default="out/rtdetr_fam_level_ablation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="debug/smoke test only; outputs are marked protocol_complete=false",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    invalid_seeds = sorted(set(args.seeds) - set(range(40, 45)))
    if invalid_seeds:
        parser.error(f"the frozen protocol only contains seeds 40-44, got {invalid_seeds}")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    device = resolve_device(args.device) if not args.dry_run else args.device
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if not args.dry_run:
        (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    checkpoints = {
        seed: resolve_local_wandb_checkpoint(
            PROJECT,
            seed,
            checkpoint="latest",
            wandb_root=REPO_ROOT / "wandb",
        )
        for seed in args.seeds
    }
    if args.dry_run:
        for seed, checkpoint in checkpoints.items():
            for condition in args.conditions:
                print(
                    f"seed={seed} condition={condition} "
                    f"active={CONDITIONS[condition]} checkpoint={checkpoint}"
                )
        return

    first_config = load_run_config(REPO_ROOT / args.config, run_index=args.seeds[0] - 40)
    dataset_params = first_config["dataset"]
    dataset_params["root"] = str((REPO_ROOT / args.dataset_root).resolve())
    dataloader_params = first_config["dataloader"]
    if args.batch_size is not None:
        dataloader_params["batch_size"] = args.batch_size
    if args.num_workers is not None:
        dataloader_params["num_workers"] = args.num_workers
    (_train_loader, _val_loader, test_loader), _denormalize = get_dataloaders(
        dataset_params,
        dataloader_params,
        seed=42,
    )

    results = []
    for seed in args.seeds:
        checkpoint = checkpoints[seed]
        pending = []
        for condition in args.conditions:
            path = result_path(output_dir, seed, condition)
            if path.is_file() and not args.force:
                payload = load_compatible_result(path, seed, condition, checkpoint)
                results.append(payload)
                print(f"[skip] {path}")
            else:
                pending.append(condition)
        if not pending:
            continue

        run_config = load_run_config(REPO_ROOT / args.config, run_index=seed - 40)
        model_params = run_config["model"]
        model_params["params"].update(
            {
                "use_fam": True,
                "freeze_fam": False,
                "fam_variant": "current_dcnv2",
                "ir_dropout_rate": 0.0,
                "spatial_jitter_std": 0.0,
            }
        )
        print(f"Loading seed {seed}: {checkpoint}")
        model = load_fusion_model(model_params, checkpoint, device)

        for condition in pending:
            active_levels = CONDITIONS[condition]
            print(
                f"[run] seed={seed} condition={condition} "
                f"active={[LEVEL_LABELS[level] for level in active_levels]}"
            )
            metrics, n_samples = evaluate(
                model,
                test_loader,
                device,
                active_levels,
                max_batches=args.max_batches,
            )
            payload = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "protocol_complete": args.max_batches is None,
                "seed": seed,
                "condition": condition,
                "active_levels": list(active_levels),
                "active_level_labels": [LEVEL_LABELS[level] for level in active_levels],
                "checkpoint": str(Path(checkpoint).resolve()),
                "split": "test",
                "input": "vis_ir",
                "n_samples": n_samples,
                "max_batches": args.max_batches,
                "metrics": metrics,
            }
            path = result_path(output_dir, seed, condition)
            with path.open("w", encoding="utf-8") as output_file:
                json.dump(payload, output_file, indent=2, sort_keys=True)
                output_file.write("\n")
            results.append(payload)
            print(
                f"[done] seed={seed} condition={condition} "
                f"mAP50={metrics['map_50']:.6f} -> {path}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    build_aggregates(
        results,
        output_dir,
        protocol_complete=args.max_batches is None,
    )


if __name__ == "__main__":
    main()
