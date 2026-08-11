#!/usr/bin/env python3
"""Audit RT-DETR FAM parameters, activations, and input-dependent offsets.

The audit reuses the 30 frozen diagnostic samples for each of the five final
FAM checkpoints. Statistics are first summarized within a checkpoint and only
then across seeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import (  # noqa: E402
    FAMCapture,
    aggregate_numeric_records,
    diagnostic_level_record,
    load_fusion_model,
    load_hf_datasets,
    load_run_config,
    net_offset_field,
)
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint  # noqa: E402
from scripts.run_rtdetr_fam_level_ablation import (  # noqa: E402
    LEVEL_LABELS,
    find_fam_modules,
    jsonable,
    resolve_device,
)


PROTOCOL_ID = "rtdetr_fam_internal_audit_v1"
PROJECT = "RTDETR_FAM_Protocol"
CONFIG_PATH = "parameters/RTDETR/rtdetr_protocol.yaml"
SAMPLE_SETS = {
    "mt_erie": {
        "split": "test",
        "indices": [35, 106, 177, 247, 318, 389, 460, 531, 601, 672],
    },
    "fhl": {
        "split": "train",
        "indices": [90, 269, 448, 627, 806, 990, 1132, 1367, 1603, 1745],
    },
    "baker": {
        "split": "train",
        "indices": [1948, 2166, 2384, 2602, 2929, 3038, 3256, 3474, 3692, 3910],
    },
}


def tensor_statistics(tensor):
    tensor = tensor.detach().double().reshape(-1)
    finite = torch.isfinite(tensor)
    values = tensor[finite]
    if values.numel() == 0:
        raise ValueError("Cannot summarize a tensor without finite values")
    return {
        "numel": int(tensor.numel()),
        "finite_fraction": float(finite.double().mean()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "abs_mean": float(values.abs().mean()),
        "max_abs": float(values.abs().max()),
        "l2_norm": float(torch.linalg.vector_norm(values)),
    }


def tensor_pair_metrics(first, second, eps=1e-12):
    if first.shape != second.shape:
        raise ValueError(f"Tensor shapes differ: {tuple(first.shape)} vs {tuple(second.shape)}")
    first = first.detach().double().reshape(-1)
    second = second.detach().double().reshape(-1)
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("Pairwise offset comparison requires finite tensors")

    difference = first - second
    scale = 0.5 * (first.abs().mean() + second.abs().mean())
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = torch.linalg.vector_norm(first_centered) * torch.linalg.vector_norm(
        second_centered
    )
    pearson_r = None
    if denominator > eps:
        pearson_r = float(torch.dot(first_centered, second_centered) / denominator)
    return {
        "mae": float(difference.abs().mean()),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "normalized_mae": float(difference.abs().mean() / max(float(scale), eps)),
        "pearson_r": pearson_r,
    }


def split_offset_predictor(module):
    weight = module.offset_conv.weight.detach()
    bias = module.offset_conv.bias.detach()
    if weight.shape[0] != 27 or bias.shape[0] != 27:
        raise RuntimeError(
            f"Expected a 27-channel DCNv2 predictor, got {tuple(weight.shape)}"
        )
    return {
        "offset_weight": weight[:18],
        "mask_weight": weight[18:],
        "offset_bias": bias[:18],
        "mask_bias": bias[18:],
        "deform_weight": module.deform_conv.weight.detach(),
        "deform_bias": module.deform_conv.bias.detach(),
    }


def parameter_audit(model):
    rows = []
    for level, module in enumerate(find_fam_modules(model)):
        for parameter_name, tensor in split_offset_predictor(module).items():
            rows.append(
                {
                    "level": level,
                    "level_label": LEVEL_LABELS[level],
                    "parameter": parameter_name,
                    "statistics": tensor_statistics(tensor),
                }
            )
    return rows


def compare_sample_tensors(tensors_by_session):
    rows = []
    for session, samples_by_level in tensors_by_session.items():
        for level, samples in samples_by_level.items():
            for (index_a, fields_a), (index_b, fields_b) in itertools.combinations(
                sorted(samples.items()), 2
            ):
                for field in ("raw_offset", "net_offset"):
                    rows.append(
                        {
                            "session": session,
                            "level": level,
                            "level_label": LEVEL_LABELS[level],
                            "field": field,
                            "sample_a": index_a,
                            "sample_b": index_b,
                            "metrics": tensor_pair_metrics(fields_a[field], fields_b[field]),
                        }
                    )
    return rows


def aggregate_pairwise_rows(rows):
    groups = {}
    for row in rows:
        key = (row["session"], row["level"], row["field"])
        groups.setdefault(key, []).append(row["metrics"])
    return [
        {
            "session": session,
            "level": level,
            "level_label": LEVEL_LABELS[level],
            "field": field,
            "n_pairs": len(metrics),
            "aggregate": aggregate_numeric_records(metrics),
        }
        for (session, level, field), metrics in sorted(groups.items())
    ]


def summarize_seed_values(seed_values):
    """Summarize checkpoint-level values without pooling their samples."""
    seed_values = sorted(seed_values)
    values = [value for _seed, value in seed_values]
    return {
        "seed_values": {str(seed): value for seed, value in seed_values},
        "seed_aggregate": {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        },
    }


def aggregate_sample_rows(rows):
    """Aggregate the ten frozen samples within each session and level."""
    groups = {}
    for row in rows:
        key = (row["session"], row["level"])
        groups.setdefault(key, []).append(row["metrics"])
    return [
        {
            "session": session,
            "level": level,
            "level_label": LEVEL_LABELS[level],
            "n_samples": len(metrics),
            "aggregate": aggregate_numeric_records(metrics),
        }
        for (session, level), metrics in sorted(groups.items())
    ]


def audit_seed(
    seed,
    checkpoint,
    config_path,
    dataset_root,
    device,
    project=PROJECT,
    fam_variant="current_dcnv2",
    datasets=None,
):
    run_config = load_run_config(config_path, run_index=seed - 40)
    model_params = run_config["model"]
    model_params["params"].update(
        {
            "use_fam": True,
            "freeze_fam": False,
            "fam_variant": fam_variant,
            "ir_dropout_rate": 0.0,
            "spatial_jitter_std": 0.0,
        }
    )
    model = load_fusion_model(model_params, checkpoint, device)
    if datasets is None:
        dataset_params = run_config["dataset"]
        dataset_params["root"] = str(dataset_root)
        datasets, _denormalize = load_hf_datasets(
            dataset_params,
            run_config["dataloader"],
        )

    capture = FAMCapture(model)
    sample_rows = []
    tensors_by_session = {}
    try:
        with torch.inference_mode():
            for session, sample_set in SAMPLE_SETS.items():
                tensors_by_session[session] = {level: {} for level in LEVEL_LABELS}
                dataset = datasets[sample_set["split"]]
                for sample_index in sample_set["indices"]:
                    capture.records.clear()
                    pixel_values = dataset[sample_index].pixel_values.unsqueeze(0).to(device)
                    model(pixel_values=pixel_values)
                    if set(capture.records) != set(LEVEL_LABELS):
                        raise RuntimeError(
                            f"Expected captures for levels {sorted(LEVEL_LABELS)}, "
                            f"got {sorted(capture.records)}"
                        )
                    for level, record in capture.records.items():
                        sample_rows.append(
                            {
                                "session": session,
                                "split": sample_set["split"],
                                "sample_index": sample_index,
                                "level": level,
                                "level_label": LEVEL_LABELS[level],
                                "metrics": diagnostic_level_record(
                                    record,
                                    input_hw=tuple(pixel_values.shape[-2:]),
                                ),
                            }
                        )
                        effective_offset = record["offset"][0].float()
                        raw_offset = record.get("raw_offset", record["offset"])[
                            0
                        ].float()
                        mask = record.get("mask")
                        mask = mask[0].float() if mask is not None else None
                        tensors_by_session[session][level][sample_index] = {
                            "raw_offset": raw_offset,
                            "net_offset": net_offset_field(
                                effective_offset,
                                record["offset_kind"],
                                mask=mask,
                            ).float(),
                        }
    finally:
        capture.remove()

    pairwise_rows = compare_sample_tensors(tensors_by_session)
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "project": project,
        "fam_variant": fam_variant,
        "seed": seed,
        "checkpoint": str(Path(checkpoint).resolve()),
        "sample_sets": SAMPLE_SETS,
        "parameter_rows": parameter_audit(model),
        "sample_rows": sample_rows,
        "sample_checkpoint_aggregates": aggregate_sample_rows(sample_rows),
        "pairwise_rows": pairwise_rows,
        "pairwise_checkpoint_aggregates": aggregate_pairwise_rows(pairwise_rows),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return jsonable(payload), datasets


def assert_compatible_existing_audit(
    payload,
    *,
    seed,
    checkpoint,
    project,
    fam_variant,
):
    """Fail closed when a resumable raw audit belongs to another campaign."""
    # Raw files produced before project/variant parameterization correspond to
    # the historical defaults and remain resumable only for those defaults.
    payload_project = payload.get("project", PROJECT)
    payload_variant = payload.get("fam_variant", "current_dcnv2")
    expected_checkpoint = str(Path(checkpoint).resolve())
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("seed") != seed
        or payload.get("checkpoint") != expected_checkpoint
        or payload_project != project
        or payload_variant != fam_variant
    ):
        raise RuntimeError("Existing audit is incompatible; inspect it before --force")


def combine_seed_audits(seed_payloads, output_dir):
    projects = {payload.get("project", PROJECT) for payload in seed_payloads}
    fam_variants = {
        payload.get("fam_variant", "current_dcnv2") for payload in seed_payloads
    }
    if len(projects) != 1 or len(fam_variants) != 1:
        raise RuntimeError("Cannot combine audits from different projects or FAM variants")

    pairwise_groups = {}
    for payload in seed_payloads:
        for row in payload["pairwise_checkpoint_aggregates"]:
            for metric, summary in row["aggregate"].items():
                key = (row["session"], row["level"], row["field"], metric)
                pairwise_groups.setdefault(key, []).append(
                    (payload["seed"], summary["mean"])
                )

    pairwise_across_seed = []
    for (session, level, field, metric), seed_values in sorted(
        pairwise_groups.items()
    ):
        pairwise_across_seed.append(
            {
                "session": session,
                "level": level,
                "level_label": LEVEL_LABELS[level],
                "field": field,
                "metric": metric,
                **summarize_seed_values(seed_values),
            }
        )

    sample_groups = {}
    for payload in seed_payloads:
        checkpoint_aggregates = payload.get("sample_checkpoint_aggregates")
        if checkpoint_aggregates is None:
            checkpoint_aggregates = aggregate_sample_rows(payload["sample_rows"])
        for row in checkpoint_aggregates:
            for metric, summary in row["aggregate"].items():
                key = (row["session"], row["level"], metric)
                sample_groups.setdefault(key, []).append(
                    (payload["seed"], summary["mean"])
                )

    sample_across_seed = []
    for (session, level, metric), seed_values in sorted(sample_groups.items()):
        sample_across_seed.append(
            {
                "session": session,
                "level": level,
                "level_label": LEVEL_LABELS[level],
                "metric": metric,
                **summarize_seed_values(seed_values),
            }
        )

    parameter_groups = {}
    for payload in seed_payloads:
        for row in payload["parameter_rows"]:
            for statistic, value in row["statistics"].items():
                key = (row["level"], row["parameter"], statistic)
                parameter_groups.setdefault(key, []).append((payload["seed"], value))

    parameter_across_seed = []
    for (level, parameter, statistic), seed_values in sorted(
        parameter_groups.items()
    ):
        parameter_across_seed.append(
            {
                "level": level,
                "level_label": LEVEL_LABELS[level],
                "parameter": parameter,
                "statistic": statistic,
                **summarize_seed_values(seed_values),
            }
        )

    combined = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "project": projects.pop(),
        "fam_variant": fam_variants.pop(),
        "sample_sets": SAMPLE_SETS,
        "experimental_unit": "checkpoint/seed",
        "seed_audits": sorted(seed_payloads, key=lambda row: row["seed"]),
        "sample_across_seed_aggregates": sample_across_seed,
        "parameter_across_seed_aggregates": parameter_across_seed,
        "pairwise_across_seed_aggregates": pairwise_across_seed,
    }
    output_path = output_dir / "rtdetr_fam_internal_audit.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(combined, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(f"Saved combined audit: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--fam-variant", default="current_dcnv2")
    parser.add_argument("--dataset-root", default="dataset/WiSARD")
    parser.add_argument("--output-dir", default="out/rtdetr_fam_internal_audit")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    invalid_seeds = sorted(set(args.seeds) - set(range(40, 45)))
    if invalid_seeds:
        parser.error(f"the frozen protocol only contains seeds 40-44, got {invalid_seeds}")

    config_path = (REPO_ROOT / args.config).resolve()
    dataset_root = (REPO_ROOT / args.dataset_root).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if not args.dry_run:
        (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    device = args.device if args.dry_run else resolve_device(args.device)

    checkpoints = {
        seed: resolve_local_wandb_checkpoint(
            args.project,
            seed,
            checkpoint="latest",
            wandb_root=REPO_ROOT / "wandb",
        )
        for seed in args.seeds
    }
    if args.dry_run:
        print(f"project={args.project} fam_variant={args.fam_variant}")
        for seed, checkpoint in checkpoints.items():
            print(f"seed={seed} checkpoint={checkpoint} device={device}")
        return

    payloads = []
    datasets = None
    for seed in args.seeds:
        raw_path = output_dir / "raw" / f"seed_{seed}.json"
        if raw_path.is_file() and not args.force:
            with raw_path.open(encoding="utf-8") as input_file:
                payload = json.load(input_file)
            try:
                assert_compatible_existing_audit(
                    payload,
                    seed=seed,
                    checkpoint=checkpoints[seed],
                    project=args.project,
                    fam_variant=args.fam_variant,
                )
            except RuntimeError as error:
                raise RuntimeError(f"{raw_path}: {error}") from error
            print(f"[skip] {raw_path}")
        else:
            print(f"[run] internal FAM audit seed={seed}")
            payload, datasets = audit_seed(
                seed,
                checkpoints[seed],
                config_path,
                dataset_root,
                device,
                project=args.project,
                fam_variant=args.fam_variant,
                datasets=datasets,
            )
            with raw_path.open("w", encoding="utf-8") as output_file:
                json.dump(payload, output_file, indent=2, sort_keys=True)
                output_file.write("\n")
            print(f"[done] {raw_path}")
        payloads.append(payload)

    combine_seed_audits(payloads, output_dir)


if __name__ == "__main__":
    main()
