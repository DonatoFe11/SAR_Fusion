#!/usr/bin/env python3
"""Run the frozen RT-DETR FAM diagnostic protocol on all final checkpoints.

The script deliberately treats the checkpoint/seed as the experimental unit:
spatial cells are summarized within each sample, samples within each
checkpoint, and only then are the five checkpoints summarized across seeds.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import aggregate_numeric_records  # noqa: E402
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint  # noqa: E402


CONFIGURATIONS = {
    "fam": {
        "project": "RTDETR_FAM_Protocol",
        "config": "parameters/RTDETR/rtdetr_fam.yaml",
        "figures": True,
    },
    "ssj": {
        "project": "RTDETR_FAM_SSJ_Protocol",
        "config": "parameters/RTDETR/rtdetr_fam_ssj.yaml",
        "figures": True,
    },
    "grid_sample": {
        "project": "RTDETR_FAM_Grid_Sample_Ablation",
        "config": "parameters/RTDETR/rtdetr_ablation_grid_sample.yaml",
        "figures": False,
    },
}

SAMPLE_SETS = {
    "mt_erie": {
        "split": "test",
        "indices": [35, 106, 177, 247, 318, 389, 460, 531, 601, 672],
        "figure_indices": [35],
    },
    "train_cross_session": {
        "split": "train",
        "indices": [
            90, 269, 448, 627, 806, 990, 1132, 1367, 1603, 1745,
            1948, 2166, 2384, 2602, 2929, 3038, 3256, 3474, 3692, 3910,
        ],
        "figure_indices": [90, 1948],
    },
}

FHL_INDICES = set(SAMPLE_SETS["train_cross_session"]["indices"][:10])
BAKER_INDICES = set(SAMPLE_SETS["train_cross_session"]["indices"][10:])


def final_session(raw_session, sample_index):
    if raw_session == "mt_erie":
        return "mt_erie"
    if sample_index in FHL_INDICES:
        return "fhl"
    if sample_index in BAKER_INDICES:
        return "baker"
    raise ValueError(f"Unregistered train sample index: {sample_index}")


def run_one_job(
    python_executable,
    label,
    configuration,
    seed,
    checkpoint,
    sample_set_name,
    sample_set,
    output_dir,
    dataset_root,
    force,
    dry_run,
):
    raw_dir = output_dir / "raw"
    figure_dir = output_dir / "figures" / label / f"seed_{seed}" / sample_set_name
    output_json = raw_dir / f"{label}_seed{seed}_{sample_set_name}.json"
    if output_json.is_file() and not force:
        print(f"[skip] {output_json} already exists")
        return output_json

    command = [
        python_executable,
        str(REPO_ROOT / "fam_alignment_check.py"),
        "--model-type", "hf",
        "--config", str(REPO_ROOT / configuration["config"]),
        "--run-index", str(seed - 40),
        "--checkpoint", checkpoint,
        "--sample-idx", *[str(index) for index in sample_set["indices"]],
        "--split", sample_set["split"],
        "--out-dir", str(figure_dir),
        "--output-json", str(output_json),
        "--configuration-label", label,
        "--seed", str(seed),
        "--session", sample_set_name,
    ]
    if dataset_root:
        command.extend(["--dataset-root", str(Path(dataset_root).resolve())])

    make_figures = configuration["figures"] and seed == 40
    if make_figures:
        command.extend(
            [
                "--figure-sample-idx",
                *[str(index) for index in sample_set["figure_indices"]],
            ]
        )
    else:
        command.append("--no-figures")

    print("[run] " + " ".join(command))
    if not dry_run:
        raw_dir.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.setdefault("MPLBACKEND", "Agg")
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    return output_json


def build_aggregates(raw_paths, output_dir):
    rows = []
    for raw_path in raw_paths:
        with raw_path.open(encoding="utf-8") as input_file:
            payload = json.load(input_file)
        for row in payload["rows"]:
            row["session"] = final_session(row["session"], row["sample_index"])
            rows.append(row)

    checkpoint_groups = {}
    for row in rows:
        key = (
            row["configuration"],
            int(row["seed"]),
            row["session"],
            int(row["level"]),
        )
        checkpoint_groups.setdefault(key, []).append(row["metrics"])

    checkpoint_aggregates = []
    for (configuration, seed, session, level), metrics in sorted(checkpoint_groups.items()):
        checkpoint_aggregates.append(
            {
                "configuration": configuration,
                "seed": seed,
                "session": session,
                "level": level,
                "n_samples": len(metrics),
                "sample_aggregate": aggregate_numeric_records(metrics),
            }
        )

    seed_groups = {}
    for aggregate in checkpoint_aggregates:
        for metric, summary in aggregate["sample_aggregate"].items():
            key = (
                aggregate["configuration"],
                aggregate["session"],
                aggregate["level"],
                metric,
            )
            seed_groups.setdefault(key, []).append(
                (aggregate["seed"], summary["mean"])
            )

    across_seed_aggregates = []
    for (configuration, session, level, metric), seed_values in sorted(seed_groups.items()):
        seed_values = sorted(seed_values)
        values = [value for _seed, value in seed_values]
        summary = aggregate_numeric_records([{"value": value} for value in values])["value"]
        across_seed_aggregates.append(
            {
                "configuration": configuration,
                "session": session,
                "level": level,
                "metric": metric,
                "seed_values": {str(seed): value for seed, value in seed_values},
                "seed_aggregate": summary,
            }
        )

    payload = {
        "schema_version": 1,
        "protocol": {
            "checkpoint": "latest",
            "seeds": [40, 41, 42, 43, 44],
            "configurations": CONFIGURATIONS,
            "sample_sets": SAMPLE_SETS,
            "experimental_unit": "checkpoint/seed",
            "aggregation_order": ["spatial cells within sample", "samples within checkpoint", "checkpoints across seeds"],
        },
        "rows": rows,
        "checkpoint_aggregates": checkpoint_aggregates,
        "across_seed_aggregates": across_seed_aggregates,
    }
    output_json = output_dir / "rtdetr_fam_diagnostics.json"
    with output_json.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    output_csv = output_dir / "rtdetr_fam_diagnostics_across_seeds.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = [
            "configuration", "session", "level", "metric", "n_seeds",
            "mean", "median", "sample_std", "min", "max",
            "seed_40", "seed_41", "seed_42", "seed_43", "seed_44",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for aggregate in across_seed_aggregates:
            summary = aggregate["seed_aggregate"]
            writer.writerow(
                {
                    "configuration": aggregate["configuration"],
                    "session": aggregate["session"],
                    "level": aggregate["level"],
                    "metric": aggregate["metric"],
                    "n_seeds": summary["n"],
                    "mean": summary["mean"],
                    "median": summary["median"],
                    "sample_std": summary["sample_std"],
                    "min": summary["min"],
                    "max": summary["max"],
                    **{
                        f"seed_{seed}": aggregate["seed_values"].get(str(seed))
                        for seed in range(40, 45)
                    },
                }
            )
    print(f"Saved combined JSON: {output_json}")
    print(f"Saved across-seed CSV: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configurations",
        choices=sorted(CONFIGURATIONS),
        nargs="+",
        default=list(CONFIGURATIONS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    parser.add_argument("--output-dir", default="out/rtdetr_fam_diagnostics")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--force", action="store_true", help="replace existing raw JSON jobs")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    args = parser.parse_args()

    invalid_seeds = sorted(set(args.seeds) - set(range(40, 45)))
    if invalid_seeds:
        parser.error(f"the frozen protocol only contains seeds 40-44, got {invalid_seeds}")

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = []
    for label in args.configurations:
        configuration = CONFIGURATIONS[label]
        for seed in args.seeds:
            checkpoint = resolve_local_wandb_checkpoint(
                configuration["project"],
                seed,
                checkpoint="latest",
                wandb_root=REPO_ROOT / "wandb",
            )
            for sample_set_name, sample_set in SAMPLE_SETS.items():
                raw_paths.append(
                    run_one_job(
                        sys.executable,
                        label,
                        configuration,
                        seed,
                        checkpoint,
                        sample_set_name,
                        sample_set,
                        output_dir,
                        args.dataset_root,
                        args.force,
                        args.dry_run,
                    )
                )

    if not args.dry_run:
        build_aggregates(raw_paths, output_dir)


if __name__ == "__main__":
    main()
