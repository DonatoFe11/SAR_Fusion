#!/usr/bin/env python3
"""Replay the frozen RT-DETRv2 Stage-A best checkpoints on validation.

The public mode evaluates the matched Additive and FAM runs in isolated child
processes, then writes the final allocation-gate report.  No test split is
evaluated and no W&B run is created or modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import torch
from safetensors import safe_open
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sarfusion.data import (
    DATASET_REGISTRY,
    get_train_val_test_params,
    seed_worker,
)
from sarfusion.data.temporal_split import select_temporal_split_items
from sarfusion.data.utils import build_preprocessor, get_collate_fn
from sarfusion.experiment.utils import WrapperModule
from sarfusion.models import build_model
from sarfusion.models.checkpoints import complete_shared_state_dict_aliases
from sarfusion.utils.metrics import build_evaluator
from sarfusion.utils.reproducibility import (
    configure_reproducibility,
    verify_training_source_manifest,
)
from sarfusion.utils.structures import DataDict


DEFAULT_GATE_DELTA = 0.010000
DEFAULT_REPLAY_TOLERANCE = 0.0002
DEFAULT_VALIDATION_SIZE = 896


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run_params(run_dir: Path) -> tuple[dict, dict]:
    files_dir = run_dir / "files"
    config_path = files_dir / "config.yaml"
    summary_path = files_dir / "wandb-summary.json"
    if not config_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Incomplete W&B run directory: {run_dir}")

    raw_config = yaml.safe_load(config_path.read_text())
    params = {
        key: entry["value"]
        for key, entry in raw_config.items()
        if key != "_wandb" and isinstance(entry, dict) and "value" in entry
    }
    summary = json.loads(summary_path.read_text())
    return params, summary


def scalar_metrics(metrics: dict) -> dict[str, float | int]:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        elif hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                continue
        if isinstance(value, bool):
            result[key] = value
        elif isinstance(value, int):
            result[key] = value
        elif isinstance(value, float):
            result[key] = value
    return result


def build_validation_loader(
    dataset_params: dict,
    dataloader_params: dict,
    *,
    data_seed: int,
) -> torch.utils.data.DataLoader:
    """Mirror the project validation loader without constructing test data."""
    dataset_params = dict(dataset_params)
    dataloader_params = dict(dataloader_params)
    evaluation_batch_size = dataloader_params.pop(
        "evaluation_batch_size", dataloader_params["batch_size"]
    )
    evaluation_dataloader_params = {
        **dataloader_params,
        "batch_size": evaluation_batch_size,
    }
    transforms, _ = build_preprocessor(dataset_params)
    dataset_name = dataset_params.pop("name")
    dataset_class = DATASET_REGISTRY[dataset_name]
    _, val_dataset_params, _ = get_train_val_test_params(
        dataset_name, dataset_params
    )
    val_dataset_params.pop("preprocessor", None)
    temporal_phase = val_dataset_params.pop("temporal_split_phase", None)
    temporal_inventory = val_dataset_params.pop(
        "temporal_split_inventory", None
    )
    val_dataset_params.pop("temporal_split_manifest", None)
    val_set = dataset_class(transform=transforms, **val_dataset_params)
    if temporal_phase:
        val_set.items = select_temporal_split_items(
            val_set.items,
            val_dataset_params["root"],
            temporal_inventory,
            temporal_phase,
        )
    generator = torch.Generator().manual_seed(int(data_seed) + 1)
    return torch.utils.data.DataLoader(
        val_set,
        collate_fn=get_collate_fn(val_set),
        worker_init_fn=seed_worker,
        generator=generator,
        **evaluation_dataloader_params,
    )


def replay_single(
    run_dir: Path,
    *,
    expect_use_fam: bool,
    tolerance: float,
    expected_validation_size: int,
) -> dict:
    run_dir = run_dir.resolve()
    params, summary = load_run_params(run_dir)
    model_params = params["model"]
    actual_use_fam = bool(model_params["params"]["use_fam"])
    if actual_use_fam != expect_use_fam:
        raise RuntimeError(
            f"use_fam={actual_use_fam} does not match expectation "
            f"{expect_use_fam} for {run_dir}"
        )

    reproducibility = params["reproducibility"]
    manifest = verify_training_source_manifest(
        reproducibility["training_source_manifest_id"],
        reproducibility["training_source_manifest_sha256"],
    )
    configure_reproducibility(
        int(params["seed"]),
        deterministic=bool(reproducibility.get("deterministic", False)),
        warn_only=bool(reproducibility.get("warn_only", False)),
    )

    val_loader = build_validation_loader(
        params["dataset"],
        params["dataloader"],
        data_seed=int(reproducibility.get("data_seed", params["seed"])),
    )
    validation_size = len(val_loader.dataset)
    if validation_size != expected_validation_size:
        raise RuntimeError(
            f"Validation size changed: {validation_size} != "
            f"{expected_validation_size}"
        )

    model_seed = int(reproducibility.get("model_seed", params["seed"]))
    set_seed(model_seed)
    model = WrapperModule(build_model(model_params))

    checkpoint_path = run_dir / "files" / "best" / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    with safe_open(checkpoint_path, framework="pt") as stream:
        weights = {key: stream.get_tensor(key) for key in stream.keys()}
        serialized_tensor_count = len(weights)
    weights, restored_aliases = complete_shared_state_dict_aliases(model, weights)
    model.load_state_dict(weights, strict=True)
    del weights

    train_params = params.get("train", {})
    accelerator = Accelerator(
        even_batches=False,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ],
        split_batches=False,
        mixed_precision=train_params.get("precision", None),
    )
    model = accelerator.prepare(model)
    val_loader = accelerator.prepare(val_loader)
    evaluator = build_evaluator(
        params.get("val_evaluation"),
        params.get("task"),
        id2class=val_loader.dataset.id2class,
    )
    evaluator = accelerator.prepare(evaluator)
    model.eval()
    evaluator.reset()

    print(
        f"Replaying {run_dir.name}: {validation_size} validation frames, "
        f"{len(val_loader)} batches",
        flush=True,
    )
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(val_loader, start=1):
            batch = DataDict(**raw_batch)
            model_input = DataDict(**dict(batch))
            model_input.labels = None
            output = model(model_input)
            evaluator.update(batch, output)
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(
                val_loader
            ):
                print(
                    f"  {run_dir.name}: batch {batch_index}/{len(val_loader)}",
                    flush=True,
                )

    accelerator.wait_for_everyone()
    metrics = scalar_metrics(evaluator.compute())
    replay_map50 = float(metrics["map_50"])
    expected_map50 = float(summary["best_map_50"])
    absolute_error = abs(replay_map50 - expected_map50)
    if not math.isfinite(replay_map50):
        raise RuntimeError(f"Non-finite replay mAP@50 for {run_dir.name}")

    report = {
        "run_id": run_dir.name.rsplit("-", 1)[-1],
        "run_dir": str(run_dir),
        "use_fam": actual_use_fam,
        "best_epoch": int(summary["best_epoch"]),
        "summary_best_map_50": expected_map50,
        "replay_metrics": metrics,
        "absolute_map_50_error": absolute_error,
        "replay_tolerance": tolerance,
        "replay_within_tolerance": absolute_error <= tolerance,
        "validation_size": validation_size,
        "validation_batches": len(val_loader),
        "checkpoint_sha256": sha256(checkpoint_path),
        "serialized_tensor_count": serialized_tensor_count,
        "strict_state_tensor_count": len(model.state_dict()),
        "restored_exact_alias_count": len(restored_aliases),
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["sha256"],
        "manifest_file_count": len(manifest["files"]),
    }

    del output, model_input, batch, evaluator, val_loader, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def orchestrate(args: argparse.Namespace) -> int:
    roles = (
        ("control", args.control_run.resolve(), False),
        ("candidate", args.candidate_run.resolve(), True),
    )
    results = {}
    with tempfile.TemporaryDirectory(prefix="rtdetr-v2-replay-") as temp_dir:
        for role, run_dir, use_fam in roles:
            child_output = Path(temp_dir) / f"{role}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single-run",
                str(run_dir),
                "--single-output",
                str(child_output),
                "--tolerance",
                str(args.tolerance),
                "--expected-validation-size",
                str(args.expected_validation_size),
            ]
            command.append("--expect-use-fam" if use_fam else "--no-expect-use-fam")
            subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
            results[role] = json.loads(child_output.read_text())

    control = results["control"]
    candidate = results["candidate"]
    summary_delta = (
        candidate["summary_best_map_50"] - control["summary_best_map_50"]
    )
    replay_delta = (
        candidate["replay_metrics"]["map_50"]
        - control["replay_metrics"]["map_50"]
    )
    replay_integrity_passed = all(
        result["replay_within_tolerance"] for result in results.values()
    )
    matched_provenance = (
        control["manifest_id"] == candidate["manifest_id"]
        and control["manifest_sha256"] == candidate["manifest_sha256"]
        and control["validation_size"] == candidate["validation_size"]
    )
    integrity_passed = replay_integrity_passed and matched_provenance
    allocation_gate_passed = (
        integrity_passed and summary_delta >= args.gate_delta
    )

    report = {
        "status": "passed" if allocation_gate_passed else "failed",
        "integrity_passed": integrity_passed,
        "replay_integrity_passed": replay_integrity_passed,
        "matched_provenance": matched_provenance,
        "allocation_gate_passed": allocation_gate_passed,
        "allocation_decision": (
            "open_seed_41_44" if allocation_gate_passed else "keep_seed_41_44_closed"
        ),
        "gate": {
            "metric": "best_validation_map_50",
            "required_delta": args.gate_delta,
            "observed_summary_delta": summary_delta,
            "observed_replay_delta": replay_delta,
            "margin_to_gate": summary_delta - args.gate_delta,
        },
        **results,
    }
    write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"Report written to {args.output.resolve()}", flush=True)
    # A negative scientific gate is a valid completed result.  Only technical
    # integrity failure makes this verification command fail.
    return 0 if integrity_passed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", type=Path)
    parser.add_argument("--candidate-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate-delta", type=float, default=DEFAULT_GATE_DELTA)
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_REPLAY_TOLERANCE
    )
    parser.add_argument(
        "--expected-validation-size",
        type=int,
        default=DEFAULT_VALIDATION_SIZE,
    )
    parser.add_argument("--single-run", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--single-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--expect-use-fam",
        action=argparse.BooleanOptionalAction,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.single_run is not None:
        if args.single_output is None or args.expect_use_fam is None:
            parser.error("internal single-run mode requires output and FAM expectation")
    elif args.control_run is None or args.candidate_run is None or args.output is None:
        parser.error("control run, candidate run, and output are required")
    return args


def main() -> None:
    args = parse_args()
    if args.single_run is not None:
        result = replay_single(
            args.single_run,
            expect_use_fam=args.expect_use_fam,
            tolerance=args.tolerance,
            expected_validation_size=args.expected_validation_size,
        )
        write_json(args.single_output.resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    raise SystemExit(orchestrate(args))


if __name__ == "__main__":
    main()
