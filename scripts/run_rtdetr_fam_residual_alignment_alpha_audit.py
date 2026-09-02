#!/usr/bin/env python3
"""Audit learned RCRA alpha maps on the frozen Stage-A validation video."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import load_fusion_model, load_run_config  # noqa: E402
from sarfusion.data import get_dataloaders  # noqa: E402
from sarfusion.models.checkpoints import (  # noqa: E402
    resolve_local_wandb_checkpoint,
)
from sarfusion.models.rtdetr_fusion import (  # noqa: E402
    ReliabilityConditionedResidualAlignment,
)
from sarfusion.utils.utils import load_yaml  # noqa: E402
from scripts.run_rtdetr_fam_reliability_gate_weight_audit import (  # noqa: E402
    WeightAccumulator,
    mask_modalities,
    prepare_batch,
    summarize_across_seeds,
)


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_residual_alignment_alpha_audit.yaml"
)


def validate_protocol(protocol):
    if protocol.get("protocol_id") != "rtdetr_fam_residual_alignment_alpha_audit_v1":
        raise ValueError("Unexpected residual-alignment audit protocol_id")
    if protocol.get("checkpoint") != "best":
        raise ValueError("The audit must use the predeclared best checkpoint")
    if protocol.get("split") != "val":
        raise ValueError("The alignment audit is restricted to validation")
    if protocol.get("modes") != ["fusion", "rgb", "ir"]:
        raise ValueError("The audit modes must be fusion, rgb, ir in that order")
    if protocol.get("seeds") != [40, 41, 42, 43, 44]:
        raise ValueError("The alignment audit requires all five campaign seeds")
    if protocol.get("level_labels") != ["P3", "P4", "P5"]:
        raise ValueError("The alignment audit requires P3--P5")


def find_alignment_gates(model):
    gates = [
        module
        for module in model.modules()
        if isinstance(module, ReliabilityConditionedResidualAlignment)
    ]
    if len(gates) != 3:
        raise RuntimeError(f"Expected three residual alignment gates, found {len(gates)}")
    return gates


class AlphaCapture:
    def __init__(self, gates, bins, low_threshold, high_threshold):
        self.accumulators = {}
        self.handles = []
        for level, gate in enumerate(gates):
            self.accumulators[level] = WeightAccumulator(
                bins, low_threshold, high_threshold
            )
            self.handles.append(
                gate.register_forward_hook(
                    self._hook(level),
                    with_kwargs=True,
                )
            )

    def _hook(self, level):
        def capture(module, args, kwargs, output):
            alpha = module.compute_alpha(
                args[0],
                args[1],
                args[2],
                rgb_present=kwargs.get("rgb_present"),
                ir_present=kwargs.get("ir_present"),
            )
            self.accumulators[level].update(alpha)

        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()


def gate_parameter_rows(seed, gates, level_labels):
    rows = []
    for level, (label, gate) in enumerate(zip(level_labels, gates)):
        rows.append(
            {
                "seed": seed,
                "level": level,
                "level_label": label,
                "logit_weight_l2": float(
                    torch.linalg.vector_norm(gate.logit_conv.weight.detach())
                ),
                "logit_bias": float(gate.logit_conv.bias.detach()[0]),
            }
        )
    return rows


def audit_mode(model, dataloader, mode, protocol, device):
    capture = AlphaCapture(
        find_alignment_gates(model),
        bins=protocol["histogram_bins"],
        low_threshold=protocol["low_alpha_threshold"],
        high_threshold=protocol["high_alpha_threshold"],
    )
    batches = 0
    frames = 0
    try:
        with torch.inference_mode():
            for batch in dataloader:
                pixel_values, pixel_mask = prepare_batch(batch, mode, device)
                model.model(pixel_values, pixel_mask=pixel_mask)
                batches += 1
                frames += int(pixel_values.shape[0])
                pixel_values = pixel_mask = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        capture.close()

    expected_batches = int(protocol["expected_validation_batches"])
    expected_frames = int(protocol["expected_validation_frames"])
    if batches != expected_batches:
        raise RuntimeError(f"Expected {expected_batches} validation batches, got {batches}")
    if frames != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} validation frames, got {frames}")
    return capture.accumulators


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "mode",
        "level",
        "level_label",
        "modality",
        "numel",
        "mean",
        "std",
        "mean_abs_delta_one",
        "min",
        "p05_approx",
        "median_approx",
        "p95_approx",
        "max",
        "fraction_below_low",
        "fraction_above_high",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(protocol_path, dry_run=False):
    protocol = load_yaml(protocol_path)
    validate_protocol(protocol)
    config_path = REPO_ROOT / protocol["training_config"]
    checkpoints = {
        seed: resolve_local_wandb_checkpoint(
            project=protocol["project"],
            seed=seed,
            checkpoint=protocol["checkpoint"],
        )
        for seed in protocol["seeds"]
    }
    if dry_run:
        print(f"Dry run OK: {len(checkpoints)} best checkpoints resolved")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_config = load_run_config(config_path, run_index=0)
    (_train, val_loader, _test), _denormalize = get_dataloaders(
        run_config["dataset"],
        run_config["dataloader"],
        seed=protocol["seeds"][0],
    )
    rows = []
    parameter_rows = []
    for run_index, seed in enumerate(protocol["seeds"]):
        seed_config = load_run_config(config_path, run_index=run_index)
        model = load_fusion_model(
            seed_config["model"], checkpoints[seed], device
        )
        gates = find_alignment_gates(model)
        parameter_rows.extend(
            gate_parameter_rows(seed, gates, protocol["level_labels"])
        )
        for mode in protocol["modes"]:
            print(f"Auditing seed={seed} mode={mode}")
            accumulators = audit_mode(model, val_loader, mode, protocol, device)
            for level, accumulator in sorted(accumulators.items()):
                rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "level": level,
                        "level_label": protocol["level_labels"][level],
                        "modality": "alignment",
                        **accumulator.summary(),
                    }
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "protocol": protocol,
        "checkpoints": checkpoints,
        "parameter_rows": parameter_rows,
        "alpha_rows": rows,
        "checkpoint_level_aggregate": summarize_across_seeds(rows),
        "protocol_complete": True,
    }
    json_path = REPO_ROOT / protocol["output_json"]
    csv_path = REPO_ROOT / protocol["output_csv"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(Path(args.protocol), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
