#!/usr/bin/env python3
"""Audit learned reliability weights on the frozen Stage-A validation video."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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
from sarfusion.models.rtdetr_fusion import ReliabilityGatedFusion  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_reliability_gate_weight_audit.yaml"
)

EXPECTED_SEEDS_BY_PROTOCOL = {
    "rtdetr_fam_reliability_gate_weight_audit_v1": [40, 41, 42, 43, 44],
    "rtdetr_fam_reliability_gate_lr10x_weight_audit_seed40_v1": [40],
}


class WeightAccumulator:
    def __init__(self, bins, low_threshold, high_threshold):
        self.bins = int(bins)
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.total_abs_delta_one = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.low_count = 0
        self.high_count = 0
        self.histogram = torch.zeros(self.bins, dtype=torch.int64)

    def update(self, tensor):
        values = tensor.detach().float().reshape(-1)
        if values.numel() == 0 or not torch.isfinite(values).all():
            raise ValueError("Reliability weights must be non-empty and finite")
        if values.min() < 0.0 or values.max() > 2.0:
            raise ValueError("Reliability weights must lie in [0, 2]")
        self.count += int(values.numel())
        self.total += float(values.double().sum())
        self.total_sq += float(values.double().square().sum())
        self.total_abs_delta_one += float((values.double() - 1.0).abs().sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.low_count += int((values < self.low_threshold).sum())
        self.high_count += int((values > self.high_threshold).sum())
        histogram = torch.histc(values, bins=self.bins, min=0.0, max=2.0)
        self.histogram += histogram.cpu().round().to(torch.int64)

    def _quantile(self, probability):
        if self.count == 0:
            raise ValueError("Cannot summarize an empty accumulator")
        target = probability * (self.count - 1)
        index = int(
            torch.searchsorted(
                self.histogram.cumsum(0),
                torch.tensor(target, dtype=torch.float64),
                right=True,
            ).clamp(max=self.bins - 1)
        )
        return (index + 0.5) * (2.0 / self.bins)

    def summary(self):
        if self.count == 0:
            raise ValueError("Cannot summarize an empty accumulator")
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return {
            "numel": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "mean_abs_delta_one": self.total_abs_delta_one / self.count,
            "min": self.minimum,
            "p05_approx": self._quantile(0.05),
            "median_approx": self._quantile(0.5),
            "p95_approx": self._quantile(0.95),
            "max": self.maximum,
            "fraction_below_low": self.low_count / self.count,
            "fraction_above_high": self.high_count / self.count,
        }


def mask_modalities(pixel_values, mode):
    masked = pixel_values.clone()
    if mode == "fusion":
        return masked
    if mode == "rgb":
        masked[:, 3:].zero_()
        return masked
    if mode == "ir":
        masked[:, :3].zero_()
        return masked
    raise ValueError(f"Unknown modality mode {mode!r}")


def prepare_batch(batch, mode, device):
    """Read the mapping returned by the unprepared production DataLoader."""
    return (
        mask_modalities(batch["pixel_values"], mode).to(device),
        batch["pixel_mask"].to(device),
    )


def find_reliability_gates(model):
    gates = [
        module
        for module in model.modules()
        if isinstance(module, ReliabilityGatedFusion)
    ]
    if len(gates) != 3:
        raise RuntimeError(f"Expected three reliability gates, found {len(gates)}")
    return gates


class GateCapture:
    def __init__(self, gates, bins, low_threshold, high_threshold):
        self.accumulators = {}
        self.handles = []
        for level, gate in enumerate(gates):
            self.handles.append(
                gate.register_forward_hook(
                    self._hook(level, bins, low_threshold, high_threshold),
                    with_kwargs=True,
                )
            )

    def _hook(self, level, bins, low_threshold, high_threshold):
        self.accumulators[(level, "rgb")] = WeightAccumulator(
            bins, low_threshold, high_threshold
        )
        self.accumulators[(level, "ir")] = WeightAccumulator(
            bins, low_threshold, high_threshold
        )

        def capture(module, args, kwargs, output):
            rgb_weight, ir_weight = module.compute_weights(
                args[0],
                args[1],
                rgb_present=kwargs.get("rgb_present"),
                ir_present=kwargs.get("ir_present"),
            )
            self.accumulators[(level, "rgb")].update(rgb_weight)
            self.accumulators[(level, "ir")].update(ir_weight)

        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()


def validate_protocol(protocol):
    protocol_id = protocol.get("protocol_id")
    if protocol_id not in EXPECTED_SEEDS_BY_PROTOCOL:
        raise ValueError("Unexpected reliability-gate audit protocol_id")
    if protocol.get("checkpoint") != "best":
        raise ValueError("The audit must use the predeclared best checkpoint")
    if protocol.get("split") != "val":
        raise ValueError("The gate audit is restricted to validation")
    if protocol.get("modes") != ["fusion", "rgb", "ir"]:
        raise ValueError("The audit modes must be fusion, rgb, ir in that order")
    if protocol.get("seeds") != EXPECTED_SEEDS_BY_PROTOCOL[protocol_id]:
        raise ValueError("Unexpected seeds for the selected audit protocol")


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
                "logit_bias_rgb": float(gate.logit_conv.bias.detach()[0]),
                "logit_bias_ir": float(gate.logit_conv.bias.detach()[1]),
            }
        )
    return rows


def audit_mode(model, dataloader, mode, protocol, device):
    gates = find_reliability_gates(model)
    capture = GateCapture(
        gates,
        bins=protocol["histogram_bins"],
        low_threshold=protocol["low_weight_threshold"],
        high_threshold=protocol["high_weight_threshold"],
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

    if batches != int(protocol["expected_validation_batches"]):
        raise RuntimeError(f"Expected 75 validation batches, got {batches}")
    if frames != int(protocol["expected_validation_frames"]):
        raise RuntimeError(f"Expected 896 validation frames, got {frames}")
    return capture.accumulators


def summarize_across_seeds(rows):
    groups = {}
    for row in rows:
        key = (row["mode"], row["level"], row["modality"])
        groups.setdefault(key, []).append((row["seed"], row["mean"]))
    output = []
    for (mode, level, modality), seed_values in sorted(groups.items()):
        values = [value for _seed, value in sorted(seed_values)]
        output.append(
            {
                "mode": mode,
                "level": level,
                "modality": modality,
                "seed_means": {
                    str(seed): value for seed, value in sorted(seed_values)
                },
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values),
                "max": max(values),
            }
        )
    return output


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
        gates = find_reliability_gates(model)
        parameter_rows.extend(
            gate_parameter_rows(seed, gates, protocol["level_labels"])
        )
        for mode in protocol["modes"]:
            print(f"Auditing seed={seed} mode={mode}")
            accumulators = audit_mode(model, val_loader, mode, protocol, device)
            for (level, modality), accumulator in sorted(accumulators.items()):
                rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "level": level,
                        "level_label": protocol["level_labels"][level],
                        "modality": modality,
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
        "weight_rows": rows,
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
