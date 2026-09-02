#!/usr/bin/env python3
"""Extract the three learned scalar residual coefficients from best checkpoints."""

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
from sarfusion.models.checkpoints import (  # noqa: E402
    resolve_local_wandb_checkpoint,
)
from sarfusion.models.rtdetr_fusion import ScalarResidualAlignment  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_scalar_alignment_control_audit.yaml"
)


def validate_protocol(protocol):
    if protocol.get("protocol_id") != "rtdetr_fam_scalar_alignment_control_audit_v1":
        raise ValueError("Unexpected scalar-alignment audit protocol_id")
    if protocol.get("checkpoint") != "best":
        raise ValueError("The scalar audit must use best checkpoints")
    if protocol.get("seeds") != [40, 41, 42, 43, 44]:
        raise ValueError("The scalar audit requires all five campaign seeds")
    if protocol.get("level_labels") != ["P3", "P4", "P5"]:
        raise ValueError("The scalar audit requires P3--P5")


def find_scalar_alignment_gates(model):
    gates = [
        module
        for module in model.modules()
        if isinstance(module, ScalarResidualAlignment)
    ]
    if len(gates) != 3:
        raise RuntimeError(f"Expected three scalar alignment gates, found {len(gates)}")
    return gates


def scalar_rows(seed, gates, level_labels):
    rows = []
    for level, (label, gate) in enumerate(zip(level_labels, gates)):
        rows.append(
            {
                "seed": seed,
                "level": level,
                "level_label": label,
                "logit": float(gate.logit.detach()),
                "alpha": float(gate.compute_alpha().detach()),
                "abs_delta_one": float(
                    (gate.compute_alpha().detach() - 1.0).abs()
                ),
            }
        )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "seed",
                "level",
                "level_label",
                "logit",
                "alpha",
                "abs_delta_one",
            ),
        )
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

    rows = []
    device = torch.device("cpu")
    for run_index, seed in enumerate(protocol["seeds"]):
        seed_config = load_run_config(config_path, run_index=run_index)
        model = load_fusion_model(
            seed_config["model"], checkpoints[seed], device
        )
        rows.extend(
            scalar_rows(
                seed,
                find_scalar_alignment_gates(model),
                protocol["level_labels"],
            )
        )
        del model

    result = {
        "protocol": protocol,
        "checkpoints": checkpoints,
        "scalar_rows": rows,
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
