#!/usr/bin/env python3
"""Run the frozen controlled synthetic IR-misalignment stress test.

Only the adapted IR channel is translated or scaled. RGB, VIS ground truth and
the pixel mask remain unchanged. Identity metrics are imported from the frozen
unused-acquisition confirmation, while every perturbed condition is saved as
an independently resumable raw JSON file.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fam_alignment_check import load_fusion_model  # noqa: E402
from sarfusion.utils.metrics import DetectionEvaluator, MetricCollection  # noqa: E402
from sarfusion.utils.structures import DataDict, WrapperModelOutput  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402
from scripts.run_rtdetr_carnation_stress_test import (  # noqa: E402
    SCALAR_METRICS,
    file_sha256,
    load_compatible_raw,
    stable_json_hash,
    summarize_values,
)
from scripts.run_rtdetr_fam_level_ablation import (  # noqa: E402
    jsonable,
    resolve_device,
    tensors_to_cpu,
)
from scripts.run_rtdetr_paired_modality_evaluation import (  # noqa: E402
    summarize_signed_values,
)
from scripts.run_rtdetr_unused_acquisition_confirmation import (  # noqa: E402
    EXPECTED_ACQUISITIONS,
    EXPECTED_CONFIGURATIONS,
    EXPECTED_SEEDS,
    build_loader,
    build_manifests,
    build_run,
    ensure_attestation_ready,
    load_payload as load_confirmation_payload,
    resolve_checkpoints,
    resolve_repo_path,
    set_evaluation_seed,
    verify_closed_stage_b,
)


DEFAULT_PROTOCOL = "parameters/RTDETR/rtdetr_synthetic_geometric_stress.yaml"
PROTOCOL_ID = "rtdetr_synthetic_geometric_stress_v1"
EXPECTED_MAGNITUDES = [8, 16, 32]
EXPECTED_DIRECTIONS = (
    ("right", 1, 0),
    ("left", -1, 0),
    ("down", 0, 1),
    ("up", 0, -1),
)
EXPECTED_SCALES = [0.9, 1.1]
EXPECTED_COMPARISONS = {
    "historical_fam_vs_additive": ("historical_fam", "historical_additive"),
    "stage_b_rcra_vs_fam": ("stage_b_rcra", "stage_b_fam"),
}


def transformation_id(kind, *, magnitude=None, direction=None, scale=None):
    if kind == "identity":
        return "identity"
    if kind == "translation":
        return f"translate_{int(magnitude):02d}_{direction}"
    if kind == "scale":
        return f"scale_{int(round(float(scale) * 100)):03d}"
    raise ValueError(f"Unknown transformation kind {kind}")


def expand_transformations(protocol):
    settings = protocol["transformations"]
    identity = {
        "id": "identity",
        "kind": "identity",
        "dx_px": 0,
        "dy_px": 0,
        "scale": 1.0,
        "magnitude_px": 0,
        "direction": "identity",
        "source": "frozen_confirmation_csv",
    }
    transformations = [identity]
    for magnitude in settings["translation_magnitudes_px"]:
        for direction in settings["translation_directions"]:
            dx = int(magnitude) * int(direction["dx_sign"])
            dy = int(magnitude) * int(direction["dy_sign"])
            transformations.append(
                {
                    "id": transformation_id(
                        "translation",
                        magnitude=magnitude,
                        direction=direction["id"],
                    ),
                    "kind": "translation",
                    "dx_px": dx,
                    "dy_px": dy,
                    "scale": 1.0,
                    "magnitude_px": int(magnitude),
                    "direction": direction["id"],
                    "source": "synthetic_inference",
                }
            )
    for scale in settings["scales"]:
        transformations.append(
            {
                "id": transformation_id("scale", scale=scale),
                "kind": "scale",
                "dx_px": 0,
                "dy_px": 0,
                "scale": float(scale),
                "magnitude_px": None,
                "direction": None,
                "source": "synthetic_inference",
            }
        )
    return transformations


def load_protocol(path):
    payload = load_yaml(path)
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Synthetic-stress YAML requires a protocol mapping")
    if protocol.get("id") != PROTOCOL_ID:
        raise ValueError("Unexpected synthetic geometric stress protocol id")
    if protocol.get("status") != "frozen_before_perturbed_inference":
        raise ValueError("Synthetic geometric stress must remain frozen")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("Synthetic stress must contain seeds 40--44 in order")
    if tuple(protocol.get("acquisitions", [])) != EXPECTED_ACQUISITIONS:
        raise ValueError("Synthetic stress acquisition set or order changed")
    if tuple(protocol.get("configurations", {})) != EXPECTED_CONFIGURATIONS:
        raise ValueError("Synthetic stress configuration set or order changed")
    if protocol.get("checkpoint") != "latest":
        raise ValueError("Synthetic stress must use latest checkpoints")
    if float(protocol.get("confidence_threshold", -1.0)) != 0.01:
        raise ValueError("Synthetic stress confidence threshold changed")

    transforms = protocol.get("transformations", {})
    if transforms.get("input_space") != "post_preprocessing_model_tensor_640x640":
        raise ValueError("Synthetic transform input space changed")
    if transforms.get("target") != "ir_channel_3_only":
        raise ValueError("Synthetic stress must transform only IR channel 3")
    if transforms.get("rgb") != "unchanged":
        raise ValueError("RGB must remain unchanged")
    if transforms.get("pixel_mask") != "unchanged":
        raise ValueError("Pixel mask must remain unchanged")
    if transforms.get("ground_truth") != "unchanged_vis":
        raise ValueError("VIS ground truth must remain unchanged")
    if transforms.get("interpolation") != "bilinear":
        raise ValueError("Synthetic interpolation changed")
    if float(transforms.get("fill_value", 1.0)) != 0.0:
        raise ValueError("Synthetic border fill must remain normalized zero")
    if transforms.get("scale_center") != "image_center":
        raise ValueError("Synthetic scales must remain center-based")
    if transforms.get("composition") != "one_perturbation_at_a_time_no_cartesian_product":
        raise ValueError("Translation and scale must not be composed")
    if transforms.get("translation_magnitudes_px") != EXPECTED_MAGNITUDES:
        raise ValueError("Translation magnitudes changed")
    directions = tuple(
        (row.get("id"), row.get("dx_sign"), row.get("dy_sign"))
        for row in transforms.get("translation_directions", [])
    )
    if directions != EXPECTED_DIRECTIONS:
        raise ValueError("Translation directions changed")
    if transforms.get("scales") != EXPECTED_SCALES:
        raise ValueError("Synthetic scales changed")
    expanded = expand_transformations(protocol)
    if len(expanded) != int(transforms.get("expected_condition_count_including_identity", -1)):
        raise ValueError("Unexpected number of synthetic transformations")
    if len({row["id"] for row in expanded}) != len(expanded):
        raise ValueError("Synthetic transformation identifiers are not unique")

    actual_comparisons = {
        name: (
            settings.get("candidate_configuration"),
            settings.get("reference_configuration"),
        )
        for name, settings in protocol.get("paired_robustness_comparisons", {}).items()
    }
    if actual_comparisons != EXPECTED_COMPARISONS:
        raise ValueError("Synthetic robustness comparisons changed")
    interpretation = protocol.get("interpretation", {})
    if interpretation.get("negative_results_must_be_reported") is not True:
        raise ValueError("Negative synthetic-stress results must be reported")
    if any(
        interpretation.get(key) is not False
        for key in interpretation
        if key != "negative_results_must_be_reported"
    ):
        raise ValueError("A forbidden synthetic-stress interpretation was enabled")
    return protocol


def load_frozen_sources(protocol, verify_hashes=True):
    source = protocol["confirmation_source"]
    confirmation_path = resolve_repo_path(source["protocol"])
    confirmation_payload = load_confirmation_payload(
        confirmation_path, require_frozen_values=verify_hashes
    )
    confirmation_protocol = confirmation_payload["protocol"]
    confirmation_hash = stable_json_hash(confirmation_protocol)
    if confirmation_hash != source["expected_protocol_sha256"]:
        raise RuntimeError("Confirmation scientific protocol hash changed")
    description = ensure_attestation_ready(confirmation_payload["attestation"])
    if description != source["required_dataset_description"]:
        raise RuntimeError("Confirmation dataset description changed")

    identity_csv = resolve_repo_path(source["identity_results_csv"])
    identity_hash = file_sha256(identity_csv)
    if verify_hashes and identity_hash != source["expected_identity_results_sha256"]:
        raise RuntimeError("Frozen identity-results CSV hash changed")
    return confirmation_payload, confirmation_hash, identity_csv, identity_hash


def load_identity_rows(
    protocol,
    confirmation_protocol,
    identity_csv,
    identity_hash,
    manifests,
    checkpoints,
):
    with Path(identity_csv).open(newline="", encoding="utf-8") as input_file:
        csv_rows = list(csv.DictReader(input_file))
    rows = []
    for row in csv_rows:
        if row["configuration"] not in EXPECTED_CONFIGURATIONS:
            continue
        if row["condition"] != "vis_ir":
            continue
        acquisition = row["acquisition"]
        seed = int(row["seed"])
        configuration = row["configuration"]
        if acquisition not in EXPECTED_ACQUISITIONS or seed not in EXPECTED_SEEDS:
            raise RuntimeError("Unexpected identity source row")
        checkpoint = checkpoints[(configuration, seed)]
        if row["checkpoint_sha256"] != checkpoint["sha256"]:
            raise RuntimeError("Identity row checkpoint differs from frozen checkpoint")
        expected_samples = manifests[acquisition]["common_frames"]
        if int(row["n_samples"]) != expected_samples:
            raise RuntimeError("Identity row sample count differs from source inventory")
        rows.append(
            {
                "schema_version": 1,
                "protocol_id": protocol["id"],
                "protocol_sha256": stable_json_hash(protocol),
                "source_confirmation_protocol_sha256": stable_json_hash(
                    confirmation_protocol
                ),
                "source_identity_results_sha256": identity_hash,
                "acquisition": acquisition,
                "source_inventory_sha256": manifests[acquisition]["inventory_sha256"],
                "configuration": configuration,
                "family": protocol["configurations"][configuration]["family"],
                "seed": seed,
                "transformation": expand_transformations(protocol)[0],
                "checkpoint": str(checkpoint["path"]),
                "checkpoint_sha256": checkpoint["sha256"],
                "batch_size": int(protocol["batch_size"]),
                "confidence_threshold": float(protocol["confidence_threshold"]),
                "n_samples": expected_samples,
                "identity_source": str(identity_csv),
                "metrics": {
                    metric: float(row[metric]) if row[metric] else None
                    for metric in SCALAR_METRICS
                },
            }
        )
    expected = {
        (acquisition, configuration, seed)
        for acquisition in EXPECTED_ACQUISITIONS
        for configuration in EXPECTED_CONFIGURATIONS
        for seed in EXPECTED_SEEDS
    }
    actual = {
        (row["acquisition"], row["configuration"], row["seed"])
        for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("Identity CSV does not contain exactly 40 frozen VIS+IR rows")
    return rows


def apply_ir_transformation(pixel_values, transformation):
    if pixel_values.ndim != 4 or pixel_values.shape[1] != 4:
        raise ValueError(f"Expected Bx4xHxW input, received {tuple(pixel_values.shape)}")
    if transformation["kind"] == "identity":
        return pixel_values
    transformed = pixel_values.clone()
    transformed[:, 3:4] = tvf.affine(
        pixel_values[:, 3:4],
        angle=0.0,
        translate=[int(transformation["dx_px"]), int(transformation["dy_px"])],
        scale=float(transformation["scale"]),
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
        center=None,
    )
    return transformed


def evaluate_transformation(model, loader, device, transformation, max_batches=None):
    evaluator = DetectionEvaluator(
        MetricCollection({}), id2class=loader.dataset.id2class
    )
    n_samples = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            base_pixels = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device, non_blocking=True)
            output = model(
                pixel_values=apply_ir_transformation(base_pixels, transformation),
                pixel_mask=pixel_mask,
            )
            evaluator.update(
                DataDict(labels=tensors_to_cpu(batch["labels"])),
                WrapperModelOutput(
                    predictions=tensors_to_cpu(output["predictions"])
                ),
            )
            n_samples += len(batch["labels"])
    return {"metrics": jsonable(evaluator.compute()), "n_samples": n_samples}


def raw_result_path(output_dir, acquisition, configuration, seed, transform_id):
    return (
        output_dir
        / "raw"
        / f"{acquisition}_{configuration}_seed_{seed}_{transform_id}.json"
    )


def _identity_index(rows):
    return {
        (row["acquisition"], row["configuration"], row["seed"]): float(
            row["metrics"]["map_50"]
        )
        for row in rows
        if row["transformation"]["id"] == "identity"
    }


def add_identity_normalized_values(rows):
    identities = _identity_index(rows)
    output = []
    for row in rows:
        key = (row["acquisition"], row["configuration"], row["seed"])
        identity = identities.get(key)
        if identity is None or identity <= 0.0:
            raise RuntimeError(f"Missing positive identity mAP@50 for {key}")
        value = float(row["metrics"]["map_50"])
        enriched = dict(row)
        enriched["identity_map_50"] = identity
        enriched["absolute_delta_map_50"] = value - identity
        enriched["relative_drop_map_50"] = (identity - value) / identity
        output.append(enriched)
    return output


def _selected(rows, *, acquisition=None, configuration=None, transform_id=None):
    return [
        row
        for row in rows
        if (acquisition is None or row["acquisition"] == acquisition)
        and (configuration is None or row["configuration"] == configuration)
        and (
            transform_id is None
            or row["transformation"]["id"] == transform_id
        )
    ]


def _seed_map(rows, field):
    mapped = {str(row["seed"]): float(row[field]) for row in rows}
    if set(mapped) != {str(seed) for seed in EXPECTED_SEEDS}:
        raise RuntimeError("A five-seed curve point is incomplete")
    return mapped


def _summarize_seed_map(values):
    return {
        "seed_values": values,
        "summary": summarize_signed_values(values.values()),
    }


def build_aggregate(
    payloads,
    protocol,
    protocol_hash,
    manifests,
    confirmation_hash,
    identity_hash,
    output_dir,
):
    rows = add_identity_normalized_values(payloads)
    rows.sort(
        key=lambda row: (
            row["acquisition"],
            row["configuration"],
            row["seed"],
            row["transformation"]["id"],
        )
    )
    transformations = expand_transformations(protocol)
    transform_by_id = {row["id"]: row for row in transformations}
    expected_jobs = {
        (acquisition, configuration, seed, transformation["id"])
        for acquisition in EXPECTED_ACQUISITIONS
        for configuration in EXPECTED_CONFIGURATIONS
        for seed in EXPECTED_SEEDS
        for transformation in transformations
    }
    actual_jobs = {
        (
            row["acquisition"],
            row["configuration"],
            row["seed"],
            row["transformation"]["id"],
        )
        for row in rows
    }
    if len(actual_jobs) != len(rows):
        raise RuntimeError("Synthetic aggregate contains duplicate jobs")
    complete = actual_jobs == expected_jobs

    per_transform = {}
    for acquisition in EXPECTED_ACQUISITIONS:
        per_transform[acquisition] = {}
        for configuration in EXPECTED_CONFIGURATIONS:
            per_transform[acquisition][configuration] = {}
            for transformation in transformations:
                selected = _selected(
                    rows,
                    acquisition=acquisition,
                    configuration=configuration,
                    transform_id=transformation["id"],
                )
                if len(selected) != len(EXPECTED_SEEDS):
                    continue
                per_transform[acquisition][configuration][transformation["id"]] = {
                    "transformation": transformation,
                    "map_50": summarize_values(
                        [row["metrics"]["map_50"] for row in selected]
                    ),
                    "absolute_delta_map_50": _summarize_seed_map(
                        _seed_map(selected, "absolute_delta_map_50")
                    ),
                    "relative_drop_map_50": _summarize_seed_map(
                        _seed_map(selected, "relative_drop_map_50")
                    ),
                }

    direction_pooled_translation = {}
    for acquisition in EXPECTED_ACQUISITIONS:
        direction_pooled_translation[acquisition] = {}
        for configuration in EXPECTED_CONFIGURATIONS:
            direction_pooled_translation[acquisition][configuration] = {}
            identity_rows = _selected(
                rows,
                acquisition=acquisition,
                configuration=configuration,
                transform_id="identity",
            )
            identity_values = {str(seed): 0.0 for seed in EXPECTED_SEEDS}
            direction_pooled_translation[acquisition][configuration]["0"] = (
                _summarize_seed_map(identity_values)
            )
            for magnitude in EXPECTED_MAGNITUDES:
                values = {}
                for seed in EXPECTED_SEEDS:
                    selected = [
                        row
                        for row in rows
                        if row["acquisition"] == acquisition
                        and row["configuration"] == configuration
                        and row["seed"] == seed
                        and row["transformation"]["kind"] == "translation"
                        and row["transformation"]["magnitude_px"] == magnitude
                    ]
                    if len(selected) != 4:
                        break
                    values[str(seed)] = statistics.fmean(
                        row["relative_drop_map_50"] for row in selected
                    )
                if len(values) == len(EXPECTED_SEEDS):
                    direction_pooled_translation[acquisition][configuration][
                        str(magnitude)
                    ] = _summarize_seed_map(values)
            del identity_rows

    acquisition_macro = {}
    for configuration in EXPECTED_CONFIGURATIONS:
        acquisition_macro[configuration] = {}
        for transformation in transformations:
            values = {}
            for seed in EXPECTED_SEEDS:
                selected = [
                    row
                    for row in rows
                    if row["configuration"] == configuration
                    and row["seed"] == seed
                    and row["transformation"]["id"] == transformation["id"]
                ]
                if len(selected) != len(EXPECTED_ACQUISITIONS):
                    break
                values[str(seed)] = statistics.fmean(
                    row["relative_drop_map_50"] for row in selected
                )
            if len(values) == len(EXPECTED_SEEDS):
                acquisition_macro[configuration][transformation["id"]] = {
                    "transformation": transformation,
                    **_summarize_seed_map(values),
                }

    robustness = {}
    for name, comparison in protocol["paired_robustness_comparisons"].items():
        candidate = comparison["candidate_configuration"]
        reference = comparison["reference_configuration"]
        robustness[name] = {}
        for acquisition in EXPECTED_ACQUISITIONS:
            robustness[name][acquisition] = {}
            for transformation in transformations:
                advantages = {}
                for seed in EXPECTED_SEEDS:
                    reference_rows = [
                        row
                        for row in rows
                        if row["acquisition"] == acquisition
                        and row["configuration"] == reference
                        and row["seed"] == seed
                        and row["transformation"]["id"] == transformation["id"]
                    ]
                    candidate_rows = [
                        row
                        for row in rows
                        if row["acquisition"] == acquisition
                        and row["configuration"] == candidate
                        and row["seed"] == seed
                        and row["transformation"]["id"] == transformation["id"]
                    ]
                    if len(reference_rows) != 1 or len(candidate_rows) != 1:
                        break
                    # Positive means the candidate loses a smaller fraction of
                    # its own identity mAP@50 and is therefore more tolerant.
                    advantages[str(seed)] = (
                        reference_rows[0]["relative_drop_map_50"]
                        - candidate_rows[0]["relative_drop_map_50"]
                    )
                if len(advantages) == len(EXPECTED_SEEDS):
                    robustness[name][acquisition][transformation["id"]] = {
                        "transformation": transformation,
                        "candidate_robustness_advantage": _summarize_seed_map(
                            advantages
                        ),
                    }

    aggregate = {
        "schema_version": 1,
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "protocol_complete": complete,
        "expected_job_count_including_reused_identity": len(expected_jobs),
        "observed_job_count_including_reused_identity": len(rows),
        "new_perturbed_inference_job_count": sum(
            row["transformation"]["id"] != "identity" for row in rows
        ),
        "source_confirmation_protocol_sha256": confirmation_hash,
        "source_identity_results_sha256": identity_hash,
        "experimental_unit": "checkpoint/seed within acquisition",
        "frame_level_independence_claim_allowed": False,
        "source_manifests": {
            name: {key: value for key, value in manifest.items() if key != "rows"}
            for name, manifest in manifests.items()
        },
        "transformations": transformations,
        "interpretation_constraints": protocol["interpretation"],
        "results": rows,
        "per_transformation_across_seed_summaries": per_transform,
        "direction_pooled_translation_relative_drop": direction_pooled_translation,
        "equal_weight_acquisition_macro_relative_drop": acquisition_macro,
        "paired_robustness_comparisons": robustness,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rtdetr_synthetic_geometric_stress.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(jsonable(aggregate), output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    csv_path = output_dir / "rtdetr_synthetic_geometric_stress.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        fields = [
            "acquisition",
            "configuration",
            "family",
            "seed",
            "transformation",
            "kind",
            "magnitude_px",
            "direction",
            "dx_px",
            "dy_px",
            "scale",
            "checkpoint_sha256",
            "n_samples",
            *SCALAR_METRICS,
            "identity_map_50",
            "absolute_delta_map_50",
            "relative_drop_map_50",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            transformation = row["transformation"]
            writer.writerow(
                {
                    "acquisition": row["acquisition"],
                    "configuration": row["configuration"],
                    "family": row["family"],
                    "seed": row["seed"],
                    "transformation": transformation["id"],
                    "kind": transformation["kind"],
                    "magnitude_px": transformation["magnitude_px"],
                    "direction": transformation["direction"],
                    "dx_px": transformation["dx_px"],
                    "dy_px": transformation["dy_px"],
                    "scale": transformation["scale"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "n_samples": row["n_samples"],
                    **{metric: row["metrics"].get(metric) for metric in SCALAR_METRICS},
                    "identity_map_50": row["identity_map_50"],
                    "absolute_delta_map_50": row["absolute_delta_map_50"],
                    "relative_drop_map_50": row["relative_drop_map_50"],
                }
            )
    print(f"Saved aggregate: {json_path}")
    print(f"Saved table: {csv_path}")
    return aggregate


def build_curves(aggregate, protocol, output_dir):
    if not aggregate["protocol_complete"]:
        return None
    import matplotlib.pyplot as plt

    labels = {
        name: settings["curve_label"]
        for name, settings in protocol["configurations"].items()
    }
    colors = {
        "historical_additive": "#7f7f7f",
        "historical_fam": "#1f77b4",
        "stage_b_fam": "#2ca02c",
        "stage_b_rcra": "#d62728",
    }
    row_names = [*EXPECTED_ACQUISITIONS, "equal-weight macro"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for row_index, row_name in enumerate(row_names):
        translation_axis, scale_axis = axes[row_index]
        for configuration in EXPECTED_CONFIGURATIONS:
            if row_name == "equal-weight macro":
                macro = aggregate["equal_weight_acquisition_macro_relative_drop"][
                    configuration
                ]
                translation_points = {}
                for magnitude in [0, *EXPECTED_MAGNITUDES]:
                    if magnitude == 0:
                        transform_ids = ["identity"]
                    else:
                        transform_ids = [
                            transformation_id(
                                "translation", magnitude=magnitude, direction=direction
                            )
                            for direction, _dx, _dy in EXPECTED_DIRECTIONS
                        ]
                    seed_values = {
                        str(seed): statistics.fmean(
                            macro[transform_id]["seed_values"][str(seed)]
                            for transform_id in transform_ids
                        )
                        for seed in EXPECTED_SEEDS
                    }
                    translation_points[str(magnitude)] = _summarize_seed_map(
                        seed_values
                    )
                scale_points = {
                    0.9: macro["scale_090"],
                    1.0: macro["identity"],
                    1.1: macro["scale_110"],
                }
            else:
                translation_points = aggregate[
                    "direction_pooled_translation_relative_drop"
                ][row_name][configuration]
                per_transform = aggregate[
                    "per_transformation_across_seed_summaries"
                ][row_name][configuration]
                scale_points = {
                    0.9: per_transform["scale_090"]["relative_drop_map_50"],
                    1.0: per_transform["identity"]["relative_drop_map_50"],
                    1.1: per_transform["scale_110"]["relative_drop_map_50"],
                }

            x_translation = [0, *EXPECTED_MAGNITUDES]
            y_translation = [
                100.0 * translation_points[str(value)]["summary"]["mean"]
                for value in x_translation
            ]
            translation_axis.plot(
                x_translation,
                y_translation,
                marker="o",
                label=labels[configuration],
                color=colors[configuration],
            )
            x_scale = [0.9, 1.0, 1.1]
            y_scale = [
                100.0 * scale_points[value]["summary"]["mean"]
                for value in x_scale
            ]
            scale_axis.plot(
                x_scale,
                y_scale,
                marker="o",
                label=labels[configuration],
                color=colors[configuration],
            )
        translation_axis.axhline(0.0, color="black", linewidth=0.8)
        scale_axis.axhline(0.0, color="black", linewidth=0.8)
        translation_axis.set_title(f"{row_name}: translation (4-direction mean)")
        scale_axis.set_title(f"{row_name}: centered scale")
        translation_axis.set_xlabel("Translation magnitude [input pixels]")
        scale_axis.set_xlabel("IR scale")
        translation_axis.set_ylabel("Relative mAP@50 drop [%]")
        scale_axis.set_ylabel("Relative mAP@50 drop [%]")
        translation_axis.grid(alpha=0.25)
        scale_axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure_path = output_dir / "rtdetr_synthetic_geometric_stress_curves.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Saved figure: {figure_path}")
    return figure_path


def _parse_selection(values, allowed, description):
    if values is None:
        return list(allowed)
    unexpected = set(values) - set(allowed)
    if unexpected:
        raise ValueError(f"Unexpected {description}: {sorted(unexpected)}")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--configurations", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--acquisitions", nargs="+")
    parser.add_argument("--transformations", nargs="+")
    args = parser.parse_args()

    protocol = load_protocol(resolve_repo_path(args.protocol))
    protocol_hash = stable_json_hash(protocol)
    (
        confirmation_payload,
        confirmation_hash,
        identity_csv,
        identity_hash,
    ) = load_frozen_sources(protocol, verify_hashes=True)
    confirmation_protocol = confirmation_payload["protocol"]
    if float(protocol["confidence_threshold"]) != float(
        confirmation_protocol["confidence_threshold"]
    ):
        raise RuntimeError("Synthetic and identity confidence thresholds differ")
    manifests = build_manifests(confirmation_protocol, verify_expected=True)
    stage_b_protocol, stage_b_runs, selection_decision = verify_closed_stage_b(
        confirmation_protocol
    )
    del stage_b_protocol
    if selection_decision["status"] != "fail_retain_fam":
        raise RuntimeError("Closed Stage-B decision changed")
    checkpoints = resolve_checkpoints(
        confirmation_protocol, stage_b_runs, verify_hashes=True
    )
    identity_rows = load_identity_rows(
        protocol,
        confirmation_protocol,
        identity_csv,
        identity_hash,
        manifests,
        checkpoints,
    )
    transformations = expand_transformations(protocol)
    transform_by_id = {row["id"]: row for row in transformations}
    perturbed_ids = [row["id"] for row in transformations if row["id"] != "identity"]

    loaders = {
        acquisition: build_loader(
            confirmation_protocol,
            manifests[acquisition],
            batch_size=args.batch_size or protocol["batch_size"],
            workers=protocol["workers"] if args.workers is None else args.workers,
        )
        for acquisition in EXPECTED_ACQUISITIONS
    }
    if args.prepare_only:
        print(
            f"Prepare-only OK: 2 inventories, 40 identity rows and "
            f"{len(transformations)} frozen transformations"
        )
        return
    if args.dry_run:
        print(
            "Dry run OK: 20 checkpoint hashes, 40 reused identity jobs and "
            "560 perturbed inference jobs"
        )
        return

    configurations = _parse_selection(
        args.configurations, EXPECTED_CONFIGURATIONS, "configurations"
    )
    seeds = _parse_selection(args.seeds, EXPECTED_SEEDS, "seeds")
    acquisitions = _parse_selection(
        args.acquisitions, EXPECTED_ACQUISITIONS, "acquisitions"
    )
    selected_transformations = _parse_selection(
        args.transformations, perturbed_ids, "perturbed transformations"
    )
    is_full_selection = (
        configurations == list(EXPECTED_CONFIGURATIONS)
        and seeds == EXPECTED_SEEDS
        and acquisitions == list(EXPECTED_ACQUISITIONS)
        and selected_transformations == perturbed_ids
        and args.max_batches is None
    )
    output_dir = resolve_repo_path(args.output_dir or protocol["output_dir"])
    if not is_full_selection and args.output_dir is None:
        raise ValueError("Partial/smoke runs require an explicit separate --output-dir")
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_evaluation_seed(int(protocol["evaluation_seed"]))
    payloads = list(identity_rows) if is_full_selection else []

    for configuration in configurations:
        for seed in seeds:
            checkpoint = checkpoints[(configuration, seed)]
            missing = {}
            for acquisition in acquisitions:
                for transform_id in selected_transformations:
                    transformation = transform_by_id[transform_id]
                    expected = {
                        "protocol_id": protocol["id"],
                        "protocol_sha256": protocol_hash,
                        "source_confirmation_protocol_sha256": confirmation_hash,
                        "source_identity_results_sha256": identity_hash,
                        "acquisition": acquisition,
                        "source_inventory_sha256": manifests[acquisition][
                            "inventory_sha256"
                        ],
                        "configuration": configuration,
                        "family": protocol["configurations"][configuration]["family"],
                        "seed": seed,
                        "transformation": transformation,
                        "checkpoint": str(checkpoint["path"]),
                        "checkpoint_sha256": checkpoint["sha256"],
                        "batch_size": int(args.batch_size or protocol["batch_size"]),
                        "confidence_threshold": float(protocol["confidence_threshold"]),
                        "max_batches": args.max_batches,
                    }
                    path = raw_result_path(
                        output_dir, acquisition, configuration, seed, transform_id
                    )
                    if path.is_file() and not args.force:
                        payloads.append(load_compatible_raw(path, expected))
                        print(f"[skip] {path}")
                    else:
                        missing.setdefault(acquisition, []).append((transformation, expected))
            if not missing:
                continue

            run = build_run(
                confirmation_protocol, configuration, seed, stage_b_runs
            )
            print(f"[load] configuration={configuration} seed={seed} device={device}")
            model = load_fusion_model(run["model"], checkpoint["path"], device)
            for acquisition, jobs in missing.items():
                for transformation, expected in jobs:
                    print(
                        f"[run] acquisition={acquisition} configuration={configuration} "
                        f"seed={seed} transformation={transformation['id']}"
                    )
                    measured = evaluate_transformation(
                        model,
                        loaders[acquisition],
                        device,
                        transformation,
                        max_batches=args.max_batches,
                    )
                    payload_row = {
                        **expected,
                        "schema_version": 1,
                        "protocol_complete": args.max_batches is None,
                        "ground_truth": "vis",
                        "n_dataset_images": len(loaders[acquisition].dataset),
                        "n_samples": measured["n_samples"],
                        "training_summary": checkpoint["training_summary"],
                        "metrics": measured["metrics"],
                    }
                    path = raw_result_path(
                        output_dir,
                        acquisition,
                        configuration,
                        seed,
                        transformation["id"],
                    )
                    with path.open("w", encoding="utf-8") as output_file:
                        json.dump(jsonable(payload_row), output_file, indent=2, sort_keys=True)
                        output_file.write("\n")
                    payloads.append(payload_row)
                    print(
                        f"[done] {acquisition}/{configuration}/seed={seed}/"
                        f"{transformation['id']} map50="
                        f"{measured['metrics']['map_50']:.6f}"
                    )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not is_full_selection:
        print(f"Partial technical run complete: {len(payloads)} perturbed results")
        return
    aggregate = build_aggregate(
        payloads,
        protocol,
        protocol_hash,
        manifests,
        confirmation_hash,
        identity_hash,
        output_dir,
    )
    build_curves(aggregate, protocol, output_dir)
    if aggregate["protocol_complete"]:
        print("Protocol complete: all 600 frozen curve points are present")


if __name__ == "__main__":
    main()
