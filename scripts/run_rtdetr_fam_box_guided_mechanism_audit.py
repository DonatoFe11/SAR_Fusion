#!/usr/bin/env python3
"""Audit the learned P3 guidance flow of the box-guided RT-DETR FAM.

The audit replays the frozen Stage-A training acquisitions in fusion mode,
with Modal Dropout disabled, and evaluates the cached common guidance flow at
the conservative VIS/IR box matches used by the auxiliary training loss.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import struct
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from PIL import Image
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sarfusion.data import get_dataloaders  # noqa: E402
from sarfusion.data.temporal_split import stable_json_hash  # noqa: E402
from sarfusion.data.wisard import (  # noqa: E402
    MULTI_MODALITY_ITEM,
    build_box_alignment_targets,
    build_wisard_items,
    load_annotations,
    yolo_to_coco_annotations,
)
from sarfusion.experiment.box_guided_alignment import (  # noqa: E402
    find_box_guided_fam,
)
from sarfusion.models import build_model  # noqa: E402
from sarfusion.models.checkpoints import (  # noqa: E402
    resolve_local_wandb_checkpoint,
)
from sarfusion.utils.grid import make_grid  # noqa: E402
from sarfusion.utils.reproducibility import (  # noqa: E402
    verify_training_source_manifest,
    verify_training_source_runtime_trace,
)
from sarfusion.utils.utils import load_yaml  # noqa: E402


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_box_guided_mechanism_audit_seed40.yaml"
)
PROTOCOL_ID = "rtdetr_fam_box_guided_mechanism_audit_v1"
FAM_VARIANT = "box_guided_common_offset_p3"
EXPECTED_PROJECT = "RTDETR_FAM_BoxGuided_SequenceVal_Seed40"
EXPECTED_TRAINING_CONFIG_SHA256 = (
    "16c1d58a4926bb6f6d1c018eb96600b52279116ca04a42d35286d63fb12ea647"
)
EXPECTED_SOURCE_INVENTORY_SHA256 = (
    "f889b5a54115f0267e0d5c087e6c3673bd2c65f63607e0c01df063620ea76a1e"
)
EXPECTED_TARGET_POPULATION_SHA256 = (
    "d519574962e81ae5b492248113247cca20d7ef15b2d189d1e3b58aebf218f3c0"
)
EXPECTED_MATCH_DISTRIBUTION = {0: 817, 1: 801, 2: 543, 3: 526, 4: 436}
EXPECTED_DIAGNOSTIC_BASELINES = {
    "zero_flow": "fixed_zero_dy_dx",
    "constant_flow": "best_smooth_l1_constant_on_frozen_train_targets",
    "constant_comparison_is_promotion_gate": False,
    "positive_improvement_interpretation": (
        "non_global_fit_on_frozen_train_targets_only"
    ),
    "does_not_establish": "input_conditioned_generalization",
}
EXPECTED_MECHANISM_GATE = {
    "minimum_guidance_relative_improvement_vs_zero": 0.2,
    "minimum_guidance_mean_abs_cells": 0.05,
    "maximum_guidance_saturation_fraction": 0.01,
    "minimum_total_relative_improvement_vs_zero": 0.0,
    "maximum_mean_positive_cancellation_ratio": 0.5,
    "maximum_fraction_guidance_at_least_half_cancelled": 0.5,
    "if_pass": "box_guidance_mechanism_supported",
    "if_fail": "box_guidance_mechanism_not_supported",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
WANDB_METADATA_KEYS = {"_wandb", "experiment", "wandb_version"}
TARGET_DIGEST_HEADER = (
    b"WISARD_BOX_ALIGNMENT_TARGETS_V1\n"
    b"mutual_nearest<=0.05\n"
    b"float32_le[x,y,dy,dx]\n"
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value, description):
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")


def _canonical_json_value(value):
    """Normalize YAML integer keys and tuples before identity comparison."""
    if isinstance(value, dict):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    return value


def scientific_config_digest(config):
    return stable_json_hash(_canonical_json_value(config))


def verify_training_config_file(config_path, expected_sha256):
    _require_sha256(expected_sha256, "training_config_sha256")
    actual = file_sha256(config_path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Training YAML hash changed: expected {expected_sha256}, got {actual}"
        )
    return actual


def validate_protocol(protocol):
    """Validate the predeclared mechanism-audit contract."""
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected box-guided mechanism audit protocol_id")
    if protocol.get("project") != EXPECTED_PROJECT:
        raise ValueError("The audit project differs from the frozen candidate")
    if protocol.get("checkpoint") != "best":
        raise ValueError("The audit must use the predeclared best checkpoint")
    if protocol.get("split") != "train":
        raise ValueError("The box-guided mechanism audit is restricted to train")
    if protocol.get("mode") != "fusion":
        raise ValueError("The box-guided mechanism audit requires fusion mode")
    if not isinstance(protocol.get("training_config"), str):
        raise ValueError("training_config must be a path string")
    _require_sha256(
        protocol.get("training_config_sha256"), "training_config_sha256"
    )
    if protocol["training_config_sha256"] != EXPECTED_TRAINING_CONFIG_SHA256:
        raise ValueError("The frozen training-config digest changed")

    seeds = protocol.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty YAML list")
    try:
        normalized_seeds = [int(seed) for seed in seeds]
    except (TypeError, ValueError) as error:
        raise ValueError("Every audit seed must be an integer") from error
    if normalized_seeds != seeds or len(set(normalized_seeds)) != len(seeds):
        raise ValueError("Audit seeds must be unique integers")
    if normalized_seeds != [40]:
        raise ValueError("This audit protocol is frozen for seed 40 only")

    matching = protocol.get("box_matching") or {}
    if matching.get("method") != "mutual_nearest_box_center":
        raise ValueError("The audit requires mutual-nearest box-center matches")
    if not math.isclose(
        float(matching.get("max_distance_normalized", -1.0)),
        0.05,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("The frozen box-match distance must be 0.05")

    if protocol.get("diagnostic_baselines") != EXPECTED_DIAGNOSTIC_BASELINES:
        raise ValueError("Unexpected diagnostic-baseline contract")

    beta = float(protocol.get("smooth_l1_beta_cells", 0.0))
    limit = float(protocol.get("guidance_limit_cells", 0.0))
    saturation = float(protocol.get("near_saturation_threshold_cells", -1.0))
    if (beta, limit, saturation) != (0.25, 4.0, 3.9):
        raise ValueError("The frozen loss/bound/saturation settings changed")

    expected_frames = int(protocol.get("expected_train_frames", 0))
    batch_size = int(protocol.get("audit_batch_size", 0))
    expected_batches = int(protocol.get("expected_train_batches", 0))
    if (expected_frames, batch_size, expected_batches) != (3123, 12, 261):
        raise ValueError("The frozen Stage-A replay size or batch size changed")

    population = protocol.get("target_population") or {}
    if population.get("pairing") != "existing_wisard_sorted_zip":
        raise ValueError("The target population must use existing sorted-zip pairing")
    if population.get("target_order") != "vis_x_vis_y_ir_minus_vis_dy_dx":
        raise ValueError("Unexpected box-guidance target order")
    if int(population.get("expected_frames", 0)) != expected_frames:
        raise ValueError("Target-population and audit frame counts differ")
    expected_population_values = {
        "expected_frames": 3123,
        "expected_frames_with_matches": 2306,
        "expected_matched_boxes": 5209,
        "expected_max_matches_per_frame": 4,
        "expected_per_frame_match_count_distribution": (
            EXPECTED_MATCH_DISTRIBUTION
        ),
        "expected_source_inventory_sha256": EXPECTED_SOURCE_INVENTORY_SHA256,
        "expected_target_population_sha256": EXPECTED_TARGET_POPULATION_SHA256,
    }
    if population != {
        "pairing": "existing_wisard_sorted_zip",
        "target_order": "vis_x_vis_y_ir_minus_vis_dy_dx",
        **expected_population_values,
    }:
        raise ValueError("The frozen target-population contract changed")
    _require_sha256(
        population.get("expected_source_inventory_sha256"),
        "expected_source_inventory_sha256",
    )
    _require_sha256(
        population.get("expected_target_population_sha256"),
        "expected_target_population_sha256",
    )

    gate = protocol.get("mechanism_gate") or {}
    if gate != EXPECTED_MECHANISM_GATE:
        raise ValueError("Unexpected mechanism-gate contract")


    for output_key, suffix in (("output_json", ".json"), ("output_csv", ".csv")):
        output = protocol.get(output_key)
        if not isinstance(output, str) or not output.endswith(f"_v1{suffix}"):
            raise ValueError(f"{output_key} must be a versioned _v1{suffix} path")


def load_seed_configs(config_path, seeds):
    """Resolve exactly one grid-search configuration for every requested seed."""
    raw = load_yaml(config_path)
    grid = make_grid(raw.get("parameters", raw))
    configs = {}
    for seed in seeds:
        matches = []
        for config in grid:
            try:
                matches_seed = int(config.get("seed")) == int(seed)
            except (TypeError, ValueError):
                matches_seed = False
            if matches_seed:
                matches.append(config)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one training configuration for seed {seed}, "
                f"found {len(matches)}"
            )
        configs[int(seed)] = matches[0]
    return configs


def validate_candidate_config(config, protocol):
    """Ensure checkpoint reconstruction and audit targets match training."""
    model_params = config.get("model", {}).get("params", {})
    if not model_params.get("use_fam"):
        raise ValueError("The audited model must enable FAM")
    if model_params.get("fam_variant") != FAM_VARIANT:
        raise ValueError(f"The audited model must use {FAM_VARIANT!r}")
    if float(model_params.get("spatial_jitter_std", 0.0)) != 0.0:
        raise ValueError("The box-guided candidate must not use spatial jitter")

    training_guidance = config.get("train", {}).get("box_guided_alignment") or {}
    if not training_guidance.get("enabled"):
        raise ValueError("The checkpoint must use box-guided alignment training")
    if not math.isclose(
        float(training_guidance.get("smooth_l1_beta_cells", -1.0)),
        float(protocol["smooth_l1_beta_cells"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Audit and training SmoothL1 beta values differ")

    dataset = config.get("dataset", {})
    if not dataset.get("box_alignment_targets"):
        raise ValueError("Training must enable box_alignment_targets")
    if not math.isclose(
        float(dataset.get("box_alignment_max_distance", -1.0)),
        float(protocol["box_matching"]["max_distance_normalized"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Audit and training box-match distances differ")


def _wandb_value(stored_config, key):
    value = stored_config.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def verify_local_run_artifacts(
    checkpoint_path,
    *,
    project,
    seed,
    scientific_config,
    expected_train_frames,
):
    """Bind a checkpoint to its exact local W&B scientific configuration."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if checkpoint_path.name != "model.safetensors" or checkpoint_path.parent.name != "best":
        raise RuntimeError("The audit checkpoint must be files/best/model.safetensors")
    files_directory = checkpoint_path.parent.parent
    if files_directory.name != "files":
        raise RuntimeError("Checkpoint is not inside a local W&B files directory")
    run_directory = files_directory.parent
    if not run_directory.name.startswith("run-"):
        raise RuntimeError("Checkpoint parent is not a local W&B run directory")

    config_path = files_directory / "config.yaml"
    summary_path = files_directory / "wandb-summary.json"
    if not config_path.is_file() or not summary_path.is_file():
        raise RuntimeError("Local W&B config and summary are required")
    stored = load_yaml(config_path)
    experiment = _wandb_value(stored, "experiment")
    if not isinstance(experiment, dict) or experiment.get("name") != project:
        raise RuntimeError("W&B project does not match the frozen protocol")

    stored_scientific_keys = set(stored) - WANDB_METADATA_KEYS
    expected_keys = set(scientific_config)
    if stored_scientific_keys != expected_keys:
        raise RuntimeError(
            "Stored W&B scientific keys differ from the training YAML: "
            f"expected {sorted(expected_keys)}, got {sorted(stored_scientific_keys)}"
        )
    stored_scientific = {
        key: _wandb_value(stored, key) for key in sorted(expected_keys)
    }
    expected_normalized = _canonical_json_value(scientific_config)
    stored_normalized = _canonical_json_value(stored_scientific)
    if stored_normalized != expected_normalized:
        raise RuntimeError("Stored W&B scientific configuration differs from YAML")
    if int(stored_scientific.get("seed", -1)) != int(seed):
        raise RuntimeError("Stored W&B seed differs from the requested seed")

    with summary_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    best_epoch = summary.get("best_epoch")
    best_map50 = summary.get("best_map_50")
    max_epochs = int(scientific_config["train"]["max_epochs"])
    expected_steps = (
        math.ceil(
            int(expected_train_frames)
            / int(scientific_config["dataloader"]["batch_size"])
        )
        * max_epochs
    )
    if (
        not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= max_epochs
        or not isinstance(best_map50, (int, float))
        or not math.isfinite(float(best_map50))
    ):
        raise RuntimeError("Local W&B summary has no valid completed best checkpoint")
    if int(summary.get("train/start_epoch", -1)) != max_epochs - 1:
        raise RuntimeError("Local W&B run did not complete its final epoch")
    if int(summary.get("train/step", -1)) != expected_steps - 1:
        raise RuntimeError("Local W&B run has an unexpected optimizer-step count")
    if any(key.startswith("test/") for key in summary):
        raise RuntimeError("Scientific training run unexpectedly evaluated test data")

    expected_digest = scientific_config_digest(scientific_config)
    stored_digest = scientific_config_digest(stored_scientific)
    if stored_digest != expected_digest:
        raise RuntimeError("Stored/local scientific configuration digests differ")
    reproducibility = scientific_config.get("reproducibility") or {}
    source_manifest = verify_training_source_manifest(
        reproducibility.get("training_source_manifest_id"),
        reproducibility.get("training_source_manifest_sha256"),
        repo_root=REPO_ROOT,
        required=True,
    )
    source_provenance = verify_training_source_runtime_trace(
        files_directory / "reproducibility_trace.jsonl",
        source_manifest,
    )
    return {
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "wandb_config_sha256": file_sha256(config_path),
        "wandb_summary_sha256": file_sha256(summary_path),
        "scientific_config_sha256": expected_digest,
        "best_epoch": best_epoch,
        "best_map_50": float(best_map50),
        "last_train_step": int(summary.get("train/step", -1)),
        "run_directory": str(run_directory),
        **source_provenance,
    }


def _alias_names_by_name(model):
    groups = {}
    named_tensors = list(model.named_parameters(remove_duplicate=False))
    named_tensors.extend(model.named_buffers(remove_duplicate=False))
    for name, tensor in named_tensors:
        groups.setdefault(id(tensor), []).append(name)
    return {
        name: tuple(names)
        for names in groups.values()
        for name in names
    }


def load_state_dict_exact_modulo_aliases(model, weights):
    """Load all checkpoint keys, allowing only genuine shared-tensor aliases."""
    try:
        incompatible = model.load_state_dict(weights, strict=False)
    except RuntimeError as error:
        raise RuntimeError(f"Checkpoint tensor shape/dtype mismatch: {error}") from error
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint keys: "
            + ", ".join(sorted(incompatible.unexpected_keys)[:10])
        )

    aliases = _alias_names_by_name(model)
    unresolved = []
    covered_aliases = []
    checkpoint_keys = set(weights)
    for missing in incompatible.missing_keys:
        alternatives = aliases.get(missing, (missing,))
        if any(alias in checkpoint_keys for alias in alternatives):
            covered_aliases.append(missing)
        else:
            unresolved.append(missing)
    if unresolved:
        raise RuntimeError(
            "Missing non-aliased checkpoint keys: "
            + ", ".join(sorted(unresolved)[:10])
        )
    return {
        "checkpoint_key_count": len(weights),
        "shared_alias_keys_covered": sorted(covered_aliases),
        "shared_alias_key_count": len(covered_aliases),
    }


def load_fusion_model_strict(model_params, checkpoint_path, device):
    """Reconstruct and load the candidate with exact state-dict coverage."""
    model = build_model(model_params)
    normalized_weights = {}
    raw_key_count = 0
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        for raw_key in checkpoint.keys():
            raw_key_count += 1
            key = raw_key[len("model.") :] if raw_key.startswith("model.") else raw_key
            if key in normalized_weights:
                raise RuntimeError(f"Checkpoint key-prefix collision for {key!r}")
            normalized_weights[key] = checkpoint.get_tensor(raw_key)
    if not normalized_weights:
        raise RuntimeError("Checkpoint contains no model tensors")
    key_audit = load_state_dict_exact_modulo_aliases(model, normalized_weights)
    key_audit["raw_checkpoint_key_count"] = raw_key_count
    model.eval().to(device)
    return model, key_audit


def audit_dataset_config(training_dataset_config, protocol):
    """Build the replay configuration without mutating the training config."""
    dataset = deepcopy(training_dataset_config)
    dataset["modal_dropout"] = False
    dataset["paired_consistency"] = False
    dataset["box_alignment_targets"] = True
    dataset["box_alignment_max_distance"] = float(
        protocol["box_matching"]["max_distance_normalized"]
    )
    return dataset


def canonical_target_population_digest(records):
    """Hash the independently frozen path/target population representation."""
    normalized = sorted(
        records,
        key=lambda record: (record["vis_id"], record["ir_id"]),
    )
    digest = hashlib.sha256()
    digest.update(TARGET_DIGEST_HEADER)
    digest.update(struct.pack(">I", len(normalized)))
    for record in normalized:
        for identifier in (record["vis_id"], record["ir_id"]):
            encoded = identifier.encode("utf-8")
            digest.update(struct.pack(">I", len(encoded)))
            digest.update(encoded)
        rows = np.asarray(record["targets"], dtype=np.float32)
        if len(rows):
            if rows.ndim != 2 or rows.shape[1] != 4:
                raise ValueError("Canonical target rows must have shape [N, 4]")
            order = np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))
            rows = rows[order]
        else:
            rows = rows.reshape(0, 4)
        digest.update(struct.pack(">I", len(rows)))
        digest.update(
            np.ascontiguousarray(rows, dtype="<f4").tobytes(order="C")
        )
    return digest.hexdigest()


def _relative_posix(path, root):
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Dataset path escapes the frozen root: {path}") from error


def _adapted_ir_size(vis_size, ir_size):
    vis_width, vis_height = vis_size
    ir_width, ir_height = ir_size
    resized_ir_width = int(ir_width * (vis_height / ir_height))
    horizontal_padding = (vis_width - resized_ir_width) // 2
    return resized_ir_width + 2 * horizontal_padding, vis_height


def _target_count_summary(records):
    counts = [int(len(record["targets"])) for record in records]
    distribution = Counter(counts)
    return {
        "frames": len(records),
        "frames_with_matches": sum(count > 0 for count in counts),
        "matched_boxes": sum(counts),
        "max_matches_per_frame": max(counts, default=0),
        "per_frame_match_count_distribution": {
            str(count): distribution[count] for count in sorted(distribution)
        },
        "target_population_sha256": canonical_target_population_digest(records),
    }


def build_target_population_manifest(dataset_config, repo_root=REPO_ROOT):
    """Rebuild source and weak-target identities without model inference."""
    dataset_root = Path(dataset_config["root"])
    if not dataset_root.is_absolute():
        dataset_root = Path(repo_root) / dataset_root
    dataset_root = dataset_root.resolve()
    folders = dataset_config.get("train_folders", dataset_config.get("folders"))
    items = build_wisard_items(dataset_root, folders)
    source_records = []
    target_records = []
    item_identifiers = {}
    max_distance = float(dataset_config["box_alignment_max_distance"])

    for sample_idx, (item_type, item) in enumerate(items):
        if item_type != MULTI_MODALITY_ITEM:
            raise RuntimeError("Target inventory contains a non-paired WiSARD item")
        (vis_image, vis_label), (ir_image, ir_label) = item
        paths = tuple(Path(path).resolve() for path in (
            vis_image,
            vis_label,
            ir_image,
            ir_label,
        ))
        vis_image, vis_label, ir_image, ir_label = paths
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError("Frozen target inventory contains a missing file")
        with Image.open(vis_image) as vis, Image.open(ir_image) as ir:
            vis_size = tuple(vis.size)
            ir_size = tuple(ir.size)
        adapted_ir_size = _adapted_ir_size(vis_size, ir_size)
        vis_annotations = yolo_to_coco_annotations(
            load_annotations(vis_label), sample_idx, *vis_size
        )
        ir_annotations = yolo_to_coco_annotations(
            load_annotations(ir_label), sample_idx, *ir_size
        )
        targets = build_box_alignment_targets(
            vis_annotations,
            ir_annotations,
            vis_size=vis_size,
            ir_size=ir_size,
            adapted_ir_size=adapted_ir_size,
            max_distance=max_distance,
        ).detach().cpu().to(torch.float32).contiguous()
        vis_id = _relative_posix(vis_image, dataset_root)
        ir_id = _relative_posix(ir_image, dataset_root)
        item_identifiers[sample_idx] = (vis_id, ir_id)
        target_records.append(
            {"vis_id": vis_id, "ir_id": ir_id, "targets": targets.numpy()}
        )
        source_records.append(
            {
                "sample_idx": sample_idx,
                "vis_image": vis_id,
                "vis_image_sha256": file_sha256(vis_image),
                "vis_size": list(vis_size),
                "vis_label": _relative_posix(vis_label, dataset_root),
                "vis_label_sha256": file_sha256(vis_label),
                "ir_image": ir_id,
                "ir_image_sha256": file_sha256(ir_image),
                "ir_size": list(ir_size),
                "ir_label": _relative_posix(ir_label, dataset_root),
                "ir_label_sha256": file_sha256(ir_label),
                "adapted_ir_size": list(adapted_ir_size),
            }
        )

    return {
        **_target_count_summary(target_records),
        "source_inventory_sha256": stable_json_hash(source_records),
        "item_identifiers": item_identifiers,
    }


def verify_target_population(summary, expected, *, require_source_inventory=True):
    """Verify frozen target counts/digest and, when available, source files.

    The independent preflight manifest contains hashes of the source files and
    must verify them.  The runtime replay is reconstructed from shuffled batch
    identifiers and target tensors, so it can independently verify only the
    target population; requiring a source-file digest from that representation
    would make every complete replay fail after inference.
    """
    actual = {
        "frames": int(summary["frames"]),
        "frames_with_matches": int(summary["frames_with_matches"]),
        "matched_boxes": int(summary["matched_boxes"]),
        "max_matches_per_frame": int(summary["max_matches_per_frame"]),
        "per_frame_match_count_distribution": {
            str(key): int(value)
            for key, value in summary["per_frame_match_count_distribution"].items()
        },
        "target_population_sha256": summary["target_population_sha256"],
    }
    frozen = {
        "frames": int(expected["expected_frames"]),
        "frames_with_matches": int(expected["expected_frames_with_matches"]),
        "matched_boxes": int(expected["expected_matched_boxes"]),
        "max_matches_per_frame": int(expected["expected_max_matches_per_frame"]),
        "per_frame_match_count_distribution": {
            str(key): int(value)
            for key, value in expected[
                "expected_per_frame_match_count_distribution"
            ].items()
        },
        "target_population_sha256": expected[
            "expected_target_population_sha256"
        ],
    }
    if actual != frozen:
        raise RuntimeError(
            "Box-guidance target population differs from the independent freeze:\n"
            + json.dumps({"expected": frozen, "actual": actual}, indent=2)
        )
    source_expected = expected.get("expected_source_inventory_sha256")
    if require_source_inventory and source_expected is not None:
        if summary.get("source_inventory_sha256") != source_expected:
            raise RuntimeError("Stage-A source file inventory digest changed")
    return actual


class RuntimeTargetPopulation:
    """Bind shuffled DataLoader targets back to the frozen dataset item ids."""

    def __init__(self, item_identifiers):
        self.item_identifiers = dict(item_identifiers)
        self.records = {}

    def update(self, sample_indices, targets):
        indices = torch.as_tensor(sample_indices).detach().cpu().reshape(-1).tolist()
        if len(indices) != len(targets):
            raise ValueError("sample_idx and target batch lengths differ")
        for sample_idx, sample_targets in zip(indices, targets):
            sample_idx = int(sample_idx)
            if sample_idx in self.records:
                raise RuntimeError(f"Duplicate DataLoader sample_idx {sample_idx}")
            if sample_idx not in self.item_identifiers:
                raise RuntimeError(f"Unknown DataLoader sample_idx {sample_idx}")
            vis_id, ir_id = self.item_identifiers[sample_idx]
            rows = (
                torch.as_tensor(sample_targets, dtype=torch.float32)
                .detach()
                .cpu()
                .contiguous()
                .numpy()
            )
            self.records[sample_idx] = {
                "vis_id": vis_id,
                "ir_id": ir_id,
                "targets": rows,
            }

    def summary(self):
        expected_indices = set(self.item_identifiers)
        actual_indices = set(self.records)
        if actual_indices != expected_indices:
            missing = sorted(expected_indices - actual_indices)[:10]
            unexpected = sorted(actual_indices - expected_indices)[:10]
            raise RuntimeError(
                "DataLoader target population is incomplete: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return _target_count_summary(list(self.records.values()))


def sample_guidance_at_targets(flow, targets):
    """Sample ``(dy, dx)`` P3 flow and form cell-space target vectors."""
    if not isinstance(flow, torch.Tensor) or flow.ndim != 4 or flow.shape[1] != 2:
        shape = getattr(flow, "shape", None)
        raise ValueError(f"guidance flow must have shape [B, 2, H, W], got {shape}")
    if not torch.isfinite(flow).all():
        raise ValueError("guidance flow must be finite")
    if not isinstance(targets, (list, tuple)) or len(targets) != flow.shape[0]:
        raise ValueError("targets must contain one tensor for every flow sample")

    height, width = flow.shape[-2:]
    sampled_terms = []
    target_terms = []
    frames_with_matches = 0
    for batch_index, sample_targets in enumerate(targets):
        sample_targets = torch.as_tensor(
            sample_targets,
            device=flow.device,
            dtype=flow.dtype,
        )
        if sample_targets.numel() == 0:
            continue
        if sample_targets.ndim != 2 or sample_targets.shape[1] != 4:
            raise ValueError(
                "each box-alignment target tensor must have shape [N, 4], got "
                f"{tuple(sample_targets.shape)}"
            )
        if not torch.isfinite(sample_targets).all():
            raise ValueError("box-alignment targets must be finite")
        positions = sample_targets[:, :2]
        if (positions < 0.0).any() or (positions > 1.0).any():
            raise ValueError("box-alignment VIS centres must lie in [0, 1]")

        sampling_grid = (2.0 * positions - 1.0).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            flow[batch_index : batch_index + 1],
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0, :, :, 0].transpose(0, 1)
        target_cells = torch.stack(
            (
                sample_targets[:, 2] * height,
                sample_targets[:, 3] * width,
            ),
            dim=-1,
        )
        sampled_terms.append(sampled)
        target_terms.append(target_cells)
        frames_with_matches += 1

    if not sampled_terms:
        empty = flow.new_empty((0, 2))
        return empty, empty.clone(), frames_with_matches
    return (
        torch.cat(sampled_terms, dim=0),
        torch.cat(target_terms, dim=0),
        frames_with_matches,
    )


def best_smooth_l1_constant(target_vectors, beta):
    """Return the component-wise constant minimizing mean SmoothL1 loss.

    The objective is convex and separable in ``(dy, dx)``. Bisection on the
    summed derivative avoids introducing an optimizer, an initialization, or
    another fitted hyperparameter into this descriptive baseline.
    """
    targets = torch.as_tensor(target_vectors, dtype=torch.float64).detach().cpu()
    if targets.ndim != 2 or targets.shape[1] != 2 or targets.shape[0] == 0:
        raise ValueError("target_vectors must have non-empty shape [N, 2]")
    if not torch.isfinite(targets).all():
        raise ValueError("target_vectors must be finite")
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("SmoothL1 beta must be finite and positive")

    constants = []
    for component in range(2):
        values = targets[:, component]
        search_lower = float(values.min()) - beta
        search_upper = float(values.max()) + beta

        lower = search_lower
        upper = search_upper
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            derivative = torch.clamp(
                (midpoint - values) / beta,
                min=-1.0,
                max=1.0,
            ).sum()
            if float(derivative) < 0.0:
                lower = midpoint
            else:
                upper = midpoint
        leftmost_minimizer = upper

        lower = search_lower
        upper = search_upper
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            derivative = torch.clamp(
                (midpoint - values) / beta,
                min=-1.0,
                max=1.0,
            ).sum()
            if float(derivative) <= 0.0:
                lower = midpoint
            else:
                upper = midpoint
        rightmost_minimizer = lower
        constants.append(0.5 * (leftmost_minimizer + rightmost_minimizer))
    return torch.tensor(constants, dtype=torch.float64)


def _vector_diagnostics(sampled_vectors, target_vectors, beta):
    """Describe bias and non-global train fit without adding a gate."""
    sampled = torch.cat(sampled_vectors, dim=0)
    targets = torch.cat(target_vectors, dim=0)
    if sampled.shape != targets.shape or sampled.ndim != 2 or sampled.shape[1] != 2:
        raise RuntimeError("Stored guidance and target vectors are inconsistent")

    constant = best_smooth_l1_constant(targets, beta)
    constant_loss = float(
        F.smooth_l1_loss(
            constant.expand_as(targets),
            targets,
            beta=beta,
            reduction="mean",
        )
    )
    learned_loss = float(
        F.smooth_l1_loss(sampled, targets, beta=beta, reduction="mean")
    )
    relative_vs_constant = (
        (constant_loss - learned_loss) / constant_loss
        if constant_loss > 0.0
        else None
    )

    guidance_mean = sampled.mean(dim=0)
    target_mean = targets.mean(dim=0)
    guidance_centered = sampled - guidance_mean
    target_centered = targets - target_mean
    centered_denominator = float(
        torch.linalg.vector_norm(guidance_centered)
        * torch.linalg.vector_norm(target_centered)
    )
    centered_cosine = (
        float((guidance_centered * target_centered).sum())
        / centered_denominator
        if centered_denominator > 1e-12
        else None
    )
    mean_guidance_l2 = float(
        torch.linalg.vector_norm(sampled, dim=1).mean()
    )
    bias_fraction = (
        float(torch.linalg.vector_norm(guidance_mean)) / mean_guidance_l2
        if mean_guidance_l2 > 1e-12
        else None
    )
    guidance_std = sampled.std(dim=0, unbiased=False)
    target_std = targets.std(dim=0, unbiased=False)
    target_median = torch.quantile(targets, 0.5, dim=0)

    return {
        "best_constant_flow_dy_cells": float(constant[0]),
        "best_constant_flow_dx_cells": float(constant[1]),
        "best_constant_smooth_l1_cells": constant_loss,
        "relative_improvement_vs_best_constant": relative_vs_constant,
        "beats_global_constant_on_train_targets": (
            relative_vs_constant is not None and relative_vs_constant > 0.0
        ),
        "sampled_guidance_mean_dy_cells": float(guidance_mean[0]),
        "sampled_guidance_mean_dx_cells": float(guidance_mean[1]),
        "sampled_guidance_std_dy_cells": float(guidance_std[0]),
        "sampled_guidance_std_dx_cells": float(guidance_std[1]),
        "target_mean_dy_cells": float(target_mean[0]),
        "target_mean_dx_cells": float(target_mean[1]),
        "target_median_dy_cells": float(target_median[0]),
        "target_median_dx_cells": float(target_median[1]),
        "target_std_dy_cells": float(target_std[0]),
        "target_std_dx_cells": float(target_std[1]),
        "centered_guidance_target_cosine": centered_cosine,
        "guidance_bias_fraction_of_mean_l2": bias_fraction,
    }


class GuidanceAuditAccumulator:
    """Accumulate exact, match-weighted guidance statistics across batches."""

    def __init__(self, smooth_l1_beta_cells, near_saturation_threshold_cells):
        self.beta = float(smooth_l1_beta_cells)
        self.saturation_threshold = float(near_saturation_threshold_cells)
        if self.beta <= 0.0:
            raise ValueError("SmoothL1 beta must be positive")
        if self.saturation_threshold < 0.0:
            raise ValueError("Near-saturation threshold must be non-negative")
        self.batches = 0
        self.frames = 0
        self.frames_with_matches = 0
        self.matches = 0
        self.scalar_components = 0
        self.learned_loss_sum = 0.0
        self.residual_common_loss_sum = 0.0
        self.total_common_loss_sum = 0.0
        self.zero_flow_loss_sum = 0.0
        self.guidance_abs_sum = 0.0
        self.residual_common_abs_sum = 0.0
        self.total_common_abs_sum = 0.0
        self.near_saturation_count = 0
        self.guidance_l2_sum = 0.0
        self.residual_common_l2_sum = 0.0
        self.total_common_l2_sum = 0.0
        self.cancellation_ratio_sum = 0.0
        self.cancellation_ratio_count = 0
        self.half_cancelled_count = 0
        self.cosine_sum = 0.0
        self.cosine_count = 0
        self.sampled_vectors = []
        self.target_vectors = []

    def update(self, flow, targets, residual_common_flow=None):
        sampled, target_cells, frames_with_matches = sample_guidance_at_targets(
            flow, targets
        )
        if residual_common_flow is None:
            residual_common_flow = torch.zeros_like(flow)
        if residual_common_flow.shape != flow.shape:
            raise ValueError("Residual-common and guidance flows must have equal shape")
        residual_sampled, residual_targets, residual_frames = (
            sample_guidance_at_targets(residual_common_flow, targets)
        )
        if residual_frames != frames_with_matches:
            raise RuntimeError("Guidance and residual target populations differ")
        if not torch.equal(residual_targets, target_cells):
            raise RuntimeError("Guidance and residual target tensors differ")
        total_sampled = sampled + residual_sampled
        self.batches += 1
        self.frames += int(flow.shape[0])
        self.frames_with_matches += int(frames_with_matches)
        if sampled.numel() == 0:
            return

        self.sampled_vectors.append(sampled.detach().double().cpu())
        self.target_vectors.append(target_cells.detach().double().cpu())

        learned_loss = F.smooth_l1_loss(
            sampled,
            target_cells,
            beta=self.beta,
            reduction="sum",
        )
        residual_common_loss = F.smooth_l1_loss(
            residual_sampled,
            target_cells,
            beta=self.beta,
            reduction="sum",
        )
        total_common_loss = F.smooth_l1_loss(
            total_sampled,
            target_cells,
            beta=self.beta,
            reduction="sum",
        )
        zero_flow_loss = F.smooth_l1_loss(
            torch.zeros_like(target_cells),
            target_cells,
            beta=self.beta,
            reduction="sum",
        )
        losses = (
            learned_loss,
            residual_common_loss,
            total_common_loss,
            zero_flow_loss,
        )
        if not all(torch.isfinite(loss) for loss in losses):
            raise ValueError("Guidance audit losses must be finite")

        self.matches += int(sampled.shape[0])
        self.scalar_components += int(sampled.numel())
        self.learned_loss_sum += float(learned_loss.double())
        self.residual_common_loss_sum += float(residual_common_loss.double())
        self.total_common_loss_sum += float(total_common_loss.double())
        self.zero_flow_loss_sum += float(zero_flow_loss.double())
        absolute = sampled.detach().double().abs()
        residual_absolute = residual_sampled.detach().double().abs()
        total_absolute = total_sampled.detach().double().abs()
        self.guidance_abs_sum += float(absolute.sum())
        self.residual_common_abs_sum += float(residual_absolute.sum())
        self.total_common_abs_sum += float(total_absolute.sum())
        self.near_saturation_count += int(
            (absolute >= self.saturation_threshold).sum()
        )

        guidance_norm = torch.linalg.vector_norm(sampled.detach().double(), dim=1)
        residual_norm = torch.linalg.vector_norm(
            residual_sampled.detach().double(), dim=1
        )
        total_norm = torch.linalg.vector_norm(total_sampled.detach().double(), dim=1)
        self.guidance_l2_sum += float(guidance_norm.sum())
        self.residual_common_l2_sum += float(residual_norm.sum())
        self.total_common_l2_sum += float(total_norm.sum())
        valid_guidance = guidance_norm > 1e-12
        if valid_guidance.any():
            cancellation = torch.clamp(
                1.0 - total_norm[valid_guidance] / guidance_norm[valid_guidance],
                min=0.0,
                max=1.0,
            )
            self.cancellation_ratio_sum += float(cancellation.sum())
            self.cancellation_ratio_count += int(cancellation.numel())
            self.half_cancelled_count += int((cancellation >= 0.5).sum())
        valid_cosine = (guidance_norm > 1e-12) & (residual_norm > 1e-12)
        if valid_cosine.any():
            cosine = F.cosine_similarity(
                sampled.detach().double()[valid_cosine],
                residual_sampled.detach().double()[valid_cosine],
                dim=1,
                eps=1e-12,
            )
            self.cosine_sum += float(cosine.sum())
            self.cosine_count += int(cosine.numel())

    def summary(self):
        if self.frames == 0:
            raise ValueError("Cannot summarize an audit with no frames")
        if self.scalar_components == 0:
            learned_loss = None
            residual_common_loss = None
            total_common_loss = None
            zero_flow_loss = None
            relative_improvement = None
            total_relative_improvement = None
            mean_abs = None
            residual_mean_abs = None
            total_mean_abs = None
            saturation_fraction = None
            mean_guidance_l2 = None
            mean_residual_l2 = None
            mean_total_l2 = None
            vector_diagnostics = {
                "best_constant_flow_dy_cells": None,
                "best_constant_flow_dx_cells": None,
                "best_constant_smooth_l1_cells": None,
                "relative_improvement_vs_best_constant": None,
                "beats_global_constant_on_train_targets": False,
                "sampled_guidance_mean_dy_cells": None,
                "sampled_guidance_mean_dx_cells": None,
                "sampled_guidance_std_dy_cells": None,
                "sampled_guidance_std_dx_cells": None,
                "target_mean_dy_cells": None,
                "target_mean_dx_cells": None,
                "target_median_dy_cells": None,
                "target_median_dx_cells": None,
                "target_std_dy_cells": None,
                "target_std_dx_cells": None,
                "centered_guidance_target_cosine": None,
                "guidance_bias_fraction_of_mean_l2": None,
            }
        else:
            learned_loss = self.learned_loss_sum / self.scalar_components
            residual_common_loss = (
                self.residual_common_loss_sum / self.scalar_components
            )
            total_common_loss = self.total_common_loss_sum / self.scalar_components
            zero_flow_loss = self.zero_flow_loss_sum / self.scalar_components
            relative_improvement = (
                (zero_flow_loss - learned_loss) / zero_flow_loss
                if zero_flow_loss > 0.0
                else None
            )
            total_relative_improvement = (
                (zero_flow_loss - total_common_loss) / zero_flow_loss
                if zero_flow_loss > 0.0
                else None
            )
            mean_abs = self.guidance_abs_sum / self.scalar_components
            residual_mean_abs = (
                self.residual_common_abs_sum / self.scalar_components
            )
            total_mean_abs = self.total_common_abs_sum / self.scalar_components
            saturation_fraction = (
                self.near_saturation_count / self.scalar_components
            )
            mean_guidance_l2 = self.guidance_l2_sum / self.matches
            mean_residual_l2 = self.residual_common_l2_sum / self.matches
            mean_total_l2 = self.total_common_l2_sum / self.matches
            vector_diagnostics = _vector_diagnostics(
                self.sampled_vectors,
                self.target_vectors,
                self.beta,
            )
        return {
            "batches": self.batches,
            "frames": self.frames,
            "frames_with_matches": self.frames_with_matches,
            "matched_boxes": self.matches,
            "matches_per_frame": self.matches / self.frames,
            "learned_smooth_l1_cells": learned_loss,
            "residual_common_smooth_l1_cells": residual_common_loss,
            "total_common_smooth_l1_cells": total_common_loss,
            "zero_flow_smooth_l1_cells": zero_flow_loss,
            "relative_improvement_vs_zero": relative_improvement,
            "total_common_relative_improvement_vs_zero": total_relative_improvement,
            "sampled_guidance_mean_abs_cells": mean_abs,
            "sampled_residual_common_mean_abs_cells": residual_mean_abs,
            "sampled_total_common_mean_abs_cells": total_mean_abs,
            "sampled_guidance_fraction_abs_ge_threshold": saturation_fraction,
            "sampled_guidance_mean_l2_cells": mean_guidance_l2,
            "sampled_residual_common_mean_l2_cells": mean_residual_l2,
            "sampled_total_common_mean_l2_cells": mean_total_l2,
            "mean_positive_cancellation_ratio": (
                self.cancellation_ratio_sum / self.cancellation_ratio_count
                if self.cancellation_ratio_count
                else None
            ),
            "fraction_guidance_at_least_half_cancelled": (
                self.half_cancelled_count / self.cancellation_ratio_count
                if self.cancellation_ratio_count
                else None
            ),
            "mean_cosine_guidance_residual_common": (
                self.cosine_sum / self.cosine_count
                if self.cosine_count
                else None
            ),
            "nonzero_guidance_vectors": self.cancellation_ratio_count,
            "sampled_scalar_components": self.scalar_components,
            **vector_diagnostics,
        }


def summarize_across_seeds(rows):
    """Use checkpoint/seed summaries as the statistical units."""
    metrics = (
        "matches_per_frame",
        "learned_smooth_l1_cells",
        "residual_common_smooth_l1_cells",
        "total_common_smooth_l1_cells",
        "zero_flow_smooth_l1_cells",
        "best_constant_flow_dy_cells",
        "best_constant_flow_dx_cells",
        "best_constant_smooth_l1_cells",
        "relative_improvement_vs_zero",
        "relative_improvement_vs_best_constant",
        "total_common_relative_improvement_vs_zero",
        "sampled_guidance_mean_abs_cells",
        "sampled_residual_common_mean_abs_cells",
        "sampled_total_common_mean_abs_cells",
        "sampled_guidance_fraction_abs_ge_threshold",
        "mean_positive_cancellation_ratio",
        "fraction_guidance_at_least_half_cancelled",
        "mean_cosine_guidance_residual_common",
        "sampled_guidance_mean_dy_cells",
        "sampled_guidance_mean_dx_cells",
        "sampled_guidance_std_dy_cells",
        "sampled_guidance_std_dx_cells",
        "target_mean_dy_cells",
        "target_mean_dx_cells",
        "target_median_dy_cells",
        "target_median_dx_cells",
        "target_std_dy_cells",
        "target_std_dx_cells",
        "centered_guidance_target_cosine",
        "guidance_bias_fraction_of_mean_l2",
    )
    summary = {}
    for metric in metrics:
        seed_values = {
            str(row["seed"]): row.get(metric)
            for row in rows
            if row.get(metric) is not None
        }
        values = list(seed_values.values())
        summary[metric] = {
            "seed_values": seed_values,
            "mean": statistics.fmean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


class ResidualCommonOffsetCapture:
    """Capture P3 residual offsets without adding persistent model state."""

    def __init__(self, guided_module):
        self.guided_module = guided_module
        self.last_raw_output = None
        self.handle = guided_module.offset_conv.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self.last_raw_output = output

    def reset(self):
        self.last_raw_output = None

    def residual_common_flow(self):
        output = self.last_raw_output
        if output is None:
            raise RuntimeError("Guided P3 offset_conv was not observed in the forward")
        if output.ndim != 4 or output.shape[1] < 18:
            raise RuntimeError("Guided P3 offset_conv returned an invalid tensor")
        residual = self.guided_module.transform_offset(output[:, :18])
        batch, channels, height, width = residual.shape
        if channels != 18:
            raise RuntimeError("Guided P3 residual offset must contain 18 channels")
        return residual.reshape(batch, 9, 2, height, width).mean(dim=1)

    def close(self):
        self.handle.remove()


def audit_seed(
    model,
    dataloader,
    protocol,
    device,
    *,
    item_identifiers,
):
    module = find_box_guided_fam(model)
    residual_capture = ResidualCommonOffsetCapture(module)
    runtime_population = RuntimeTargetPopulation(item_identifiers)
    accumulator = GuidanceAuditAccumulator(
        protocol["smooth_l1_beta_cells"],
        protocol["near_saturation_threshold_cells"],
    )
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(dataloader, start=1):
                modes = batch.get("modality_mode")
                if modes is None or any(mode != "fusion" for mode in modes):
                    raise RuntimeError("Audit replay produced a non-fusion sample")
                if "box_alignment_targets" not in batch:
                    raise RuntimeError(
                        "Audit replay did not produce box-alignment targets"
                    )
                if "sample_idx" not in batch:
                    raise RuntimeError("Audit replay did not preserve sample_idx")

                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                residual_capture.reset()
                model.model(pixel_values, pixel_mask=pixel_mask)
                flow = module.last_guidance_flow
                if flow is None:
                    raise RuntimeError("The guided P3 FAM did not cache its flow")
                residual_common = residual_capture.residual_common_flow()
                accumulator.update(
                    flow,
                    batch["box_alignment_targets"],
                    residual_common_flow=residual_common,
                )
                runtime_population.update(
                    batch["sample_idx"], batch["box_alignment_targets"]
                )

                if batch_index % 50 == 0:
                    print(
                        f"  audited {accumulator.frames}/"
                        f"{protocol['expected_train_frames']} frames"
                    )
                pixel_values = pixel_mask = flow = residual_common = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        residual_capture.close()

    summary = accumulator.summary()
    if summary["frames"] != int(protocol["expected_train_frames"]):
        raise RuntimeError(
            f"Expected {protocol['expected_train_frames']} train frames, got "
            f"{summary['frames']}"
        )
    if summary["batches"] != int(protocol["expected_train_batches"]):
        raise RuntimeError(
            f"Expected {protocol['expected_train_batches']} train batches, got "
            f"{summary['batches']}"
        )
    if summary["matched_boxes"] == 0:
        raise RuntimeError("No conservative box matches were found during the audit")
    population_summary = runtime_population.summary()
    verify_target_population(
        population_summary,
        protocol["target_population"],
        require_source_inventory=False,
    )
    return summary, population_summary


def mechanism_gate(summary, rule):
    guidance_improvement = summary["relative_improvement_vs_zero"]
    guidance_mean_abs = summary["sampled_guidance_mean_abs_cells"]
    saturation_fraction = summary[
        "sampled_guidance_fraction_abs_ge_threshold"
    ]
    total_improvement = summary["total_common_relative_improvement_vs_zero"]
    cancellation = summary["mean_positive_cancellation_ratio"]
    half_cancelled = summary["fraction_guidance_at_least_half_cancelled"]
    checks = {
        "guidance_improves_over_zero": (
            guidance_improvement is not None
            and guidance_improvement
            >= float(rule["minimum_guidance_relative_improvement_vs_zero"])
        ),
        "guidance_is_non_degenerate": (
            guidance_mean_abs is not None
            and guidance_mean_abs
            > float(rule["minimum_guidance_mean_abs_cells"])
        ),
        "guidance_is_not_saturated": (
            saturation_fraction is not None
            and saturation_fraction
            < float(rule["maximum_guidance_saturation_fraction"])
        ),
        "total_common_not_worse_than_zero": (
            total_improvement is not None
            and total_improvement
            >= float(rule["minimum_total_relative_improvement_vs_zero"])
        ),
        "mean_cancellation_not_substantial": (
            cancellation is not None
            and cancellation
            <= float(rule["maximum_mean_positive_cancellation_ratio"])
        ),
        "half_cancellation_not_widespread": (
            half_cancelled is not None
            and half_cancelled
            <= float(
                rule["maximum_fraction_guidance_at_least_half_cancelled"]
            )
        ),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "status": rule["if_pass"] if passed else rule["if_fail"],
        "checks": checks,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    fieldnames = [
        "protocol_id",
        "seed",
        "checkpoint",
        "batches",
        "frames",
        "frames_with_matches",
        "matched_boxes",
        "matches_per_frame",
        "learned_smooth_l1_cells",
        "residual_common_smooth_l1_cells",
        "total_common_smooth_l1_cells",
        "zero_flow_smooth_l1_cells",
        "best_constant_flow_dy_cells",
        "best_constant_flow_dx_cells",
        "best_constant_smooth_l1_cells",
        "relative_improvement_vs_zero",
        "relative_improvement_vs_best_constant",
        "beats_global_constant_on_train_targets",
        "total_common_relative_improvement_vs_zero",
        "sampled_guidance_mean_abs_cells",
        "sampled_residual_common_mean_abs_cells",
        "sampled_total_common_mean_abs_cells",
        "sampled_guidance_fraction_abs_ge_threshold",
        "sampled_guidance_mean_l2_cells",
        "sampled_residual_common_mean_l2_cells",
        "sampled_total_common_mean_l2_cells",
        "mean_positive_cancellation_ratio",
        "fraction_guidance_at_least_half_cancelled",
        "mean_cosine_guidance_residual_common",
        "sampled_guidance_mean_dy_cells",
        "sampled_guidance_mean_dx_cells",
        "sampled_guidance_std_dy_cells",
        "sampled_guidance_std_dx_cells",
        "target_mean_dy_cells",
        "target_mean_dx_cells",
        "target_median_dy_cells",
        "target_median_dx_cells",
        "target_std_dy_cells",
        "target_std_dx_cells",
        "centered_guidance_target_cosine",
        "guidance_bias_fraction_of_mean_l2",
        "nonzero_guidance_vectors",
        "sampled_scalar_components",
        "near_saturation_threshold_cells",
        "checkpoint_sha256",
        "scientific_config_sha256",
        "wandb_config_sha256",
        "reproducibility_trace_sha256",
        "training_source_manifest_sha256",
        "target_population_sha256",
        "source_inventory_sha256",
        "mechanism_gate_passed",
        "mechanism_gate_status",
    ]
    with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_results_atomic(json_path, csv_path, result, rows):
    """Prepare both outputs and publish JSON last as the completion marker."""
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(
            "Mechanism-audit outputs already exist; preserve the versioned evidence"
        )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json_path = json_path.with_name(json_path.name + ".tmp")
    temporary_csv_path = csv_path.with_name(csv_path.name + ".tmp")
    try:
        temporary_json_path.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        write_csv(temporary_csv_path, rows)
        temporary_csv_path.replace(csv_path)
        temporary_json_path.replace(json_path)
    finally:
        temporary_json_path.unlink(missing_ok=True)
        temporary_csv_path.unlink(missing_ok=True)


def run(protocol_path, dry_run=False):
    protocol = load_yaml(protocol_path)
    validate_protocol(protocol)
    protocol_digest = stable_json_hash(_canonical_json_value(protocol))
    json_path = REPO_ROOT / protocol["output_json"]
    csv_path = REPO_ROOT / protocol["output_csv"]
    if not dry_run and (json_path.exists() or csv_path.exists()):
        raise FileExistsError(
            "Mechanism-audit output already exists; refusing to rerun or overwrite"
        )
    config_path = (REPO_ROOT / protocol["training_config"]).resolve()
    training_config_digest = verify_training_config_file(
        config_path, protocol["training_config_sha256"]
    )
    configs = load_seed_configs(config_path, protocol["seeds"])
    for config in configs.values():
        validate_candidate_config(config, protocol)

    first_config = configs[int(protocol["seeds"][0])]
    frozen_dataset = audit_dataset_config(first_config["dataset"], protocol)
    for config in configs.values():
        candidate_dataset = audit_dataset_config(config["dataset"], protocol)
        if _canonical_json_value(candidate_dataset) != _canonical_json_value(
            frozen_dataset
        ):
            raise RuntimeError("Seed configurations use different audit datasets")
    population_manifest = build_target_population_manifest(frozen_dataset)
    verify_target_population(population_manifest, protocol["target_population"])

    checkpoints = {
        int(seed): resolve_local_wandb_checkpoint(
            project=protocol["project"],
            seed=seed,
            checkpoint=protocol["checkpoint"],
            wandb_root=REPO_ROOT / "wandb",
        )
        for seed in protocol["seeds"]
    }
    artifact_audits = {
        int(seed): verify_local_run_artifacts(
            checkpoints[int(seed)],
            project=protocol["project"],
            seed=seed,
            scientific_config=configs[int(seed)],
            expected_train_frames=protocol["expected_train_frames"],
        )
        for seed in protocol["seeds"]
    }
    if dry_run:
        print(
            f"Dry run OK: {len(checkpoints)} completed best checkpoint(s), "
            "exact W&B/YAML identity and frozen source/target digests verified; "
            "no result files written"
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in protocol["seeds"]:
        seed = int(seed)
        config = configs[seed]
        dataset_config = audit_dataset_config(config["dataset"], protocol)
        dataloader_config = deepcopy(config["dataloader"])
        dataloader_config["batch_size"] = int(protocol["audit_batch_size"])
        (train_loader, _val, _test), _denormalize = get_dataloaders(
            dataset_config,
            dataloader_config,
            seed=seed,
        )
        model, state_dict_audit = load_fusion_model_strict(
            config["model"], checkpoints[seed], device
        )
        print(f"Auditing seed={seed} on frozen Stage-A train fusion replay")
        mechanism_summary, runtime_population = audit_seed(
            model,
            train_loader,
            protocol,
            device,
            item_identifiers=population_manifest["item_identifiers"],
        )
        gate = mechanism_gate(mechanism_summary, protocol["mechanism_gate"])
        row = {
            "protocol_id": protocol["protocol_id"],
            "seed": seed,
            "checkpoint": checkpoints[seed],
            **mechanism_summary,
            "near_saturation_threshold_cells": protocol[
                "near_saturation_threshold_cells"
            ],
            "checkpoint_sha256": artifact_audits[seed]["checkpoint_sha256"],
            "scientific_config_sha256": artifact_audits[seed][
                "scientific_config_sha256"
            ],
            "wandb_config_sha256": artifact_audits[seed]["wandb_config_sha256"],
            "reproducibility_trace_sha256": artifact_audits[seed][
                "reproducibility_trace_sha256"
            ],
            "training_source_manifest_sha256": artifact_audits[seed][
                "training_source_manifest_sha256"
            ],
            "target_population_sha256": runtime_population[
                "target_population_sha256"
            ],
            "source_inventory_sha256": population_manifest[
                "source_inventory_sha256"
            ],
            "mechanism_gate_passed": gate["passed"],
            "mechanism_gate_status": gate["status"],
        }
        rows.append(row)
        artifact_audits[seed]["state_dict"] = state_dict_audit
        artifact_audits[seed]["mechanism_gate"] = gate
        del model, train_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "protocol": protocol,
        "protocol_sha256": protocol_digest,
        "training_config_sha256": training_config_digest,
        "source_inventory": {
            key: value
            for key, value in population_manifest.items()
            if key != "item_identifiers"
        },
        "checkpoint_audits": {
            str(seed): {
                "path": checkpoints[seed],
                **artifact_audits[seed],
            }
            for seed in sorted(checkpoints)
        },
        "seed_rows": rows,
        "cross_seed_summary": summarize_across_seeds(rows),
        "protocol_complete": True,
        "mechanism_gate_all_passed": all(
            row["mechanism_gate_passed"] for row in rows
        ),
    }
    write_results_atomic(json_path, csv_path, result, rows)
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
