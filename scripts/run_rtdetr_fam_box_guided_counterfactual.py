#!/usr/bin/env python3
"""Counterfactual inference audit for the seed-40 box-guided FAM.

The selected best checkpoint is loaded exactly once and evaluated on the
frozen 896-frame Stage-A validation acquisition in two conditions.  The first
keeps the learned box-guidance branch active.  The second temporarily zeros
only the final guidance predictor convolution, making its ``(dy, dx)`` output
exactly zero, then restores the original tensors byte-for-byte.

No result file is created until both full passes and every identity check have
completed.  This audit does not access or construct the Mt Erie test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sarfusion.data.temporal_split import stable_json_hash  # noqa: E402
from sarfusion.data.utils import build_preprocessor, get_collate_fn  # noqa: E402
from sarfusion.data.wisard import MULTI_MODALITY_ITEM, WiSARDDataset  # noqa: E402
from sarfusion.experiment.box_guided_alignment import (  # noqa: E402
    find_box_guided_fam,
)
from sarfusion.models import build_model  # noqa: E402
from sarfusion.models.checkpoints import (  # noqa: E402
    resolve_local_wandb_checkpoint,
)
from sarfusion.utils.grid import make_grid  # noqa: E402
from sarfusion.utils.metrics import DetectionEvaluator, MetricCollection  # noqa: E402
from sarfusion.utils.reproducibility import (  # noqa: E402
    verify_training_source_manifest,
    verify_training_source_runtime_trace,
)
from sarfusion.utils.structures import DataDict, WrapperModelOutput  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_box_guided_counterfactual_seed40.yaml"
)
PROTOCOL_ID = "rtdetr_fam_box_guided_counterfactual_v1"
PROJECT = "RTDETR_FAM_BoxGuided_SequenceVal_Seed40"
FAM_VARIANT = "box_guided_common_offset_p3"
CONDITIONS = ("active", "zero")
EXPECTED_TRAINING_CONFIG_SHA256 = (
    "16c1d58a4926bb6f6d1c018eb96600b52279116ca04a42d35286d63fb12ea647"
)
EXPECTED_SOURCE_INVENTORY_SHA256 = (
    "47e2f348ebdc202cb749b1bbf2741fc868d8681eb24ffa4356cf4f584dfa4ec4"
)
EXPECTED_CONTENT_INVENTORY_SHA256 = (
    "6c7748af3be2761a3a466b548af64aae925b693fbca795edf695072e28f17141"
)
EXPECTED_SAMPLE_ORDER_SHA256 = (
    "49415f065575c869087c78f842591096b74a0ea3a16ca2e4ce765e26958badcd"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
WANDB_METADATA_KEYS = {"_wandb", "experiment", "wandb_version"}
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


def _require_sha256(value, description):
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")


def _canonical_json_value(value):
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


def resolve_repo_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def validate_protocol(protocol):
    """Validate the frozen counterfactual contract, including its fail gate."""
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected box-guided counterfactual protocol_id")
    if protocol.get("status") != "frozen_before_counterfactual_inference":
        raise ValueError("The counterfactual protocol is not frozen")
    if protocol.get("purpose") != (
        "isolate_inference_time_contribution_of_learned_box_guidance"
    ):
        raise ValueError("Unexpected counterfactual purpose")
    if protocol.get("project") != PROJECT:
        raise ValueError("The audit must resolve the seed-40 candidate project")
    if protocol.get("checkpoint") != "best" or int(protocol.get("seed", -1)) != 40:
        raise ValueError("The audit requires the selected best seed-40 checkpoint")
    if protocol.get("split") != "validation":
        raise ValueError("The counterfactual audit is restricted to validation")
    if protocol.get("mode") != "fusion" or protocol.get("ground_truth") != "vis":
        raise ValueError("The counterfactual audit requires fusion input and VIS GT")
    if not math.isclose(
        float(protocol.get("confidence_threshold", -1.0)),
        0.01,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("The frozen confidence threshold must be 0.01")
    if not isinstance(protocol.get("training_config"), str):
        raise ValueError("training_config must be a path string")
    _require_sha256(protocol.get("training_config_sha256"), "training_config_sha256")
    if protocol["training_config_sha256"] != EXPECTED_TRAINING_CONFIG_SHA256:
        raise ValueError("The frozen candidate training-config digest changed")

    expected_frames = int(protocol.get("source", {}).get("expected_paired_frames", 0))
    batch_size = int(protocol.get("batch_size", 0))
    if expected_frames != 896 or batch_size <= 0:
        raise ValueError("The audit must contain 896 frames and a positive batch size")
    if int(protocol.get("expected_batches", 0)) != math.ceil(
        expected_frames / batch_size
    ):
        raise ValueError("expected_batches does not match the frozen population")

    source = protocol.get("source") or {}
    if source.get("pairing") != "existing_wisard_sorted_zip":
        raise ValueError("The audit must reproduce WiSARD sorted-zip pairing")
    expected_pair = [[
        "210924_FHL_Enterprise_VIS_0401",
        "210924_FHL_Enterprise_IR_0402",
    ]]
    if source.get("paired_folders") != expected_pair:
        raise ValueError("Only FHL 0401/0402 is allowed in this audit")
    if any("mterie" in str(value).lower() for value in source.values()):
        raise ValueError("Mt Erie is forbidden in the counterfactual audit")
    for key in (
        "expected_inventory_sha256",
        "expected_content_inventory_sha256",
        "expected_sample_order_sha256",
    ):
        _require_sha256(source.get(key), f"source.{key}")
    expected_source_hashes = {
        "expected_inventory_sha256": EXPECTED_SOURCE_INVENTORY_SHA256,
        "expected_content_inventory_sha256": EXPECTED_CONTENT_INVENTORY_SHA256,
        "expected_sample_order_sha256": EXPECTED_SAMPLE_ORDER_SHA256,
    }
    observed_source_hashes = {
        key: source.get(key) for key in expected_source_hashes
    }
    if observed_source_hashes != expected_source_hashes:
        raise ValueError("The frozen FHL source/order digests changed")

    counterfactual = protocol.get("counterfactual") or {}
    if counterfactual.get("order") != list(CONDITIONS):
        raise ValueError("Conditions must be evaluated in active, zero order")
    if counterfactual.get("zero") != (
        "zero_and_restore_guidance_predictor_final_conv_weight_and_bias"
    ):
        raise ValueError("Unexpected counterfactual intervention")
    if counterfactual.get("require_same_checkpoint_instance") is not True:
        raise ValueError("Both conditions must use one checkpoint instance")
    if (
        counterfactual.get("require_same_sample_order_and_ground_truth")
        is not True
    ):
        raise ValueError("Sample order and ground truth identity must be enforced")

    reproduction = protocol.get("active_reproduction") or {}
    if reproduction.get("metric") != "map_50":
        raise ValueError("Active replay must reproduce W&B map_50")
    tolerance = float(reproduction.get("absolute_tolerance", -1.0))
    if not 0.0 <= tolerance <= 0.001:
        raise ValueError("Active-reproduction tolerance must be in [0, 0.001]")

    gate = protocol.get("diagnostic_gate") or {}
    if (
        gate.get("metric") != "active_minus_zero_map_50"
        or gate.get("comparator") != "greater_than_or_equal"
        or float(gate.get("minimum", math.nan)) != 0.0
    ):
        raise ValueError("The diagnostic gate must be active-minus-zero >= 0.0")
    interpretation = protocol.get("interpretation") or {}
    if not interpretation or any(value is not False for value in interpretation.values()):
        raise ValueError("Counterfactual interpretation constraints were relaxed")

    json_path = Path(protocol.get("output_json", ""))
    csv_path = Path(protocol.get("output_csv", ""))
    expected_stem = "rtdetr_fam_box_guided_counterfactual_v1"
    if json_path.stem != expected_stem or csv_path.stem != expected_stem:
        raise ValueError("Counterfactual outputs must carry the v1 version suffix")
    if json_path.suffix != ".json" or csv_path.suffix != ".csv":
        raise ValueError("Counterfactual output extensions must be JSON and CSV")
    return protocol


def load_protocol(path=DEFAULT_PROTOCOL):
    protocol = load_yaml(path)
    if not isinstance(protocol, dict):
        raise ValueError("Counterfactual protocol YAML must contain a mapping")
    return validate_protocol(protocol)


def verify_training_config_file(config_path, expected_sha256):
    _require_sha256(expected_sha256, "training_config_sha256")
    actual = file_sha256(config_path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Training YAML hash changed: expected {expected_sha256}, got {actual}"
        )
    return actual


def load_candidate_config(config_path, protocol):
    payload = load_yaml(config_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("parameters"), dict):
        raise ValueError("Training YAML must contain a parameters mapping")
    experiment = payload.get("experiment") or {}
    if experiment.get("name") != protocol["project"]:
        raise ValueError("Training YAML project differs from the protocol")
    runs = make_grid(payload["parameters"])
    if len(runs) != 1:
        raise ValueError("Counterfactual training YAML must expand to exactly one run")
    config = runs[0]
    if int(config.get("seed", -1)) != int(protocol["seed"]):
        raise ValueError("Training YAML does not describe seed 40")
    validate_candidate_config(config, protocol)
    return config


def validate_candidate_config(config, protocol):
    model = config.get("model") or {}
    params = model.get("params") or {}
    if model.get("name") != "fusion_rtdetr":
        raise ValueError("Candidate must use fusion_rtdetr")
    if params.get("use_fam") is not True or params.get("fam_variant") != FAM_VARIANT:
        raise ValueError(f"Candidate must reconstruct {FAM_VARIANT}")
    if bool(params.get("use_p2", False)):
        raise ValueError("Box-guided counterfactual is restricted to P3--P5")
    if float(params.get("spatial_jitter_std", 0.0)) != 0.0:
        raise ValueError("Candidate must not use spatial jitter")
    if not math.isclose(
        float(params.get("threshold", -1.0)),
        float(protocol["confidence_threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Training and audit confidence thresholds differ")

    train = config.get("train") or {}
    guidance = train.get("box_guided_alignment") or {}
    if not guidance.get("enabled") or int(train.get("max_epochs", 0)) != 10:
        raise ValueError("Candidate must be the completed ten-epoch guided screen")
    if config.get("run_test") is not False:
        raise ValueError("Candidate training must not evaluate the test split")
    dataset = config.get("dataset") or {}
    if not dataset.get("box_alignment_targets"):
        raise ValueError("Candidate training must enable box-alignment targets")
    if not dataset.get("modal_dropout"):
        raise ValueError("Candidate training must retain historical Modal Dropout")
    if dataset.get("val_folders") != protocol["source"]["paired_folders"]:
        raise ValueError("Candidate validation acquisition differs from the audit")
    if dataset.get("root") != protocol["source"]["dataset_root"]:
        raise ValueError("Candidate dataset root differs from the audit")
    serialized = json.dumps(_canonical_json_value(config), sort_keys=True).lower()
    if "mterie" in serialized or "210417_mterie" in serialized:
        raise ValueError("Candidate scientific configuration unexpectedly names Mt Erie")


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
    """Bind a unique best checkpoint to its complete local W&B run."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if checkpoint_path.name != "model.safetensors" or checkpoint_path.parent.name != "best":
        raise RuntimeError("Checkpoint must be files/best/model.safetensors")
    files_directory = checkpoint_path.parent.parent
    if files_directory.name != "files":
        raise RuntimeError("Checkpoint is outside a local W&B files directory")
    run_directory = files_directory.parent
    if not run_directory.name.startswith("run-"):
        raise RuntimeError("Checkpoint parent is not a local W&B run directory")

    config_path = files_directory / "config.yaml"
    summary_path = files_directory / "wandb-summary.json"
    if not config_path.is_file() or not summary_path.is_file():
        raise RuntimeError("Local W&B config and summary are both required")
    stored = load_yaml(config_path)
    if not isinstance(stored, dict):
        raise RuntimeError("Local W&B config is not a mapping")
    experiment = _wandb_value(stored, "experiment")
    if not isinstance(experiment, dict) or experiment.get("name") != project:
        raise RuntimeError("W&B project differs from the frozen protocol")

    expected_keys = set(scientific_config)
    stored_keys = set(stored) - WANDB_METADATA_KEYS
    if stored_keys != expected_keys:
        raise RuntimeError(
            "Stored W&B scientific keys differ from the training YAML: "
            f"expected {sorted(expected_keys)}, got {sorted(stored_keys)}"
        )
    stored_scientific = {
        key: _wandb_value(stored, key) for key in sorted(expected_keys)
    }
    if _canonical_json_value(stored_scientific) != _canonical_json_value(
        scientific_config
    ):
        raise RuntimeError("Stored W&B scientific configuration differs from YAML")
    if int(stored_scientific.get("seed", -1)) != int(seed):
        raise RuntimeError("Stored W&B seed differs from the requested seed")

    with summary_path.open(encoding="utf-8") as input_file:
        summary = json.load(input_file)
    max_epochs = int(scientific_config["train"]["max_epochs"])
    expected_steps = (
        math.ceil(
            int(expected_train_frames)
            / int(scientific_config["dataloader"]["batch_size"])
        )
        * max_epochs
    )
    best_epoch = summary.get("best_epoch")
    best_map50 = summary.get("best_map_50")
    if (
        not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= max_epochs
        or not isinstance(best_map50, (int, float))
        or not math.isfinite(float(best_map50))
    ):
        raise RuntimeError("W&B summary has no valid completed best checkpoint")
    if int(summary.get("train/start_epoch", -1)) != max_epochs - 1:
        raise RuntimeError("W&B run did not complete its final epoch")
    if int(summary.get("train/step", -1)) != expected_steps - 1:
        raise RuntimeError("W&B run has an unexpected optimizer-step count")
    if any(key.startswith("test/") for key in summary):
        raise RuntimeError("Candidate training unexpectedly evaluated test data")

    expected_digest = scientific_config_digest(scientific_config)
    stored_digest = scientific_config_digest(stored_scientific)
    if stored_digest != expected_digest:
        raise RuntimeError("Stored and YAML scientific-config hashes differ")
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
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "wandb_config_sha256": file_sha256(config_path),
        "wandb_summary_sha256": file_sha256(summary_path),
        "scientific_config_sha256": expected_digest,
        "best_epoch": best_epoch,
        "best_map_50": float(best_map50),
        "last_train_step": int(summary["train/step"]),
        "expected_optimizer_steps": expected_steps,
        "run_directory": str(run_directory),
        **source_provenance,
    }


def sorted_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(path for path in directory.iterdir() if path.is_file())


def temporal_index(path):
    return Path(path).stem.rsplit("_", 1)[-1]


def build_source_inventory(protocol):
    """Reproduce and hash the exact FHL sorted zip, including image bytes."""
    source = protocol["source"]
    dataset_root = resolve_repo_path(source["dataset_root"])
    historical_rows = []
    content_rows = []
    vis_inventory = 0
    ir_inventory = 0
    vis_boxes = 0
    ir_boxes = 0
    vis_empty = 0
    ir_empty = 0
    unpaired_vis = []

    for vis_folder, ir_folder in source["paired_folders"]:
        vis_images = sorted_files(dataset_root / vis_folder / "images")
        vis_labels = sorted_files(dataset_root / vis_folder / "labels")
        ir_images = sorted_files(dataset_root / ir_folder / "images")
        ir_labels = sorted_files(dataset_root / ir_folder / "labels")
        if len(vis_images) != len(vis_labels):
            raise RuntimeError("VIS image/label inventory is unbalanced")
        if len(ir_images) != len(ir_labels):
            raise RuntimeError("IR image/label inventory is unbalanced")
        vis_inventory += len(vis_images)
        ir_inventory += len(ir_images)
        unpaired_vis.extend(path.name for path in vis_images[len(ir_images) :])

        for vis_image, vis_label, ir_image, ir_label in zip(
            vis_images, vis_labels, ir_images, ir_labels
        ):
            if temporal_index(vis_image) != temporal_index(ir_image):
                raise RuntimeError(
                    f"Sorted-zip temporal shift: {vis_image.name} vs {ir_image.name}"
                )
            vis_label_bytes = vis_label.read_bytes()
            ir_label_bytes = ir_label.read_bytes()
            vis_lines = [
                line for line in vis_label_bytes.decode().splitlines() if line.strip()
            ]
            ir_lines = [
                line for line in ir_label_bytes.decode().splitlines() if line.strip()
            ]
            vis_boxes += len(vis_lines)
            ir_boxes += len(ir_lines)
            vis_empty += int(not vis_lines)
            ir_empty += int(not ir_lines)
            historical = {
                "vis_image": str(vis_image.relative_to(dataset_root)),
                "vis_image_size": vis_image.stat().st_size,
                "vis_label": str(vis_label.relative_to(dataset_root)),
                "vis_label_sha256": hashlib.sha256(vis_label_bytes).hexdigest(),
                "ir_image": str(ir_image.relative_to(dataset_root)),
                "ir_image_size": ir_image.stat().st_size,
                "ir_label": str(ir_label.relative_to(dataset_root)),
                "ir_label_sha256": hashlib.sha256(ir_label_bytes).hexdigest(),
            }
            historical_rows.append(historical)
            content = dict(historical)
            content["vis_image_sha256"] = file_sha256(vis_image)
            content["ir_image_sha256"] = file_sha256(ir_image)
            content_rows.append(content)

    inventory = {
        "dataset_root": str(dataset_root),
        "pairing": source["pairing"],
        "ground_truth": "vis",
        "vis_inventory": vis_inventory,
        "ir_inventory": ir_inventory,
        "paired_frames": len(historical_rows),
        "unpaired_vis": unpaired_vis,
        "vis_boxes": vis_boxes,
        "vis_empty_frames": vis_empty,
        "ir_boxes": ir_boxes,
        "ir_empty_frames": ir_empty,
        "inventory_sha256": stable_json_hash(historical_rows),
        "content_inventory_sha256": stable_json_hash(content_rows),
        "sample_order_sha256": stable_json_hash(
            [row["vis_image"] for row in historical_rows]
        ),
        "rows": content_rows,
    }
    expected = {
        "vis_inventory": int(source["expected_vis_inventory"]),
        "ir_inventory": int(source["expected_ir_inventory"]),
        "paired_frames": int(source["expected_paired_frames"]),
        "unpaired_vis": [source["expected_unpaired_vis_terminal"]],
        "vis_boxes": int(source["expected_vis_boxes"]),
        "vis_empty_frames": int(source["expected_vis_empty_frames"]),
        "ir_boxes": int(source["expected_ir_boxes"]),
        "ir_empty_frames": int(source["expected_ir_empty_frames"]),
        "inventory_sha256": source["expected_inventory_sha256"],
        "content_inventory_sha256": source[
            "expected_content_inventory_sha256"
        ],
        "sample_order_sha256": source["expected_sample_order_sha256"],
    }
    actual = {key: inventory[key] for key in expected}
    if actual != expected:
        raise RuntimeError(
            "FHL validation inventory differs from the freeze:\n"
            + json.dumps({"expected": expected, "actual": actual}, indent=2)
        )
    return inventory


def _dataset_item_paths(item, dataset_root):
    item_type, payload = item
    if item_type != MULTI_MODALITY_ITEM:
        raise RuntimeError("Counterfactual dataset contains a non-paired item")
    (vis_image, vis_label), (ir_image, ir_label) = payload
    return {
        "vis_image": str(Path(vis_image).resolve().relative_to(dataset_root)),
        "vis_label": str(Path(vis_label).resolve().relative_to(dataset_root)),
        "ir_image": str(Path(ir_image).resolve().relative_to(dataset_root)),
        "ir_label": str(Path(ir_label).resolve().relative_to(dataset_root)),
    }


def build_validation_loader(config, protocol, inventory):
    """Construct only the Stage-A FHL validation loader; never Mt Erie."""
    dataset_params = deepcopy(config["dataset"])
    transform, _denormalize = build_preprocessor(dataset_params)
    dataset_root = Path(inventory["dataset_root"]).resolve()
    dataset = WiSARDDataset(
        root=str(dataset_root),
        folders=protocol["source"]["paired_folders"],
        transform=transform,
        single_class=True,
        modal_dropout=False,
        paired_consistency=False,
        box_alignment_targets=False,
        use_tiling=False,
        test_all_tiles=False,
    )
    if len(dataset) != int(protocol["source"]["expected_paired_frames"]):
        raise RuntimeError("Validation dataset length differs from the freeze")
    expected_paths = [
        {key: row[key] for key in ("vis_image", "vis_label", "ir_image", "ir_label")}
        for row in inventory["rows"]
    ]
    actual_paths = [_dataset_item_paths(item, dataset_root) for item in dataset.items]
    if actual_paths != expected_paths:
        raise RuntimeError("WiSARDDataset order/content differs from the inventory")

    generator = torch.Generator().manual_seed(int(protocol["evaluation_seed"]))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(protocol["batch_size"]),
        num_workers=int(protocol["workers"]),
        shuffle=False,
        collate_fn=get_collate_fn(dataset),
        generator=generator,
    )


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
    checkpoint_keys = set(weights)
    unresolved = []
    covered = []
    for missing in incompatible.missing_keys:
        alternatives = aliases.get(missing, (missing,))
        if any(alias in checkpoint_keys for alias in alternatives):
            covered.append(missing)
        else:
            unresolved.append(missing)
    if unresolved:
        raise RuntimeError(
            "Missing non-aliased checkpoint keys: "
            + ", ".join(sorted(unresolved)[:10])
        )
    return {
        "checkpoint_key_count": len(weights),
        "shared_alias_keys_covered": sorted(covered),
        "shared_alias_key_count": len(covered),
    }


def load_fusion_model_strict(model_params, checkpoint_path, device):
    model = build_model(model_params)
    normalized = {}
    raw_key_count = 0
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        for raw_key in checkpoint.keys():
            raw_key_count += 1
            key = raw_key[len("model.") :] if raw_key.startswith("model.") else raw_key
            if key in normalized:
                raise RuntimeError(f"Checkpoint key-prefix collision for {key!r}")
            normalized[key] = checkpoint.get_tensor(raw_key)
    if not normalized:
        raise RuntimeError("Checkpoint contains no model tensors")
    key_audit = load_state_dict_exact_modulo_aliases(model, normalized)
    key_audit["raw_checkpoint_key_count"] = raw_key_count
    model.eval().to(device)
    return model, key_audit


def _tensor_bundle_sha256(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def guidance_final_layer(module):
    predictor = getattr(module, "guidance_predictor", None)
    if predictor is None or len(predictor) == 0:
        raise RuntimeError("Guided FAM has no guidance_predictor")
    final = predictor[-1]
    if not isinstance(final, torch.nn.Conv2d) or final.out_channels != 2:
        raise RuntimeError("Guidance predictor must end in a two-channel Conv2d")
    if final.bias is None:
        raise RuntimeError("Guidance predictor final Conv2d must have a bias")
    return final


@contextmanager
def temporarily_zero_guidance_output(module):
    """Zero and restore final guidance tensors, even if inference raises."""
    final = guidance_final_layer(module)
    saved_weight = final.weight.detach().clone()
    saved_bias = final.bias.detach().clone()
    active_sha256 = _tensor_bundle_sha256(
        {"weight": saved_weight, "bias": saved_bias}
    )
    intervention = {
        "active_final_layer_sha256": active_sha256,
        "zero_final_layer_sha256": None,
        "restored_final_layer_sha256": None,
        "restored_exactly": False,
    }
    try:
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()
        if torch.count_nonzero(final.weight).item() != 0 or torch.count_nonzero(
            final.bias
        ).item() != 0:
            raise RuntimeError("Guidance intervention did not produce exact zeros")
        intervention["zero_final_layer_sha256"] = _tensor_bundle_sha256(
            {"weight": final.weight, "bias": final.bias}
        )
        yield intervention
    finally:
        with torch.no_grad():
            final.weight.copy_(saved_weight)
            final.bias.copy_(saved_bias)
        intervention["restored_exactly"] = bool(
            torch.equal(final.weight.detach(), saved_weight)
            and torch.equal(final.bias.detach(), saved_bias)
        )
        intervention["restored_final_layer_sha256"] = _tensor_bundle_sha256(
            {"weight": final.weight, "bias": final.bias}
        )
        if (
            not intervention["restored_exactly"]
            or intervention["restored_final_layer_sha256"] != active_sha256
        ):
            raise RuntimeError("Guidance final layer was not restored exactly")


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _tensors_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensors_to_cpu(item) for item in value]
    return value


def _update_value_digest(digest, value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value):
            digest.update(str(key).encode())
            digest.update(b"\0")
            _update_value_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"list\0")
        for item in value:
            _update_value_digest(digest, item)
    else:
        digest.update(json.dumps(_jsonable(value), sort_keys=True).encode())
        digest.update(b"\0")


def set_evaluation_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def evaluate_condition(model, loader, device, *, condition, guided_module):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown counterfactual condition: {condition}")
    evaluator = DetectionEvaluator(MetricCollection({}), id2class=loader.dataset.id2class)
    sample_indices = []
    ground_truth_digest = hashlib.sha256(b"BOX_GUIDED_COUNTERFACTUAL_GT_V1\n")
    n_batches = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            n_batches += 1
            indices = [int(index) for index in batch["sample_idx"].tolist()]
            modes = list(batch["modality_mode"])
            if any(mode != "fusion" for mode in modes):
                raise RuntimeError("Counterfactual loader emitted a non-fusion sample")
            sample_indices.extend(indices)
            for index, label in zip(indices, batch["labels"]):
                ground_truth_digest.update(index.to_bytes(8, "big", signed=False))
                _update_value_digest(ground_truth_digest, label)

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device, non_blocking=True)
            output = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            if condition == "zero":
                flow = guided_module.last_guidance_flow
                if flow is None or torch.count_nonzero(flow).item() != 0:
                    raise RuntimeError("Zero intervention produced non-zero guidance")
            evaluator.update(
                DataDict(labels=_tensors_to_cpu(batch["labels"])),
                WrapperModelOutput(
                    predictions=_tensors_to_cpu(output["predictions"])
                ),
            )

    expected_frames = int(loader.dataset.__len__())
    expected_indices = list(range(expected_frames))
    if sample_indices != expected_indices:
        raise RuntimeError("Evaluation sample order is not the frozen 0..895 order")
    metrics = _jsonable(evaluator.compute())
    if "map_50" not in metrics or not math.isfinite(float(metrics["map_50"])):
        raise RuntimeError("Evaluation did not produce a finite map_50")
    return {
        "condition": condition,
        "n_samples": len(sample_indices),
        "n_batches": n_batches,
        "sample_indices_sha256": stable_json_hash(sample_indices),
        "ground_truth_sha256": ground_truth_digest.hexdigest(),
        "metrics": metrics,
    }


def compare_pass_identity(active, zero, protocol):
    expected_frames = int(protocol["source"]["expected_paired_frames"])
    expected_batches = int(protocol["expected_batches"])
    for result in (active, zero):
        if result["n_samples"] != expected_frames:
            raise RuntimeError("A counterfactual pass did not evaluate all 896 frames")
        if result["n_batches"] != expected_batches:
            raise RuntimeError("A counterfactual pass has an unexpected batch count")
    if active["sample_indices_sha256"] != zero["sample_indices_sha256"]:
        raise RuntimeError("Counterfactual sample orders differ")
    if active["ground_truth_sha256"] != zero["ground_truth_sha256"]:
        raise RuntimeError("Counterfactual ground-truth tensors differ")
    return {
        "same_sample_order": True,
        "same_ground_truth": True,
        "sample_indices_sha256": active["sample_indices_sha256"],
        "ground_truth_sha256": active["ground_truth_sha256"],
    }


def diagnostic_gate(active_map50, zero_map50, rule):
    active_map50 = float(active_map50)
    zero_map50 = float(zero_map50)
    if not math.isfinite(active_map50) or not math.isfinite(zero_map50):
        raise RuntimeError("Counterfactual gate received a non-finite map_50")
    delta = active_map50 - zero_map50
    minimum = float(rule["minimum"])
    passed = delta >= minimum
    return {
        "metric": rule["metric"],
        "active_map_50": active_map50,
        "zero_map_50": zero_map50,
        "active_minus_zero_map_50": delta,
        "minimum": minimum,
        "comparator": rule["comparator"],
        "passed": passed,
        "status": rule["if_pass"] if passed else rule["if_fail"],
    }


def verify_active_reproduction(active, artifact_audit, protocol):
    observed = float(active["metrics"]["map_50"])
    expected = float(artifact_audit["best_map_50"])
    absolute_error = abs(observed - expected)
    tolerance = float(protocol["active_reproduction"]["absolute_tolerance"])
    if not math.isfinite(absolute_error) or absolute_error > tolerance:
        raise RuntimeError(
            "Active replay does not reproduce W&B best_map_50: "
            f"observed={observed}, expected={expected}, "
            f"absolute_error={absolute_error}, tolerance={tolerance}"
        )
    return {
        "metric": "map_50",
        "observed": observed,
        "wandb_best": expected,
        "absolute_error": absolute_error,
        "absolute_tolerance": tolerance,
        "passed": True,
    }


def implementation_source_hashes():
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "sarfusion" / "models" / "rtdetr_fusion.py",
        REPO_ROOT / "sarfusion" / "experiment" / "box_guided_alignment.py",
        REPO_ROOT / "sarfusion" / "data" / "wisard.py",
        REPO_ROOT / "sarfusion" / "utils" / "metrics.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in paths
    }


def build_complete_result(
    *,
    protocol,
    protocol_sha256,
    training_config_sha256,
    artifact_audit,
    state_dict_audit,
    inventory,
    active,
    zero,
    pass_identity,
    reproduction,
    intervention,
):
    if [active.get("condition"), zero.get("condition")] != list(CONDITIONS):
        raise RuntimeError("Counterfactual result set is incomplete")
    if not intervention.get("restored_exactly"):
        raise RuntimeError("Cannot complete result before exact weight restoration")
    gate = diagnostic_gate(
        active["metrics"]["map_50"],
        zero["metrics"]["map_50"],
        protocol["diagnostic_gate"],
    )
    inventory_summary = {key: value for key, value in inventory.items() if key != "rows"}
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "protocol_complete": True,
        "purpose": protocol["purpose"],
        "checkpoint": artifact_audit,
        "configuration_hashes": {
            "training_yaml_sha256": training_config_sha256,
            "scientific_config_sha256": artifact_audit[
                "scientific_config_sha256"
            ],
            "wandb_config_sha256": artifact_audit["wandb_config_sha256"],
            "wandb_summary_sha256": artifact_audit["wandb_summary_sha256"],
            "reproducibility_trace_sha256": artifact_audit[
                "reproducibility_trace_sha256"
            ],
            "training_source_manifest_sha256": artifact_audit[
                "training_source_manifest_sha256"
            ],
        },
        "implementation_source_sha256": implementation_source_hashes(),
        "source_inventory": inventory_summary,
        "evaluation": {
            "split": protocol["split"],
            "mode": protocol["mode"],
            "ground_truth": protocol["ground_truth"],
            "confidence_threshold": float(protocol["confidence_threshold"]),
            "condition_order": list(CONDITIONS),
            "identity": pass_identity,
            "active_reproduction": reproduction,
            "guidance_final_layer_intervention": intervention,
        },
        "results": [active, zero],
        "diagnostic_gate": gate,
        "interpretation": protocol["interpretation"],
        "state_dict_audit": state_dict_audit,
    }


def _fsync_directory(directory):
    descriptor = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_text(path, text):
    path = Path(path)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    try:
        with temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        return Path(temporary.name)
    except BaseException:
        Path(temporary.name).unlink(missing_ok=True)
        raise


def _csv_text(result):
    import io

    output = io.StringIO(newline="")
    fieldnames = [
        "condition",
        "n_samples",
        "n_batches",
        *SCALAR_METRICS,
        "active_minus_zero_map_50",
        "diagnostic_gate_passed",
        "diagnostic_gate_status",
        "checkpoint_sha256",
        "training_yaml_sha256",
        "source_inventory_sha256",
        "content_inventory_sha256",
        "sample_indices_sha256",
        "ground_truth_sha256",
        "reproducibility_trace_sha256",
        "training_source_manifest_sha256",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    gate = result["diagnostic_gate"]
    for row in result["results"]:
        writer.writerow(
            {
                "condition": row["condition"],
                "n_samples": row["n_samples"],
                "n_batches": row["n_batches"],
                **{metric: row["metrics"].get(metric) for metric in SCALAR_METRICS},
                "active_minus_zero_map_50": gate[
                    "active_minus_zero_map_50"
                ],
                "diagnostic_gate_passed": gate["passed"],
                "diagnostic_gate_status": gate["status"],
                "checkpoint_sha256": result["checkpoint"]["checkpoint_sha256"],
                "training_yaml_sha256": result["configuration_hashes"][
                    "training_yaml_sha256"
                ],
                "source_inventory_sha256": result["source_inventory"][
                    "inventory_sha256"
                ],
                "content_inventory_sha256": result["source_inventory"][
                    "content_inventory_sha256"
                ],
                "sample_indices_sha256": row["sample_indices_sha256"],
                "ground_truth_sha256": row["ground_truth_sha256"],
                "reproducibility_trace_sha256": result[
                    "configuration_hashes"
                ]["reproducibility_trace_sha256"],
                "training_source_manifest_sha256": result[
                    "configuration_hashes"
                ]["training_source_manifest_sha256"],
            }
        )
    return output.getvalue()


def write_complete_outputs(result, json_path, csv_path):
    """Publish JSON and CSV only for a fully verified two-condition run."""
    if result.get("protocol_complete") is not True:
        raise ValueError("Refusing to write an incomplete counterfactual result")
    if [row.get("condition") for row in result.get("results", [])] != list(
        CONDITIONS
    ):
        raise ValueError("Refusing to write an incomplete condition set")
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(
            "Counterfactual outputs already exist; preserve the versioned evidence"
        )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    csv_text = _csv_text(result)
    staged_json = _stage_text(json_path, json_text)
    staged_csv = _stage_text(csv_path, csv_text)
    try:
        os.replace(staged_json, json_path)
        os.replace(staged_csv, csv_path)
        _fsync_directory(json_path.parent)
        if csv_path.parent != json_path.parent:
            _fsync_directory(csv_path.parent)
    finally:
        staged_json.unlink(missing_ok=True)
        staged_csv.unlink(missing_ok=True)


def resolve_device(requested=None):
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare(protocol_path=DEFAULT_PROTOCOL, *, wandb_root=None):
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    protocol_sha256 = file_sha256(protocol_path)
    config_path = resolve_repo_path(protocol["training_config"])
    training_hash = verify_training_config_file(
        config_path, protocol["training_config_sha256"]
    )
    config = load_candidate_config(config_path, protocol)
    inventory = build_source_inventory(protocol)
    checkpoint = resolve_local_wandb_checkpoint(
        project=protocol["project"],
        seed=int(protocol["seed"]),
        checkpoint=protocol["checkpoint"],
        wandb_root=wandb_root or (REPO_ROOT / "wandb"),
    )
    artifact_audit = verify_local_run_artifacts(
        checkpoint,
        project=protocol["project"],
        seed=int(protocol["seed"]),
        scientific_config=config,
        expected_train_frames=int(protocol["expected_train_frames"]),
    )
    return {
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": protocol_sha256,
        "config": config,
        "config_path": config_path,
        "training_config_sha256": training_hash,
        "inventory": inventory,
        "checkpoint": Path(checkpoint).resolve(),
        "artifact_audit": artifact_audit,
    }


def run(
    protocol_path=DEFAULT_PROTOCOL,
    *,
    device=None,
    wandb_root=None,
    dry_run=False,
):
    prepared = prepare(protocol_path, wandb_root=wandb_root)
    protocol = prepared["protocol"]
    if dry_run:
        return {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": prepared["protocol_sha256"],
            "checkpoint_sha256": prepared["artifact_audit"]["checkpoint_sha256"],
            "source_inventory_sha256": prepared["inventory"]["inventory_sha256"],
            "content_inventory_sha256": prepared["inventory"][
                "content_inventory_sha256"
            ],
            "dry_run": True,
        }

    evaluation_device = resolve_device(device)
    loader = build_validation_loader(
        prepared["config"], protocol, prepared["inventory"]
    )
    model, state_dict_audit = load_fusion_model_strict(
        prepared["config"]["model"], prepared["checkpoint"], evaluation_device
    )
    if not math.isclose(
        float(model.threshold),
        float(protocol["confidence_threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Reconstructed model threshold differs from protocol")
    guided_module = find_box_guided_fam(model)
    checkpoint_instance_id = id(model)

    set_evaluation_seed(protocol["evaluation_seed"])
    if getattr(loader, "generator", None) is not None:
        loader.generator.manual_seed(int(protocol["evaluation_seed"]))
    active = evaluate_condition(
        model,
        loader,
        evaluation_device,
        condition="active",
        guided_module=guided_module,
    )
    reproduction = verify_active_reproduction(
        active, prepared["artifact_audit"], protocol
    )

    with temporarily_zero_guidance_output(guided_module) as intervention:
        if id(model) != checkpoint_instance_id:
            raise RuntimeError("Counterfactual model instance changed")
        set_evaluation_seed(protocol["evaluation_seed"])
        if getattr(loader, "generator", None) is not None:
            loader.generator.manual_seed(int(protocol["evaluation_seed"]))
        zero = evaluate_condition(
            model,
            loader,
            evaluation_device,
            condition="zero",
            guided_module=guided_module,
        )

    if id(model) != checkpoint_instance_id:
        raise RuntimeError("Counterfactual model instance changed")
    identity = compare_pass_identity(active, zero, protocol)
    result = build_complete_result(
        protocol=protocol,
        protocol_sha256=prepared["protocol_sha256"],
        training_config_sha256=prepared["training_config_sha256"],
        artifact_audit=prepared["artifact_audit"],
        state_dict_audit=state_dict_audit,
        inventory=prepared["inventory"],
        active=active,
        zero=zero,
        pass_identity=identity,
        reproduction=reproduction,
        intervention=intervention,
    )
    write_complete_outputs(
        result,
        resolve_repo_path(protocol["output_json"]),
        resolve_repo_path(protocol["output_csv"]),
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb-root", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify the frozen source and completed local checkpoint without inference.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run(
        args.protocol,
        device=args.device,
        wandb_root=args.wandb_root,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        gate = result["diagnostic_gate"]
        print(
            "Counterfactual complete: "
            f"active-zero mAP50={gate['active_minus_zero_map_50']:.8f}; "
            f"gate={gate['status']}"
        )


if __name__ == "__main__":
    main()
