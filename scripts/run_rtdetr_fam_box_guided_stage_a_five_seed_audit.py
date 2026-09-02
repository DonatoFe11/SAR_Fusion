#!/usr/bin/env python3
"""Fail-closed five-seed Stage-A audit for box-guided RT-DETR FAM.

Seed 40 is resolved from the dedicated screen projects, while seeds 41--44
are resolved from the conditional expansion projects.  Candidate checkpoints
receive the frozen mechanism replay; matched controls are used only for the
paired best-validation-mAP comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from copy import deepcopy
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sarfusion.data import get_dataloaders  # noqa: E402
from sarfusion.data.temporal_split import stable_json_hash  # noqa: E402
from sarfusion.models.checkpoints import (  # noqa: E402
    resolve_local_wandb_checkpoint,
)
from sarfusion.utils.grid import make_grid  # noqa: E402
from sarfusion.utils.utils import load_yaml  # noqa: E402
from scripts.run_rtdetr_fam_box_guided_mechanism_audit import (  # noqa: E402
    EXPECTED_DIAGNOSTIC_BASELINES,
    EXPECTED_MATCH_DISTRIBUTION,
    EXPECTED_MECHANISM_GATE,
    EXPECTED_SOURCE_INVENTORY_SHA256,
    EXPECTED_TARGET_POPULATION_SHA256,
    _canonical_json_value,
    audit_dataset_config,
    audit_seed,
    build_target_population_manifest,
    file_sha256,
    load_fusion_model_strict,
    mechanism_gate,
    summarize_across_seeds,
    validate_candidate_config,
    verify_local_run_artifacts,
    verify_target_population,
    verify_training_config_file,
)


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_box_guided_stage_a_five_seed_audit_v2.yaml"
)
PROTOCOL_ID = "rtdetr_fam_box_guided_stage_a_five_seed_audit_v2"
EXPECTED_STATUS = "conditional_frozen_before_seeds_41_44_training"
EXPECTED_SEEDS = [40, 41, 42, 43, 44]
EXPECTED_IMPLEMENTATION_SOURCE_PATHS = (
    "scripts/run_rtdetr_fam_box_guided_stage_a_five_seed_audit.py",
    "scripts/run_rtdetr_fam_box_guided_mechanism_audit.py",
    "scripts/run_rtdetr_fam_box_guided_counterfactual.py",
)
EXPECTED_COUNTERFACTUAL_RESULT_IMPLEMENTATION_PATHS = (
    "scripts/run_rtdetr_fam_box_guided_counterfactual.py",
    "sarfusion/models/rtdetr_fusion.py",
    "sarfusion/experiment/box_guided_alignment.py",
    "sarfusion/data/wisard.py",
    "sarfusion/utils/metrics.py",
)

EXPECTED_CANDIDATE_SOURCES = [
    {
        "source_id": "candidate_seed40_screen",
        "project": "RTDETR_FAM_BoxGuided_SequenceVal_Seed40",
        "group": "RTDETR_fam_box_guided_sequence_validation_v1",
        "training_config": (
            "parameters/RTDETR/"
            "rtdetr_fam_box_guided_sequence_validation_seed40.yaml"
        ),
        "training_config_sha256": (
            "16c1d58a4926bb6f6d1c018eb96600b52279116ca04a42d35286d63fb12ea647"
        ),
        "expected_grid_seeds": [40],
        "selected_runs": [{"seed": 40, "grid_index": 0}],
        "required_run_metadata": {
            "start_from_grid": 0,
            "start_from_run": 0,
            "search": "grid",
            "isolate_runs": True,
        },
    },
    {
        "source_id": "candidate_seeds41_44_expansion",
        "project": "RTDETR_FAM_BoxGuided_SequenceVal_FiveSeed",
        "group": "RTDETR_fam_box_guided_sequence_validation_v1",
        "training_config": (
            "parameters/RTDETR/"
            "rtdetr_fam_box_guided_sequence_validation_five_seed.yaml"
        ),
        "training_config_sha256": (
            "8080f8f384626a49bcc2fe0440294e584aeceac7b33e0a07e86d0280dd85c6b7"
        ),
        "expected_grid_seeds": [40, 41, 42, 43, 44],
        "selected_runs": [
            {"seed": 41, "grid_index": 1},
            {"seed": 42, "grid_index": 2},
            {"seed": 43, "grid_index": 3},
            {"seed": 44, "grid_index": 4},
        ],
        "required_run_metadata": {
            "start_from_grid": 0,
            "start_from_run": 1,
            "search": "grid",
            "isolate_runs": True,
        },
    },
]

EXPECTED_CONTROL_SOURCES = [
    {
        "source_id": "control_seed40_screen",
        "project": "RTDETR_FAM_BoxGuided_MatchedControl_Seed40",
        "group": "RTDETR_fam_box_guided_matched_control_v1",
        "training_config": (
            "parameters/RTDETR/"
            "rtdetr_fam_box_guided_matched_control_seed40.yaml"
        ),
        "training_config_sha256": (
            "811c5c15ede894337b7dee2fe862bf50a9fd24fb31026fdf2b6c5c61eac4fef6"
        ),
        "expected_grid_seeds": [40],
        "selected_runs": [{"seed": 40, "grid_index": 0}],
        "required_run_metadata": {
            "start_from_grid": 0,
            "start_from_run": 0,
            "search": "grid",
            "isolate_runs": True,
        },
    },
    {
        "source_id": "control_seeds41_44_expansion",
        "project": "RTDETR_FAM_BoxGuided_MatchedControl_Seeds41to44",
        "group": "RTDETR_fam_box_guided_matched_control_v1",
        "training_config": (
            "parameters/RTDETR/"
            "rtdetr_fam_box_guided_matched_control_seeds41_44.yaml"
        ),
        "training_config_sha256": (
            "917653f36fa4bb5eb78e990ae6a05ef7b55b0116b4ea4398540dcbbfa94263a4"
        ),
        "expected_grid_seeds": [41, 42, 43, 44],
        "selected_runs": [
            {"seed": 41, "grid_index": 0},
            {"seed": 42, "grid_index": 1},
            {"seed": 43, "grid_index": 2},
            {"seed": 44, "grid_index": 3},
        ],
        "required_run_metadata": {
            "start_from_grid": 0,
            "start_from_run": 0,
            "search": "grid",
            "isolate_runs": True,
        },
    },
]

EXPECTED_TARGET_POPULATION = {
    "pairing": "existing_wisard_sorted_zip",
    "target_order": "vis_x_vis_y_ir_minus_vis_dy_dx",
    "expected_frames": 3123,
    "expected_frames_with_matches": 2306,
    "expected_matched_boxes": 5209,
    "expected_max_matches_per_frame": 4,
    "expected_per_frame_match_count_distribution": EXPECTED_MATCH_DISTRIBUTION,
    "expected_source_inventory_sha256": EXPECTED_SOURCE_INVENTORY_SHA256,
    "expected_target_population_sha256": EXPECTED_TARGET_POPULATION_SHA256,
}

EXPECTED_COUNTERFACTUAL_PREREQUISITE = {
    "protocol_id": "rtdetr_fam_box_guided_counterfactual_v1",
    "protocol_path": (
        "parameters/RTDETR/rtdetr_fam_box_guided_counterfactual_seed40.yaml"
    ),
    "protocol_file_sha256": (
        "37eed97e8c468b21c1e581fa07d6d1d7bbbf2b790ac62417e14ce1255de67acd"
    ),
    "protocol_payload_sha256": (
        "b891a2395996f2c6a95992bdd272a211625eb19a805020d7bb9e60cfe7f9ec0c"
    ),
    "result_json": (
        "notes/Search_and_Rescue/results/"
        "rtdetr_fam_box_guided_counterfactual_v1.json"
    ),
    "schema_version": 1,
    "seed": 40,
    "project": "RTDETR_FAM_BoxGuided_SequenceVal_Seed40",
    "checkpoint": "best",
    "training_config_sha256": (
        "16c1d58a4926bb6f6d1c018eb96600b52279116ca04a42d35286d63fb12ea647"
    ),
    "require_protocol_complete": True,
    "require_diagnostic_gate_passed": True,
    "require_active_reproduction_passed": True,
    "require_guidance_weights_restored": True,
}

EXPECTED_MECHANISM_PREREQUISITE = {
    "protocol_id": "rtdetr_fam_box_guided_mechanism_audit_v1",
    "protocol_path": (
        "parameters/RTDETR/rtdetr_fam_box_guided_mechanism_audit_seed40.yaml"
    ),
    "protocol_file_sha256": (
        "2b7a5d97fafb7d0cd822bbf58edf90eda5f87020843b63f223f265a7c0cb7835"
    ),
    "protocol_payload_sha256": (
        "7278a05a279730d483e7108c3a8fe9d09f9400c6fbf1b0bfffbf783d2004e4f5"
    ),
    "result_json": (
        "notes/Search_and_Rescue/results/"
        "rtdetr_fam_box_guided_mechanism_audit_v1.json"
    ),
    "schema_version": 1,
    "seed": 40,
    "project": "RTDETR_FAM_BoxGuided_SequenceVal_Seed40",
    "checkpoint": "best",
    "training_config_sha256": (
        "16c1d58a4926bb6f6d1c018eb96600b52279116ca04a42d35286d63fb12ea647"
    ),
    "require_protocol_complete": True,
    "require_mechanism_gate_passed": True,
}

EXPECTED_PROMOTION_GATE = {
    "metric": "best_map_50",
    "delta": "candidate_minus_matched_control",
    "minimum_seed40_gain": 0.01,
    "minimum_mean_gain": 0.01,
    "minimum_positive_seed_wins": 4,
    "require_mechanism_all_passed": True,
    "rule": "all_conditions_required",
    "if_pass": "promote_box_guided_candidate_to_stage_b",
    "if_fail": "do_not_promote_box_guided_candidate",
}

EXPECTED_TOP_LEVEL_KEYS = {
    "protocol_id",
    "status",
    "seeds",
    "checkpoint",
    "split",
    "mode",
    "candidate_sources",
    "control_sources",
    "seed40_counterfactual_prerequisite",
    "seed40_mechanism_prerequisite",
    "implementation_sources",
    "box_matching",
    "diagnostic_baselines",
    "smooth_l1_beta_cells",
    "guidance_limit_cells",
    "near_saturation_threshold_cells",
    "expected_train_frames",
    "audit_batch_size",
    "expected_train_batches",
    "target_population",
    "mechanism_gate",
    "stage_a_promotion_gate",
    "output_json",
    "output_csv",
}

MECHANISM_CSV_FIELDS = [
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
]

RUNTIME_FINGERPRINT_KEYS = (
    "python",
    "torch",
    "cuda_runtime",
    "cuda_available",
    "cudnn",
    "gpu",
    "deterministic_algorithms",
    "deterministic_warn_only",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "cublas_workspace_config",
)


def validate_protocol(protocol):
    """Require the exact predeclared five-seed protocol."""
    if set(protocol) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("Unexpected five-seed audit protocol fields")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unexpected five-seed audit protocol_id")
    if protocol.get("status") != EXPECTED_STATUS:
        raise ValueError("Unexpected five-seed audit freeze status")
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("The audit requires ordered seeds 40--44")
    if (protocol.get("checkpoint"), protocol.get("split"), protocol.get("mode")) != (
        "best",
        "train",
        "fusion",
    ):
        raise ValueError("The audit requires best/train/fusion")
    if protocol.get("candidate_sources") != EXPECTED_CANDIDATE_SOURCES:
        raise ValueError("Unexpected candidate-source contract")
    if protocol.get("control_sources") != EXPECTED_CONTROL_SOURCES:
        raise ValueError("Unexpected control-source contract")
    if (
        protocol.get("seed40_counterfactual_prerequisite")
        != EXPECTED_COUNTERFACTUAL_PREREQUISITE
    ):
        raise ValueError("Unexpected seed-40 counterfactual prerequisite")
    if (
        protocol.get("seed40_mechanism_prerequisite")
        != EXPECTED_MECHANISM_PREREQUISITE
    ):
        raise ValueError("Unexpected seed-40 mechanism prerequisite")
    implementation_sources = protocol.get("implementation_sources")
    if not isinstance(implementation_sources, dict) or set(
        implementation_sources
    ) != set(EXPECTED_IMPLEMENTATION_SOURCE_PATHS):
        raise ValueError("Unexpected implementation-source paths")
    for digest in implementation_sources.values():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Implementation-source hashes must be lowercase SHA-256")
    if protocol.get("box_matching") != {
        "method": "mutual_nearest_box_center",
        "max_distance_normalized": 0.05,
    }:
        raise ValueError("Unexpected frozen box-matching contract")
    if protocol.get("diagnostic_baselines") != EXPECTED_DIAGNOSTIC_BASELINES:
        raise ValueError("Unexpected diagnostic-baseline contract")
    if (
        protocol.get("smooth_l1_beta_cells"),
        protocol.get("guidance_limit_cells"),
        protocol.get("near_saturation_threshold_cells"),
        protocol.get("expected_train_frames"),
        protocol.get("audit_batch_size"),
        protocol.get("expected_train_batches"),
    ) != (0.25, 4.0, 3.9, 3123, 12, 261):
        raise ValueError("Unexpected frozen mechanism replay settings")
    if protocol.get("target_population") != EXPECTED_TARGET_POPULATION:
        raise ValueError("Unexpected frozen target-population contract")
    if protocol.get("mechanism_gate") != EXPECTED_MECHANISM_GATE:
        raise ValueError("Unexpected mechanism-gate contract")
    if protocol.get("stage_a_promotion_gate") != EXPECTED_PROMOTION_GATE:
        raise ValueError("Unexpected Stage-A promotion gate")
    for key, suffix in (("output_json", ".json"), ("output_csv", ".csv")):
        output = protocol.get(key)
        if not isinstance(output, str) or not output.endswith(f"_v2{suffix}"):
            raise ValueError(f"{key} must be a versioned _v2{suffix} path")


def load_source_configs(sources, expected_seeds):
    """Bind every selected seed to an exact YAML grid item and launch source."""
    resolved = {}
    source_audits = {}
    for source in sources:
        config_path = (REPO_ROOT / source["training_config"]).resolve()
        digest = verify_training_config_file(
            config_path, source["training_config_sha256"]
        )
        raw = load_yaml(config_path)
        experiment = raw.get("experiment") or {}
        if experiment.get("name") != source["project"]:
            raise RuntimeError("Source project differs from its training YAML")
        if experiment.get("group") != source["group"]:
            raise RuntimeError("Source group differs from its training YAML")
        raw_launch = {
            key: experiment.get(key)
            for key in ("start_from_grid", "start_from_run", "search", "isolate_runs")
        }
        if raw_launch != {
            "start_from_grid": 0,
            "start_from_run": 0,
            "search": "grid",
            "isolate_runs": True,
        }:
            raise RuntimeError("Training YAML has unexpected base launch metadata")

        grid = make_grid(raw["parameters"])
        try:
            grid_seeds = [int(config["seed"]) for config in grid]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Training grid has an invalid seed expansion") from error
        if grid_seeds != source["expected_grid_seeds"]:
            raise RuntimeError("Training grid order differs from the frozen source")
        if len(grid) != len(grid_seeds) or len(set(grid_seeds)) != len(grid_seeds):
            raise RuntimeError("Training source must have one grid item per seed")

        for selection in source["selected_runs"]:
            seed = int(selection["seed"])
            index = int(selection["grid_index"])
            if index < 0 or index >= len(grid):
                raise RuntimeError("Selected training grid index is out of bounds")
            if int(grid[index]["seed"]) != seed:
                raise RuntimeError("Selected grid index does not map to its frozen seed")
            if seed in resolved:
                raise RuntimeError(f"Seed {seed} is assigned to multiple sources")
            resolved[seed] = {
                "config": grid[index],
                "grid_index": index,
                "source": deepcopy(source),
            }
        source_audits[source["source_id"]] = {
            "training_config": source["training_config"],
            "training_config_sha256": digest,
            "expanded_grid_seeds": grid_seeds,
        }

    if list(sorted(resolved)) != list(expected_seeds):
        raise RuntimeError(
            f"Source coverage differs from frozen seeds: got {sorted(resolved)}"
        )
    return resolved, source_audits


def verify_implementation_sources(protocol, *, repo_root=REPO_ROOT):
    """Bind this aggregator and both prerequisite audit implementations."""
    verified = {}
    for relative_path in EXPECTED_IMPLEMENTATION_SOURCE_PATHS:
        source_path = (Path(repo_root) / relative_path).resolve()
        try:
            source_path.relative_to(Path(repo_root).resolve())
        except ValueError as error:
            raise RuntimeError("Implementation source escapes repository root") from error
        actual = file_sha256(source_path)
        expected = protocol["implementation_sources"][relative_path]
        if actual != expected:
            raise RuntimeError(
                f"Implementation source changed: {relative_path}; "
                f"expected {expected}, got {actual}"
            )
        verified[relative_path] = actual
    return verified


def validate_matched_pair(candidate, control, protocol):
    """Require a FAM control differing only by the declared intervention."""
    validate_candidate_config(candidate, protocol)
    control_params = control.get("model", {}).get("params", {})
    if not control_params.get("use_fam"):
        raise ValueError("Matched control must enable FAM")
    if control_params.get("fam_variant") != "current_dcnv2":
        raise ValueError("Matched control must use historical DCNv2 FAM")
    if control.get("train", {}).get("box_guided_alignment") is not None:
        raise ValueError("Matched control unexpectedly enables guidance loss")
    if control.get("dataset", {}).get("box_alignment_targets") is not None:
        raise ValueError("Matched control unexpectedly enables guidance targets")

    normalized_candidate = deepcopy(candidate)
    normalized_control = deepcopy(control)
    normalized_candidate["model"]["params"]["fam_variant"] = "current_dcnv2"
    required_removals = (
        (normalized_candidate["train"], "box_guidance_lr"),
        (normalized_candidate["train"], "box_guided_alignment"),
        (normalized_candidate["dataset"], "box_alignment_targets"),
        (normalized_candidate["dataset"], "box_alignment_max_distance"),
    )
    for container, key in required_removals:
        if key not in container:
            raise ValueError(f"Candidate is missing declared intervention field {key}")
        container.pop(key)
    normalized_candidate.get("tracker", {}).pop("tags", None)
    normalized_control.get("tracker", {}).pop("tags", None)
    if _canonical_json_value(normalized_candidate) != _canonical_json_value(
        normalized_control
    ):
        raise RuntimeError("Candidate/control differ beyond the declared intervention")


def _stored_wandb_value(config, key):
    value = config.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def verify_stored_experiment_metadata(checkpoint_path, source):
    """Bind project/group and CLI grid offset to the stored W&B run."""
    checkpoint_path = Path(checkpoint_path).resolve()
    config_path = checkpoint_path.parent.parent / "config.yaml"
    if not config_path.is_file():
        raise RuntimeError("Stored W&B config is required for launch metadata")
    stored = load_yaml(config_path)
    experiment = _stored_wandb_value(stored, "experiment")
    if not isinstance(experiment, dict):
        raise RuntimeError("Stored W&B experiment metadata is invalid")
    expected = {
        "name": source["project"],
        "group": source["group"],
        **source["required_run_metadata"],
    }
    actual = {key: experiment.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            "Stored W&B launch metadata differs from the frozen source: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )
    return actual


def verify_runtime_trace_identity(artifact, seed):
    """Bind the run seed and environment to the first runtime trace event."""
    run_directory = artifact.get("run_directory")
    if not isinstance(run_directory, str):
        raise RuntimeError("Artifact audit did not expose a W&B run directory")
    trace_path = Path(run_directory) / "files" / "reproducibility_trace.jsonl"
    if not trace_path.is_file():
        raise RuntimeError("Reproducibility trace is missing from the W&B run")
    first_event = None
    with trace_path.open(encoding="utf-8") as trace_file:
        for line in trace_file:
            if line.strip():
                try:
                    first_event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError("First reproducibility trace event is invalid") from error
                break
    if not isinstance(first_event, dict) or first_event.get("event") != "runtime":
        raise RuntimeError("First reproducibility trace event must be runtime")
    expected_identity = {
        "seed": int(seed),
        "data_seed": int(seed),
        "training_seed": int(seed),
        "repetition": 0,
        "model_seed": None,
    }
    actual_identity = {key: first_event.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise RuntimeError(
            "Runtime trace seed identity differs from the frozen run: "
            + json.dumps(
                {"expected": expected_identity, "actual": actual_identity},
                sort_keys=True,
            )
        )
    missing = [key for key in RUNTIME_FINGERPRINT_KEYS if key not in first_event]
    if missing:
        raise RuntimeError(
            "Runtime trace is missing environment fingerprint fields: "
            + ", ".join(missing)
        )
    fingerprint = {key: first_event[key] for key in RUNTIME_FINGERPRINT_KEYS}
    return {"seed_identity": actual_identity, "environment": fingerprint}


def verify_common_runtime_environment(candidate_artifacts, control_artifacts):
    """Require one environment fingerprint across all ten matched runs."""
    fingerprints = {}
    for arm, artifacts in (
        ("candidate", candidate_artifacts),
        ("matched_control", control_artifacts),
    ):
        for seed in EXPECTED_SEEDS:
            runtime = artifacts[seed].get("runtime_identity") or {}
            fingerprints[f"{arm}:{seed}"] = runtime.get("environment")
    reference_key = "candidate:40"
    reference = fingerprints[reference_key]
    if not isinstance(reference, dict):
        raise RuntimeError("Candidate seed-40 runtime fingerprint is missing")
    mismatches = {
        key: value
        for key, value in fingerprints.items()
        if value != reference
    }
    if mismatches:
        raise RuntimeError(
            "The ten Stage-A runs do not share one runtime environment: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return reference


def _load_json_object(path, description):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required {description} is missing: {path.resolve()}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Required {description} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Required {description} must be a JSON object")
    return value


def verify_counterfactual_prerequisite(
    prerequisite,
    candidate_checkpoint,
    candidate_artifact,
    *,
    repo_root=REPO_ROOT,
):
    """Verify the passed seed-40 validation counterfactual and checkpoint identity."""
    protocol_path = (Path(repo_root) / prerequisite["protocol_path"]).resolve()
    if file_sha256(protocol_path) != prerequisite["protocol_file_sha256"]:
        raise RuntimeError("Seed-40 counterfactual protocol file changed")
    frozen_protocol = load_yaml(protocol_path)
    if stable_json_hash(_canonical_json_value(frozen_protocol)) != prerequisite[
        "protocol_payload_sha256"
    ]:
        raise RuntimeError("Seed-40 counterfactual protocol payload changed")
    protocol_identity = {
        "protocol_id": frozen_protocol.get("protocol_id"),
        "project": frozen_protocol.get("project"),
        "training_config_sha256": frozen_protocol.get("training_config_sha256"),
        "seed": frozen_protocol.get("seed"),
        "checkpoint": frozen_protocol.get("checkpoint"),
        "result_json": frozen_protocol.get("output_json"),
    }
    expected_identity = {
        key: prerequisite[key]
        for key in (
            "protocol_id",
            "project",
            "training_config_sha256",
            "seed",
            "checkpoint",
            "result_json",
        )
    }
    if protocol_identity != expected_identity:
        raise RuntimeError("Seed-40 counterfactual protocol identity changed")

    result_path = (Path(repo_root) / prerequisite["result_json"]).resolve()
    result = _load_json_object(result_path, "seed-40 counterfactual result")
    if result.get("schema_version") != prerequisite["schema_version"]:
        raise RuntimeError("Counterfactual result schema differs from the freeze")
    if result.get("protocol_id") != prerequisite["protocol_id"]:
        raise RuntimeError("Counterfactual result protocol_id differs from the freeze")
    # Counterfactual v1 records the raw protocol-file hash (see its prepare()).
    if result.get("protocol_sha256") != prerequisite["protocol_file_sha256"]:
        raise RuntimeError("Counterfactual result points to a different protocol")
    if prerequisite["require_protocol_complete"] and result.get(
        "protocol_complete"
    ) is not True:
        raise RuntimeError("Seed-40 counterfactual result is incomplete")

    checkpoint = result.get("checkpoint") or {}
    if checkpoint.get("checkpoint_sha256") != candidate_artifact.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError("Counterfactual used a different seed-40 checkpoint")
    if Path(checkpoint.get("checkpoint_path", "")).resolve() != Path(
        candidate_checkpoint
    ).resolve():
        raise RuntimeError("Counterfactual checkpoint path differs from five-seed audit")
    configuration_hashes = result.get("configuration_hashes") or {}
    if configuration_hashes.get("training_yaml_sha256") != prerequisite[
        "training_config_sha256"
    ]:
        raise RuntimeError("Counterfactual used a different training YAML")
    if checkpoint.get("training_source_manifest_sha256") != candidate_artifact.get(
        "training_source_manifest_sha256"
    ):
        raise RuntimeError("Counterfactual training-source manifest differs")

    condition_rows = result.get("results") or []
    if [row.get("condition") for row in condition_rows] != ["active", "zero"]:
        raise RuntimeError("Counterfactual active/zero result set is incomplete")
    try:
        active_map = float(condition_rows[0]["metrics"]["map_50"])
        zero_map = float(condition_rows[1]["metrics"]["map_50"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Counterfactual map_50 values are invalid") from error
    if not math.isfinite(active_map) or not math.isfinite(zero_map):
        raise RuntimeError("Counterfactual map_50 values are non-finite")
    if not 0.0 <= active_map <= 1.0 or not 0.0 <= zero_map <= 1.0:
        raise RuntimeError("Counterfactual map_50 values must lie in [0, 1]")
    expected_samples = int(frozen_protocol["source"]["expected_paired_frames"])
    expected_batches = int(frozen_protocol["expected_batches"])
    for condition in condition_rows:
        if (
            int(condition.get("n_samples", -1)) != expected_samples
            or int(condition.get("n_batches", -1)) != expected_batches
        ):
            raise RuntimeError("Counterfactual condition replay is incomplete")
    evaluation = result.get("evaluation") or {}
    identity = evaluation.get("identity") or {}
    if (
        identity.get("same_sample_order") is not True
        or identity.get("same_ground_truth") is not True
    ):
        raise RuntimeError("Counterfactual condition identity checks did not pass")
    for digest_key in ("sample_indices_sha256", "ground_truth_sha256"):
        values = [condition.get(digest_key) for condition in condition_rows]
        if (
            len(set(values)) != 1
            or values[0] != identity.get(digest_key)
            or not isinstance(values[0], str)
            or len(values[0]) != 64
        ):
            raise RuntimeError("Counterfactual condition identity digests differ")
    expected_sample_indices_sha256 = stable_json_hash(list(range(expected_samples)))
    if identity["sample_indices_sha256"] != expected_sample_indices_sha256:
        raise RuntimeError("Counterfactual sample-index order differs from 0..895")

    source_inventory = result.get("source_inventory") or {}
    frozen_source = frozen_protocol["source"]
    expected_inventory = {
        "vis_inventory": int(frozen_source["expected_vis_inventory"]),
        "ir_inventory": int(frozen_source["expected_ir_inventory"]),
        "paired_frames": expected_samples,
        "unpaired_vis": [frozen_source["expected_unpaired_vis_terminal"]],
        "vis_boxes": int(frozen_source["expected_vis_boxes"]),
        "vis_empty_frames": int(frozen_source["expected_vis_empty_frames"]),
        "ir_boxes": int(frozen_source["expected_ir_boxes"]),
        "ir_empty_frames": int(frozen_source["expected_ir_empty_frames"]),
        "inventory_sha256": frozen_source["expected_inventory_sha256"],
        "content_inventory_sha256": frozen_source[
            "expected_content_inventory_sha256"
        ],
        "sample_order_sha256": frozen_source["expected_sample_order_sha256"],
    }
    if {key: source_inventory.get(key) for key in expected_inventory} != expected_inventory:
        raise RuntimeError("Counterfactual source inventory differs from the freeze")

    implementation_hashes = result.get("implementation_source_sha256") or {}
    if set(implementation_hashes) != set(
        EXPECTED_COUNTERFACTUAL_RESULT_IMPLEMENTATION_PATHS
    ):
        raise RuntimeError("Counterfactual implementation-source set differs")
    for relative_path in EXPECTED_COUNTERFACTUAL_RESULT_IMPLEMENTATION_PATHS:
        if implementation_hashes[relative_path] != file_sha256(
            Path(repo_root) / relative_path
        ):
            raise RuntimeError(
                f"Counterfactual implementation source changed: {relative_path}"
            )
    gate = result.get("diagnostic_gate") or {}
    observed_delta = active_map - zero_map
    if not math.isclose(
        float(gate.get("active_minus_zero_map_50", math.nan)),
        observed_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Counterfactual gate delta does not match its results")
    if prerequisite["require_diagnostic_gate_passed"] and (
        gate.get("passed") is not True or observed_delta < 0.0
    ):
        raise RuntimeError("Seed-40 counterfactual gate did not pass")

    reproduction = evaluation.get("active_reproduction") or {}
    tolerance = float(reproduction.get("absolute_tolerance", math.nan))
    expected_best = float(candidate_artifact.get("best_map_50", math.nan))
    reproduction_error = abs(active_map - expected_best)
    if prerequisite["require_active_reproduction_passed"] and (
        reproduction.get("passed") is not True
        or not math.isfinite(tolerance)
        or tolerance != 0.0002
        or reproduction_error > tolerance
    ):
        raise RuntimeError("Counterfactual active replay did not reproduce best_map_50")
    intervention = evaluation.get("guidance_final_layer_intervention") or {}
    if prerequisite["require_guidance_weights_restored"] and intervention.get(
        "restored_exactly"
    ) is not True:
        raise RuntimeError("Counterfactual did not restore guidance weights exactly")
    return {
        "result_path": str(result_path),
        "result_sha256": file_sha256(result_path),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "active_minus_zero_map_50": observed_delta,
        "passed": True,
    }


def verify_mechanism_prerequisite(
    prerequisite,
    candidate_checkpoint,
    candidate_artifact,
    *,
    repo_root=REPO_ROOT,
):
    """Verify the passed seed-40 mechanism artifact used before expansion."""
    protocol_path = (Path(repo_root) / prerequisite["protocol_path"]).resolve()
    if file_sha256(protocol_path) != prerequisite["protocol_file_sha256"]:
        raise RuntimeError("Seed-40 mechanism protocol file changed")
    frozen_protocol = load_yaml(protocol_path)
    if stable_json_hash(_canonical_json_value(frozen_protocol)) != prerequisite[
        "protocol_payload_sha256"
    ]:
        raise RuntimeError("Seed-40 mechanism protocol payload changed")
    protocol_identity = {
        "protocol_id": frozen_protocol.get("protocol_id"),
        "project": frozen_protocol.get("project"),
        "training_config_sha256": frozen_protocol.get("training_config_sha256"),
        "checkpoint": frozen_protocol.get("checkpoint"),
        "result_json": frozen_protocol.get("output_json"),
    }
    expected_identity = {
        key: prerequisite[key]
        for key in (
            "protocol_id",
            "project",
            "training_config_sha256",
            "checkpoint",
            "result_json",
        )
    }
    if protocol_identity != expected_identity or frozen_protocol.get("seeds") != [
        prerequisite["seed"]
    ]:
        raise RuntimeError("Seed-40 mechanism protocol identity changed")

    result_path = (Path(repo_root) / prerequisite["result_json"]).resolve()
    result = _load_json_object(result_path, "seed-40 mechanism result")
    if result.get("schema_version") != prerequisite["schema_version"]:
        raise RuntimeError("Mechanism result schema differs from the freeze")
    if result.get("protocol_sha256") != prerequisite["protocol_payload_sha256"]:
        raise RuntimeError("Mechanism result points to a different protocol")
    if _canonical_json_value(result.get("protocol")) != _canonical_json_value(
        frozen_protocol
    ):
        raise RuntimeError("Mechanism result embedded protocol differs from its file")
    if result.get("training_config_sha256") != prerequisite[
        "training_config_sha256"
    ]:
        raise RuntimeError("Mechanism result used a different training YAML")
    if prerequisite["require_protocol_complete"] and result.get(
        "protocol_complete"
    ) is not True:
        raise RuntimeError("Seed-40 mechanism result is incomplete")
    if prerequisite["require_mechanism_gate_passed"] and result.get(
        "mechanism_gate_all_passed"
    ) is not True:
        raise RuntimeError("Seed-40 mechanism gate did not pass")

    rows = result.get("seed_rows") or []
    if len(rows) != 1 or int(rows[0].get("seed", -1)) != prerequisite["seed"]:
        raise RuntimeError("Mechanism result does not contain exactly seed 40")
    if rows[0].get("mechanism_gate_passed") is not True:
        raise RuntimeError("Seed-40 mechanism row did not pass")
    recomputed_gate = mechanism_gate(rows[0], frozen_protocol["mechanism_gate"])
    if recomputed_gate.get("passed") is not True:
        raise RuntimeError("Seed-40 mechanism metrics do not pass the frozen gate")

    population = frozen_protocol["target_population"]
    expected_row_population = {
        "batches": int(frozen_protocol["expected_train_batches"]),
        "frames": int(population["expected_frames"]),
        "frames_with_matches": int(population["expected_frames_with_matches"]),
        "matched_boxes": int(population["expected_matched_boxes"]),
    }
    try:
        observed_row_population = {
            key: int(rows[0].get(key, -1)) for key in expected_row_population
        }
    except (TypeError, ValueError) as error:
        raise RuntimeError("Seed-40 mechanism replay population is invalid") from error
    if observed_row_population != expected_row_population:
        raise RuntimeError("Seed-40 mechanism replay population differs from the freeze")

    expected_population_digests = {
        "target_population_sha256": population[
            "expected_target_population_sha256"
        ],
        "source_inventory_sha256": population[
            "expected_source_inventory_sha256"
        ],
    }
    if {
        key: rows[0].get(key) for key in expected_population_digests
    } != expected_population_digests:
        raise RuntimeError("Seed-40 mechanism row population digests differ")

    source_inventory = result.get("source_inventory") or {}
    expected_source_inventory = {
        "frames": population["expected_frames"],
        "frames_with_matches": population["expected_frames_with_matches"],
        "matched_boxes": population["expected_matched_boxes"],
        "max_matches_per_frame": population["expected_max_matches_per_frame"],
        "per_frame_match_count_distribution": population[
            "expected_per_frame_match_count_distribution"
        ],
        **expected_population_digests,
    }
    observed_source_inventory = {
        key: source_inventory.get(key) for key in expected_source_inventory
    }
    if _canonical_json_value(observed_source_inventory) != _canonical_json_value(
        expected_source_inventory
    ):
        raise RuntimeError("Seed-40 mechanism source inventory differs from the freeze")

    checkpoint = (result.get("checkpoint_audits") or {}).get("40") or {}
    if checkpoint.get("checkpoint_sha256") != candidate_artifact.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError("Mechanism audit used a different seed-40 checkpoint")
    if Path(checkpoint.get("path", "")).resolve() != Path(
        candidate_checkpoint
    ).resolve():
        raise RuntimeError("Mechanism checkpoint path differs from five-seed audit")
    if checkpoint.get("training_source_manifest_sha256") != candidate_artifact.get(
        "training_source_manifest_sha256"
    ):
        raise RuntimeError("Mechanism training-source manifest differs")
    checkpoint_gate = checkpoint.get("mechanism_gate") or {}
    if checkpoint_gate.get("passed") is not True:
        raise RuntimeError("Mechanism checkpoint audit did not record a passed gate")
    return {
        "result_path": str(result_path),
        "result_sha256": file_sha256(result_path),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "passed": True,
    }


def resolve_arm_artifacts(entries, protocol):
    checkpoints = {}
    artifacts = {}
    for seed in protocol["seeds"]:
        entry = entries[int(seed)]
        source = entry["source"]
        checkpoint = resolve_local_wandb_checkpoint(
            project=source["project"],
            seed=seed,
            checkpoint=protocol["checkpoint"],
            wandb_root=REPO_ROOT / "wandb",
        )
        artifact = verify_local_run_artifacts(
            checkpoint,
            project=source["project"],
            seed=seed,
            scientific_config=entry["config"],
            expected_train_frames=protocol["expected_train_frames"],
        )
        best_map50 = artifact.get("best_map_50")
        if (
            not isinstance(best_map50, (int, float))
            or not math.isfinite(float(best_map50))
            or not 0.0 <= float(best_map50) <= 1.0
        ):
            raise RuntimeError("Stored best_map_50 must be finite and lie in [0, 1]")
        artifact["stored_experiment_metadata"] = verify_stored_experiment_metadata(
            checkpoint, source
        )
        artifact["runtime_identity"] = verify_runtime_trace_identity(artifact, seed)
        artifact["source_id"] = source["source_id"]
        artifact["project"] = source["project"]
        artifact["training_config"] = source["training_config"]
        artifact["training_config_sha256"] = source["training_config_sha256"]
        artifact["grid_index"] = entry["grid_index"]
        checkpoints[int(seed)] = checkpoint
        artifacts[int(seed)] = artifact
    return checkpoints, artifacts


def strict_load_control_checkpoints(entries, checkpoints, artifacts, device):
    """Prove exact state-dict coverage for every matched-control checkpoint."""
    for seed in EXPECTED_SEEDS:
        model, state_dict_audit = load_fusion_model_strict(
            entries[seed]["config"]["model"], checkpoints[seed], device
        )
        artifacts[seed]["state_dict"] = state_dict_audit
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def paired_stage_a_summary(rows, gate):
    """Aggregate paired seed deltas and apply the predeclared promotion rule."""
    if [int(row["seed"]) for row in rows] != EXPECTED_SEEDS:
        raise ValueError("Paired Stage-A summary requires ordered seeds 40--44")
    deltas = [float(row["best_map_50_delta"]) for row in rows]
    if not all(math.isfinite(value) for value in deltas):
        raise ValueError("Paired best-mAP deltas must be finite")
    mean_gain = statistics.fmean(deltas)
    wins = sum(value > 0.0 for value in deltas)
    mechanism_passes = sum(bool(row["mechanism_gate_passed"]) for row in rows)
    checks = {
        "seed40_gain_at_least_minimum": deltas[0]
        >= float(gate["minimum_seed40_gain"]),
        "mean_gain_at_least_minimum": mean_gain
        >= float(gate["minimum_mean_gain"]),
        "positive_seed_wins_at_least_minimum": wins
        >= int(gate["minimum_positive_seed_wins"]),
        "mechanism_passed_all_seeds": (
            mechanism_passes == len(EXPECTED_SEEDS)
            if gate["require_mechanism_all_passed"]
            else True
        ),
    }
    passed = all(checks.values())
    return {
        "experimental_unit": "paired_checkpoint_seed",
        "seed_deltas": {
            str(row["seed"]): float(row["best_map_50_delta"]) for row in rows
        },
        "mean_gain": mean_gain,
        "sample_std": statistics.stdev(deltas),
        "positive_seed_wins": wins,
        "mechanism_seed_passes": mechanism_passes,
        "checks": checks,
        "passed": passed,
        "status": gate["if_pass"] if passed else gate["if_fail"],
    }


def _write_csv_temp(path, rows):
    fieldnames = [
        "protocol_id",
        "seed",
        "candidate_project",
        "control_project",
        "candidate_grid_index",
        "control_grid_index",
        "candidate_checkpoint",
        "control_checkpoint",
        "candidate_best_map_50",
        "control_best_map_50",
        "best_map_50_delta",
        *MECHANISM_CSV_FIELDS,
        "mechanism_gate_passed",
        "mechanism_gate_status",
        "candidate_checkpoint_sha256",
        "control_checkpoint_sha256",
        "candidate_scientific_config_sha256",
        "control_scientific_config_sha256",
        "candidate_training_config_sha256",
        "control_training_config_sha256",
        "target_population_sha256",
        "source_inventory_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_results_atomic(json_path, csv_path, result, rows):
    """Prepare both files fully and publish JSON last as the completion marker."""
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(
            "Five-seed audit outputs already exist; preserve the versioned evidence"
        )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = json_path.with_name(json_path.name + ".tmp")
    temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
    try:
        temporary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        _write_csv_temp(temporary_csv, rows)
        temporary_csv.replace(csv_path)
        temporary_json.replace(json_path)
    finally:
        temporary_json.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)


def run(protocol_path=DEFAULT_PROTOCOL, dry_run=False):
    protocol = load_yaml(protocol_path)
    validate_protocol(protocol)
    protocol_digest = stable_json_hash(_canonical_json_value(protocol))
    json_path = REPO_ROOT / protocol["output_json"]
    csv_path = REPO_ROOT / protocol["output_csv"]
    if not dry_run and (json_path.exists() or csv_path.exists()):
        raise FileExistsError(
            "Five-seed audit output already exists; refusing to rerun or overwrite"
        )
    implementation_sources = verify_implementation_sources(protocol)

    candidate_entries, candidate_source_audits = load_source_configs(
        protocol["candidate_sources"], protocol["seeds"]
    )
    control_entries, control_source_audits = load_source_configs(
        protocol["control_sources"], protocol["seeds"]
    )
    for seed in protocol["seeds"]:
        validate_matched_pair(
            candidate_entries[int(seed)]["config"],
            control_entries[int(seed)]["config"],
            protocol,
        )

    first_candidate = candidate_entries[EXPECTED_SEEDS[0]]["config"]
    frozen_dataset = audit_dataset_config(first_candidate["dataset"], protocol)
    for seed in protocol["seeds"]:
        candidate_dataset = audit_dataset_config(
            candidate_entries[int(seed)]["config"]["dataset"], protocol
        )
        if _canonical_json_value(candidate_dataset) != _canonical_json_value(
            frozen_dataset
        ):
            raise RuntimeError("Candidate seeds use different audit datasets")
    population_manifest = build_target_population_manifest(frozen_dataset)
    verify_target_population(population_manifest, protocol["target_population"])

    candidate_checkpoints, candidate_artifacts = resolve_arm_artifacts(
        candidate_entries, protocol
    )
    control_checkpoints, control_artifacts = resolve_arm_artifacts(
        control_entries, protocol
    )
    common_runtime_environment = verify_common_runtime_environment(
        candidate_artifacts, control_artifacts
    )
    seed40_prerequisites = {
        "counterfactual": verify_counterfactual_prerequisite(
            protocol["seed40_counterfactual_prerequisite"],
            candidate_checkpoints[40],
            candidate_artifacts[40],
        ),
        "mechanism": verify_mechanism_prerequisite(
            protocol["seed40_mechanism_prerequisite"],
            candidate_checkpoints[40],
            candidate_artifacts[40],
        ),
    }
    if dry_run:
        print(
            "Dry run OK: 5 candidate and 5 matched-control checkpoints, exact "
            "YAML/grid/launch/source identity and frozen target population verified; "
            "no result files written"
        )
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strict_load_control_checkpoints(
        control_entries,
        control_checkpoints,
        control_artifacts,
        device,
    )
    rows = []
    for seed in protocol["seeds"]:
        seed = int(seed)
        candidate_entry = candidate_entries[seed]
        candidate_config = candidate_entry["config"]
        dataset_config = audit_dataset_config(candidate_config["dataset"], protocol)
        dataloader_config = deepcopy(candidate_config["dataloader"])
        dataloader_config["batch_size"] = int(protocol["audit_batch_size"])
        (train_loader, _val, _test), _denormalize = get_dataloaders(
            dataset_config,
            dataloader_config,
            seed=seed,
        )
        model, state_dict_audit = load_fusion_model_strict(
            candidate_config["model"], candidate_checkpoints[seed], device
        )
        print(f"Auditing seed={seed} on frozen Stage-A train fusion replay")
        mechanism_summary, runtime_population = audit_seed(
            model,
            train_loader,
            protocol,
            device,
            item_identifiers=population_manifest["item_identifiers"],
        )
        mechanism_decision = mechanism_gate(
            mechanism_summary, protocol["mechanism_gate"]
        )
        candidate_best = float(candidate_artifacts[seed]["best_map_50"])
        control_best = float(control_artifacts[seed]["best_map_50"])
        row = {
            "protocol_id": protocol["protocol_id"],
            "seed": seed,
            "candidate_project": candidate_entry["source"]["project"],
            "control_project": control_entries[seed]["source"]["project"],
            "candidate_grid_index": candidate_entry["grid_index"],
            "control_grid_index": control_entries[seed]["grid_index"],
            "candidate_checkpoint": candidate_checkpoints[seed],
            "control_checkpoint": control_checkpoints[seed],
            "candidate_best_map_50": candidate_best,
            "control_best_map_50": control_best,
            "best_map_50_delta": candidate_best - control_best,
            **mechanism_summary,
            "mechanism_gate_passed": mechanism_decision["passed"],
            "mechanism_gate_status": mechanism_decision["status"],
            "candidate_checkpoint_sha256": candidate_artifacts[seed][
                "checkpoint_sha256"
            ],
            "control_checkpoint_sha256": control_artifacts[seed][
                "checkpoint_sha256"
            ],
            "candidate_scientific_config_sha256": candidate_artifacts[seed][
                "scientific_config_sha256"
            ],
            "control_scientific_config_sha256": control_artifacts[seed][
                "scientific_config_sha256"
            ],
            "candidate_training_config_sha256": candidate_entry["source"][
                "training_config_sha256"
            ],
            "control_training_config_sha256": control_entries[seed]["source"][
                "training_config_sha256"
            ],
            "target_population_sha256": runtime_population[
                "target_population_sha256"
            ],
            "source_inventory_sha256": population_manifest[
                "source_inventory_sha256"
            ],
        }
        rows.append(row)
        candidate_artifacts[seed]["state_dict"] = state_dict_audit
        candidate_artifacts[seed]["mechanism_gate"] = mechanism_decision
        del model, train_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    promotion = paired_stage_a_summary(rows, protocol["stage_a_promotion_gate"])
    result = {
        "schema_version": 2,
        "protocol": protocol,
        "protocol_sha256": protocol_digest,
        "implementation_sources_sha256": implementation_sources,
        "training_sources": {
            "candidate": candidate_source_audits,
            "matched_control": control_source_audits,
        },
        "common_runtime_environment": common_runtime_environment,
        "seed40_screen_prerequisites": seed40_prerequisites,
        "source_inventory": {
            key: value
            for key, value in population_manifest.items()
            if key != "item_identifiers"
        },
        "checkpoint_audits": {
            "candidate": {
                str(seed): {
                    "path": candidate_checkpoints[seed],
                    **candidate_artifacts[seed],
                }
                for seed in EXPECTED_SEEDS
            },
            "matched_control": {
                str(seed): {
                    "path": control_checkpoints[seed],
                    **control_artifacts[seed],
                }
                for seed in EXPECTED_SEEDS
            },
        },
        "seed_rows": rows,
        "cross_seed_mechanism_summary": summarize_across_seeds(rows),
        "stage_a_promotion": promotion,
        "protocol_complete": len(rows) == 5
        and [row["seed"] for row in rows] == EXPECTED_SEEDS,
    }
    write_results_atomic(json_path, csv_path, result, rows)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(Path(args.protocol), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
