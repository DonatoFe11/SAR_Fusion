import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from sarfusion.data.temporal_split import stable_json_hash
from sarfusion.utils.utils import load_yaml
from scripts.run_rtdetr_fam_box_guided_mechanism_audit import (
    file_sha256 as real_file_sha256,
)
from scripts.run_rtdetr_fam_box_guided_stage_a_five_seed_audit import (
    DEFAULT_PROTOCOL,
    EXPECTED_COUNTERFACTUAL_RESULT_IMPLEMENTATION_PATHS,
    EXPECTED_SEEDS,
    load_source_configs,
    paired_stage_a_summary,
    resolve_arm_artifacts,
    run,
    strict_load_control_checkpoints,
    validate_matched_pair,
    validate_protocol,
    verify_common_runtime_environment,
    verify_counterfactual_prerequisite,
    verify_mechanism_prerequisite,
    verify_implementation_sources,
    verify_runtime_trace_identity,
    verify_stored_experiment_metadata,
    write_results_atomic,
)


class TestBoxGuidedFiveSeedStageAAudit(unittest.TestCase):
    def test_protocol_and_real_grid_indices_are_frozen(self):
        protocol = load_yaml(DEFAULT_PROTOCOL)
        validate_protocol(protocol)
        candidates, _ = load_source_configs(
            protocol["candidate_sources"], protocol["seeds"]
        )
        controls, _ = load_source_configs(
            protocol["control_sources"], protocol["seeds"]
        )
        self.assertEqual(sorted(candidates), EXPECTED_SEEDS)
        self.assertEqual(
            [candidates[seed]["grid_index"] for seed in EXPECTED_SEEDS],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [controls[seed]["grid_index"] for seed in EXPECTED_SEEDS],
            [0, 0, 1, 2, 3],
        )
        self.assertEqual(
            candidates[41]["source"]["required_run_metadata"]["start_from_run"],
            1,
        )
        self.assertEqual(
            controls[41]["source"]["required_run_metadata"]["start_from_run"],
            0,
        )

    def test_protocol_rejects_source_or_gate_drift(self):
        protocol = load_yaml(DEFAULT_PROTOCOL)
        changed = deepcopy(protocol)
        changed["candidate_sources"][1]["selected_runs"][0]["grid_index"] = 0
        with self.assertRaisesRegex(ValueError, "candidate-source"):
            validate_protocol(changed)

        changed = deepcopy(protocol)
        changed["stage_a_promotion_gate"]["minimum_mean_gain"] = 0.0
        with self.assertRaisesRegex(ValueError, "promotion"):
            validate_protocol(changed)

        changed = deepcopy(protocol)
        first_path = next(iter(changed["implementation_sources"]))
        changed["implementation_sources"][first_path] = "0" * 64
        validate_protocol(changed)
        with self.assertRaisesRegex(RuntimeError, "Implementation source changed"):
            verify_implementation_sources(changed)

    def test_every_candidate_has_an_exact_matched_control(self):
        protocol = load_yaml(DEFAULT_PROTOCOL)
        candidates, _ = load_source_configs(
            protocol["candidate_sources"], protocol["seeds"]
        )
        controls, _ = load_source_configs(
            protocol["control_sources"], protocol["seeds"]
        )
        for seed in EXPECTED_SEEDS:
            validate_matched_pair(
                candidates[seed]["config"], controls[seed]["config"], protocol
            )

    def test_stored_wandb_metadata_binds_cli_start_offset(self):
        protocol = load_yaml(DEFAULT_PROTOCOL)
        source = protocol["candidate_sources"][1]
        with TemporaryDirectory() as temporary_directory:
            files = Path(temporary_directory) / "run-test" / "files"
            checkpoint = files / "best" / "model.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            experiment = {
                "name": source["project"],
                "group": source["group"],
                **source["required_run_metadata"],
            }
            (files / "config.yaml").write_text(
                yaml.safe_dump({"experiment": {"value": experiment}}),
                encoding="utf-8",
            )
            verified = verify_stored_experiment_metadata(checkpoint, source)
            self.assertEqual(verified["start_from_run"], 1)

            experiment["start_from_run"] = 0
            (files / "config.yaml").write_text(
                yaml.safe_dump({"experiment": {"value": experiment}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "launch metadata"):
                verify_stored_experiment_metadata(checkpoint, source)

    def test_checkpoint_best_map50_is_range_checked(self):
        protocol = load_yaml(DEFAULT_PROTOCOL)
        candidates, _ = load_source_configs(
            protocol["candidate_sources"], protocol["seeds"]
        )
        module = "scripts.run_rtdetr_fam_box_guided_stage_a_five_seed_audit."
        with (
            patch(
                module + "resolve_local_wandb_checkpoint",
                return_value="/tmp/files/best/model.safetensors",
            ),
            patch(
                module + "verify_local_run_artifacts",
                return_value={"best_map_50": 1.01},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, r"lie in \[0, 1\]"):
                resolve_arm_artifacts(candidates, protocol)

    def test_all_control_checkpoints_are_strict_loaded_without_inference(self):
        protocol = load_yaml(DEFAULT_PROTOCOL)
        controls, _ = load_source_configs(
            protocol["control_sources"], protocol["seeds"]
        )
        checkpoints = {seed: f"/tmp/control-{seed}.safetensors" for seed in EXPECTED_SEEDS}
        artifacts = {seed: {} for seed in EXPECTED_SEEDS}
        module = "scripts.run_rtdetr_fam_box_guided_stage_a_five_seed_audit."
        with (
            patch(
                module + "load_fusion_model_strict",
                side_effect=[
                    (object(), {"checkpoint_key_count": 1000 + seed})
                    for seed in EXPECTED_SEEDS
                ],
            ) as loader,
            patch(module + "torch.cuda.is_available", return_value=False),
        ):
            strict_load_control_checkpoints(
                controls, checkpoints, artifacts, "cpu"
            )
        self.assertEqual(loader.call_count, 5)
        self.assertEqual(
            artifacts[44]["state_dict"]["checkpoint_key_count"], 1044
        )

    def test_paired_gate_requires_mean_wins_and_mechanism_five_of_five(self):
        protocol = load_yaml(DEFAULT_PROTOCOL)
        deltas = [0.02, 0.02, 0.01, 0.01, -0.01]
        rows = [
            {
                "seed": seed,
                "best_map_50_delta": delta,
                "mechanism_gate_passed": True,
            }
            for seed, delta in zip(EXPECTED_SEEDS, deltas)
        ]
        summary = paired_stage_a_summary(
            rows, protocol["stage_a_promotion_gate"]
        )
        self.assertAlmostEqual(summary["mean_gain"], 0.01)
        self.assertEqual(summary["positive_seed_wins"], 4)
        self.assertTrue(summary["passed"])

        rows[0]["mechanism_gate_passed"] = False
        summary = paired_stage_a_summary(
            rows, protocol["stage_a_promotion_gate"]
        )
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["checks"]["mechanism_passed_all_seeds"])

        rows = [
            {
                "seed": seed,
                "best_map_50_delta": delta,
                "mechanism_gate_passed": True,
            }
            for seed, delta in zip(
                EXPECTED_SEEDS, [0.009, 0.02, 0.02, 0.02, -0.001]
            )
        ]
        summary = paired_stage_a_summary(
            rows, protocol["stage_a_promotion_gate"]
        )
        self.assertGreater(summary["mean_gain"], 0.01)
        self.assertEqual(summary["positive_seed_wins"], 4)
        self.assertFalse(summary["checks"]["seed40_gain_at_least_minimum"])
        self.assertFalse(summary["passed"])

    def test_runtime_trace_binds_seed_and_one_environment(self):
        with TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / "run-test"
            files = run_directory / "files"
            files.mkdir(parents=True)
            runtime = {
                "event": "runtime",
                "seed": 40,
                "data_seed": 40,
                "training_seed": 40,
                "repetition": 0,
                "model_seed": None,
                "python": "3.12.2",
                "torch": "2.4.0",
                "cuda_runtime": "12.1",
                "cuda_available": True,
                "cudnn": 90100,
                "gpu": "GPU",
                "deterministic_algorithms": False,
                "deterministic_warn_only": False,
                "cudnn_deterministic": False,
                "cudnn_benchmark": False,
                "cublas_workspace_config": ":4096:8",
            }
            trace = files / "reproducibility_trace.jsonl"
            trace.write_text(json.dumps(runtime) + "\n", encoding="utf-8")
            identity = verify_runtime_trace_identity(
                {"run_directory": str(run_directory)}, 40
            )
            self.assertEqual(identity["seed_identity"]["training_seed"], 40)

            artifacts = {
                seed: {"runtime_identity": deepcopy(identity)}
                for seed in EXPECTED_SEEDS
            }
            environment = verify_common_runtime_environment(artifacts, artifacts)
            self.assertEqual(environment["gpu"], "GPU")
            artifacts[44]["runtime_identity"]["environment"]["torch"] = "changed"
            with self.assertRaisesRegex(RuntimeError, "one runtime environment"):
                verify_common_runtime_environment(artifacts, deepcopy(artifacts))

            runtime["training_seed"] = 41
            trace.write_text(json.dumps(runtime) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "seed identity"):
                verify_runtime_trace_identity(
                    {"run_directory": str(run_directory)}, 40
                )

    def test_seed40_prerequisites_require_same_checkpoint_and_passed_gates(self):
        protocol = load_yaml(DEFAULT_PROTOCOL)
        checkpoint_path = "/tmp/candidate-seed40/files/best/model.safetensors"
        artifact = {
            "checkpoint_sha256": "a" * 64,
            "best_map_50": 0.2,
            "training_source_manifest_sha256": "b" * 64,
        }
        counterfactual = {
            "schema_version": 1,
            "protocol_id": protocol["seed40_counterfactual_prerequisite"][
                "protocol_id"
            ],
            "protocol_sha256": protocol["seed40_counterfactual_prerequisite"][
                "protocol_file_sha256"
            ],
            "protocol_complete": True,
            "checkpoint": {
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": "a" * 64,
                "training_source_manifest_sha256": "b" * 64,
            },
            "configuration_hashes": {
                "training_yaml_sha256": protocol[
                    "seed40_counterfactual_prerequisite"
                ]["training_config_sha256"]
            },
            "results": [
                {
                    "condition": "active",
                    "n_samples": 896,
                    "n_batches": 75,
                    "sample_indices_sha256": stable_json_hash(list(range(896))),
                    "ground_truth_sha256": "e" * 64,
                    "metrics": {"map_50": 0.2},
                },
                {
                    "condition": "zero",
                    "n_samples": 896,
                    "n_batches": 75,
                    "sample_indices_sha256": stable_json_hash(list(range(896))),
                    "ground_truth_sha256": "e" * 64,
                    "metrics": {"map_50": 0.19},
                },
            ],
            "diagnostic_gate": {
                "active_minus_zero_map_50": 0.01,
                "passed": True,
            },
            "evaluation": {
                "identity": {
                    "same_sample_order": True,
                    "same_ground_truth": True,
                    "sample_indices_sha256": stable_json_hash(list(range(896))),
                    "ground_truth_sha256": "e" * 64,
                },
                "active_reproduction": {
                    "passed": True,
                    "absolute_tolerance": 0.0002,
                },
                "guidance_final_layer_intervention": {"restored_exactly": True},
            },
        }
        frozen_counterfactual = load_yaml(
            Path(__file__).resolve().parents[1]
            / protocol["seed40_counterfactual_prerequisite"]["protocol_path"]
        )
        source = frozen_counterfactual["source"]
        counterfactual["source_inventory"] = {
            "vis_inventory": source["expected_vis_inventory"],
            "ir_inventory": source["expected_ir_inventory"],
            "paired_frames": source["expected_paired_frames"],
            "unpaired_vis": [source["expected_unpaired_vis_terminal"]],
            "vis_boxes": source["expected_vis_boxes"],
            "vis_empty_frames": source["expected_vis_empty_frames"],
            "ir_boxes": source["expected_ir_boxes"],
            "ir_empty_frames": source["expected_ir_empty_frames"],
            "inventory_sha256": source["expected_inventory_sha256"],
            "content_inventory_sha256": source[
                "expected_content_inventory_sha256"
            ],
            "sample_order_sha256": source["expected_sample_order_sha256"],
        }
        repo_root = Path(__file__).resolve().parents[1]
        counterfactual["implementation_source_sha256"] = {
            relative_path: real_file_sha256(repo_root / relative_path)
            for relative_path in EXPECTED_COUNTERFACTUAL_RESULT_IMPLEMENTATION_PATHS
        }
        module = "scripts.run_rtdetr_fam_box_guided_stage_a_five_seed_audit."

        def fake_result_hash(path):
            path = Path(path)
            if path.name.endswith("_v1.json"):
                return "d" * 64
            return real_file_sha256(path)

        with (
            patch(module + "_load_json_object", return_value=counterfactual),
            patch(module + "file_sha256", side_effect=fake_result_hash),
        ):
            verified = verify_counterfactual_prerequisite(
                protocol["seed40_counterfactual_prerequisite"],
                checkpoint_path,
                artifact,
            )
            self.assertTrue(verified["passed"])
            counterfactual["diagnostic_gate"]["passed"] = False
            with self.assertRaisesRegex(RuntimeError, "gate did not pass"):
                verify_counterfactual_prerequisite(
                    protocol["seed40_counterfactual_prerequisite"],
                    checkpoint_path,
                    artifact,
                )

        mechanism_protocol = load_yaml(
            Path(__file__).resolve().parents[1]
            / protocol["seed40_mechanism_prerequisite"]["protocol_path"]
        )
        # JSON publication stringifies the integer keys in the frozen target
        # match-count distribution; the verifier must compare canonically.
        serialized_mechanism_protocol = json.loads(json.dumps(mechanism_protocol))
        population = mechanism_protocol["target_population"]
        mechanism = {
            "schema_version": 1,
            "protocol": serialized_mechanism_protocol,
            "protocol_sha256": protocol["seed40_mechanism_prerequisite"][
                "protocol_payload_sha256"
            ],
            "training_config_sha256": protocol[
                "seed40_mechanism_prerequisite"
            ]["training_config_sha256"],
            "protocol_complete": True,
            "mechanism_gate_all_passed": True,
            "seed_rows": [
                {
                    "seed": 40,
                    "mechanism_gate_passed": True,
                    "batches": mechanism_protocol["expected_train_batches"],
                    "frames": population["expected_frames"],
                    "frames_with_matches": population[
                        "expected_frames_with_matches"
                    ],
                    "matched_boxes": population["expected_matched_boxes"],
                    "target_population_sha256": population[
                        "expected_target_population_sha256"
                    ],
                    "source_inventory_sha256": population[
                        "expected_source_inventory_sha256"
                    ],
                }
            ],
            "source_inventory": {
                "frames": population["expected_frames"],
                "frames_with_matches": population["expected_frames_with_matches"],
                "matched_boxes": population["expected_matched_boxes"],
                "max_matches_per_frame": population[
                    "expected_max_matches_per_frame"
                ],
                "per_frame_match_count_distribution": population[
                    "expected_per_frame_match_count_distribution"
                ],
                "target_population_sha256": population[
                    "expected_target_population_sha256"
                ],
                "source_inventory_sha256": population[
                    "expected_source_inventory_sha256"
                ],
            },
            "checkpoint_audits": {
                "40": {
                    "path": checkpoint_path,
                    "checkpoint_sha256": "a" * 64,
                    "training_source_manifest_sha256": "b" * 64,
                    "mechanism_gate": {"passed": True},
                }
            },
        }
        mechanism["seed_rows"][0].update(
            {
                "relative_improvement_vs_zero": 0.2,
                "sampled_guidance_mean_abs_cells": 0.051,
                "sampled_guidance_fraction_abs_ge_threshold": 0.009,
                "total_common_relative_improvement_vs_zero": 0.0,
                "mean_positive_cancellation_ratio": 0.5,
                "fraction_guidance_at_least_half_cancelled": 0.5,
            }
        )
        with (
            patch(module + "_load_json_object", return_value=mechanism),
            patch(module + "file_sha256", side_effect=fake_result_hash),
        ):
            verified = verify_mechanism_prerequisite(
                protocol["seed40_mechanism_prerequisite"],
                checkpoint_path,
                artifact,
            )
            self.assertTrue(verified["passed"])
            mechanism["seed_rows"][0]["relative_improvement_vs_zero"] = 0.199
            with self.assertRaisesRegex(RuntimeError, "metrics do not pass"):
                verify_mechanism_prerequisite(
                    protocol["seed40_mechanism_prerequisite"],
                    checkpoint_path,
                    artifact,
                )
            mechanism["seed_rows"][0]["relative_improvement_vs_zero"] = 0.2
            mechanism["seed_rows"][0]["matched_boxes"] -= 1
            with self.assertRaisesRegex(RuntimeError, "replay population differs"):
                verify_mechanism_prerequisite(
                    protocol["seed40_mechanism_prerequisite"],
                    checkpoint_path,
                    artifact,
                )
            mechanism["seed_rows"][0]["matched_boxes"] += 1
            mechanism["checkpoint_audits"]["40"]["checkpoint_sha256"] = "c" * 64
            with self.assertRaisesRegex(RuntimeError, "different seed-40 checkpoint"):
                verify_mechanism_prerequisite(
                    protocol["seed40_mechanism_prerequisite"],
                    checkpoint_path,
                    artifact,
                )

    def test_result_publication_refuses_overwrite(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            json_path = root / "result.json"
            csv_path = root / "result.csv"
            json_path.write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exist"):
                write_results_atomic(json_path, csv_path, {}, [])
            self.assertEqual(json_path.read_text(encoding="utf-8"), "old")
            self.assertFalse(csv_path.exists())

            protocol = load_yaml(DEFAULT_PROTOCOL)
            protocol_path = root / "protocol.yaml"
            protocol["output_json"] = str(root / "audit_v2.json")
            protocol["output_csv"] = str(root / "audit_v2.csv")
            Path(protocol["output_json"]).write_text("old", encoding="utf-8")
            protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
            module = "scripts.run_rtdetr_fam_box_guided_stage_a_five_seed_audit."
            with patch(module + "load_source_configs") as source_loader:
                with self.assertRaisesRegex(FileExistsError, "refusing to rerun"):
                    run(protocol_path, dry_run=False)
            source_loader.assert_not_called()

    def test_dry_run_resolves_both_source_arms_and_writes_nothing(self):
        protocol = load_yaml(DEFAULT_PROTOCOL)
        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            protocol_path = temporary_root / "protocol.yaml"
            json_path = temporary_root / "five_seed_audit_v2.json"
            csv_path = temporary_root / "five_seed_audit_v2.csv"
            protocol["output_json"] = str(json_path)
            protocol["output_csv"] = str(csv_path)
            protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")

            population = protocol["target_population"]
            fake_population = {
                "frames": population["expected_frames"],
                "frames_with_matches": population["expected_frames_with_matches"],
                "matched_boxes": population["expected_matched_boxes"],
                "max_matches_per_frame": population["expected_max_matches_per_frame"],
                "per_frame_match_count_distribution": population[
                    "expected_per_frame_match_count_distribution"
                ],
                "source_inventory_sha256": population[
                    "expected_source_inventory_sha256"
                ],
                "target_population_sha256": population[
                    "expected_target_population_sha256"
                ],
                "item_identifiers": {},
            }

            module = (
                "scripts.run_rtdetr_fam_box_guided_stage_a_five_seed_audit."
            )

            def fake_checkpoint(project, seed, checkpoint, wandb_root):
                return (
                    f"/tmp/{project}-{seed}/files/{checkpoint}/model.safetensors"
                )

            with (
                patch(module + "build_target_population_manifest", return_value=fake_population),
                patch(module + "resolve_local_wandb_checkpoint", side_effect=fake_checkpoint) as resolver,
                patch(
                    module + "verify_local_run_artifacts",
                    return_value={
                        "checkpoint_sha256": "a" * 64,
                        "scientific_config_sha256": "b" * 64,
                        "best_map_50": 0.2,
                    },
                ),
                patch(
                    module + "verify_stored_experiment_metadata",
                    return_value={"start_from_run": 0},
                ),
                patch(
                    module + "verify_runtime_trace_identity",
                    return_value={
                        "seed_identity": {},
                        "environment": {"runtime": "identical"},
                    },
                ),
                patch(
                    module + "verify_counterfactual_prerequisite",
                    return_value={"passed": True},
                ),
                patch(
                    module + "verify_mechanism_prerequisite",
                    return_value={"passed": True},
                ),
            ):
                self.assertIsNone(run(protocol_path, dry_run=True))

            self.assertEqual(resolver.call_count, 10)
            projects = [call.kwargs["project"] for call in resolver.call_args_list]
            self.assertEqual(
                projects.count("RTDETR_FAM_BoxGuided_SequenceVal_Seed40"), 1
            )
            self.assertEqual(
                projects.count("RTDETR_FAM_BoxGuided_SequenceVal_FiveSeed"), 4
            )
            self.assertEqual(
                projects.count("RTDETR_FAM_BoxGuided_MatchedControl_Seed40"), 1
            )
            self.assertEqual(
                projects.count(
                    "RTDETR_FAM_BoxGuided_MatchedControl_Seeds41to44"
                ),
                4,
            )
            self.assertFalse(json_path.exists())
            self.assertFalse(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
