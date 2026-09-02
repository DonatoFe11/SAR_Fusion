import unittest
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import yaml
from torch import nn

from scripts.run_rtdetr_fam_box_guided_mechanism_audit import (
    GuidanceAuditAccumulator,
    ResidualCommonOffsetCapture,
    audit_dataset_config,
    best_smooth_l1_constant,
    build_target_population_manifest,
    load_state_dict_exact_modulo_aliases,
    load_seed_configs,
    mechanism_gate,
    run,
    sample_guidance_at_targets,
    summarize_across_seeds,
    validate_candidate_config,
    validate_protocol,
    verify_local_run_artifacts,
    verify_target_population,
    verify_training_config_file,
    write_results_atomic,
)
from sarfusion.utils.utils import load_yaml
from sarfusion.utils.reproducibility import (
    ReproducibilityTrace,
    build_training_source_manifest,
    training_source_runtime_fields,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_box_guided_mechanism_audit_seed40.yaml"
)


class TestBoxGuidedMechanismAudit(unittest.TestCase):
    def test_result_publication_refuses_overwrite_and_publishes_json_last(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            json_path = root / "mechanism_v1.json"
            csv_path = root / "mechanism_v1.csv"
            json_path.write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exist"):
                write_results_atomic(json_path, csv_path, {}, [])
            self.assertEqual(json_path.read_text(encoding="utf-8"), "old")
            self.assertFalse(csv_path.exists())

    def test_protocol_freezes_train_fusion_replay_and_match_rule(self):
        protocol = load_yaml(PROTOCOL_PATH)
        validate_protocol(protocol)
        self.assertEqual(protocol["seeds"], [40])
        self.assertEqual(protocol["checkpoint"], "best")
        self.assertEqual(protocol["split"], "train")
        self.assertEqual(protocol["mode"], "fusion")
        self.assertEqual(
            protocol["box_matching"],
            {
                "method": "mutual_nearest_box_center",
                "max_distance_normalized": 0.05,
            },
        )
        self.assertEqual(protocol["expected_train_frames"], 3123)
        self.assertEqual(protocol["expected_train_batches"], 261)
        self.assertEqual(
            protocol["diagnostic_baselines"],
            {
                "zero_flow": "fixed_zero_dy_dx",
                "constant_flow": (
                    "best_smooth_l1_constant_on_frozen_train_targets"
                ),
                "constant_comparison_is_promotion_gate": False,
                "positive_improvement_interpretation": (
                    "non_global_fit_on_frozen_train_targets_only"
                ),
                "does_not_establish": "input_conditioned_generalization",
            },
        )
        population = protocol["target_population"]
        self.assertEqual(population["expected_matched_boxes"], 5209)
        self.assertEqual(population["expected_frames_with_matches"], 2306)
        self.assertEqual(
            protocol["mechanism_gate"][
                "minimum_guidance_relative_improvement_vs_zero"
            ],
            0.2,
        )
        self.assertEqual(
            protocol["mechanism_gate"]["minimum_guidance_mean_abs_cells"],
            0.05,
        )
        self.assertEqual(
            protocol["mechanism_gate"][
                "maximum_guidance_saturation_fraction"
            ],
            0.01,
        )
        self.assertEqual(
            protocol["mechanism_gate"][
                "maximum_mean_positive_cancellation_ratio"
            ],
            0.5,
        )
        self.assertEqual(
            protocol["mechanism_gate"][
                "maximum_fraction_guidance_at_least_half_cancelled"
            ],
            0.5,
        )

    def test_seed40_protocol_rejects_expansion_or_duplicate_seeds(self):
        protocol = load_yaml(PROTOCOL_PATH)
        protocol["seeds"] = [40, 42, 44]
        with self.assertRaisesRegex(ValueError, "seed 40 only"):
            validate_protocol(protocol)
        protocol["seeds"] = [40, 40]
        with self.assertRaisesRegex(ValueError, "unique integers"):
            validate_protocol(protocol)

    def test_audit_replay_disables_dropout_without_mutating_training_config(self):
        protocol = load_yaml(PROTOCOL_PATH)
        training = {
            "name": "wisard",
            "modal_dropout": True,
            "paired_consistency": True,
            "box_alignment_targets": True,
            "box_alignment_max_distance": 0.05,
        }
        original = deepcopy(training)
        replay = audit_dataset_config(training, protocol)
        self.assertEqual(training, original)
        self.assertFalse(replay["modal_dropout"])
        self.assertFalse(replay["paired_consistency"])
        self.assertTrue(replay["box_alignment_targets"])
        self.assertEqual(replay["box_alignment_max_distance"], 0.05)

    def test_candidate_config_must_reconstruct_the_guided_variant(self):
        protocol = load_yaml(PROTOCOL_PATH)
        config = {
            "model": {
                "params": {
                    "use_fam": True,
                    "fam_variant": "box_guided_common_offset_p3",
                    "spatial_jitter_std": 0.0,
                }
            },
            "train": {
                "box_guided_alignment": {
                    "enabled": True,
                    "smooth_l1_beta_cells": 0.25,
                }
            },
            "dataset": {
                "box_alignment_targets": True,
                "box_alignment_max_distance": 0.05,
            },
        }
        validate_candidate_config(config, protocol)
        config["model"]["params"]["fam_variant"] = "current_dcnv2"
        with self.assertRaisesRegex(ValueError, "box_guided_common_offset_p3"):
            validate_candidate_config(config, protocol)

    def test_frozen_training_yaml_resolves_one_matching_candidate(self):
        protocol = load_yaml(PROTOCOL_PATH)
        verify_training_config_file(
            REPO_ROOT / protocol["training_config"],
            protocol["training_config_sha256"],
        )
        configs = load_seed_configs(
            REPO_ROOT / protocol["training_config"], protocol["seeds"]
        )
        self.assertEqual(set(configs), {40})
        validate_candidate_config(configs[40], protocol)

    def test_dry_run_resolves_checkpoints_without_writing_results(self):
        protocol = load_yaml(PROTOCOL_PATH)
        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            protocol_path = temporary_root / "protocol.yaml"
            json_path = temporary_root / "mechanism_audit_v1.json"
            csv_path = temporary_root / "mechanism_audit_v1.csv"
            protocol["output_json"] = str(json_path)
            protocol["output_csv"] = str(csv_path)
            protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")

            population = protocol["target_population"]
            fake_manifest = {
                "frames": population["expected_frames"],
                "frames_with_matches": population["expected_frames_with_matches"],
                "matched_boxes": population["expected_matched_boxes"],
                "max_matches_per_frame": population[
                    "expected_max_matches_per_frame"
                ],
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
            module = "scripts.run_rtdetr_fam_box_guided_mechanism_audit."
            with (
                patch(
                    module + "build_target_population_manifest",
                    return_value=fake_manifest,
                ),
                patch(
                    module + "resolve_local_wandb_checkpoint",
                    return_value="/tmp/fake-best/model.safetensors",
                ) as resolver,
                patch(
                    module + "verify_local_run_artifacts",
                    return_value={"checkpoint_sha256": "a" * 64},
                ),
            ):
                run(protocol_path, dry_run=True)

            resolver.assert_called_once_with(
                project=protocol["project"],
                seed=40,
                checkpoint="best",
                wandb_root=REPO_ROOT / "wandb",
            )
            self.assertFalse(json_path.exists())
            self.assertFalse(csv_path.exists())

    def test_sampling_matches_training_cell_conversion(self):
        flow = torch.empty(1, 2, 8, 10)
        flow[:, 0].fill_(1.0)
        flow[:, 1].fill_(-2.0)
        targets = [torch.tensor([[0.5, 0.5, 0.125, -0.2]])]
        sampled, target_cells, frames_with_matches = sample_guidance_at_targets(
            flow, targets
        )
        torch.testing.assert_close(sampled, torch.tensor([[1.0, -2.0]]))
        torch.testing.assert_close(target_cells, torch.tensor([[1.0, -2.0]]))
        self.assertEqual(frames_with_matches, 1)

    def test_best_smooth_l1_constant_is_exact_and_component_wise(self):
        targets = torch.tensor(
            [[-2.0, 1.0], [0.0, 1.0], [2.0, 1.0]],
            dtype=torch.float64,
        )
        constant = best_smooth_l1_constant(targets, beta=0.25)
        torch.testing.assert_close(
            constant,
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            atol=1e-12,
            rtol=0.0,
        )

    def test_constant_diagnostic_distinguishes_input_dependent_flow(self):
        accumulator = GuidanceAuditAccumulator(0.25, 3.9)
        flow = torch.zeros(1, 2, 2, 2)
        flow[0, 0, 0, 0] = -1.0
        flow[0, 0, 1, 1] = 1.0
        targets = [
            torch.tensor(
                [
                    [0.25, 0.25, -0.5, 0.0],
                    [0.75, 0.75, 0.5, 0.0],
                ]
            )
        ]
        accumulator.update(flow, targets)
        summary = accumulator.summary()
        self.assertAlmostEqual(summary["best_constant_flow_dy_cells"], 0.0)
        self.assertAlmostEqual(summary["best_constant_flow_dx_cells"], 0.0)
        self.assertAlmostEqual(
            summary["best_constant_smooth_l1_cells"], 0.4375
        )
        self.assertAlmostEqual(
            summary["relative_improvement_vs_best_constant"], 1.0
        )
        self.assertTrue(summary["beats_global_constant_on_train_targets"])
        self.assertAlmostEqual(summary["sampled_guidance_std_dy_cells"], 1.0)
        self.assertAlmostEqual(summary["sampled_guidance_std_dx_cells"], 0.0)
        self.assertAlmostEqual(summary["centered_guidance_target_cosine"], 1.0)
        self.assertAlmostEqual(summary["guidance_bias_fraction_of_mean_l2"], 0.0)

        constant_accumulator = GuidanceAuditAccumulator(0.25, 3.9)
        constant_accumulator.update(torch.zeros_like(flow), targets)
        constant_summary = constant_accumulator.summary()
        self.assertAlmostEqual(
            constant_summary["relative_improvement_vs_best_constant"], 0.0
        )
        self.assertFalse(
            constant_summary["beats_global_constant_on_train_targets"]
        )
        self.assertIsNone(
            constant_summary["centered_guidance_target_cosine"]
        )

    def test_accumulator_reports_exact_loss_improvement_and_flow_statistics(self):
        accumulator = GuidanceAuditAccumulator(
            smooth_l1_beta_cells=0.25,
            near_saturation_threshold_cells=3.9,
        )
        matching_flow = torch.empty(1, 2, 8, 10)
        matching_flow[:, 0].fill_(1.0)
        matching_flow[:, 1].fill_(-2.0)
        accumulator.update(
            matching_flow,
            [torch.tensor([[0.5, 0.5, 0.125, -0.2]])],
        )
        accumulator.update(torch.zeros(1, 2, 8, 10), [torch.empty(0, 4)])

        summary = accumulator.summary()
        self.assertEqual(summary["batches"], 2)
        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["frames_with_matches"], 1)
        self.assertEqual(summary["matched_boxes"], 1)
        self.assertAlmostEqual(summary["matches_per_frame"], 0.5)
        self.assertAlmostEqual(summary["learned_smooth_l1_cells"], 0.0)
        # SmoothL1(.25) at residual magnitudes one and two: (.875+1.875)/2.
        self.assertAlmostEqual(summary["zero_flow_smooth_l1_cells"], 1.375)
        self.assertAlmostEqual(summary["relative_improvement_vs_zero"], 1.0)
        self.assertAlmostEqual(
            summary["sampled_guidance_mean_abs_cells"], 1.5
        )
        self.assertAlmostEqual(
            summary["sampled_guidance_fraction_abs_ge_threshold"], 0.0
        )

    def test_accumulator_counts_near_saturation_per_sampled_component(self):
        accumulator = GuidanceAuditAccumulator(0.25, 3.9)
        flow = torch.empty(1, 2, 4, 4)
        flow[:, 0].fill_(4.0)
        flow[:, 1].fill_(-3.8)
        accumulator.update(
            flow,
            [torch.tensor([[0.5, 0.5, 0.0, 0.0]])],
        )
        summary = accumulator.summary()
        self.assertAlmostEqual(
            summary["sampled_guidance_fraction_abs_ge_threshold"], 0.5
        )

    def test_residual_capture_measures_mean_common_offset(self):
        class Guided(nn.Module):
            def __init__(self):
                super().__init__()
                self.offset_conv = nn.Conv2d(1, 27, 1, bias=False)

            @staticmethod
            def transform_offset(offset):
                return offset

        guided = Guided()
        capture = ResidualCommonOffsetCapture(guided)
        try:
            with torch.no_grad():
                guided.offset_conv.weight[:18].copy_(
                    torch.arange(18, dtype=torch.float32).view(18, 1, 1, 1)
                )
                guided.offset_conv.weight[18:].zero_()
            guided.offset_conv(torch.ones(1, 1, 2, 2))
            common = capture.residual_common_flow()
        finally:
            capture.close()
        expected = torch.tensor([8.0, 9.0]).view(1, 2, 1, 1).expand_as(common)
        torch.testing.assert_close(common, expected)

    def test_total_common_and_cancellation_gate_fail_closed(self):
        accumulator = GuidanceAuditAccumulator(0.25, 3.9)
        guidance = torch.ones(1, 2, 4, 4)
        residual = -guidance
        targets = [torch.tensor([[0.5, 0.5, 0.25, 0.25]])]
        accumulator.update(guidance, targets, residual_common_flow=residual)
        summary = accumulator.summary()
        self.assertAlmostEqual(summary["mean_positive_cancellation_ratio"], 1.0)
        self.assertAlmostEqual(
            summary["fraction_guidance_at_least_half_cancelled"], 1.0
        )
        decision = mechanism_gate(
            summary,
            load_yaml(PROTOCOL_PATH)["mechanism_gate"],
        )
        self.assertFalse(decision["passed"])

    def test_guidance_quality_checks_are_all_fail_closed(self):
        rule = load_yaml(PROTOCOL_PATH)["mechanism_gate"]
        passing = {
            "relative_improvement_vs_zero": 0.2,
            "sampled_guidance_mean_abs_cells": 0.051,
            "sampled_guidance_fraction_abs_ge_threshold": 0.009,
            "total_common_relative_improvement_vs_zero": 0.0,
            "mean_positive_cancellation_ratio": 0.5,
            "fraction_guidance_at_least_half_cancelled": 0.5,
        }
        self.assertTrue(mechanism_gate(passing, rule)["passed"])

        failures = (
            ("relative_improvement_vs_zero", 0.199),
            ("sampled_guidance_mean_abs_cells", 0.05),
            ("sampled_guidance_fraction_abs_ge_threshold", 0.01),
            ("total_common_relative_improvement_vs_zero", -1e-6),
            ("mean_positive_cancellation_ratio", 0.501),
            ("fraction_guidance_at_least_half_cancelled", 0.501),
        )
        for key, value in failures:
            with self.subTest(key=key):
                changed = dict(passing)
                changed[key] = value
                self.assertFalse(mechanism_gate(changed, rule)["passed"])

    def test_exact_state_dict_allows_only_real_shared_aliases(self):
        class SharedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.primary = nn.Linear(2, 2)
                self.alias = self.primary

        source = SharedModel()
        weights = {
            "primary.weight": source.primary.weight.detach().clone(),
            "primary.bias": source.primary.bias.detach().clone(),
        }
        audit = load_state_dict_exact_modulo_aliases(SharedModel(), weights)
        self.assertEqual(audit["shared_alias_key_count"], 2)

        with self.assertRaisesRegex(RuntimeError, "Missing non-aliased"):
            load_state_dict_exact_modulo_aliases(
                SharedModel(), {"primary.weight": weights["primary.weight"]}
            )
        with self.assertRaisesRegex(RuntimeError, "Unexpected checkpoint"):
            load_state_dict_exact_modulo_aliases(
                SharedModel(), {**weights, "not_a_model_key": torch.zeros(1)}
            )

    def test_local_wandb_config_is_bound_to_checkpoint_and_completed_run(self):
        scientific = {
            "seed": 40,
            "train": {"max_epochs": 1},
            "dataloader": {"batch_size": 1},
        }
        source_manifest = build_training_source_manifest()
        scientific["reproducibility"] = {
            "trace": True,
            "training_source_manifest_id": source_manifest["manifest_id"],
            "training_source_manifest_sha256": source_manifest["sha256"],
        }
        with TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory) / "run-20260101-test"
            files = run_root / "files"
            best = files / "best"
            best.mkdir(parents=True)
            checkpoint = best / "model.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            stored = {
                "wandb_version": 1,
                "_wandb": {"value": {}},
                "experiment": {"value": {"name": "project"}},
                **{key: {"value": value} for key, value in scientific.items()},
            }
            (files / "config.yaml").write_text(
                yaml.safe_dump(stored), encoding="utf-8"
            )
            (files / "wandb-summary.json").write_text(
                json.dumps(
                    {
                        "best_epoch": 1,
                        "best_map_50": 0.2,
                        "train/start_epoch": 0,
                        "train/step": 1,
                    }
                ),
                encoding="utf-8",
            )
            trace_path = files / "reproducibility_trace.jsonl"
            ReproducibilityTrace(trace_path).write(
                "runtime",
                seed=40,
                **training_source_runtime_fields(source_manifest),
            )
            audit = verify_local_run_artifacts(
                checkpoint,
                project="project",
                seed=40,
                scientific_config=scientific,
                expected_train_frames=2,
            )
            self.assertEqual(len(audit["checkpoint_sha256"]), 64)
            self.assertEqual(
                audit["training_source_manifest_sha256"],
                source_manifest["sha256"],
            )
            self.assertEqual(len(audit["reproducibility_trace_sha256"]), 64)

            changed = deepcopy(scientific)
            changed["seed"] = 41
            with self.assertRaisesRegex(RuntimeError, "configuration differs"):
                verify_local_run_artifacts(
                    checkpoint,
                    project="project",
                    seed=41,
                    scientific_config=changed,
                    expected_train_frames=2,
                )

    def test_real_stage_a_target_population_matches_independent_freeze(self):
        protocol = load_yaml(PROTOCOL_PATH)
        config = load_seed_configs(
            REPO_ROOT / protocol["training_config"], protocol["seeds"]
        )[40]
        manifest = build_target_population_manifest(
            audit_dataset_config(config["dataset"], protocol)
        )
        verified = verify_target_population(
            manifest, protocol["target_population"]
        )
        self.assertEqual(verified["matched_boxes"], 5209)
        self.assertEqual(
            verified["target_population_sha256"],
            "d519574962e81ae5b492248113247cca20d7ef15b2d189d1e3b58aebf218f3c0",
        )

        runtime_summary = {
            key: value
            for key, value in manifest.items()
            if key not in ("source_inventory_sha256", "item_identifiers")
        }
        with self.assertRaisesRegex(RuntimeError, "source file inventory"):
            verify_target_population(
                runtime_summary, protocol["target_population"]
            )
        runtime_verified = verify_target_population(
            runtime_summary,
            protocol["target_population"],
            require_source_inventory=False,
        )
        self.assertEqual(runtime_verified, verified)

    def test_cross_seed_summary_uses_checkpoint_rows_as_units(self):
        base = {
            "matches_per_frame": 1.0,
            "learned_smooth_l1_cells": 0.5,
            "residual_common_smooth_l1_cells": 0.8,
            "total_common_smooth_l1_cells": 0.4,
            "zero_flow_smooth_l1_cells": 1.0,
            "relative_improvement_vs_zero": 0.5,
            "total_common_relative_improvement_vs_zero": 0.6,
            "sampled_guidance_mean_abs_cells": 0.25,
            "sampled_residual_common_mean_abs_cells": 0.2,
            "sampled_total_common_mean_abs_cells": 0.3,
            "sampled_guidance_fraction_abs_ge_threshold": 0.0,
            "mean_positive_cancellation_ratio": 0.1,
            "fraction_guidance_at_least_half_cancelled": 0.0,
            "mean_cosine_guidance_residual_common": -0.2,
        }
        rows = [
            {"seed": 40, **base},
            {
                "seed": 41,
                **{
                    **base,
                    "relative_improvement_vs_zero": 0.7,
                },
            },
        ]
        summary = summarize_across_seeds(rows)["relative_improvement_vs_zero"]
        self.assertEqual(summary["seed_values"], {"40": 0.5, "41": 0.7})
        self.assertAlmostEqual(summary["mean"], 0.6)
        self.assertAlmostEqual(summary["sample_std"], 2 ** 0.5 / 10)


if __name__ == "__main__":
    unittest.main()
