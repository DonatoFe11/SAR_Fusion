import copy
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import yaml
from torch import nn

from sarfusion.data.temporal_split import stable_json_hash
from sarfusion.utils.reproducibility import (
    ReproducibilityTrace,
    build_training_source_manifest,
    training_source_runtime_fields,
)
from scripts.run_rtdetr_fam_box_guided_counterfactual import (
    CONDITIONS,
    build_source_inventory,
    compare_pass_identity,
    diagnostic_gate,
    file_sha256,
    load_candidate_config,
    load_protocol,
    load_state_dict_exact_modulo_aliases,
    run,
    temporarily_zero_guidance_output,
    validate_protocol,
    verify_local_run_artifacts,
    verify_training_config_file,
    write_complete_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_box_guided_counterfactual_seed40.yaml"
)


class TinyGuidedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.guidance_predictor = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(4, 2, 1),
        )


def fake_pass(condition, map50):
    return {
        "condition": condition,
        "n_samples": 896,
        "n_batches": 75,
        "sample_indices_sha256": "a" * 64,
        "ground_truth_sha256": "b" * 64,
        "metrics": {
            "map": map50 - 0.01,
            "map_50": map50,
            "map_75": map50 - 0.02,
            "map_small": -1.0,
            "map_medium": map50,
            "map_large": map50,
            "mar_1": map50,
            "mar_10": map50,
            "mar_100": map50,
        },
    }


class TestBoxGuidedCounterfactual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(PROTOCOL_PATH)

    def test_protocol_freezes_one_seed_one_checkpoint_and_fail_closed_gate(self):
        protocol = self.protocol
        self.assertEqual(protocol["seed"], 40)
        self.assertEqual(protocol["checkpoint"], "best")
        self.assertEqual(protocol["split"], "validation")
        self.assertEqual(protocol["mode"], "fusion")
        self.assertEqual(protocol["ground_truth"], "vis")
        self.assertEqual(protocol["confidence_threshold"], 0.01)
        self.assertEqual(protocol["counterfactual"]["order"], list(CONDITIONS))
        self.assertEqual(
            protocol["diagnostic_gate"],
            {
                "metric": "active_minus_zero_map_50",
                "comparator": "greater_than_or_equal",
                "minimum": 0.0,
                "if_pass": "guidance_non_degrading_on_stage_a_validation",
                "if_fail": "guidance_not_supported_as_inference_contributor",
            },
        )
        self.assertFalse(protocol["interpretation"]["permits_mt_erie"])
        self.assertNotIn(
            "mterie", json.dumps(protocol["source"], sort_keys=True).lower()
        )

    def test_training_yaml_hash_and_candidate_identity_are_exact(self):
        config_path = REPO_ROOT / self.protocol["training_config"]
        actual_hash = verify_training_config_file(
            config_path, self.protocol["training_config_sha256"]
        )
        self.assertEqual(actual_hash, file_sha256(config_path))
        config = load_candidate_config(config_path, self.protocol)
        self.assertEqual(config["seed"], 40)
        self.assertEqual(
            config["model"]["params"]["fam_variant"],
            "box_guided_common_offset_p3",
        )
        self.assertEqual(
            config["dataset"]["val_folders"],
            [[
                "210924_FHL_Enterprise_VIS_0401",
                "210924_FHL_Enterprise_IR_0402",
            ]],
        )

    def test_protocol_rejects_rewritten_frozen_hashes(self):
        changed = copy.deepcopy(self.protocol)
        changed["training_config_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "training-config digest"):
            validate_protocol(changed)

        changed = copy.deepcopy(self.protocol)
        changed["source"]["expected_content_inventory_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source/order digests"):
            validate_protocol(changed)

    def test_inventory_hashes_exact_source_bytes_and_excludes_terminal_vis(self):
        inventory = build_source_inventory(self.protocol)
        self.assertEqual(inventory["paired_frames"], 896)
        self.assertEqual(inventory["vis_inventory"], 897)
        self.assertEqual(inventory["ir_inventory"], 896)
        self.assertEqual(
            inventory["unpaired_vis"],
            ["210924_FHL_Enterprise_VIS_0401.mp4_00896.jpg"],
        )
        self.assertEqual(
            inventory["inventory_sha256"],
            "47e2f348ebdc202cb749b1bbf2741fc868d8681eb24ffa4356cf4f584dfa4ec4",
        )
        self.assertEqual(
            inventory["content_inventory_sha256"],
            "6c7748af3be2761a3a466b548af64aae925b693fbca795edf695072e28f17141",
        )
        self.assertEqual(
            inventory["sample_order_sha256"],
            "49415f065575c869087c78f842591096b74a0ea3a16ca2e4ce765e26958badcd",
        )
        first = inventory["rows"][0]
        self.assertRegex(first["vis_image_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(first["ir_image_sha256"], r"^[0-9a-f]{64}$")
        changed = copy.deepcopy(inventory["rows"])
        changed[0]["vis_image_sha256"] = "0" * 64
        self.assertNotEqual(
            stable_json_hash(changed), inventory["content_inventory_sha256"]
        )

    def test_final_guidance_conv_is_exactly_zeroed_and_restored_on_success(self):
        module = TinyGuidedModule()
        final = module.guidance_predictor[-1]
        original_weight = final.weight.detach().clone()
        original_bias = final.bias.detach().clone()
        with temporarily_zero_guidance_output(module) as intervention:
            self.assertEqual(torch.count_nonzero(final.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(final.bias).item(), 0)
            self.assertFalse(intervention["restored_exactly"])
        self.assertTrue(intervention["restored_exactly"])
        self.assertTrue(torch.equal(final.weight, original_weight))
        self.assertTrue(torch.equal(final.bias, original_bias))
        self.assertEqual(
            intervention["active_final_layer_sha256"],
            intervention["restored_final_layer_sha256"],
        )

    def test_final_guidance_conv_is_restored_if_inference_raises(self):
        module = TinyGuidedModule()
        final = module.guidance_predictor[-1]
        original_weight = final.weight.detach().clone()
        original_bias = final.bias.detach().clone()
        intervention = None
        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            with temporarily_zero_guidance_output(module) as intervention:
                raise RuntimeError("synthetic inference failure")
        self.assertTrue(intervention["restored_exactly"])
        self.assertTrue(torch.equal(final.weight, original_weight))
        self.assertTrue(torch.equal(final.bias, original_bias))

    def test_gate_is_active_minus_zero_and_boundary_passes(self):
        rule = self.protocol["diagnostic_gate"]
        boundary = diagnostic_gate(0.2, 0.2, rule)
        self.assertEqual(boundary["active_minus_zero_map_50"], 0.0)
        self.assertTrue(boundary["passed"])
        failed = diagnostic_gate(0.199, 0.2, rule)
        self.assertAlmostEqual(failed["active_minus_zero_map_50"], -0.001)
        self.assertFalse(failed["passed"])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            diagnostic_gate(math.nan, 0.2, rule)

    def test_pass_identity_requires_same_896_samples_batches_and_ground_truth(self):
        active = fake_pass("active", 0.2)
        zero = fake_pass("zero", 0.19)
        identity = compare_pass_identity(active, zero, self.protocol)
        self.assertTrue(identity["same_sample_order"])
        self.assertTrue(identity["same_ground_truth"])
        changed = copy.deepcopy(zero)
        changed["ground_truth_sha256"] = "c" * 64
        with self.assertRaisesRegex(RuntimeError, "ground-truth"):
            compare_pass_identity(active, changed, self.protocol)

    def test_state_dict_loader_rejects_non_alias_missing_keys(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        weights = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if key != "1.bias"
        }
        with self.assertRaisesRegex(RuntimeError, "Missing non-aliased"):
            load_state_dict_exact_modulo_aliases(model, weights)

    def test_wandb_artifacts_are_bound_to_exact_config_and_completed_run(self):
        config_path = REPO_ROOT / self.protocol["training_config"]
        scientific = copy.deepcopy(load_candidate_config(config_path, self.protocol))
        source_manifest = build_training_source_manifest()
        scientific["reproducibility"].update(
            {
                "training_source_manifest_id": source_manifest["manifest_id"],
                "training_source_manifest_sha256": source_manifest["sha256"],
            }
        )
        with TemporaryDirectory() as temporary_directory:
            files = Path(temporary_directory) / "run-abc" / "files"
            best = files / "best"
            best.mkdir(parents=True)
            checkpoint = best / "model.safetensors"
            checkpoint.write_bytes(b"synthetic checkpoint identity")
            stored = {
                key: {"value": value} for key, value in scientific.items()
            }
            stored["experiment"] = {"value": {"name": self.protocol["project"]}}
            (files / "config.yaml").write_text(
                yaml.safe_dump(stored, sort_keys=False), encoding="utf-8"
            )
            expected_steps = math.ceil(3123 / 4) * 10
            summary_path = files / "wandb-summary.json"
            summary = {
                "best_epoch": 4,
                "best_map_50": 0.123,
                "train/start_epoch": 9,
                "train/step": expected_steps - 1,
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            ReproducibilityTrace(
                files / "reproducibility_trace.jsonl"
            ).write(
                "runtime",
                seed=40,
                **training_source_runtime_fields(source_manifest),
            )
            audit = verify_local_run_artifacts(
                checkpoint,
                project=self.protocol["project"],
                seed=40,
                scientific_config=scientific,
                expected_train_frames=3123,
            )
            self.assertEqual(audit["expected_optimizer_steps"], 7810)
            self.assertEqual(audit["last_train_step"], 7809)
            self.assertEqual(audit["best_map_50"], 0.123)
            self.assertEqual(
                audit["training_source_manifest_sha256"],
                source_manifest["sha256"],
            )
            self.assertEqual(len(audit["reproducibility_trace_sha256"]), 64)

            summary["test/map_50"] = 0.5
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "test data"):
                verify_local_run_artifacts(
                    checkpoint,
                    project=self.protocol["project"],
                    seed=40,
                    scientific_config=scientific,
                    expected_train_frames=3123,
                )

    def test_outputs_are_refused_until_complete_then_atomically_published(self):
        result = {
            "protocol_complete": True,
            "results": [fake_pass("active", 0.2), fake_pass("zero", 0.19)],
            "diagnostic_gate": diagnostic_gate(
                0.2, 0.19, self.protocol["diagnostic_gate"]
            ),
            "checkpoint": {"checkpoint_sha256": "c" * 64},
            "configuration_hashes": {
                "training_yaml_sha256": "d" * 64,
                "reproducibility_trace_sha256": "1" * 64,
                "training_source_manifest_sha256": "2" * 64,
            },
            "source_inventory": {
                "inventory_sha256": "e" * 64,
                "content_inventory_sha256": "f" * 64,
            },
        }
        with TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "result_v1.json"
            csv_path = Path(temporary_directory) / "result_v1.csv"
            incomplete = copy.deepcopy(result)
            incomplete["protocol_complete"] = False
            with self.assertRaisesRegex(ValueError, "incomplete"):
                write_complete_outputs(incomplete, json_path, csv_path)
            self.assertFalse(json_path.exists())
            self.assertFalse(csv_path.exists())

            write_complete_outputs(result, json_path, csv_path)
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertEqual(json.loads(json_path.read_text()), result)
            self.assertEqual(len(csv_path.read_text().splitlines()), 3)
            self.assertFalse(list(Path(temporary_directory).glob("*.tmp")))

    def test_dry_run_performs_no_output_write(self):
        prepared = {
            "protocol": self.protocol,
            "protocol_sha256": "1" * 64,
            "artifact_audit": {"checkpoint_sha256": "2" * 64},
            "inventory": {
                "inventory_sha256": "3" * 64,
                "content_inventory_sha256": "4" * 64,
            },
        }
        with patch(
            "scripts.run_rtdetr_fam_box_guided_counterfactual.prepare",
            return_value=prepared,
        ), patch(
            "scripts.run_rtdetr_fam_box_guided_counterfactual.write_complete_outputs"
        ) as writer:
            result = run(PROTOCOL_PATH, dry_run=True)
        self.assertTrue(result["dry_run"])
        writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
