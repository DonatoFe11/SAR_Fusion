import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_rtdetr_fam_rcra_full_data_stage_b_evaluation import (
    EXPECTED_SEEDS,
    build_aggregate,
    load_protocol,
    stage_b_decision,
    validate_training_configs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_rcra_full_data_stage_b_evaluation.yaml"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_temporal_validation_split.json"
)


class TestRTDETRFAMRCRAFullDataStageB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(PROTOCOL_PATH)
        cls.runs = validate_training_configs(cls.protocol)

    def test_protocol_freezes_latest_and_paired_decision(self):
        self.assertEqual(self.protocol["checkpoint"], "latest")
        self.assertEqual(self.protocol["seeds"], EXPECTED_SEEDS)
        self.assertEqual(tuple(self.protocol["configurations"]), ("fam", "rcra"))
        rule = self.protocol["primary_comparison"]
        self.assertEqual(rule["metric"], "map_50")
        self.assertEqual(rule["minimum_mean_gain"], 0.01)
        self.assertEqual(rule["minimum_positive_seed_wins"], 4)
        self.assertFalse(
            self.protocol["interpretation"][
                "historical_fam_as_primary_comparator_allowed"
            ]
        )

    def test_grids_have_five_complete_fixed_budget_runs(self):
        self.assertEqual(len(self.runs), 10)
        for configuration in ("fam", "rcra"):
            self.assertEqual(
                [
                    int(self.runs[configuration, seed]["seed"])
                    for seed in EXPECTED_SEEDS
                ],
                EXPECTED_SEEDS,
            )
            for seed in EXPECTED_SEEDS:
                run = self.runs[configuration, seed]
                train = run["train"]
                self.assertFalse(run["run_test"])
                self.assertEqual(run["test_checkpoint"], "latest")
                self.assertEqual(train["max_epochs"], 10)
                self.assertFalse(train["run_validation"])
                self.assertTrue(train["save_final_checkpoint_only"])
                self.assertNotIn("early_stopping_patience", train)
                self.assertNotIn("max_steps_per_epoch", train)
                self.assertEqual(run["dataloader"]["batch_size"], 4)

    def test_only_declared_candidate_recipe_differs(self):
        for seed in EXPECTED_SEEDS:
            fam = copy.deepcopy(self.runs["fam", seed])
            rcra = copy.deepcopy(self.runs["rcra", seed])
            self.assertFalse(
                fam["model"]["params"]["use_residual_alignment_gating"]
            )
            self.assertTrue(
                rcra["model"]["params"]["use_residual_alignment_gating"]
            )
            self.assertNotIn("alignment_gate_lr", fam["train"])
            self.assertEqual(rcra["train"]["alignment_gate_lr"], 2e-4)

            fam.pop("tracker")
            rcra.pop("tracker")
            rcra["train"].pop("alignment_gate_lr")
            rcra["model"]["params"]["use_residual_alignment_gating"] = False
            self.assertEqual(fam, rcra)

    def test_full_training_inventory_is_frozen_at_4019_frames(self):
        with MANIFEST_PATH.open(encoding="utf-8") as input_file:
            manifest = json.load(input_file)
        source = self.protocol["training_source"]
        self.assertEqual(manifest["expected"]["n_source_frames"], 4019)
        self.assertEqual(source["expected_frames"], 4019)
        self.assertEqual(
            manifest["expected"]["source_inventory_sha256"],
            source["expected_inventory_sha256"],
        )
        self.assertEqual(
            [
                [sequence["vis_folder"], sequence["ir_folder"]]
                for sequence in manifest["sequences"]
            ],
            source["paired_folders"],
        )

    def test_decision_requires_both_mean_and_win_thresholds(self):
        rule = self.protocol["primary_comparison"]
        passed = stage_b_decision([0.02, 0.02, 0.02, 0.02, -0.01], rule)
        self.assertEqual(passed["status"], "pass_rcra")
        self.assertEqual(passed["selected_architecture"], "rcra")

        only_three_wins = stage_b_decision(
            [0.05, 0.05, 0.05, -0.01, -0.01], rule
        )
        self.assertTrue(only_three_wins["passes_mean_gain"])
        self.assertFalse(only_three_wins["passes_win_count"])
        self.assertEqual(only_three_wins["selected_architecture"], "fam")

        too_small = stage_b_decision([0.005] * 5, rule)
        self.assertFalse(too_small["passes_mean_gain"])
        self.assertTrue(too_small["passes_win_count"])
        self.assertEqual(too_small["selected_architecture"], "fam")

    def test_complete_aggregate_applies_rule_and_writes_both_tables(self):
        payloads = []
        baseline = [0.30, 0.31, 0.32, 0.33, 0.34]
        deltas = [0.02, 0.02, 0.02, 0.02, -0.01]
        for configuration in ("fam", "rcra"):
            for index, seed in enumerate(EXPECTED_SEEDS):
                map_50 = baseline[index]
                if configuration == "rcra":
                    map_50 += deltas[index]
                payloads.append(
                    {
                        "configuration": configuration,
                        "seed": seed,
                        "checkpoint_sha256": f"{configuration}-{seed}",
                        "n_samples": 708,
                        "training_summary": {"run_id": f"{configuration}{seed}"},
                        "metrics": {
                            "map": map_50 / 3,
                            "map_50": map_50,
                            "map_75": map_50 / 4,
                            "map_small": map_50 / 3,
                            "map_medium": -1.0,
                            "map_large": -1.0,
                            "mar_1": map_50 / 5,
                            "mar_10": map_50 / 2,
                            "mar_100": map_50 / 1.5,
                        },
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            aggregate = build_aggregate(
                payloads,
                self.protocol,
                "protocol-hash",
                {"n_frames": 708},
                {"n_frames": 4019},
                output_dir,
            )
            self.assertTrue(aggregate["protocol_complete"])
            decision = aggregate["rcra_minus_fam_map50"]["decision"]
            self.assertEqual(decision["status"], "pass_rcra")
            self.assertEqual(decision["candidate_wins"], 4)
            self.assertTrue(
                (output_dir / "rtdetr_fam_rcra_full_data_stage_b_evaluation.json").is_file()
            )
            self.assertTrue(
                (output_dir / "rtdetr_fam_rcra_full_data_stage_b_evaluation.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
