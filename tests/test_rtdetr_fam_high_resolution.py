import csv
import statistics
import unittest
from pathlib import Path

from sarfusion.utils.grid import make_grid
from sarfusion.utils.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_sequence_validation_fixed10_protocol.yaml"
)
HIGHRES_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_800_sequence_validation_five_seed.yaml"
)
PROBE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_800_runtime_probe.yaml"
)
RESULTS_PATH = (
    REPO_ROOT
    / "notes"
    / "Search_and_Rescue"
    / "results"
    / "rtdetr_fam_800_stage_a_validation.csv"
)


class TestRTDetrFAMHighResolution(unittest.TestCase):
    def test_grid_expands_to_five_seed_runs_at_800_square(self):
        protocol = load_yaml(HIGHRES_PATH)
        runs = make_grid(protocol["parameters"])

        self.assertEqual([run["seed"] for run in runs], [40, 41, 42, 43, 44])
        for run in runs:
            self.assertEqual(
                run["dataset"]["preprocessor"]["size"],
                {"height": 800, "width": 800},
            )
            self.assertEqual(run["dataloader"]["batch_size"], 2)
            self.assertEqual(run["dataloader"]["evaluation_batch_size"], 8)
            self.assertEqual(run["train"]["gradient_accumulation_steps"], 2)
            self.assertFalse(run["run_test"])

    def test_architecture_split_and_optimizer_match_fam_baseline(self):
        baseline = load_yaml(BASELINE_PATH)["parameters"]
        highres = load_yaml(HIGHRES_PATH)["parameters"]

        self.assertEqual(highres["model"], baseline["model"])
        self.assertEqual(highres["loss"], baseline["loss"])
        self.assertEqual(highres["dataset"]["train_folders"], baseline["dataset"]["train_folders"])
        self.assertEqual(highres["dataset"]["val_folders"], baseline["dataset"]["val_folders"])
        self.assertEqual(highres["train"]["initial_lr"], baseline["train"]["initial_lr"])
        self.assertEqual(highres["train"]["optimizer"], baseline["train"]["optimizer"])
        self.assertEqual(highres["train"]["max_epochs"], [10])
        self.assertNotIn("early_stopping_patience", highres["train"])

    def test_probe_is_short_checkpoint_free_and_campaign_equivalent(self):
        probe = load_yaml(PROBE_PATH)["parameters"]
        campaign = load_yaml(HIGHRES_PATH)["parameters"]

        self.assertEqual(probe["seed"], [40])
        self.assertEqual(probe["train"]["max_epochs"], [1])
        self.assertEqual(probe["train"]["max_steps_per_epoch"], [20])
        self.assertEqual(probe["train"]["save_checkpoints"], [False])
        self.assertIn("ExcludeFromCampaign", probe["tracker"]["tags"][0])
        self.assertEqual(probe["model"], campaign["model"])
        self.assertEqual(probe["dataset"], campaign["dataset"])
        self.assertEqual(probe["dataloader"], campaign["dataloader"])

    def test_five_seed_results_are_complete_and_fail_promotion_rule(self):
        with RESULTS_PATH.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        self.assertEqual([int(row["seed"]) for row in rows], [40, 41, 42, 43, 44])
        self.assertEqual(len({row["highres_run_id"] for row in rows}), 5)
        highres = [float(row["highres_best_map50"]) for row in rows]
        deltas = [float(row["highres_minus_baseline"]) for row in rows]

        self.assertAlmostEqual(statistics.fmean(highres), 0.1446889848, places=9)
        self.assertAlmostEqual(statistics.stdev(highres), 0.0151117409, places=9)
        self.assertAlmostEqual(statistics.fmean(deltas), -0.0198736727, places=9)
        self.assertEqual(sum(delta > 0 for delta in deltas), 1)


if __name__ == "__main__":
    unittest.main()
