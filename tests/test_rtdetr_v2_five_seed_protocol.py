"""Check the real grid expansion before launching ten expensive v2 trainings."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import unittest

from sarfusion.experiment.experiment import Experimenter
from sarfusion.utils.reproducibility import verify_training_source_manifest
from sarfusion.utils.utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ROOT / "parameters" / "RTDETR"


class TestRTDetrV2FiveSeedProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocols = {}
        cls.experiments = {}
        for branch in ("additive", "fam"):
            protocol = load_yaml(
                PARAMETERS
                / f"rtdetr_v2_{branch}_sequence_validation_five_seed_v2.yaml"
            )
            experiment = Experimenter()
            with redirect_stdout(StringIO()):
                experiment.calculate_runs(protocol)
            cls.protocols[branch] = protocol
            cls.experiments[branch] = experiment

    def test_five_runs_have_coupled_seeds_and_fresh_projects(self):
        for branch, experiment in self.experiments.items():
            with self.subTest(branch=branch):
                self.assertEqual([len(grid) for grid in experiment.grids], [1] * 5)
                self.assertEqual(experiment.gs.total_runs_to_run, 5)
                self.assertTrue(experiment.exp_settings.isolate_runs)
                self.assertFalse(experiment.exp_settings.continue_with_errors)
                self.assertEqual(experiment.exp_settings.start_from_grid, 0)
                self.assertEqual(experiment.exp_settings.start_from_run, 0)
                for seed, grid in zip(range(40, 45), experiment.grids):
                    run = grid[0]
                    self.assertEqual(run["seed"], seed)
                    for key in ("data_seed", "model_seed", "training_seed"):
                        self.assertEqual(run["reproducibility"][key], seed)

    def test_paired_runs_differ_only_in_fam_and_tracking_identity(self):
        control = self.experiments["additive"]
        candidate = self.experiments["fam"]
        for control_grid, candidate_grid in zip(control.grids, candidate.grids):
            left, right = deepcopy(control_grid[0]), deepcopy(candidate_grid[0])
            with self.subTest(seed=left["seed"]):
                self.assertIs(left["model"]["params"].pop("use_fam"), False)
                self.assertIs(right["model"]["params"].pop("use_fam"), True)
                left.pop("tracker")
                right.pop("tracker")
                self.assertEqual(left, right)

    def test_stage_a_best_with_full_budget_and_latest_diagnostic(self):
        for branch, experiment in self.experiments.items():
            for grid in experiment.grids:
                run = grid[0]
                with self.subTest(branch=branch, seed=run["seed"]):
                    self.assertIs(run["run_test"], False)
                    self.assertEqual(run["test_checkpoint"], "best")
                    self.assertIn("StageA", run["tracker"]["tags"])
                    self.assertIn("BestPrimary", run["tracker"]["tags"])
                    self.assertIs(run["strict_checkpoint_loading"], True)
                    self.assertIs(run["reproducibility"]["deterministic"], False)
                    train = run["train"]
                    self.assertEqual(train["max_epochs"], 10)
                    self.assertNotIn("early_stopping_patience", train)
                    self.assertNotIn("max_steps_per_epoch", train)
                    self.assertIs(train["run_validation"], True)
                    self.assertEqual(train["val_frequency"], 1)
                    self.assertIs(train["save_checkpoints"], True)
                    self.assertIs(train["save_final_checkpoint_only"], True)
                    self.assertEqual(train["watch_metric"], "map_50")
                    self.assertEqual(train["checkpoint_min_delta"], 0.001)


    def test_manifest_matches_current_sources_for_both_branches(self):
        for branch, experiment in self.experiments.items():
            with self.subTest(branch=branch):
                repro = experiment.grids[0][0]["reproducibility"]
                manifest = verify_training_source_manifest(
                    repro["training_source_manifest_id"],
                    repro["training_source_manifest_sha256"],
                    repo_root=ROOT,
                    required=True,
                )
                self.assertIn("sarfusion/models/rtdetr_v2_fusion.py", manifest["files"])


if __name__ == "__main__":
    unittest.main()
