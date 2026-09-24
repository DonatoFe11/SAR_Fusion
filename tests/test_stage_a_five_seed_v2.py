"""Check the five-seed Stage A grids, training contracts and launcher."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import subprocess
import unittest

from sarfusion.experiment.experiment import Experimenter
from sarfusion.experiment.box_guided_alignment import validate_box_guided_training_contract
from sarfusion.experiment.modality_consistency import validate_modality_consistency_config
from sarfusion.utils.reproducibility import verify_training_source_manifest
from sarfusion.utils.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "baseline": "rtdetr_fam_stage_a_five_seed_v2.yaml",
    "box_guided": "rtdetr_fam_box_guided_stage_a_five_seed_v2.yaml",
    "mixed_consistency": "rtdetr_fam_mixed_consistency_stage_a_five_seed_v2.yaml",
}


class StageAFiveSeedV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {}
        for name, filename in CONFIGS.items():
            experiment = Experimenter()
            with redirect_stdout(StringIO()):
                experiment.calculate_runs(load_yaml(ROOT / "parameters/RTDETR" / filename))
            cls.experiments[name] = experiment

    def test_exactly_fifteen_fresh_rtdetr_runs_with_coupled_seeds(self):
        for experiment in self.experiments.values():
            self.assertEqual([len(grid) for grid in experiment.grids], [1] * 5)
            self.assertEqual(experiment.gs.total_runs_to_run, 5)
            settings = experiment.exp_settings
            self.assertIn("StageA_FiveSeed_V2", settings.name)
            self.assertTrue(settings.isolate_runs)
            self.assertFalse(settings.continue_with_errors)
            self.assertEqual(settings.start_from_grid, 0)
            self.assertEqual(settings.start_from_run, 0)
            for seed, grid in zip(range(40, 45), experiment.grids):
                run = grid[0]
                self.assertEqual(run["seed"], seed)
                for key in ("data_seed", "model_seed", "training_seed"):
                    self.assertEqual(run["reproducibility"][key], seed)

    def test_complete_stage_a_budget_and_checkpoint_rule(self):
        for experiment in self.experiments.values():
            for grid in experiment.grids:
                run = grid[0]
                train = run["train"]
                self.assertFalse(run["run_test"])
                self.assertEqual(run["test_checkpoint"], "best")
                self.assertTrue(run["strict_checkpoint_loading"])
                self.assertEqual(train["max_epochs"], 10)
                self.assertTrue(train["run_validation"])
                self.assertEqual(train["val_frequency"], 1)
                self.assertEqual(train["watch_metric"], "map_50")
                self.assertEqual(train["checkpoint_min_delta"], 0.001)
                self.assertTrue(train["save_checkpoints"])
                self.assertTrue(train["save_final_checkpoint_only"])
                self.assertNotIn("early_stopping_patience", train)
                self.assertNotIn("max_steps_per_epoch", train)
                self.assertEqual(len(run["dataset"]["train_folders"]), 2)
                self.assertEqual(len(run["dataset"]["val_folders"]), 1)
                self.assertEqual(run["dataset"]["modal_dropout_coordinate_contract"], "native")

    def test_candidates_differ_only_in_the_declared_intervention(self):
        for index in range(5):
            baseline = deepcopy(self.experiments["baseline"].grids[index][0])
            baseline.pop("tracker")
            box = deepcopy(self.experiments["box_guided"].grids[index][0])
            box.pop("tracker")
            self.assertEqual(box["model"]["params"]["fam_variant"], "box_guided_common_offset_p3")
            box["model"]["params"]["fam_variant"] = "current_dcnv2"
            self.assertEqual(box["train"].pop("box_guidance_lr"), 0.0001)
            box["train"].pop("box_guided_alignment")
            self.assertTrue(box["dataset"].pop("box_alignment_targets"))
            self.assertEqual(box["dataset"].pop("box_alignment_max_distance"), 0.05)
            self.assertEqual(box, baseline)
            mixed = deepcopy(self.experiments["mixed_consistency"].grids[index][0])
            mixed.pop("tracker")
            mixed["train"].pop("modality_consistency")
            self.assertTrue(mixed["dataset"].pop("paired_consistency"))
            self.assertEqual(mixed["dataset"].pop("paired_consistency_student_probs"), [0.5, 0.5])
            self.assertEqual(mixed, baseline)

    def test_existing_training_contracts_and_current_manifest(self):
        for name, experiment in self.experiments.items():
            run = experiment.grids[0][0]
            consistency = validate_modality_consistency_config(run["train"].get("modality_consistency"))
            validate_box_guided_training_contract(
                run["train"].get("box_guided_alignment"), run["dataset"], run["model"],
                modality_consistency_enabled=consistency["enabled"],
            )
            repro = run["reproducibility"]
            verify_training_source_manifest(repro["training_source_manifest_id"],
                                           repro["training_source_manifest_sha256"], required=True)


    def test_launcher_has_all_campaigns_and_never_calls_old_screens(self):
        result = subprocess.run(["bash", "scripts/run_stage_a_five_seed_v2.sh", "--dry-run"],
                                cwd=ROOT, check=True, capture_output=True, text=True)
        commands = [line for line in result.stdout.splitlines() if line.startswith("Running:")]
        self.assertEqual(len(commands), 13)  # three 5-run RT-DETR grids + ten isolated YOLO runs
        for arm in ("additive", "fam"):
            for seed in range(40, 45):
                self.assertEqual(sum(f"yolo26s_{arm}_stage_a_five_seed_v2.yaml --seed {seed} " in c
                                     for c in commands), 1)
        self.assertNotIn("audit", "\n".join(commands))
        self.assertNotIn("probe_evaluation", "\n".join(commands))


if __name__ == "__main__":
    unittest.main()
