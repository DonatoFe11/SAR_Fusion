from contextlib import nullcontext
import unittest
from pathlib import Path
from unittest.mock import Mock

from sarfusion.data.temporal_split import (
    load_temporal_split_manifest,
    manifest_folder_pairs,
    select_temporal_split_items,
)
from sarfusion.data.wisard import build_wisard_items
from sarfusion.experiment.run import Run
from sarfusion.utils.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset" / "WiSARD"
MANIFEST_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_temporal_validation_split.json"
)
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_temporal_validation_protocol.yaml"
)
SMOKE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_temporal_validation_smoke.yaml"
)


class TestRTDETRTemporalValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.inventory = load_temporal_split_manifest(
            MANIFEST_PATH, DATASET_ROOT, verify=True
        )

    def test_manifest_is_train_derived_frozen_and_has_embargo(self):
        self.assertEqual(self.manifest["status"], "frozen_before_training")
        self.assertFalse(
            self.manifest["selection_rule"]["model_predictions_inspected_before_freeze"]
        )
        self.assertEqual(
            {sequence["ranges"]["embargo"][1] - sequence["ranges"]["embargo"][0]
             for sequence in self.manifest["sequences"]},
            {30},
        )
        folders = manifest_folder_pairs(self.manifest)
        self.assertEqual(len(folders), 3)
        self.assertFalse(any("MtErie" in folder for pair in folders for folder in pair))

    def test_frozen_counts_and_hashes_match_local_dataset(self):
        self.assertEqual(self.inventory["n_source_frames"], 4019)
        self.assertEqual(self.inventory["phases"]["train"]["n_frames"], 3125)
        self.assertEqual(self.inventory["phases"]["embargo"]["n_frames"], 90)
        self.assertEqual(self.inventory["phases"]["val"]["n_frames"], 804)
        self.assertEqual(self.inventory["phases"]["val"]["n_vis_boxes"], 1799)
        self.assertEqual(
            self.inventory["source_inventory_sha256"],
            "5f684ff047c2fd41e5e6440daa4e5ca61a89b34d907f6f156638bb7f740ca8eb",
        )

    def test_selected_train_and_validation_items_are_disjoint(self):
        items = build_wisard_items(DATASET_ROOT, manifest_folder_pairs(self.manifest))
        train = select_temporal_split_items(
            items, DATASET_ROOT, self.inventory, "train"
        )
        val = select_temporal_split_items(items, DATASET_ROOT, self.inventory, "val")
        self.assertEqual(len(train), 3125)
        self.assertEqual(len(val), 804)
        self.assertTrue(set(map(str, train)).isdisjoint(set(map(str, val))))

    def test_protocol_freezes_checkpoint_and_early_stopping_rule(self):
        protocol = load_yaml(PROTOCOL_PATH)["parameters"]
        train = protocol["train"]
        self.assertEqual(train["max_epochs"], [10])
        self.assertEqual(train["watch_metric"], ["map_50"])
        self.assertEqual(train["checkpoint_min_delta"], [0.001])
        self.assertEqual(train["early_stopping_patience"], [5])
        self.assertEqual(train["val_frequency"], [1])
        self.assertEqual(train["save_final_checkpoint_only"], [True])
        self.assertEqual(protocol["run_test"], [False])
        self.assertEqual(protocol["test_checkpoint"], ["best"])
        self.assertEqual(protocol["seed"], [40, 41, 42, 43, 44])
        self.assertEqual(protocol["dataloader"]["batch_size"], [4])
        self.assertEqual(protocol["dataloader"]["evaluation_batch_size"], [12])

    def test_smoke_is_one_seed_two_epochs_and_not_a_campaign_run(self):
        smoke = load_yaml(SMOKE_PATH)
        params = smoke["parameters"]
        self.assertEqual(smoke["experiment"]["name"], "RTDETR_FAM_TemporalVal_Smoke")
        self.assertEqual(params["seed"], [40])
        self.assertEqual(params["train"]["max_epochs"], [2])
        self.assertTrue(params["train"]["run_validation"][0])
        self.assertFalse(params["run_test"][0])
        self.assertEqual(params["dataloader"]["batch_size"], [4])
        self.assertEqual(params["dataloader"]["evaluation_batch_size"], [12])

    def test_checkpoint_min_delta_keeps_earliest_near_tie(self):
        run = Run()
        run.train_params = {"checkpoint_min_delta": 0.001}
        run.watch_metric = "map_50"
        run.greater_is_better = True

        self.assertTrue(run._update_best_metric(0, {"map_50": 0.30}))
        self.assertFalse(run._update_best_metric(1, {"map_50": 0.3005}))
        self.assertEqual(run.best_epoch, 0)
        self.assertAlmostEqual(run.best_metric, 0.30)
        self.assertTrue(run._update_best_metric(2, {"map_50": 0.302}))
        self.assertEqual(run.best_epoch, 2)

    def test_best_and_latest_are_saved_as_separate_states(self):
        run = Run()
        run.tracker = Mock()
        run.watch_metric = "map_50"
        run.best_metric = 0.42
        run.best_epoch = 3

        run.save_training_state(3, improved=True, save_latest=True)

        self.assertEqual(
            run.tracker.log_training_state.call_args_list[0].kwargs["subfolder"],
            "best",
        )
        self.assertEqual(
            run.tracker.log_training_state.call_args_list[1].kwargs["subfolder"],
            "latest",
        )
        run.tracker.add_summary.assert_called_once_with(
            {"best_epoch": 4, "best_map_50": 0.42}
        )

    def test_early_stopping_counts_only_qualifying_improvements(self):
        run = Run()
        run.train_params = {
            "max_epochs": 10,
            "run_validation": True,
            "val_frequency": 1,
            "save_checkpoints": True,
            "save_final_checkpoint_only": True,
            "early_stopping_patience": 2,
            "checkpoint_min_delta": 0.001,
        }
        run.params = {"experiment": {"name": "unit-test"}, "run_test": False}
        run.watch_metric = "map_50"
        run.greater_is_better = True
        run.val_loader = object()
        run.test_loader = None
        run.scheduler = None
        run.scheduler_step_moment = None
        run.tracker = Mock()
        run.tracker.train.return_value = nullcontext()
        run.tracker.validate.return_value = nullcontext()
        run.train_epoch = Mock()
        run.validate_epoch = Mock(
            side_effect=[
                {"map_50": 0.3000},
                {"map_50": 0.3005},
                {"map_50": 0.3006},
            ]
        )
        run.end = Mock()

        run.launch()

        self.assertEqual(run.train_epoch.call_count, 3)
        self.assertEqual(run.validate_epoch.call_count, 3)
        self.assertEqual(run.best_epoch, 0)
        self.assertAlmostEqual(run.best_metric, 0.3)
        saved_subfolders = [
            call.kwargs["subfolder"]
            for call in run.tracker.log_training_state.call_args_list
        ]
        self.assertEqual(saved_subfolders, ["best", "latest"])
        run.end.assert_called_once()


if __name__ == "__main__":
    unittest.main()
