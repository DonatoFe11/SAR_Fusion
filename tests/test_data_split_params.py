import unittest

from sarfusion.data import get_train_val_test_params
from sarfusion.data.wisard import build_wisard_items


class TestWiSARDPhaseParameters(unittest.TestCase):
    def test_modal_dropout_is_only_enabled_for_training(self):
        dataset_params = {
            "root": "dataset/WiSARD",
            "folders": "vis_ir",
            "single_class": True,
            "modal_dropout": True,
            "modal_dropout_probs": [0.2, 0.2, 0.6],
            "modal_dropout_coordinate_contract": "paired_vis",
            "use_tiling": False,
        }

        train_params, val_params, test_params = get_train_val_test_params(
            "wisard", dataset_params
        )

        self.assertTrue(train_params["modal_dropout"])
        self.assertFalse(val_params["modal_dropout"])
        self.assertFalse(test_params["modal_dropout"])
        self.assertEqual(train_params["modal_dropout_probs"], [0.2, 0.2, 0.6])
        self.assertEqual(val_params["modal_dropout_probs"], [0.2, 0.2, 0.6])
        self.assertEqual(test_params["modal_dropout_probs"], [0.2, 0.2, 0.6])
        self.assertEqual(
            train_params["modal_dropout_coordinate_contract"], "paired_vis"
        )
        self.assertEqual(
            val_params["modal_dropout_coordinate_contract"], "paired_vis"
        )
        self.assertEqual(
            test_params["modal_dropout_coordinate_contract"], "paired_vis"
        )

        # Splitting phase parameters must not mutate the shared YAML values.
        self.assertTrue(dataset_params["modal_dropout"])

    def test_explicit_whole_sequence_overrides_bypass_default_phase_filter(self):
        train_folders = [
            ["210924_FHL_Enterprise_VIS_0405", "210924_FHL_Enterprise_IR_0406"],
            ["220109_Baker_Enterprise_VIS_1", "220109_Baker_Enterprise_IR_1"],
        ]
        val_folders = [
            ["210924_FHL_Enterprise_VIS_0401", "210924_FHL_Enterprise_IR_0402"]
        ]
        dataset_params = {
            "root": "dataset/WiSARD",
            "folders": "vis_ir",
            "train_folders": train_folders,
            "val_folders": val_folders,
            "single_class": True,
            "modal_dropout": True,
            "use_tiling": False,
        }

        train, val, test = get_train_val_test_params("wisard", dataset_params)

        expected_train = [tuple(pair) for pair in train_folders]
        expected_val = [tuple(pair) for pair in val_folders]
        self.assertEqual(train["folders"], expected_train)
        self.assertEqual(val["folders"], expected_val)
        self.assertNotIn("train_folders", train)
        self.assertNotIn("val_folders", val)
        self.assertEqual(
            len(build_wisard_items(dataset_params["root"], train["folders"])),
            3123,
        )
        self.assertEqual(
            len(build_wisard_items(dataset_params["root"], val["folders"])),
            896,
        )
        self.assertEqual(
            len(build_wisard_items(dataset_params["root"], test["folders"])),
            708,
        )


if __name__ == "__main__":
    unittest.main()
