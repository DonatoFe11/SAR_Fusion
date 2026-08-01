import unittest

from sarfusion.data import get_train_val_test_params


class TestWiSARDPhaseParameters(unittest.TestCase):
    def test_modal_dropout_is_only_enabled_for_training(self):
        dataset_params = {
            "root": "dataset/WiSARD",
            "folders": "vis_ir",
            "single_class": True,
            "modal_dropout": True,
            "modal_dropout_probs": [0.2, 0.2, 0.6],
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

        # Splitting phase parameters must not mutate the shared YAML values.
        self.assertTrue(dataset_params["modal_dropout"])


if __name__ == "__main__":
    unittest.main()
