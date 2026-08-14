import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from ultralytics.cfg import IterableSimpleNamespace

from sarfusion.experiment.yolo import (
    WisardTrainer,
    _guard_wandb_plot_curve,
    build_wisard_validator_args,
    install_wandb_empty_curve_guard,
)
from sarfusion.utils.grid import make_grid
from sarfusion.utils.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeValidator:
    def __init__(self):
        self.args = SimpleNamespace(plots=None)
        self.dataloader = None
        self.checkpoint = None
        self.mode = None

    def __call__(self, model, mode):
        self.checkpoint = model
        self.mode = mode
        return {"metrics/mAP50(B)": 0.5, "fitness": 0.4}


class TestYOLOFinalProtocol(unittest.TestCase):
    def _grid(self, filename):
        config = load_yaml(REPO_ROOT / "parameters" / "YOLO" / filename)
        return config, make_grid(config["parameters"])

    def test_final_configs_have_five_paired_fixed_horizon_runs(self):
        expected = {
            "yolov10_additive_protocol.yaml": False,
            "yolov10_fam_protocol.yaml": True,
        }
        for filename, use_fam in expected.items():
            with self.subTest(filename=filename):
                config, grid = self._grid(filename)
                self.assertTrue(config["experiment"]["isolate_runs"])
                self.assertEqual([run["seed"] for run in grid], [40, 41, 42, 43, 44])
                for run in grid:
                    self.assertEqual(run["epochs"], 200)
                    self.assertEqual(run["patience"], 0)
                    self.assertFalse(run["val"])
                    self.assertEqual(run["test_checkpoint"], "last")
                    self.assertTrue(run["modal_dropout"])
                    self.assertEqual(run["modal_dropout_strategy"], "feature")
                    self.assertEqual(run["modal_dropout_probs"], [0.2, 0.2, 0.6])
                    self.assertEqual(run["model"]["params"]["use_fam"], use_fam)
                    self.assertEqual(run["model"]["params"]["spatial_jitter_std"], 0.0)

    def test_final_eval_uses_last_checkpoint_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = WisardTrainer.__new__(WisardTrainer)
            trainer.last = root / "last.pt"
            trainer.best = root / "best.pt"
            trainer.last.touch()
            trainer.best.touch()
            trainer.batch_size = 4
            trainer.args = SimpleNamespace(
                task="detect", test_checkpoint="last", plots=False
            )
            trainer.data = {"test": "test-images"}
            trainer.get_dataloader = lambda *args, **kwargs: "test-loader"
            trainer.validator = _FakeValidator()
            trainer.run_callbacks = lambda name: None

            with patch("sarfusion.experiment.yolo.strip_optimizer"):
                trainer.final_eval()

            self.assertEqual(trainer.validator.checkpoint, trainer.last)
            self.assertEqual(trainer.validator.mode, "test")
            self.assertEqual(trainer.validator.dataloader, "test-loader")
            self.assertEqual(trainer.metrics, {"test/mAP50(B)": 0.5})

    def test_trainer_only_checkpoint_selector_is_not_forwarded_to_validator(self):
        trainer_args = IterableSimpleNamespace(
            task="detect",
            augment_vis_ir=False,
            modal_dropout=True,
            modal_dropout_probs=[0.2, 0.2, 0.6],
            modal_dropout_strategy="feature",
            test_checkpoint="last",
        )

        validator_args = build_wisard_validator_args(trainer_args)

        self.assertEqual(validator_args.task, "detect")
        self.assertFalse(hasattr(validator_args, "test_checkpoint"))
        self.assertFalse(hasattr(validator_args, "modal_dropout"))

    def test_final_eval_rejects_unknown_checkpoint_selector(self):
        trainer = WisardTrainer.__new__(WisardTrainer)
        trainer.args = SimpleNamespace(test_checkpoint="latest")
        with self.assertRaisesRegex(ValueError, "best.*last"):
            trainer.final_eval()

    def test_wandb_curve_guard_skips_empty_precision_recall_curve(self):
        plot_curve = Mock(return_value="plotted")
        guarded = _guard_wandb_plot_curve(plot_curve)

        result = guarded(
            np.linspace(0.0, 1.0, 1000),
            np.array([]),
            title="Precision-Recall(B)",
        )

        self.assertIsNone(result)
        plot_curve.assert_not_called()

    def test_wandb_curve_guard_preserves_valid_curves(self):
        plot_curve = Mock(return_value="plotted")
        guarded = _guard_wandb_plot_curve(plot_curve)
        x = np.linspace(0.0, 1.0, 3)
        y = np.array([[1.0, 0.5, 0.0]])

        result = guarded(x, y, title="Precision-Recall(B)")

        self.assertEqual(result, "plotted")
        plot_curve.assert_called_once_with(x, y, title="Precision-Recall(B)")

    def test_wandb_curve_guard_installation_is_idempotent(self):
        plot_curve = Mock(return_value="plotted")

        with patch("ultralytics.utils.callbacks.wb._plot_curve", plot_curve):
            self.assertTrue(install_wandb_empty_curve_guard())
            from ultralytics.utils.callbacks import wb as wb_callback

            installed = wb_callback._plot_curve
            self.assertFalse(install_wandb_empty_curve_guard())
            self.assertTrue(
                installed.__dict__.get("_sarfusion_empty_curve_guard", False)
            )


if __name__ == "__main__":
    unittest.main()
