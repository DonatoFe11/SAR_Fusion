import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from safetensors.torch import save_file

from sarfusion.models import MODEL_REGISTRY, build_model
from sarfusion.models.checkpoints import resolve_local_wandb_checkpoint


class TestLocalWandBCheckpointResolution(unittest.TestCase):
    def _create_run(self, root, run_name, project, seed, checkpoint="latest"):
        files = Path(root) / f"run-20260806_000000-{run_name}" / "files"
        target = files / checkpoint / "model.safetensors"
        target.parent.mkdir(parents=True)
        target.touch()
        with (files / "config.yaml").open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(
                {
                    "experiment": {"value": {"name": project}},
                    "seed": {"value": seed},
                },
                config_file,
            )
        return target

    def test_resolves_exact_project_seed_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = self._create_run(directory, "abc123", "RTDETR_FAM", 42)
            self._create_run(directory, "other1", "RTDETR_FAM", 41)
            self._create_run(directory, "other2", "RTDETR_BASE", 42)

            actual = resolve_local_wandb_checkpoint(
                "RTDETR_FAM", 42, wandb_root=directory
            )

            self.assertEqual(Path(actual), expected)

    def test_missing_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_local_wandb_checkpoint(
                    "RTDETR_FAM", 40, wandb_root=directory
                )

    def test_duplicate_completed_runs_fail_instead_of_selecting_one(self):
        with tempfile.TemporaryDirectory() as directory:
            self._create_run(directory, "first1", "RTDETR_FAM", 40)
            self._create_run(directory, "second", "RTDETR_FAM", 40)

            with self.assertRaises(RuntimeError):
                resolve_local_wandb_checkpoint(
                    "RTDETR_FAM", 40, wandb_root=directory
                )

    def test_full_match_requirement_accepts_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            save_file(
                {
                    "weight": torch.full((1, 2), 3.0),
                    "bias": torch.tensor([4.0]),
                },
                checkpoint,
            )
            with patch.dict(
                MODEL_REGISTRY,
                {"test_linear": lambda: torch.nn.Linear(2, 1)},
            ):
                model = build_model(
                    {
                        "name": "test_linear",
                        "params": {
                            "pretrained_path": str(checkpoint),
                            "require_full_pretrained_match": True,
                        },
                    }
                )

            torch.testing.assert_close(model.weight, torch.full((1, 2), 3.0))
            torch.testing.assert_close(model.bias, torch.tensor([4.0]))

    def test_full_match_requirement_rejects_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            save_file({"weight": torch.ones(1, 2)}, checkpoint)
            with patch.dict(
                MODEL_REGISTRY,
                {"test_linear": lambda: torch.nn.Linear(2, 1)},
            ):
                with self.assertRaises(RuntimeError):
                    build_model(
                        {
                            "name": "test_linear",
                            "params": {
                                "pretrained_path": str(checkpoint),
                                "require_full_pretrained_match": True,
                            },
                        }
                    )


if __name__ == "__main__":
    unittest.main()
