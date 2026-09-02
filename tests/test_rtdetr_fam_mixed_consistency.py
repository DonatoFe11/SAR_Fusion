import copy
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from PIL import Image
import torchvision.transforms.functional as tvF

from sarfusion.data import get_train_val_test_params
from sarfusion.data.wisard import MULTI_MODALITY_ITEM, WiSARDDataset
from sarfusion.experiment.modality_consistency import (
    matched_detection_consistency_loss,
    modality_consistency_epoch_scale,
    validate_modality_consistency_config,
)
from sarfusion.utils.grid import make_grid
from sarfusion.utils.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_sequence_validation_fixed10_protocol.yaml"
)
SEED40_CONFIG = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_mixed_consistency_sequence_validation_seed40.yaml"
)
FIVE_SEED_CONFIG = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_mixed_consistency_sequence_validation_five_seed.yaml"
)


class _FakeProcessor:
    def __call__(self, image, annotations, return_tensors):
        pixels = (
            tvF.pil_to_tensor(image)
            if isinstance(image, Image.Image)
            else torch.as_tensor(image)
        ).float()
        height, width = pixels.shape[-2:]
        return {
            "pixel_values": pixels.unsqueeze(0),
            "labels": [
                {
                    "orig_size": torch.tensor([height, width]),
                    "annotation_source": annotations["annotation_source"],
                }
            ],
        }


def _paired_consistency_dataset():
    dataset = object.__new__(WiSARDDataset)
    dataset.items = [
        (
            MULTI_MODALITY_ITEM,
            (("vis.jpg", "vis.txt"), ("ir.jpg", "ir.txt")),
        )
    ]
    dataset.expanded_items = None
    dataset.transform = _FakeProcessor()
    dataset.ir_transform = dataset.transform
    dataset.image_size = 640
    dataset.augment = False
    dataset.return_path = True
    dataset.single_class = True
    dataset.modal_dropout = True
    dataset.modal_dropout_probs = [0.2, 0.2, 0.6]
    dataset.modal_dropout_coordinate_contract = "native"
    dataset.paired_consistency = True
    dataset.paired_consistency_student_probs = [0.5, 0.5]
    dataset.use_tiling = False
    dataset.test_all_tiles = False
    dataset._load_rgb = lambda _path: Image.new(
        "RGB", (8, 6), color=(10, 20, 30)
    )
    dataset._load_ir = lambda _path: Image.new(
        "RGB", (4, 3), color=(40, 40, 40)
    )
    dataset._load_annotations = lambda path, _image, _index: {
        "image_id": 0,
        "annotations": [],
        "annotation_source": path,
    }
    return dataset


def _expanded_run(path, index=0):
    return copy.deepcopy(make_grid(load_yaml(path)["parameters"])[index])


def _remove_consistency_intervention(run):
    run = copy.deepcopy(run)
    run.pop("tracker")
    run["train"].pop("modality_consistency")
    run["dataset"].pop("paired_consistency")
    run["dataset"].pop("paired_consistency_student_probs")
    run["dataset"].pop("modal_dropout_coordinate_contract")
    return run


class TestRTDetrFamMixedConsistency(TestCase):
    def test_seed40_and_five_seed_configs_match_frozen_fam_baseline(self):
        baseline_runs = make_grid(load_yaml(BASELINE_CONFIG)["parameters"])
        seed40_runs = make_grid(load_yaml(SEED40_CONFIG)["parameters"])
        candidate_runs = make_grid(load_yaml(FIVE_SEED_CONFIG)["parameters"])

        self.assertEqual(len(seed40_runs), 1)
        self.assertEqual([run["seed"] for run in candidate_runs], [40, 41, 42, 43, 44])
        self.assertEqual(
            _remove_consistency_intervention(seed40_runs[0]),
            {key: value for key, value in baseline_runs[0].items() if key != "tracker"},
        )
        for baseline, candidate in zip(baseline_runs, candidate_runs):
            self.assertEqual(
                _remove_consistency_intervention(candidate),
                {key: value for key, value in baseline.items() if key != "tracker"},
            )

    def test_native_supervision_is_preserved_while_paired_student_is_masked(self):
        dataset = _paired_consistency_dataset()
        with patch(
            "sarfusion.data.wisard.random.choices",
            side_effect=[["ir"], ["rgb"]],
        ):
            sample = dataset[0]

        self.assertEqual(sample.modality_mode, "ir")
        self.assertEqual(sample.labels["annotation_source"], "ir.txt")
        self.assertEqual(sample.path, "ir.jpg")
        self.assertTrue(
            torch.equal(
                sample.pixel_values[:3], torch.zeros_like(sample.pixel_values[:3])
            )
        )
        self.assertGreater(
            torch.count_nonzero(sample.consistency_teacher_pixel_values).item(),
            0,
        )
        self.assertEqual(sample.consistency_student_mode, "rgb")
        self.assertTrue(
            torch.equal(
                sample.consistency_student_pixel_values[3:4],
                torch.zeros_like(sample.consistency_student_pixel_values[3:4]),
            )
        )
        self.assertTrue(
            torch.equal(
                sample.consistency_student_pixel_values[:3],
                sample.consistency_teacher_pixel_values[:3],
            )
        )

    def test_consistency_generation_is_disabled_outside_training(self):
        dataset_params = {
            "root": "dataset/WiSARD",
            "folders": "vis_ir",
            "single_class": True,
            "modal_dropout": True,
            "modal_dropout_coordinate_contract": "native",
            "paired_consistency": True,
            "paired_consistency_student_probs": [0.5, 0.5],
        }
        train, val, test = get_train_val_test_params("wisard", dataset_params)
        self.assertTrue(train["paired_consistency"])
        self.assertFalse(val["paired_consistency"])
        self.assertFalse(test["paired_consistency"])

    def test_consistency_config_and_warmup_are_strict(self):
        config = validate_modality_consistency_config(
            {"enabled": True, "start_epoch": 1, "warmup_epochs": 2}
        )
        self.assertEqual(modality_consistency_epoch_scale(config, 0), 0.0)
        self.assertEqual(modality_consistency_epoch_scale(config, 1), 0.5)
        self.assertEqual(modality_consistency_epoch_scale(config, 2), 1.0)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_modality_consistency_config({"unknown": 1})
        with self.assertRaisesRegex(ValueError, "confidence_threshold"):
            validate_modality_consistency_config({"confidence_threshold": 1.1})

    def test_hungarian_matching_is_invariant_to_query_permutation(self):
        teacher = SimpleNamespace(
            logits=torch.tensor([[[3.0], [2.0], [-5.0]]]),
            pred_boxes=torch.tensor(
                [[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]]]
            ),
        )
        student = SimpleNamespace(
            logits=teacher.logits[:, [1, 0, 2]].clone().requires_grad_(True),
            pred_boxes=teacher.pred_boxes[:, [1, 0, 2]].clone().requires_grad_(True),
        )
        config = validate_modality_consistency_config(
            {"enabled": True, "confidence_threshold": 0.2}
        )

        loss = matched_detection_consistency_loss(teacher, student, config)

        self.assertAlmostEqual(loss.value.item(), 0.0, places=6)
        self.assertEqual(loss.components["consistency_matched_queries"].item(), 2)
        loss.value.backward()
        self.assertTrue(torch.isfinite(student.logits.grad).all())
        self.assertTrue(torch.isfinite(student.pred_boxes.grad).all())

    def test_localization_difference_produces_student_gradients(self):
        teacher = SimpleNamespace(
            logits=torch.tensor([[[3.0], [2.0]]]),
            pred_boxes=torch.tensor(
                [[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.2, 0.2]]]
            ),
        )
        student = SimpleNamespace(
            logits=torch.tensor([[[2.5], [1.5]]], requires_grad=True),
            pred_boxes=torch.tensor(
                [[[0.3, 0.2, 0.1, 0.1], [0.7, 0.8, 0.2, 0.2]]],
                requires_grad=True,
            ),
        )
        config = validate_modality_consistency_config(
            {"enabled": True, "confidence_threshold": 0.2}
        )

        loss = matched_detection_consistency_loss(teacher, student, config)
        self.assertGreater(loss.value.item(), 0.0)
        loss.value.backward()
        self.assertGreater(student.logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(student.pred_boxes.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
