import copy
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import torch
from PIL import Image
import torchvision.transforms.functional as tvF

from sarfusion.data.wisard import MULTI_MODALITY_ITEM, WiSARDDataset
from sarfusion.utils.grid import make_grid
from sarfusion.utils.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONFIG = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_paired_vis_modal_dropout_sequence_validation_seed40.yaml"
)
BASELINE_CONFIG = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_sequence_validation_fixed10_protocol.yaml"
)


class _FakeProcessor:
    """Minimal processor exposing whether VIS or IR annotations were used."""

    def __call__(self, image, annotations, return_tensors):
        if isinstance(image, Image.Image):
            pixels = tvF.pil_to_tensor(image)
        else:
            pixels = torch.as_tensor(image)
        pixels = pixels.float()
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


def _paired_dataset(contract):
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
    dataset.modal_dropout_coordinate_contract = contract
    dataset.use_tiling = False
    dataset.test_all_tiles = False
    dataset._load_rgb = lambda _path: Image.new("RGB", (8, 6), color=(10, 20, 30))
    dataset._load_ir = lambda _path: Image.new("RGB", (4, 3), color=(40, 40, 40))
    dataset._load_annotations = lambda path, _image, _index: {
        "image_id": 0,
        "annotations": [],
        "annotation_source": path,
    }
    return dataset


class TestRTDetrPairedVisModalDropout(TestCase):
    def test_seed40_probe_is_matched_to_the_frozen_fam_baseline(self):
        candidate_grid = make_grid(load_yaml(CANDIDATE_CONFIG)["parameters"])
        baseline_grid = make_grid(load_yaml(BASELINE_CONFIG)["parameters"])
        self.assertEqual(len(candidate_grid), 1)
        self.assertEqual([run["seed"] for run in baseline_grid], [40, 41, 42, 43, 44])

        candidate = copy.deepcopy(candidate_grid[0])
        baseline = copy.deepcopy(baseline_grid[0])
        self.assertEqual(candidate["seed"], 40)
        self.assertEqual(
            candidate["dataset"].pop("modal_dropout_coordinate_contract"),
            "paired_vis",
        )
        candidate.pop("tracker")
        baseline.pop("tracker")
        self.assertEqual(candidate, baseline)

    def test_paired_vis_ir_draw_masks_rgb_and_keeps_vis_target_contract(self):
        dataset = _paired_dataset("paired_vis")
        with patch("sarfusion.data.wisard.random.choices", return_value=["ir"]):
            sample = dataset[0]

        self.assertTrue(torch.equal(sample.pixel_values[:3], torch.zeros_like(sample.pixel_values[:3])))
        self.assertGreater(torch.count_nonzero(sample.pixel_values[3:4]).item(), 0)
        self.assertEqual(sample.labels["annotation_source"], "vis.txt")
        self.assertEqual(sample.path, "vis.jpg")
        self.assertEqual(sample.modality_mode, "ir")

    def test_native_default_preserves_historical_ir_target_contract(self):
        dataset = _paired_dataset("native")
        with patch("sarfusion.data.wisard.random.choices", return_value=["ir"]):
            sample = dataset[0]

        self.assertTrue(torch.equal(sample.pixel_values[:3], torch.zeros_like(sample.pixel_values[:3])))
        self.assertGreater(torch.count_nonzero(sample.pixel_values[3:4]).item(), 0)
        self.assertEqual(sample.labels["annotation_source"], "ir.txt")
        self.assertEqual(sample.path, "ir.jpg")

    def test_fusion_and_rgb_draws_are_unchanged_by_coordinate_contract(self):
        for mode in ("fusion", "rgb"):
            with self.subTest(mode=mode):
                native = _paired_dataset("native")
                paired_vis = _paired_dataset("paired_vis")
                with patch(
                    "sarfusion.data.wisard.random.choices", return_value=[mode]
                ):
                    native_sample = native[0]
                    paired_sample = paired_vis[0]

                self.assertTrue(
                    torch.equal(native_sample.pixel_values, paired_sample.pixel_values)
                )
                self.assertTrue(
                    torch.equal(
                        native_sample.labels["orig_size"],
                        paired_sample.labels["orig_size"],
                    )
                )
                self.assertEqual(
                    native_sample.labels["annotation_source"],
                    paired_sample.labels["annotation_source"],
                )
                self.assertEqual(native_sample.path, paired_sample.path)

    def test_channel_mask_validates_shape_and_mode(self):
        image = torch.ones(4, 5, 7)
        ir_only = WiSARDDataset._apply_paired_modal_dropout(image, "ir")
        rgb_only = WiSARDDataset._apply_paired_modal_dropout(image, "rgb")

        self.assertTrue(torch.equal(ir_only[:3], torch.zeros_like(ir_only[:3])))
        self.assertTrue(torch.equal(ir_only[3:4], torch.ones_like(ir_only[3:4])))
        self.assertTrue(torch.equal(rgb_only[:3], torch.ones_like(rgb_only[:3])))
        self.assertTrue(torch.equal(rgb_only[3:4], torch.zeros_like(rgb_only[3:4])))
        self.assertTrue(torch.equal(image, torch.ones_like(image)))
        with self.assertRaisesRegex(ValueError, "4-channel"):
            WiSARDDataset._apply_paired_modal_dropout(torch.ones(3, 5, 7), "ir")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            WiSARDDataset._apply_paired_modal_dropout(image, "unknown")

    def test_configuration_validation_is_strict_and_case_normalized(self):
        item = [(MULTI_MODALITY_ITEM, (("vis.jpg", "vis.txt"), ("ir.jpg", "ir.txt")))]
        with patch("sarfusion.data.wisard.build_wisard_items", return_value=item):
            dataset = WiSARDDataset(
                root="unused",
                folders=[],
                modal_dropout_probs=(0.2, 0.2, 0.6),
                modal_dropout_coordinate_contract="PAIRED_VIS",
            )
            self.assertEqual(dataset.modal_dropout_coordinate_contract, "paired_vis")

            with self.assertRaisesRegex(ValueError, "sum to 1.0"):
                WiSARDDataset(
                    root="unused",
                    folders=[],
                    modal_dropout_probs=[0.2, 0.2, 0.5],
                )
            with self.assertRaisesRegex(ValueError, "native.*paired_vis"):
                WiSARDDataset(
                    root="unused",
                    folders=[],
                    modal_dropout_coordinate_contract="other",
                )


if __name__ == "__main__":
    import unittest

    unittest.main()
