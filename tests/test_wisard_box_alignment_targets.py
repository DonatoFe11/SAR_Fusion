from unittest import TestCase
from unittest.mock import patch

import torch
from PIL import Image
import torchvision.transforms.functional as tvF

from sarfusion.data import get_train_val_test_params
from sarfusion.data.wisard import (
    MULTI_MODALITY_ITEM,
    RGB_ITEM,
    WiSARDDataset,
    adapt_ir2rgb,
    build_box_alignment_targets,
)


def _annotations(*centers, width=100, height=100):
    annotations = []
    for annotation_id, (center_x, center_y) in enumerate(centers):
        box_width = width * 0.02
        box_height = height * 0.02
        annotations.append(
            {
                "id": annotation_id,
                "bbox": [
                    center_x * width - box_width / 2.0,
                    center_y * height - box_height / 2.0,
                    box_width,
                    box_height,
                ],
            }
        )
    return {"image_id": 0, "annotations": annotations}


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
            "labels": [{"orig_size": torch.tensor([height, width])}],
        }

    def pad(self, pixel_values, return_tensors, input_data_format):
        values = torch.stack(pixel_values)
        return {
            "pixel_values": values,
            "pixel_mask": torch.ones(
                values.shape[0], values.shape[2], values.shape[3], dtype=torch.int64
            ),
        }


def _paired_item():
    return [
        (
            MULTI_MODALITY_ITEM,
            (("vis.jpg", "vis.txt"), ("ir.jpg", "ir.txt")),
        )
    ]


class TestWiSARDBoxAlignmentTargets(TestCase):
    def test_mutual_nearest_matching_and_target_order(self):
        targets = build_box_alignment_targets(
            _annotations((0.20, 0.50), (0.80, 0.50)),
            _annotations((0.22, 0.51), (0.79, 0.49), (0.50, 0.50)),
            vis_size=(100, 100),
            ir_size=(100, 100),
            adapted_ir_size=(100, 100),
            max_distance=0.05,
        )

        self.assertEqual(tuple(targets.shape), (2, 4))
        self.assertTrue(
            torch.allclose(
                targets,
                torch.tensor(
                    [
                        [0.20, 0.50, 0.01, 0.02],
                        [0.80, 0.50, -0.01, -0.01],
                    ]
                ),
                atol=1e-6,
            )
        )

    def test_non_mutual_and_distant_pairs_are_excluded(self):
        non_mutual = build_box_alignment_targets(
            _annotations((0.20, 0.50), (0.24, 0.50)),
            _annotations((0.21, 0.50)),
            vis_size=(100, 100),
            ir_size=(100, 100),
            adapted_ir_size=(100, 100),
            max_distance=0.05,
        )
        distant = build_box_alignment_targets(
            _annotations((0.20, 0.20)),
            _annotations((0.40, 0.40)),
            vis_size=(100, 100),
            ir_size=(100, 100),
            adapted_ir_size=(100, 100),
            max_distance=0.05,
        )

        self.assertEqual(tuple(non_mutual.shape), (1, 4))
        self.assertEqual(tuple(distant.shape), (0, 4))
        self.assertEqual(distant.dtype, torch.float32)

    def test_ir_centers_follow_actual_adapt_ir2rgb_geometry(self):
        vis_image = Image.new("RGB", (192, 108))
        ir_image = Image.new("RGB", (64, 51))
        _, adapted_ir = adapt_ir2rgb(vis_image, ir_image)
        adapted_ir_size = (adapted_ir.shape[-1], adapted_ir.shape[-2])

        targets = build_box_alignment_targets(
            _annotations((0.50, 0.50), width=192, height=108),
            _annotations((0.52, 0.50), width=64, height=51),
            vis_size=vis_image.size,
            ir_size=ir_image.size,
            adapted_ir_size=adapted_ir_size,
            max_distance=0.05,
        )

        resized_width = int(64 * (108 / 51))
        left_padding = (adapted_ir_size[0] - resized_width) / 2.0
        expected_ir_x = (
            0.52 * resized_width + left_padding
        ) / adapted_ir_size[0]
        self.assertEqual(tuple(targets.shape), (1, 4))
        self.assertAlmostEqual(targets[0, 3].item(), expected_ir_x - 0.50, places=6)

    def test_configuration_rejects_non_paired_or_tiled_datasets(self):
        with patch(
            "sarfusion.data.wisard.build_wisard_items",
            return_value=[(RGB_ITEM, ("vis.jpg", "vis.txt"))],
        ):
            with self.assertRaisesRegex(ValueError, "only paired"):
                WiSARDDataset(
                    root="unused",
                    folders=[],
                    box_alignment_targets=True,
                )

        with patch(
            "sarfusion.data.wisard.build_wisard_items", return_value=_paired_item()
        ):
            with self.assertRaisesRegex(ValueError, "tiled"):
                WiSARDDataset(
                    root="unused",
                    folders=[],
                    box_alignment_targets=True,
                    use_tiling=True,
                )
            with self.assertRaisesRegex(ValueError, "sqrt"):
                WiSARDDataset(
                    root="unused",
                    folders=[],
                    box_alignment_max_distance=float("inf"),
                )

    def test_only_fusion_samples_carry_targets_and_collate_keeps_a_list(self):
        with patch(
            "sarfusion.data.wisard.build_wisard_items", return_value=_paired_item()
        ):
            dataset = WiSARDDataset(
                root="unused",
                folders=[],
                transform=_FakeProcessor(),
                box_alignment_targets=True,
            )

        dataset._load_rgb = lambda _path: Image.new("RGB", (100, 100))
        dataset._load_ir = lambda _path: Image.new("RGB", (100, 100))
        dataset._load_annotations = lambda path, _image, _index: (
            _annotations((0.40, 0.50))
            if path == "vis.txt"
            else _annotations((0.42, 0.50))
        )

        fusion_sample = dataset[0]
        dataset.modal_dropout = True
        dataset.modal_dropout_probs = [0.0, 1.0, 0.0]
        with patch("sarfusion.data.wisard.random.choices", return_value=["rgb"]):
            rgb_sample = dataset[0]

        self.assertEqual(fusion_sample.modality_mode, "fusion")
        self.assertEqual(tuple(fusion_sample.box_alignment_targets.shape), (1, 4))
        self.assertEqual(rgb_sample.modality_mode, "rgb")
        self.assertEqual(tuple(rgb_sample.box_alignment_targets.shape), (0, 4))

        batch = dataset.collate_fn([fusion_sample, rgb_sample])
        self.assertIsInstance(batch["box_alignment_targets"], list)
        self.assertEqual(len(batch["box_alignment_targets"]), 2)

    def test_validation_and_test_force_targets_off(self):
        dataset_params = {
            "root": "dataset/WiSARD",
            "folders": "vis_ir",
            "box_alignment_targets": True,
            "box_alignment_max_distance": 0.05,
        }
        train, val, test = get_train_val_test_params("wisard", dataset_params)

        self.assertTrue(train["box_alignment_targets"])
        self.assertFalse(val["box_alignment_targets"])
        self.assertFalse(test["box_alignment_targets"])
