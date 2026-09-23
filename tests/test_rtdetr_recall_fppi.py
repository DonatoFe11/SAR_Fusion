import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from scripts import run_rtdetr_recall_fppi as runner


REPO_ROOT = Path(__file__).resolve().parents[1]


def cache_fixture():
    arrays = {
        "gt_boxes": np.array([[0.0, 0.0, 2.0, 2.0]]),
        "gt_offsets": np.array([0, 1, 1], dtype=np.int64),
        "pred_boxes": np.array([
            [0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 4.0, 4.0],
            [0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 4.0, 4.0],
        ]),
        "pred_offsets": np.array([0, 2, 4], dtype=np.int64),
        "scores": np.array([0.9, 0.1, 0.8, 0.2]),
        "sample_indices": np.array([0, 1], dtype=np.int64),
    }
    metadata = {
        "job": "test", "native_queries": 2, "seed": 40,
        "n_images": 2, "n_gt": 1, "n_empty": 1,
        "checkpoint_sha256": "a" * 64,
    }
    return arrays, metadata


def fake_batch(indices=(0, 1), with_mask=True):
    pixels = torch.zeros(len(indices), 4, 4, 4)
    for row, index in enumerate(indices):
        pixels[row, 0, 0, 0] = index
    batch = {
        "pixel_values": pixels,
        "sample_idx": torch.tensor(indices, dtype=torch.int64),
        "labels": [
            {"boxes": torch.tensor([[1.0, 1.0, 2.0, 2.0]])
             if index == 0 else torch.empty(0, 4)}
            for index in indices
        ],
    }
    if with_mask:
        batch["pixel_mask"] = torch.ones(len(indices), 4, 4, dtype=torch.bool)
    return batch


class FakeModel(torch.nn.Module):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
        self.calls = []

    def forward(self, pixel_values, pixel_mask=None):
        self.calls.append({
            "grad_enabled": torch.is_grad_enabled(),
            "inference_mode": torch.is_inference_mode_enabled(),
            "mask": pixel_mask,
        })
        predictions = []
        for sample in pixel_values:
            index = int(sample[0, 0, 0])
            predictions.append({
                "boxes": torch.tensor([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 2.0, 2.0]]),
                "scores": torch.tensor([0.9, 0.1] if index == 0 else [0.8, 0.2]),
                "labels": torch.zeros(2, dtype=torch.int64),
            })
        if self.transform is not None:
            predictions = self.transform(predictions)
        return {"predictions": predictions}


class RecallFPPIProtocolTests(unittest.TestCase):
    def test_current_protocol_and_critical_values_are_frozen(self):
        protocol = runner.load_protocol(REPO_ROOT / runner.DEFAULT_PROTOCOL)
        self.assertEqual(protocol["seeds"], list(range(40, 45)))
        self.assertEqual(protocol["configurations"], ["historical_additive", "historical_fam"])
        self.assertEqual(protocol["acquisitions"], ["mterie", "carnation_0025_0026", "fhl_0407_0408"])
        changes = {
            "checkpoint": "best", "seeds": [40], "collection_threshold": 0.01,
            "iou_threshold": 0.75, "fppi_budgets": [0.5],
            "matching": "iou_descending", "ground_truth": "ir",
        }
        for key, value in changes.items():
            changed = copy.deepcopy(protocol)
            changed[key] = value
            with self.subTest(key=key), patch.object(runner, "load_yaml", return_value={"protocol": changed}):
                with self.assertRaisesRegex(ValueError, key):
                    runner.load_protocol("unused.yaml")
        changed = copy.deepcopy(protocol)
        changed["interpretation"]["deployment_threshold_selection_allowed"] = True
        with patch.object(runner, "load_yaml", return_value={"protocol": changed}):
            with self.assertRaisesRegex(ValueError, "interpretation"):
                runner.load_protocol("unused.yaml")

    def test_xywh_conversion_uses_centres_and_does_not_clip(self):
        boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.0, 0.0, 0.2, 0.2]])
        original = boxes.copy()
        np.testing.assert_allclose(
            runner.xywh_to_xyxy(boxes), [[0.4, 0.3, 0.6, 0.7], [-0.1, -0.1, 0.1, 0.1]]
        )
        np.testing.assert_array_equal(boxes, original)
        self.assertEqual(runner.xywh_to_xyxy(np.empty((0, 4))).shape, (0, 4))

    def test_xywh_conversion_rejects_malformed_shape(self):
        for boxes in (np.zeros((2, 2, 2)), np.zeros((2, 3))):
            with self.subTest(shape=boxes.shape), self.assertRaises(ValueError):
                runner.xywh_to_xyxy(boxes)


class RecallFPPICacheTests(unittest.TestCase):
    def test_atomic_roundtrip_and_metadata_mismatch(self):
        arrays, metadata = cache_fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "cache.npz"
            runner.atomic_npz(path, metadata=json.dumps(metadata), **arrays)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(".npz.tmp").exists())
            restored, restored_metadata = runner.load_cache(path, {"seed": 40, "native_queries": 2})
            self.assertEqual(metadata, restored_metadata)
            for name, value in arrays.items():
                np.testing.assert_array_equal(value, restored[name])
            with self.assertRaisesRegex(RuntimeError, "seed"):
                runner.load_cache(path, {"seed": 41})
            with self.assertRaisesRegex(RuntimeError, "checkpoint_sha256"):
                runner.load_cache(path, {"checkpoint_sha256": "b" * 64})

    def test_malformed_offsets_are_rejected_including_unsigned_wrap(self):
        cases = {
            "wrong_start": ("pred_offsets", np.array([1, 2, 4])),
            "wrong_end": ("pred_offsets", np.array([0, 2, 3])),
            "wrong_shape": ("pred_offsets", np.array([0, 4])),
            "wrong_dtype": ("pred_offsets", np.array([0.0, 2.0, 4.0])),
            "decreasing_unsigned": ("pred_offsets", np.array([0, 5, 4], dtype=np.uint64)),
            "native_queries_missing": ("pred_offsets", np.array([0, 1, 4])),
            "gt_offsets_wrong": ("gt_offsets", np.array([0, 0, 0])),
        }
        for label, (name, value) in cases.items():
            arrays, metadata = cache_fixture()
            arrays[name] = value
            with self.subTest(label=label), self.assertRaises(ValueError):
                runner.validate_arrays(arrays, metadata)

    def test_malformed_scores_boxes_and_indices_are_rejected(self):
        cases = {
            "score_nan": ("scores", np.array([0.9, np.nan, 0.8, 0.2])),
            "score_high": ("scores", np.array([1.1, 0.1, 0.8, 0.2])),
            "score_low": ("scores", np.array([-0.1, 0.1, 0.8, 0.2])),
            "score_count": ("scores", np.array([0.9, 0.1])),
            "indices_shuffle": ("sample_indices", np.array([1, 0])),
            "indices_float": ("sample_indices", np.array([0.0, 1.0])),
            "indices_duplicate": ("sample_indices", np.array([0, 0])),
            "indices_shape": ("sample_indices", np.array([[0, 1]])),
            "gt_nonfinite": ("gt_boxes", np.array([[0.0, 0.0, np.inf, 2.0]])),
            "gt_degenerate": ("gt_boxes", np.array([[0.0, 0.0, 0.0, 2.0]])),
        }
        for label, (name, value) in cases.items():
            arrays, metadata = cache_fixture()
            arrays[name] = value
            with self.subTest(label=label), self.assertRaises(ValueError):
                runner.validate_arrays(arrays, metadata)

    def test_metadata_counts_must_agree_with_arrays(self):
        for name, value in (
            ("n_images", 3), ("n_gt", 2), ("n_empty", 0),
            ("n_images", True), ("n_gt", 1.0), ("n_empty", False),
            ("native_queries", 0),
        ):
            arrays, metadata = cache_fixture()
            metadata[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.validate_arrays(arrays, metadata)

    def test_make_curve_matches_image_local_gt_and_counts_empty_frames(self):
        arrays, metadata = cache_fixture()
        curve = runner.make_curve(arrays, metadata, iou_threshold=0.5)
        np.testing.assert_array_equal(curve["threshold"], [np.inf, 0.9, 0.8, 0.2, 0.1])
        np.testing.assert_array_equal(curve["tp"], [0, 1, 1, 1, 1])
        np.testing.assert_array_equal(curve["fp"], [0, 0, 1, 2, 3])
        np.testing.assert_array_equal(curve["recall"], [0, 1, 1, 1, 1])
        np.testing.assert_array_equal(curve["fppi"], [0, 0, 0.5, 1.0, 1.5])


class RecallFPPIPredictionCollectionTests(unittest.TestCase):
    def test_collect_preserves_native_queries_and_runs_without_training_loss(self):
        model = FakeModel().eval()
        arrays, metadata = runner.collect_predictions(
            model, [fake_batch()], "cpu", {"job": "fake", "native_queries": 2}
        )
        expected, _ = cache_fixture()
        for name, value in expected.items():
            np.testing.assert_allclose(arrays[name], value)
        self.assertEqual((metadata["n_images"], metadata["n_gt"], metadata["n_empty"]), (2, 1, 1))
        self.assertGreaterEqual(metadata["elapsed_seconds"], 0)
        self.assertTrue(model.calls[0]["inference_mode"])
        self.assertFalse(model.calls[0]["grad_enabled"])
        self.assertIsNotNone(model.calls[0]["mask"])

    def test_smoke_limit_and_absent_mask(self):
        model = FakeModel().eval()
        arrays, metadata = runner.collect_predictions(
            model, [fake_batch((0,), with_mask=False), fake_batch((1,))],
            "cpu", {"job": "fake", "native_queries": 2}, max_batches=1,
        )
        self.assertEqual(metadata["n_images"], 1)
        self.assertEqual(len(model.calls), 1)
        self.assertIsNone(model.calls[0]["mask"])
        np.testing.assert_array_equal(arrays["sample_indices"], [0])

    def test_missing_queries_classes_and_batch_sizes_fail(self):
        def shortened(predictions):
            for prediction in predictions:
                for key in prediction:
                    prediction[key] = prediction[key][:1]
            return predictions

        def wrong_class(predictions):
            predictions[0]["labels"][0] = 1
            return predictions

        def wrong_batch(predictions):
            return predictions[:1]

        def wrong_label_shape(predictions):
            predictions[0]["labels"] = predictions[0]["labels"][:1]
            return predictions

        for transform in (shortened, wrong_class, wrong_batch, wrong_label_shape):
            with self.subTest(transform=transform.__name__), self.assertRaises(RuntimeError):
                runner.collect_predictions(
                    FakeModel(transform).eval(), [fake_batch()], "cpu",
                    {"job": "fake", "native_queries": 2},
                )

    def test_shuffled_or_incomplete_sample_indices_fail(self):
        shuffled = fake_batch((1, 0))
        truncated = fake_batch()
        truncated["sample_idx"] = truncated["sample_idx"][:1]
        for batch in (shuffled, truncated):
            with self.subTest(indices=batch["sample_idx"]), self.assertRaises(RuntimeError):
                runner.collect_predictions(
                    FakeModel().eval(), [batch], "cpu", {"job": "fake", "native_queries": 2}
                )

    def test_empty_loader_fails(self):
        with self.assertRaisesRegex(RuntimeError, "No frames"):
            runner.collect_predictions(FakeModel().eval(), [], "cpu", {"job": "fake", "native_queries": 2})


class RecallFPPICheckpointAuditTests(unittest.TestCase):
    def test_shared_parameter_alias_is_recovered_and_unknown_key_fails(self):
        model = torch.nn.Module()
        model.head = torch.nn.Linear(2, 1, bias=False)
        model.decoder_head = model.head
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.safetensors"
            save_file({"model.head.weight": model.head.weight.detach().clone()}, str(path))
            runner.verify_checkpoint_keys(model, path)
            save_file({"model.wrong.weight": model.head.weight.detach().clone()}, str(path))
            with self.assertRaisesRegex(RuntimeError, "Checkpoint mismatch"):
                runner.verify_checkpoint_keys(model, path)

    def test_shared_buffer_alias_is_recovered(self):
        model = torch.nn.Module()
        buffer = torch.ones(2)
        model.register_buffer("running_a", buffer)
        model.register_buffer("running_b", buffer)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.safetensors"
            save_file({"model.running_a": buffer.clone()}, str(path))
            runner.verify_checkpoint_keys(model, path)


if __name__ == "__main__":
    unittest.main()
