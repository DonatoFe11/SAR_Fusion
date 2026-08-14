import unittest
import tempfile
from pathlib import Path

import torch
from PIL import Image

from scripts.run_rtdetr_error_analysis import (
    CONFIDENCE_THRESHOLDS,
    PRIMARY_CONFIDENCE,
    analyze_image,
    greedy_match,
    render_qualitative_figures,
    render_threshold_sensitivity,
    select_qualitative_samples,
    size_category_from_normalized_box,
    summarize_rows,
    xywh_to_xyxy,
)


class TestRTDetrErrorAnalysis(unittest.TestCase):
    def test_thresholds_include_the_final_evaluation_setting_as_primary(self):
        self.assertEqual(PRIMARY_CONFIDENCE, 0.01)
        self.assertEqual(CONFIDENCE_THRESHOLDS, (0.01, 0.05, 0.10, 0.25, 0.50))

    def test_size_categories_use_frozen_coco_limits_at_640(self):
        self.assertEqual(size_category_from_normalized_box([0.5, 0.5, 31 / 640, 31 / 640]), "small")
        self.assertEqual(size_category_from_normalized_box([0.5, 0.5, 32 / 640, 32 / 640]), "medium")
        self.assertEqual(size_category_from_normalized_box([0.5, 0.5, 96 / 640, 96 / 640]), "large")

    def test_greedy_matching_is_one_to_one(self):
        gt = xywh_to_xyxy([[0.5, 0.5, 0.2, 0.2]])
        predictions = xywh_to_xyxy(
            [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]
        )
        matches = greedy_match(gt, predictions)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["gt_index"], 0)
        self.assertEqual(matches[0]["pred_index"], 0)

    def test_confidence_filter_and_error_counts(self):
        result = analyze_image(
            gt_boxes=torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            pred_boxes=torch.tensor(
                [[0.5, 0.5, 0.1, 0.1], [0.1, 0.1, 0.05, 0.05]]
            ),
            pred_scores=torch.tensor([0.9, 0.4]),
            confidence_threshold=0.5,
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
        self.assertEqual(result["n_predictions"], 1)
        self.assertEqual(result["matched_prediction_indices"], [0])

    def test_empty_frame_predictions_are_false_positives(self):
        result = analyze_image(
            gt_boxes=torch.empty((0, 4)),
            pred_boxes=torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            pred_scores=torch.tensor([0.9]),
            confidence_threshold=0.5,
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 0))
        self.assertFalse(result["has_annotations"])

    def test_summary_preserves_frame_and_object_denominators(self):
        rows = [
            {
                "n_gt": 1,
                "n_predictions": 1,
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "no_prediction": False,
                "has_annotations": True,
                "matched_iou_sum": 0.8,
                "matched_iou_count": 1,
                "size_gt": {"small": 1, "medium": 0, "large": 0},
                "size_tp": {"small": 1, "medium": 0, "large": 0},
            },
            {
                "n_gt": 0,
                "n_predictions": 2,
                "tp": 0,
                "fp": 2,
                "fn": 0,
                "no_prediction": False,
                "has_annotations": False,
                "matched_iou_sum": 0.0,
                "matched_iou_count": 0,
                "size_gt": {"small": 0, "medium": 0, "large": 0},
                "size_tp": {"small": 0, "medium": 0, "large": 0},
            },
        ]
        summary = summarize_rows(rows)
        self.assertAlmostEqual(summary["precision"], 1 / 3)
        self.assertAlmostEqual(summary["recall"], 1.0)
        self.assertAlmostEqual(summary["fp_per_image"], 1.0)
        self.assertAlmostEqual(summary["empty_frame_fp_fraction"], 1.0)
        self.assertAlmostEqual(summary["small_recall"], 1.0)
        self.assertIsNone(summary["medium_recall"])

    def test_qualitative_selection_uses_fixed_positions_per_session(self):
        records = []
        for session in ("a", "b"):
            for position in range(6):
                records.append(
                    {
                        "sample_index": len(records),
                        "session": session,
                        "session_position": position,
                        "n_gt": 0 if position in {3, 4} else 1,
                        "size_counts": {
                            "small": 1 if position in {0, 1, 2, 5} else 0,
                            "medium": 0,
                            "large": 0,
                        },
                    }
                )
        selected = select_qualitative_samples(records)
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            [(row["session"], row["selection_category"], row["session_position"]) for row in selected],
            [("a", "small_target", 2), ("a", "empty", 3), ("b", "small_target", 2), ("b", "empty", 3)],
        )

    def test_qualitative_renderer_uses_same_manifest_frame_for_both_models(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            Image.new("RGB", (100, 60), "gray").save(root / "rgb.png")
            Image.new("L", (80, 60), 128).save(root / "ir.png")
            manifest = {
                "selected_samples": [
                    {
                        "sample_index": 7,
                        "selection_category": "small_target",
                        "vis_path": "rgb.png",
                        "ir_path": "ir.png",
                    }
                ]
            }
            sample = {
                "gt_boxes": [[0.5, 0.5, 0.2, 0.2]],
                "pred_boxes": [[0.5, 0.5, 0.2, 0.2]],
                "pred_scores": [0.9],
                "matches": [{"gt_index": 0, "pred_index": 0, "iou": 1.0}],
                "matched_gt_indices": [0],
                "matched_prediction_indices": [0],
                "counts": {"n_gt": 1, "n_predictions": 1, "tp": 1, "fp": 0, "fn": 0},
            }
            payloads = [
                {
                    "configuration": configuration,
                    "seed": 43,
                    "figure_samples": {"0.01": {"7": sample}},
                }
                for configuration in ("additive", "fam")
            ]
            render_qualitative_figures(payloads, manifest, root, root / "output")
            self.assertTrue(
                (root / "output/figures/qualitative/sample_007_small_target_conf_001.png").is_file()
            )

    def test_threshold_sensitivity_renderer_writes_figure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metrics = {
                "precision": {"mean": 0.5, "sample_std": 0.1},
                "recall": {"mean": 0.6, "sample_std": 0.1},
                "fp_per_image": {"mean": 1.0, "sample_std": 0.2},
                "nonempty_miss_frame_fraction": {"mean": 0.4, "sample_std": 0.1},
            }
            combined = {
                "across_seed_summaries": {
                    f"{threshold:.2f}": {
                        configuration: metrics
                        for configuration in ("additive", "fam")
                    }
                    for threshold in CONFIDENCE_THRESHOLDS
                }
            }
            render_threshold_sensitivity(combined, root)
            self.assertTrue(
                (root / "figures/rtdetr_additive_fam_threshold_sensitivity.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
