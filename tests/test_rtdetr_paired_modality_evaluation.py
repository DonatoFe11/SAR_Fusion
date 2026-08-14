import tempfile
import unittest
from pathlib import Path

import torch

from scripts.run_rtdetr_paired_modality_evaluation import (
    build_aggregates,
    check_fusion_reference,
    load_protocol,
    mask_modalities,
    stable_json_hash,
    summarize_signed_values,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRTDETRPairedModalityEvaluation(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(
            REPO_ROOT
            / "parameters"
            / "RTDETR"
            / "rtdetr_paired_modality_evaluation.yaml"
        )

    def test_protocol_fixes_same_frames_vis_ground_truth_and_six_configs(self):
        self.assertEqual(self.protocol["source"]["expected_frames"], 708)
        self.assertEqual(self.protocol["source"]["ground_truth"], "vis")
        self.assertEqual(self.protocol["source"]["expected_vis_boxes"], 1770)
        self.assertEqual(len(self.protocol["configurations"]), 6)
        self.assertFalse(self.protocol["interpretation"]["model_selection_allowed"])

    def test_channel_masks_are_exact_and_do_not_mutate_source(self):
        pixels = torch.arange(2 * 4 * 3 * 3, dtype=torch.float32).reshape(2, 4, 3, 3)
        original = pixels.clone()

        fusion = mask_modalities(pixels, "vis_ir")
        vis = mask_modalities(pixels, "vis")
        ir = mask_modalities(pixels, "ir")

        self.assertTrue(torch.equal(fusion, original))
        self.assertTrue(torch.equal(vis[:, :3], original[:, :3]))
        self.assertTrue(torch.count_nonzero(vis[:, 3:]) == 0)
        self.assertTrue(torch.count_nonzero(ir[:, :3]) == 0)
        self.assertTrue(torch.equal(ir[:, 3:], original[:, 3:]))
        self.assertTrue(torch.equal(pixels, original))

    def test_channel_mask_rejects_wrong_input_or_modality(self):
        with self.assertRaisesRegex(ValueError, "Bx4xHxW"):
            mask_modalities(torch.zeros(1, 3, 8, 8), "vis")
        with self.assertRaisesRegex(ValueError, "Unknown modality"):
            mask_modalities(torch.zeros(1, 4, 8, 8), "thermal")

    def test_fusion_reference_has_frozen_tolerance(self):
        result = check_fusion_reference(self.protocol, "additive", 40, 0.2570)
        self.assertAlmostEqual(result["difference"], 0.0004)
        with self.assertRaisesRegex(RuntimeError, "sanity check failed"):
            check_fusion_reference(self.protocol, "additive", 40, 0.30)

    def test_signed_summary_keeps_negative_deltas(self):
        summary = summarize_signed_values([-0.2, 0.1, 0.4])
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["mean"], 0.1)
        self.assertEqual(summary["min"], -0.2)

    def test_complete_aggregate_requires_all_ninety_units(self):
        payloads = []
        for configuration in self.protocol["configurations"]:
            for seed in self.protocol["seeds"]:
                for modality in self.protocol["modalities"]:
                    score = 0.24 if modality == "vis_ir" else 0.20
                    if configuration == "fam":
                        score += 0.01
                    payloads.append(
                        {
                            "configuration": configuration,
                            "seed": seed,
                            "modality": modality,
                            "n_samples": 708,
                            "metrics": {metric: score for metric in (
                                "map", "map_50", "map_75", "map_small", "map_medium",
                                "map_large", "mar_1", "mar_10", "mar_100",
                            )},
                        }
                    )
        manifest = {"n_frames": 708, "ground_truth": "vis"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            combined = build_aggregates(
                payloads,
                self.protocol,
                stable_json_hash(self.protocol),
                manifest,
                Path(temporary_directory),
                complete=True,
            )
        self.assertTrue(combined["protocol_complete"])
        self.assertEqual(len(combined["results"]), 90)
        self.assertEqual(combined["fusion_minus_best_single_map50"]["additive"]["fusion_wins"], 5)


if __name__ == "__main__":
    unittest.main()
