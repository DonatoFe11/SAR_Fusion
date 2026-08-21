import unittest
from pathlib import Path

import torch

from scripts.run_rtdetr_fam_reliability_gate_weight_audit import (
    WeightAccumulator,
    mask_modalities,
    prepare_batch,
    summarize_across_seeds,
    validate_protocol,
)
from sarfusion.utils.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_reliability_gate_weight_audit.yaml"
)


class TestReliabilityGateWeightAudit(unittest.TestCase):
    def test_protocol_is_validation_only_and_uses_all_best_checkpoints(self):
        protocol = load_yaml(PROTOCOL_PATH)
        validate_protocol(protocol)
        self.assertEqual(protocol["expected_validation_frames"], 896)
        self.assertEqual(protocol["expected_validation_batches"], 75)

    def test_modality_masks_preserve_the_requested_channels(self):
        pixels = torch.ones(2, 4, 3, 3)
        fusion = mask_modalities(pixels, "fusion")
        rgb = mask_modalities(pixels, "rgb")
        ir = mask_modalities(pixels, "ir")
        torch.testing.assert_close(fusion, pixels)
        self.assertTrue((rgb[:, :3] == 1).all())
        self.assertTrue((rgb[:, 3:] == 0).all())
        self.assertTrue((ir[:, :3] == 0).all())
        self.assertTrue((ir[:, 3:] == 1).all())
        torch.testing.assert_close(pixels, torch.ones_like(pixels))

    def test_production_collation_contract_uses_mapping_keys(self):
        batch = {
            "pixel_values": torch.ones(2, 4, 3, 3),
            "pixel_mask": torch.ones(2, 3, 3, dtype=torch.bool),
        }
        pixel_values, pixel_mask = prepare_batch(batch, "rgb", torch.device("cpu"))
        self.assertEqual(tuple(pixel_values.shape), (2, 4, 3, 3))
        self.assertEqual(tuple(pixel_mask.shape), (2, 3, 3))
        self.assertTrue((pixel_values[:, 3:] == 0).all())

    def test_accumulator_reports_exact_moments_and_threshold_fractions(self):
        accumulator = WeightAccumulator(400, 0.9, 1.1)
        accumulator.update(torch.tensor([0.5, 1.0, 1.5]))
        summary = accumulator.summary()
        self.assertEqual(summary["numel"], 3)
        self.assertAlmostEqual(summary["mean"], 1.0)
        self.assertAlmostEqual(summary["mean_abs_delta_one"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["fraction_below_low"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["fraction_above_high"], 1.0 / 3.0)

    def test_cross_seed_summary_uses_seed_means_as_units(self):
        rows = [
            {"seed": 40, "mode": "fusion", "level": 0, "modality": "rgb", "mean": 0.9},
            {"seed": 41, "mode": "fusion", "level": 0, "modality": "rgb", "mean": 1.1},
        ]
        summary = summarize_across_seeds(rows)[0]
        self.assertAlmostEqual(summary["mean"], 1.0)
        self.assertAlmostEqual(summary["sample_std"], 2 ** 0.5 / 10)
        self.assertEqual(summary["seed_means"], {"40": 0.9, "41": 1.1})


if __name__ == "__main__":
    unittest.main()
