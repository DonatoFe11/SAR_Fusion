import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torchvision.ops import DeformConv2d

from scripts.run_rtdetr_compute_benchmark import (
    fam_conventional_cost,
    load_protocol,
    parameter_summary,
    percentile,
    summarize_values,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyFAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.offset_conv = nn.Conv2d(2 * channels, 27, 3, padding=1)
        self.deform_conv = DeformConv2d(channels, channels, 3, padding=1)


class FeatureAlignmentModule(TinyFAM):
    pass


class TestRTDETRComputeBenchmark(unittest.TestCase):
    def test_frozen_protocol_scope_and_order(self):
        protocol = load_protocol(
            REPO_ROOT
            / "parameters"
            / "RTDETR"
            / "rtdetr_additive_fam_compute_benchmark.yaml"
        )

        self.assertEqual(protocol["checkpoint_seed"], 43)
        self.assertEqual(protocol["checkpoint"], "latest")
        self.assertEqual(protocol["fam_variant"], "current_dcnv2")
        self.assertEqual(
            protocol["execution"]["trial_order"],
            [
                ["additive", "fam"],
                ["fam", "additive"],
                ["additive", "fam"],
            ],
        )
        self.assertFalse(protocol["execution"]["preprocessing_included"])
        self.assertFalse(protocol["execution"]["postprocessing_included"])
        self.assertFalse(protocol["execution"]["tf32"])

    def test_percentile_and_summary(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        self.assertEqual(percentile(values, 0.0), 1.0)
        self.assertEqual(percentile(values, 0.5), 3.0)
        self.assertEqual(percentile(values, 1.0), 5.0)
        summary = summarize_values(values)
        self.assertEqual(summary["n"], 5)
        self.assertEqual(summary["mean"], 3.0)
        self.assertEqual(summary["median"], 3.0)

    def test_parameter_summary_separates_fam_parameters(self):
        model = nn.Sequential(nn.Conv2d(4, 8, 1), FeatureAlignmentModule(8))

        summary = parameter_summary(model)
        expected_fam = sum(parameter.numel() for parameter in model[1].parameters())
        expected_total = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(summary["fam_parameters"], expected_fam)
        self.assertEqual(summary["total_parameters"], expected_total)
        self.assertEqual(summary["non_fam_parameters"], expected_total - expected_fam)
        self.assertEqual(summary["parameter_bytes"], expected_total * 4)

    def test_conventional_fam_cost_matches_dense_mac_formula(self):
        channels = 8
        height = width = 10
        module = FeatureAlignmentModule(channels)
        shapes = [
            {
                "level": 0,
                "rgb_shape": [1, channels, height, width],
                "ir_shape": [1, channels, height, width],
            }
        ]

        result = fam_conventional_cost([module], shapes)
        expected_offset = height * width * 27 * (2 * channels) * 3 * 3
        expected_deform = height * width * channels * channels * 3 * 3

        self.assertEqual(result["offset_conv_macs"], expected_offset)
        self.assertEqual(result["deform_conv_macs"], expected_deform)
        self.assertEqual(result["total_macs"], expected_offset + expected_deform)
        self.assertEqual(
            result["conventional_flops_two_per_mac"],
            2 * (expected_offset + expected_deform),
        )


if __name__ == "__main__":
    unittest.main()
