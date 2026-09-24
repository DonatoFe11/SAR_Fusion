import math
import unittest
from unittest.mock import Mock

import torch
from torch import nn

from fam_alignment_check import (
    FAMCapture,
    aggregate_numeric_records,
    net_offset_field,
    offset_statistics,
    offset_vectors,
    plot_offset_field,
)
from sarfusion.models.rtdetr_fusion import (
    BoundedFeatureAlignmentModule,
    BoxGuidedCommonOffsetFeatureAlignmentModule,
)


class GridSampleFeatureAlignmentModule(nn.Module):
    """Minimal namesake used to verify class-name-based hook registration."""

    def __init__(self):
        super().__init__()
        self.offset_conv = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, rgb, ir):
        self.offset_conv(torch.cat([rgb, ir], dim=1))
        return ir


class TestFAMAlignmentDiagnostics(unittest.TestCase):
    def test_grid_sample_offsets_keep_one_vector_per_cell(self):
        offset = torch.zeros(2, 2, 4)
        offset[0] = 1.0
        offset[1] = 2.0

        vectors = offset_vectors(offset, "grid_sample")
        net = net_offset_field(offset, "grid_sample")
        stats = offset_statistics(
            offset,
            "grid_sample",
            input_hw=(20, 40),
        )

        self.assertEqual(tuple(vectors.shape), (1, 2, 2, 4))
        self.assertTrue(torch.equal(net, offset))
        self.assertAlmostEqual(
            stats["net_vector_magnitude_input_px"]["mean"],
            math.sqrt(10.0 ** 2 + 20.0 ** 2),
            places=5,
        )
        self.assertNotIn("mask", stats)

    def test_dcnv2_net_field_is_mask_weighted_across_nine_points(self):
        vectors = torch.zeros(9, 2, 1, 1)
        vectors[0, 0] = 9.0
        offset = vectors.reshape(18, 1, 1)
        mask = torch.zeros(9, 1, 1)
        mask[0] = 1.0

        net = net_offset_field(offset, "dcnv2_3x3", mask=mask)

        self.assertAlmostEqual(net[0, 0, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(net[1, 0, 0].item(), 9.0, places=5)

    def test_dcnv2_coordinates_scale_with_the_correct_image_axis(self):
        offset = torch.zeros(18, 2, 4)
        offset[0::2] = 2.0  # dy
        offset[1::2] = 3.0  # dx
        stats = offset_statistics(
            offset, "dcnv2_3x3", input_hw=(20, 80), mask=torch.ones(9, 2, 4)
        )
        self.assertEqual(stats["signed_coordinate_mean_feature_px"], [3.0, 2.0])
        self.assertAlmostEqual(
            stats["net_vector_magnitude_input_px"]["mean"],
            math.hypot(3 * 20, 2 * 10), places=4,
        )

    def test_quiver_keeps_positive_dy_on_the_downward_image_axis(self):
        for kind, channels in (("dcnv2_3x3", 18), ("grid_sample", 2)):
            with self.subTest(kind=kind):
                offset = torch.zeros(channels, 2, 4)
                offset[0::2] = 2.0 if kind == "dcnv2_3x3" else 3.0
                offset[1::2] = 3.0 if kind == "dcnv2_3x3" else 2.0
                axes = Mock()
                mask = torch.ones(9, 2, 4) if channels == 18 else None
                plot_offset_field(axes, offset, mask, offset_kind=kind, stride=1)
                _, _, dx, dy = axes.quiver.call_args.args
                self.assertTrue((dx == 3.0).all())
                self.assertTrue((dy == 2.0).all())
                axes.set_ylim.assert_called_once_with(2, 0)

    def test_capture_supports_grid_sample_variant(self):
        model = nn.Sequential(GridSampleFeatureAlignmentModule())
        capture = FAMCapture(model)
        rgb = torch.randn(1, 2, 3, 3)
        ir = torch.randn(1, 2, 3, 3)

        model[0](rgb, ir)

        self.assertEqual(capture.records[0]["offset_kind"], "grid_sample")
        self.assertEqual(tuple(capture.records[0]["offset"].shape), (1, 2, 3, 3))
        capture.remove()

    def test_capture_reports_effective_bounded_offset_and_keeps_raw_value(self):
        model = nn.Sequential(BoundedFeatureAlignmentModule(2))
        with torch.no_grad():
            model[0].offset_conv.weight.zero_()
            model[0].offset_conv.bias[:18].fill_(100.0)
        capture = FAMCapture(model)
        rgb = torch.randn(1, 2, 3, 3)
        ir = torch.randn(1, 2, 3, 3)

        model[0](rgb, ir)

        record = capture.records[0]
        self.assertEqual(record["offset_kind"], "dcnv2_3x3")
        self.assertAlmostEqual(record["raw_offset"].mean().item(), 100.0)
        self.assertLessEqual(record["offset"].abs().max().item(), 4.0)
        capture.remove()

    def test_capture_reports_box_guidance_and_effective_total_offset(self):
        module = BoxGuidedCommonOffsetFeatureAlignmentModule(2)
        with torch.no_grad():
            module.offset_conv.weight.zero_()
            module.offset_conv.bias.zero_()
            module.offset_conv.bias[:18].fill_(1.0)
            module.guidance_predictor[-1].weight.zero_()
            # 4*tanh(raw/4) == 2 cells.
            module.guidance_predictor[-1].bias.fill_(
                4.0 * torch.atanh(torch.tensor(0.5)).item()
            )
        capture = FAMCapture(nn.Sequential(module))
        rgb = torch.randn(1, 2, 3, 3)
        ir = torch.randn(1, 2, 3, 3)

        module(rgb, ir, both_present=torch.tensor([True]))

        record = capture.records[0]
        self.assertEqual(record["offset_kind"], "dcnv2_3x3")
        torch.testing.assert_close(
            record["residual_offset"],
            torch.ones_like(record["residual_offset"]),
        )
        torch.testing.assert_close(
            record["guidance_flow"],
            torch.full_like(record["guidance_flow"], 2.0),
        )
        torch.testing.assert_close(
            record["offset"],
            torch.full_like(record["offset"], 3.0),
        )
        capture.remove()

    def test_aggregate_keeps_record_count(self):
        aggregate = aggregate_numeric_records(
            [{"offset": {"mean": 1.0}}, {"offset": {"mean": 3.0}}]
        )

        self.assertEqual(aggregate["offset.mean"]["n"], 2)
        self.assertAlmostEqual(aggregate["offset.mean"]["mean"], 2.0)
        self.assertAlmostEqual(
            aggregate["offset.mean"]["sample_std"],
            math.sqrt(2.0),
        )


if __name__ == "__main__":
    unittest.main()
