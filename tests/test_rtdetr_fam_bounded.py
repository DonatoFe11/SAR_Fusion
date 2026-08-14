import unittest

import torch

from sarfusion.models.rtdetr_fusion import (
    BoundedFeatureAlignmentModule,
    FAM_VARIANTS,
)


class TestBoundedFeatureAlignmentModule(unittest.TestCase):
    def test_variant_is_registered_with_fixed_limit(self):
        self.assertIs(
            FAM_VARIANTS["bounded_dcnv2_4"],
            BoundedFeatureAlignmentModule,
        )
        self.assertEqual(BoundedFeatureAlignmentModule.OFFSET_LIMIT_CELLS, 4.0)

    def test_transformed_offsets_are_strictly_bounded_and_finite(self):
        module = BoundedFeatureAlignmentModule(4)
        raw = torch.tensor(
            [-1.0e6, -8.0, -4.0, 0.0, 4.0, 8.0, 1.0e6]
        )
        transformed = module.transform_offset(raw)

        self.assertTrue(torch.isfinite(transformed).all())
        self.assertLessEqual(transformed.abs().max().item(), 4.0)
        self.assertEqual(transformed[3].item(), 0.0)

    def test_transform_has_unit_gradient_at_zero(self):
        module = BoundedFeatureAlignmentModule(4)
        raw = torch.zeros(1, requires_grad=True)
        module.transform_offset(raw).sum().backward()
        torch.testing.assert_close(raw.grad, torch.ones_like(raw))

    def test_initial_offset_prediction_remains_zero(self):
        torch.manual_seed(42)
        module = BoundedFeatureAlignmentModule(4)
        rgb = torch.randn(2, 4, 9, 8)
        ir = torch.randn(2, 4, 9, 8)

        raw = module.offset_conv(torch.cat([rgb, ir], dim=1))[:, :18]
        transformed = module.transform_offset(raw)
        torch.testing.assert_close(transformed, torch.zeros_like(transformed))


if __name__ == "__main__":
    unittest.main()
