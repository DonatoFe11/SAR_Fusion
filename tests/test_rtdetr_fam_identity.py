import unittest

import torch

from sarfusion.models.rtdetr_fusion import (
    GridSampleFeatureAlignmentModule,
    IdentityInitializedFeatureAlignmentModule,
)


MODULE_CLASSES = (
    IdentityInitializedFeatureAlignmentModule,
    GridSampleFeatureAlignmentModule,
)
PYRAMID_SHAPES = (
    (2, 8, 80, 80),
    (2, 8, 40, 40),
    (2, 8, 20, 20),
    (2, 8, 81, 79),
    (2, 8, 41, 39),
    (2, 8, 21, 19),
)


def _relative_l2_error(actual, expected):
    return torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(
        expected
    ).clamp_min(torch.finfo(expected.dtype).eps)


class TestFAMIdentityInitialization(unittest.TestCase):
    def test_initial_forward_is_ir_identity(self):
        for module_class in MODULE_CLASSES:
            for shape in PYRAMID_SHAPES:
                with self.subTest(module=module_class.__name__, shape=shape):
                    torch.manual_seed(42)
                    rgb = torch.randn(shape)
                    ir = torch.randn(shape)
                    module = module_class(shape[1])
                    module.eval()

                    with torch.no_grad():
                        first = module(rgb, ir)
                        repeated = module(rgb, ir)
                        module.train()
                        training_mode = module(rgb, ir)

                    self.assertEqual(first.shape, ir.shape)
                    self.assertTrue(torch.isfinite(first).all())
                    self.assertLess((first - ir).abs().max().item(), 1e-5)
                    self.assertLess(_relative_l2_error(first, ir).item(), 1e-5)
                    torch.testing.assert_close(
                        repeated, first, atol=0.0, rtol=0.0
                    )
                    torch.testing.assert_close(
                        training_mode, first, atol=0.0, rtol=0.0
                    )

    def test_gradients_are_finite_and_optimizer_can_leave_identity(self):
        for module_class in MODULE_CLASSES:
            with self.subTest(module=module_class.__name__):
                torch.manual_seed(42)
                rgb = torch.randn(2, 4, 9, 8)
                ir = torch.randn(2, 4, 9, 8, requires_grad=True)
                module = module_class(4)
                optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2)

                before = module(rgb, ir)
                torch.testing.assert_close(before, ir, atol=1e-5, rtol=1e-5)

                loss = before.square().mean()
                loss.backward()

                self.assertIsNotNone(ir.grad)
                self.assertTrue(torch.isfinite(ir.grad).all())
                trainable_parameters = [
                    parameter
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ]
                parameter_grads = [
                    parameter.grad for parameter in trainable_parameters
                ]
                self.assertTrue(trainable_parameters)
                self.assertTrue(
                    all(gradient is not None for gradient in parameter_grads)
                )
                self.assertTrue(
                    all(
                        torch.isfinite(gradient).all()
                        for gradient in parameter_grads
                    )
                )
                self.assertTrue(
                    any(
                        gradient.abs().max().item() > 0
                        for gradient in parameter_grads
                    )
                )

                optimizer.step()
                after = module(rgb, ir.detach())

                self.assertTrue(torch.isfinite(after).all())
                self.assertFalse(
                    torch.allclose(after, ir.detach(), atol=1e-5, rtol=1e-5)
                )

    def test_identity_variants_reject_spatial_jitter(self):
        for module_class in MODULE_CLASSES:
            with self.subTest(module=module_class.__name__):
                with self.assertRaisesRegex(
                    ValueError, "spatial_jitter_std=0.0"
                ):
                    module_class(4, spatial_jitter_std=0.5)


if __name__ == "__main__":
    unittest.main()
