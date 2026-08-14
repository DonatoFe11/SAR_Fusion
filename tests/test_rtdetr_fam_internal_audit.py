import unittest

import torch

from scripts.run_rtdetr_fam_internal_audit import (
    PROTOCOL_ID,
    assert_compatible_existing_audit,
    summarize_seed_values,
    tensor_pair_metrics,
    tensor_statistics,
)


class TestRTDetrFAMInternalAudit(unittest.TestCase):
    def test_identical_nonconstant_fields_have_zero_error_and_unit_correlation(self):
        field = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        metrics = tensor_pair_metrics(field, field.clone())
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["pearson_r"], 1.0)

    def test_constant_shift_preserves_correlation_but_not_mae(self):
        first = torch.tensor([1.0, 2.0, 3.0])
        second = first + 2.0
        metrics = tensor_pair_metrics(first, second)
        self.assertAlmostEqual(metrics["mae"], 2.0)
        self.assertAlmostEqual(metrics["rmse"], 2.0)
        self.assertAlmostEqual(metrics["pearson_r"], 1.0)

    def test_constant_fields_have_undefined_pearson(self):
        metrics = tensor_pair_metrics(torch.ones(4), torch.ones(4))
        self.assertIsNone(metrics["pearson_r"])

    def test_shape_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            tensor_pair_metrics(torch.ones(2), torch.ones(3))

    def test_tensor_statistics_report_nonfinite_fraction(self):
        stats = tensor_statistics(torch.tensor([1.0, 3.0, float("nan")]))
        self.assertEqual(stats["numel"], 3)
        self.assertAlmostEqual(stats["finite_fraction"], 2 / 3)
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["abs_mean"], 2.0)

    def test_seed_summary_keeps_checkpoints_as_units(self):
        summary = summarize_seed_values([(42, 3.0), (40, 1.0), (41, 2.0)])
        self.assertEqual(list(summary["seed_values"]), ["40", "41", "42"])
        self.assertEqual(summary["seed_aggregate"]["n"], 3)
        self.assertAlmostEqual(summary["seed_aggregate"]["mean"], 2.0)

    def test_legacy_raw_is_compatible_only_with_historical_defaults(self):
        payload = {
            "protocol_id": PROTOCOL_ID,
            "seed": 40,
            "checkpoint": "/tmp/checkpoint.safetensors",
        }
        assert_compatible_existing_audit(
            payload,
            seed=40,
            checkpoint="/tmp/checkpoint.safetensors",
            project="RTDETR_FAM_Protocol",
            fam_variant="current_dcnv2",
        )
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            assert_compatible_existing_audit(
                payload,
                seed=40,
                checkpoint="/tmp/checkpoint.safetensors",
                project="RTDETR_FAM_Bounded4_Protocol",
                fam_variant="bounded_dcnv2_4",
            )


if __name__ == "__main__":
    unittest.main()
