import unittest
from pathlib import Path

import torch

from scripts.run_rtdetr_fam_residual_alignment_alpha_audit import (
    AlphaCapture,
    find_alignment_gates,
    validate_protocol,
)
from sarfusion.models.rtdetr_fusion import (
    ReliabilityConditionedResidualAlignment,
)
from sarfusion.utils.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_residual_alignment_alpha_audit.yaml"
)


class TestResidualAlignmentAlphaAudit(unittest.TestCase):
    def test_protocol_is_validation_only_and_five_seed(self):
        protocol = load_yaml(PROTOCOL_PATH)
        validate_protocol(protocol)
        self.assertEqual(protocol["seeds"], [40, 41, 42, 43, 44])
        self.assertEqual(protocol["checkpoint"], "best")
        self.assertEqual(protocol["expected_validation_frames"], 896)
        self.assertEqual(protocol["expected_validation_batches"], 75)

    def test_capture_records_neutral_alpha_once_per_level(self):
        model = torch.nn.Sequential(
            ReliabilityConditionedResidualAlignment(hidden_channels=2),
            ReliabilityConditionedResidualAlignment(hidden_channels=2),
            ReliabilityConditionedResidualAlignment(hidden_channels=2),
        )
        gates = find_alignment_gates(model)
        capture = AlphaCapture(gates, bins=400, low_threshold=0.9, high_threshold=1.1)
        feature = torch.randn(2, 4, 5, 6)
        try:
            for gate in gates:
                gate(feature, feature, feature)
        finally:
            capture.close()

        self.assertEqual(len(capture.accumulators), 3)
        for accumulator in capture.accumulators.values():
            summary = accumulator.summary()
            self.assertEqual(summary["numel"], 2 * 5 * 6)
            self.assertEqual(summary["mean"], 1.0)
            self.assertEqual(summary["mean_abs_delta_one"], 0.0)

    def test_wrong_gate_count_is_rejected(self):
        model = torch.nn.Sequential(
            ReliabilityConditionedResidualAlignment(hidden_channels=2)
        )
        with self.assertRaisesRegex(RuntimeError, "Expected three"):
            find_alignment_gates(model)


if __name__ == "__main__":
    unittest.main()
