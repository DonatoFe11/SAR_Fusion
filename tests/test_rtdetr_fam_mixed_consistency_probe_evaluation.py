import copy
from pathlib import Path
from unittest import TestCase

from scripts.run_rtdetr_fam_mixed_consistency_probe_evaluation import (
    NATIVE_CONDITION,
    load_protocol,
    promotion_decision,
    validate_training_configs,
)
from scripts.run_rtdetr_fam_paired_vis_modal_dropout_probe_evaluation import (
    build_inventory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_mixed_consistency_probe_evaluation.yaml"
)


def decision_rows(*, fusion_delta, paired_ir_delta, native_ir_delta):
    baseline = {
        "paired_vis_ir": 0.15,
        "paired_vis": 0.14,
        "paired_ir_vis_gt": 0.02,
        NATIVE_CONDITION: 0.50,
    }
    deltas = {
        "paired_vis_ir": fusion_delta,
        "paired_vis": 0.0,
        "paired_ir_vis_gt": paired_ir_delta,
        NATIVE_CONDITION: native_ir_delta,
    }
    rows = []
    for condition, value in baseline.items():
        for configuration, delta in (
            ("baseline", 0.0),
            ("mixed_consistency", deltas[condition]),
        ):
            rows.append(
                {
                    "configuration": configuration,
                    "condition": condition,
                    "metrics": {"map_50": value + delta},
                }
            )
    return rows


class TestMixedConsistencyProbeEvaluation(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(PROTOCOL_PATH)

    def test_protocol_is_frozen_before_training_and_excludes_other_selectors(self):
        self.assertEqual(
            self.protocol["status"], "frozen_before_scientific_training"
        )
        constraints = self.protocol["interpretation"]
        self.assertFalse(constraints["mt_erie_allowed"])
        self.assertFalse(constraints["confirmation_acquisitions_allowed"])
        self.assertFalse(constraints["synthetic_stress_allowed"])

    def test_training_configs_match_except_for_the_declared_intervention(self):
        runs = validate_training_configs(self.protocol)
        self.assertEqual(runs["baseline"]["seed"], 40)
        candidate = runs["mixed_consistency"]
        self.assertEqual(candidate["seed"], 40)
        self.assertEqual(
            candidate["dataset"]["modal_dropout_coordinate_contract"], "native"
        )
        self.assertTrue(candidate["dataset"]["paired_consistency"])
        self.assertTrue(candidate["train"]["modality_consistency"]["enabled"])

    def test_fhl_inventory_remains_the_stage_a_validation_inventory(self):
        inventory = build_inventory(self.protocol)
        self.assertEqual(inventory["paired_frames"], 896)
        self.assertEqual(
            inventory["inventory_sha256"],
            "47e2f348ebdc202cb749b1bbf2741fc868d8681eb24ffa4356cf4f584dfa4ec4",
        )

    def test_promotion_requires_all_three_frozen_thresholds(self):
        rule = self.protocol["promotion_rule"]
        passed = promotion_decision(
            decision_rows(
                fusion_delta=-0.01,
                paired_ir_delta=0.03,
                native_ir_delta=-0.03,
            ),
            rule,
        )
        self.assertTrue(passed["passed"])
        for failed_delta in (
            {"fusion_delta": -0.0101, "paired_ir_delta": 0.03, "native_ir_delta": -0.03},
            {"fusion_delta": -0.01, "paired_ir_delta": 0.0299, "native_ir_delta": -0.03},
            {"fusion_delta": -0.01, "paired_ir_delta": 0.03, "native_ir_delta": -0.0301},
        ):
            with self.subTest(**failed_delta):
                failed = promotion_decision(decision_rows(**failed_delta), rule)
                self.assertFalse(failed["passed"])

    def test_training_yaml_hashes_are_enforced(self):
        changed = copy.deepcopy(self.protocol)
        changed["configurations"]["mixed_consistency"][
            "training_config_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash differs"):
            validate_training_configs(changed)


if __name__ == "__main__":
    import unittest

    unittest.main()
