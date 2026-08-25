from pathlib import Path
from unittest import TestCase

from scripts.run_rtdetr_fam_paired_vis_modal_dropout_probe_evaluation import (
    NATIVE_CONDITION,
    build_inventory,
    load_protocol,
    promotion_decision,
    validate_training_configs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_paired_vis_modal_dropout_probe_evaluation.yaml"
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
        rows.append(
            {
                "configuration": "baseline",
                "condition": condition,
                "metrics": {"map_50": value},
            }
        )
        rows.append(
            {
                "configuration": "paired_vis_dropout",
                "condition": condition,
                "metrics": {"map_50": value + deltas[condition]},
            }
        )
    return rows


class TestPairedVisDropoutProbeEvaluation(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(PROTOCOL_PATH)

    def test_protocol_restates_the_pretraining_rule_and_excludes_mt_erie(self):
        self.assertEqual(
            self.protocol["frozen_rule_commit"],
            "e7ccd39bc9b38507747089306ec43164b13c7e0c",
        )
        self.assertFalse(self.protocol["interpretation"]["mt_erie_allowed"])
        folders = self.protocol["source"]["paired_folders"]
        self.assertTrue(all("MtErie" not in folder for pair in folders for folder in pair))

    def test_training_configs_are_matched_except_for_the_coordinate_contract(self):
        runs = validate_training_configs(self.protocol)
        self.assertEqual(runs["baseline"]["seed"], 40)
        self.assertEqual(runs["paired_vis_dropout"]["seed"], 40)
        self.assertEqual(
            runs["paired_vis_dropout"]["dataset"][
                "modal_dropout_coordinate_contract"
            ],
            "paired_vis",
        )

    def test_fhl_inventory_is_exact_and_has_no_temporal_shift(self):
        inventory = build_inventory(self.protocol)
        self.assertEqual(inventory["paired_frames"], 896)
        self.assertEqual(inventory["unpaired_vis"], [
            "210924_FHL_Enterprise_VIS_0401.mp4_00896.jpg"
        ])
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
        self.assertEqual(passed["status"], rule["if_pass"])

        for failed_delta in (
            {"fusion_delta": -0.0101, "paired_ir_delta": 0.03, "native_ir_delta": -0.03},
            {"fusion_delta": -0.01, "paired_ir_delta": 0.0299, "native_ir_delta": -0.03},
            {"fusion_delta": -0.01, "paired_ir_delta": 0.03, "native_ir_delta": -0.0301},
        ):
            with self.subTest(**failed_delta):
                failed = promotion_decision(decision_rows(**failed_delta), rule)
                self.assertFalse(failed["passed"])
                self.assertEqual(failed["status"], rule["if_fail"])


if __name__ == "__main__":
    import unittest

    unittest.main()
