import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_rtdetr_carnation_stress_test import SCALAR_METRICS, stable_json_hash
from scripts.run_rtdetr_fam_full_data_paired_modality_evaluation import (
    EXPECTED_SEEDS,
    build_aggregate,
    check_fusion_reference,
    load_protocol,
    verify_closed_selection,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_full_data_paired_modality_evaluation.yaml"
)
RESULTS_PATH = (
    REPO_ROOT
    / "notes"
    / "Search_and_Rescue"
    / "results"
    / "rtdetr_fam_full_data_paired_modality_evaluation.csv"
)


class TestRTDETRFAMFullDataPairedModalityEvaluation(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(PROTOCOL_PATH)

    def test_protocol_is_descriptive_and_keeps_common_population(self):
        self.assertEqual(self.protocol["seeds"], EXPECTED_SEEDS)
        self.assertEqual(self.protocol["checkpoint"], "latest")
        self.assertEqual(self.protocol["source"]["expected_frames"], 708)
        self.assertEqual(self.protocol["source"]["expected_vis_boxes"], 1770)
        self.assertEqual(self.protocol["source"]["ground_truth"], "vis")
        self.assertEqual(
            set(self.protocol["modalities"]), {"vis_ir", "vis", "ir"}
        )
        self.assertTrue(
            all(value is False for value in self.protocol["interpretation"].values())
        )

    def test_versioned_stage_b_result_still_selects_fam(self):
        audit, stage_b_protocol = verify_closed_selection(self.protocol)
        self.assertEqual(audit["decision"]["status"], "fail_retain_fam")
        self.assertEqual(audit["decision"]["selected_architecture"], "fam")
        self.assertEqual(audit["decision"]["candidate_wins"], 3)
        self.assertEqual(
            stage_b_protocol["configurations"]["fam"]["project"],
            self.protocol["project"],
        )

    def test_fusion_reference_uses_frozen_stage_b_value(self):
        expected = self.protocol["expected_fusion_map50"]["values"][40]
        result = check_fusion_reference(self.protocol, 40, expected + 0.0001)
        self.assertAlmostEqual(result["difference"], 0.0001)
        with self.assertRaisesRegex(RuntimeError, "reconstruction failed"):
            check_fusion_reference(self.protocol, 40, expected + 0.001)

    def test_complete_aggregate_requires_all_fifteen_units(self):
        payloads = []
        for seed in EXPECTED_SEEDS:
            for modality, score in (("vis_ir", 0.40), ("vis", 0.35), ("ir", 0.02)):
                payloads.append(
                    {
                        "configuration": "fam",
                        "seed": seed,
                        "modality": modality,
                        "n_samples": 708,
                        "training_summary": {"run_id": f"fam{seed}"},
                        "metrics": {metric: score for metric in SCALAR_METRICS},
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            aggregate = build_aggregate(
                payloads,
                self.protocol,
                stable_json_hash(self.protocol),
                {"n_frames": 708, "ground_truth": "vis"},
                {"decision": {"selected_architecture": "fam"}},
                Path(directory),
                complete=True,
            )
        self.assertTrue(aggregate["protocol_complete"])
        self.assertEqual(len(aggregate["results"]), 15)
        fusion_delta = aggregate["paired_map50_deltas"]["fusion_minus_best_single"]
        self.assertAlmostEqual(fusion_delta["summary"]["mean"], 0.05)
        self.assertEqual(fusion_delta["positive_seed_count"], 5)

    def test_versioned_result_has_five_paired_fusion_wins(self):
        with RESULTS_PATH.open(newline="", encoding="utf-8") as input_file:
            rows = list(csv.DictReader(input_file))
        self.assertEqual(len(rows), 5)
        self.assertEqual([int(row["seed"]) for row in rows], EXPECTED_SEEDS)

        fusion = [float(row["vis_ir_map50"]) for row in rows]
        vis = [float(row["vis_map50"]) for row in rows]
        masked_ir = [float(row["masked_ir_vis_gt_map50"]) for row in rows]
        deltas = [
            float(row["fusion_minus_best_paired_intervention"]) for row in rows
        ]
        self.assertTrue(
            all(
                fused > visible > thermal
                for fused, visible, thermal in zip(fusion, vis, masked_ir)
            )
        )
        self.assertEqual(sum(delta > 0 for delta in deltas), 5)
        self.assertAlmostEqual(sum(fusion) / len(fusion), 0.3568793118)
        self.assertAlmostEqual(sum(vis) / len(vis), 0.3196012497)
        self.assertAlmostEqual(sum(masked_ir) / len(masked_ir), 0.0214871965)
        self.assertAlmostEqual(sum(deltas) / len(deltas), 0.0372780621)


if __name__ == "__main__":
    unittest.main()
