import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_rtdetr_carnation_stress_test import SCALAR_METRICS, stable_json_hash
from scripts.run_rtdetr_fam_full_data_native_ir_coordinate_diagnostic import (
    EXPECTED_SEEDS,
    build_aggregate,
    load_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_full_data_native_ir_coordinate_diagnostic.yaml"
)
RESULTS_PATH = (
    REPO_ROOT
    / "notes"
    / "Search_and_Rescue"
    / "results"
    / "rtdetr_fam_full_data_native_ir_coordinate_diagnostic.csv"
)


class TestRTDETRFAMFullDataNativeIRCoordinateDiagnostic(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(PROTOCOL_PATH)

    def test_protocol_is_post_hoc_and_keeps_native_ir_contract(self):
        self.assertEqual(self.protocol["status"], "post_hoc_diagnostic")
        self.assertEqual(self.protocol["seeds"], EXPECTED_SEEDS)
        source = self.protocol["source"]
        self.assertEqual(source["expected_frames"], 708)
        self.assertEqual(source["expected_ir_boxes"], 1824)
        self.assertEqual(source["expected_ir_empty_frames"], 3)
        self.assertEqual(source["ground_truth"], "ir")
        self.assertEqual(source["preprocessing"], "native_ir_without_adapt_ir2rgb")
        self.assertTrue(
            all(value is False for value in self.protocol["interpretation"].values())
        )

    def test_complete_aggregate_requires_all_five_seeds(self):
        payloads = []
        for seed in EXPECTED_SEEDS:
            payloads.append(
                {
                    "seed": seed,
                    "n_samples": 708,
                    "training_summary": {"run_id": f"fam{seed}"},
                    "metrics": {metric: 0.5 for metric in SCALAR_METRICS},
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            aggregate = build_aggregate(
                payloads,
                self.protocol,
                stable_json_hash(self.protocol),
                {"rows": [], "inventory_sha256": "paired"},
                {"n_frames": 708},
                Path(directory),
            )
        self.assertTrue(aggregate["protocol_complete"])
        self.assertEqual(aggregate["protocol_status"], "post_hoc_diagnostic")
        self.assertAlmostEqual(aggregate["across_seed_summary"]["map_50"]["mean"], 0.5)

    def test_versioned_native_ir_result_does_not_show_branch_collapse(self):
        with RESULTS_PATH.open(newline="", encoding="utf-8") as input_file:
            rows = list(csv.DictReader(input_file))
        self.assertEqual(len(rows), 5)
        self.assertEqual([int(row["seed"]) for row in rows], EXPECTED_SEEDS)
        values = [float(row["map50"]) for row in rows]
        self.assertTrue(all(value > 0.5 for value in values))
        self.assertAlmostEqual(sum(values) / len(values), 0.5618445158)


if __name__ == "__main__":
    unittest.main()
