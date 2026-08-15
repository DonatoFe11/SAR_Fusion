import tempfile
import unittest
from pathlib import Path

from scripts.run_rtdetr_carnation_stress_test import SCALAR_METRICS, stable_json_hash
from scripts.run_rtdetr_sequence_checkpoint_evaluation import (
    build_aggregate,
    load_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRTDETRSequenceCheckpointEvaluation(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(
            REPO_ROOT
            / "parameters"
            / "RTDETR"
            / "rtdetr_fam_sequence_validation_checkpoint_evaluation.yaml"
        )

    def test_protocol_freezes_best_as_primary_and_latest_as_diagnostic(self):
        self.assertEqual(self.protocol["seeds"], [40, 41, 42, 43, 44])
        self.assertEqual(self.protocol["checkpoints"], ["best", "latest"])
        self.assertEqual(self.protocol["primary_checkpoint"], "best")
        self.assertEqual(self.protocol["diagnostic_checkpoint"], "latest")
        self.assertEqual(self.protocol["source"]["expected_frames"], 708)
        self.assertEqual(self.protocol["source"]["ground_truth"], "vis")
        self.assertFalse(
            self.protocol["interpretation"]["model_selection_from_mterie_allowed"]
        )

    def test_complete_aggregate_is_paired_by_seed(self):
        payloads = []
        for seed in self.protocol["seeds"]:
            training_summary = {
                "run_id": f"run{seed}",
                "run_dir": f"/tmp/run{seed}",
                "best_epoch": seed - 39,
                "best_validation_map50": 0.2,
                "latest_epoch": 10,
                "latest_validation_map50": 0.1,
                "runtime_seconds": 9000.0 + seed,
            }
            for checkpoint, score in (("best", 0.4), ("latest", 0.3)):
                payloads.append(
                    {
                        "seed": seed,
                        "checkpoint_kind": checkpoint,
                        "checkpoint_sha256": f"hash-{seed}-{checkpoint}",
                        "training_summary": training_summary,
                        "metrics": {metric: score for metric in SCALAR_METRICS},
                    }
                )
        manifest = {"n_frames": 708, "ground_truth": "vis"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            aggregate = build_aggregate(
                payloads,
                self.protocol,
                stable_json_hash(self.protocol),
                manifest,
                Path(temporary_directory),
            )
        self.assertTrue(aggregate["protocol_complete"])
        self.assertEqual(len(aggregate["results"]), 10)
        comparison = aggregate["best_minus_latest_map50"]
        self.assertEqual(comparison["best_wins"], 5)
        self.assertAlmostEqual(comparison["summary"]["mean"], 0.1)


if __name__ == "__main__":
    unittest.main()
