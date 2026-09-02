import copy
import csv
import tempfile
import unittest
from pathlib import Path

from sarfusion.data.utils import load_annotations
from scripts.run_rtdetr_carnation_stress_test import SCALAR_METRICS, stable_json_hash
from scripts.run_rtdetr_unused_acquisition_confirmation import (
    EXPECTED_ACQUISITIONS,
    EXPECTED_CONFIGURATIONS,
    EXPECTED_SEEDS,
    build_aggregate,
    ensure_attestation_ready,
    load_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_unused_acquisition_confirmation.yaml"
)


class TestRTDETRUnusedAcquisitionConfirmation(unittest.TestCase):
    def test_empty_annotation_sentinel_means_unlabelled_stream(self):
        self.assertEqual(load_annotations(""), [])

    def test_protocol_freezes_sources_jobs_and_interpretation(self):
        payload = load_payload(PROTOCOL_PATH)
        protocol = payload["protocol"]
        self.assertEqual(tuple(protocol["acquisitions"]), EXPECTED_ACQUISITIONS)
        self.assertEqual(tuple(protocol["configurations"]), EXPECTED_CONFIGURATIONS)
        self.assertEqual(protocol["seeds"], EXPECTED_SEEDS)
        self.assertEqual(protocol["checkpoint"], "latest")
        self.assertEqual(protocol["ground_truth"], "vis")
        self.assertEqual(protocol["confidence_threshold"], 0.01)
        self.assertFalse(protocol["interpretation"]["model_selection_allowed"])
        self.assertTrue(
            protocol["interpretation"]["negative_results_must_be_reported"]
        )

    def test_pending_attestation_blocks_only_scientific_inference(self):
        pending = {
            "status": "pending",
            "statement": None,
            "recorded_on": None,
        }
        with self.assertRaisesRegex(RuntimeError, "attestation is pending"):
            ensure_attestation_ready(pending)

        unused = {
            "status": "no_prior_model_or_manual_experimental_use",
            "statement": "No prior use.",
            "recorded_on": "2026-08-30",
        }
        self.assertEqual(
            ensure_attestation_ready(unused),
            "previously_unused_internal_acquisition_confirmation",
        )
        additional = copy.deepcopy(unused)
        additional["status"] = "prior_or_uncertain_use"
        self.assertEqual(
            ensure_attestation_ready(additional), "additional_internal_acquisitions"
        )

    def test_complete_aggregate_requires_all_fifty_jobs(self):
        payload = load_payload(PROTOCOL_PATH)
        protocol = payload["protocol"]
        attestation = {
            "status": "no_prior_model_or_manual_experimental_use",
            "statement": "No prior use.",
            "recorded_on": "2026-08-30",
        }
        rows = []
        for acquisition_index, acquisition in enumerate(EXPECTED_ACQUISITIONS):
            for configuration, settings in protocol["configurations"].items():
                for seed in EXPECTED_SEEDS:
                    for condition in settings["conditions"]:
                        value = 0.20 + 0.01 * acquisition_index + 0.001 * (seed - 40)
                        if configuration == "historical_fam":
                            value += 0.02
                        elif configuration == "stage_b_fam":
                            value += 0.10 if condition == "vis_ir" else 0.05
                        elif configuration == "stage_b_rcra":
                            value += 0.11
                        rows.append(
                            {
                                "acquisition": acquisition,
                                "configuration": configuration,
                                "seed": seed,
                                "condition": condition,
                                "checkpoint_sha256": f"{configuration}-{seed}",
                                "n_samples": 1315 if acquisition_index == 0 else 1035,
                                "metrics": {
                                    metric: value for metric in SCALAR_METRICS
                                },
                            }
                        )

        manifests = {
            acquisition: {
                "acquisition": acquisition,
                "inventory_sha256": stable_json_hash(acquisition),
                "rows": [],
            }
            for acquisition in EXPECTED_ACQUISITIONS
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            aggregate = build_aggregate(
                rows,
                protocol,
                attestation,
                stable_json_hash(protocol),
                manifests,
                {"status": "fail_retain_fam", "selected_architecture": "fam"},
                output_dir,
            )
            with (
                output_dir / "rtdetr_unused_acquisition_confirmation.csv"
            ).open(newline="", encoding="utf-8") as input_file:
                csv_rows = list(csv.DictReader(input_file))

        self.assertTrue(aggregate["protocol_complete"])
        self.assertEqual(len(csv_rows), 50)
        historical = aggregate["paired_map50_comparisons"][
            "historical_fam_minus_additive"
        ]
        for acquisition in EXPECTED_ACQUISITIONS:
            self.assertEqual(historical[acquisition]["positive_seed_count"], 5)
            self.assertAlmostEqual(
                historical[acquisition]["summary"]["mean"], 0.02
            )
        rcra_macro = aggregate["equal_weight_acquisition_macro"][
            "stage_b_rcra_minus_fam"
        ]
        self.assertEqual(rcra_macro["positive_seed_count"], 5)
        self.assertAlmostEqual(rcra_macro["summary"]["mean"], 0.01)


if __name__ == "__main__":
    unittest.main()
