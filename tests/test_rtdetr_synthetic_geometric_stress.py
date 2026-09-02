import tempfile
import unittest
from pathlib import Path

import torch

from scripts.run_rtdetr_carnation_stress_test import SCALAR_METRICS, stable_json_hash
from scripts.run_rtdetr_synthetic_geometric_stress import (
    EXPECTED_ACQUISITIONS,
    EXPECTED_CONFIGURATIONS,
    EXPECTED_SEEDS,
    apply_ir_transformation,
    build_aggregate,
    expand_transformations,
    load_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_synthetic_geometric_stress.yaml"
)


class TestRTDETRSyntheticGeometricStress(unittest.TestCase):
    def test_protocol_freezes_independent_translation_and_scale_curves(self):
        protocol = load_protocol(PROTOCOL_PATH)
        transformations = expand_transformations(protocol)
        self.assertEqual(len(transformations), 15)
        self.assertEqual(transformations[0]["id"], "identity")
        self.assertEqual(
            [row["id"] for row in transformations if row["kind"] == "scale"],
            ["scale_090", "scale_110"],
        )
        translations = [
            row for row in transformations if row["kind"] == "translation"
        ]
        self.assertEqual(len(translations), 12)
        self.assertEqual(
            {(row["dx_px"], row["dy_px"]) for row in translations},
            {
                (8, 0),
                (-8, 0),
                (0, 8),
                (0, -8),
                (16, 0),
                (-16, 0),
                (0, 16),
                (0, -16),
                (32, 0),
                (-32, 0),
                (0, 32),
                (0, -32),
            },
        )
        self.assertFalse(
            protocol["interpretation"]["transformation_selection_allowed"]
        )
        self.assertTrue(
            protocol["interpretation"]["negative_results_must_be_reported"]
        )

    def test_translation_changes_only_ir_channel(self):
        pixels = torch.zeros(1, 4, 5, 5)
        pixels[:, :3] = torch.arange(75, dtype=torch.float32).reshape(1, 3, 5, 5)
        pixels[0, 3, 2, 2] = 1.0
        original = pixels.clone()
        transformation = {
            "kind": "translation",
            "dx_px": 1,
            "dy_px": 0,
            "scale": 1.0,
        }
        shifted = apply_ir_transformation(pixels, transformation)
        self.assertTrue(torch.equal(shifted[:, :3], original[:, :3]))
        self.assertEqual(float(shifted[0, 3, 2, 3]), 1.0)
        self.assertEqual(float(shifted[0, 3].sum()), 1.0)
        self.assertTrue(torch.equal(pixels, original))

    def test_identity_is_exact_and_invalid_shape_fails(self):
        pixels = torch.randn(2, 4, 7, 9)
        identity = {"kind": "identity"}
        self.assertIs(apply_ir_transformation(pixels, identity), pixels)
        with self.assertRaisesRegex(ValueError, "Bx4xHxW"):
            apply_ir_transformation(torch.randn(2, 3, 7, 9), identity)

    def test_complete_aggregate_requires_all_six_hundred_curve_points(self):
        protocol = load_protocol(PROTOCOL_PATH)
        transformations = expand_transformations(protocol)
        factor = {
            "historical_additive": 1.0,
            "historical_fam": 0.5,
            "stage_b_fam": 0.8,
            "stage_b_rcra": 0.6,
        }
        direction_bias = {"right": 0.001, "left": -0.001, "down": 0.002, "up": -0.002}
        rows = []
        for acquisition in EXPECTED_ACQUISITIONS:
            for configuration in EXPECTED_CONFIGURATIONS:
                for seed in EXPECTED_SEEDS:
                    identity_map50 = 0.4 + 0.001 * (seed - 40)
                    for transformation in transformations:
                        if transformation["kind"] == "identity":
                            drop = 0.0
                        elif transformation["kind"] == "translation":
                            drop = factor[configuration] * (
                                transformation["magnitude_px"] / 1000.0
                                + direction_bias[transformation["direction"]]
                            )
                        else:
                            drop = factor[configuration] * abs(
                                transformation["scale"] - 1.0
                            )
                        value = identity_map50 * (1.0 - drop)
                        rows.append(
                            {
                                "acquisition": acquisition,
                                "configuration": configuration,
                                "family": protocol["configurations"][configuration][
                                    "family"
                                ],
                                "seed": seed,
                                "transformation": transformation,
                                "checkpoint_sha256": f"{configuration}-{seed}",
                                "n_samples": 10,
                                "metrics": {
                                    metric: value for metric in SCALAR_METRICS
                                },
                            }
                        )

        manifests = {
            acquisition: {
                "acquisition": acquisition,
                "inventory_sha256": stable_json_hash(acquisition),
            }
            for acquisition in EXPECTED_ACQUISITIONS
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            aggregate = build_aggregate(
                rows,
                protocol,
                stable_json_hash(protocol),
                manifests,
                "confirmation-hash",
                "identity-hash",
                output_dir,
            )
            csv_lines = (
                output_dir / "rtdetr_synthetic_geometric_stress.csv"
            ).read_text(encoding="utf-8").splitlines()

        self.assertTrue(aggregate["protocol_complete"])
        self.assertEqual(aggregate["observed_job_count_including_reused_identity"], 600)
        self.assertEqual(aggregate["new_perturbed_inference_job_count"], 560)
        self.assertEqual(len(csv_lines), 601)
        pooled = aggregate["direction_pooled_translation_relative_drop"][
            "carnation_0025_0026"
        ]["historical_additive"]["8"]
        self.assertAlmostEqual(pooled["summary"]["mean"], 0.008)
        advantage = aggregate["paired_robustness_comparisons"][
            "historical_fam_vs_additive"
        ]["carnation_0025_0026"]["translate_32_right"][
            "candidate_robustness_advantage"
        ]["summary"]["mean"]
        self.assertGreater(advantage, 0.0)


if __name__ == "__main__":
    unittest.main()
