import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_rtdetr_carnation_stress_test import (
    SCALAR_METRICS,
    build_aggregates,
    load_compatible_raw,
    load_protocol,
    numeric_frame_id,
    render_paired_map50,
    stable_json_hash,
    summarize_values,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRTDETRCarnationStressTest(unittest.TestCase):
    def test_frozen_protocol_is_one_shot_and_uses_final_checkpoints(self):
        protocol = load_protocol(
            REPO_ROOT
            / "parameters"
            / "RTDETR"
            / "rtdetr_carnation_stress_test.yaml"
        )

        self.assertEqual(protocol["checkpoint"], "latest")
        self.assertEqual(protocol["seeds"], [40, 41, 42, 43, 44])
        self.assertEqual(
            protocol["modalities"],
            {"vis_ir": "fusion", "vis": "rgb", "ir": "ir"},
        )
        self.assertEqual(protocol["source"]["expected_common_frames"], 739)
        self.assertFalse(protocol["interpretation"]["model_selection_allowed"])
        self.assertFalse(protocol["interpretation"]["threshold_tuning_allowed"])
        self.assertEqual(protocol["fam_variant"], "current_dcnv2")

    def test_numeric_frame_id(self):
        self.assertEqual(
            numeric_frame_id("Carnation_VIS_0023_00001044.jpg"), 1044
        )
        with self.assertRaisesRegex(ValueError, "numeric frame suffix"):
            numeric_frame_id("frame_without_number.jpg")

    def test_summary_uses_checkpoint_level_sample_std_and_t_interval(self):
        summary = summarize_values([0.1, 0.2, 0.3, 0.4, 0.5])

        self.assertEqual(summary["n"], 5)
        self.assertAlmostEqual(summary["mean"], 0.3)
        self.assertAlmostEqual(summary["median"], 0.3)
        self.assertAlmostEqual(summary["sample_std"], 0.15811388300841897)
        self.assertEqual(len(summary["ci95_t"]), 2)

    def test_undefined_torchmetrics_sentinel_is_not_averaged(self):
        self.assertIsNone(summarize_values([-1.0, -1.0]))
        self.assertEqual(summarize_values([-1.0, 0.25])["mean"], 0.25)

    def test_existing_raw_result_must_match_frozen_job(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "raw.json"
            payload = {
                "protocol_id": "carnation-v1",
                "checkpoint_sha256": "abc",
                "metrics": {"map_50": 0.5},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(
                load_compatible_raw(
                    path,
                    {"protocol_id": "carnation-v1", "checkpoint_sha256": "abc"},
                ),
                payload,
            )
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                load_compatible_raw(
                    path,
                    {"protocol_id": "carnation-v1", "checkpoint_sha256": "changed"},
                )

    def test_aggregate_requires_all_thirty_frozen_units_for_completion(self):
        protocol = load_protocol(
            REPO_ROOT
            / "parameters"
            / "RTDETR"
            / "rtdetr_carnation_stress_test.yaml"
        )
        payloads = []
        for modality_index, modality in enumerate(protocol["modalities"]):
            for configuration in protocol["configurations"]:
                for seed in protocol["seeds"]:
                    base = 0.1 + modality_index * 0.01 + (seed - 40) * 0.001
                    if configuration == "fam":
                        base += 0.02
                    payloads.append(
                        {
                            "configuration": configuration,
                            "seed": seed,
                            "modality": modality,
                            "n_samples": 739,
                            "metrics": {metric: base for metric in SCALAR_METRICS},
                        }
                    )

        source_manifest = {"common_frame_count": 739}
        with tempfile.TemporaryDirectory() as temporary_directory:
            combined = build_aggregates(
                payloads,
                protocol,
                stable_json_hash(protocol),
                source_manifest,
                Path(temporary_directory),
                complete=True,
            )

        self.assertTrue(combined["protocol_complete"])
        for modality in protocol["modalities"]:
            delta = combined["paired_deltas_fam_minus_additive"][modality]["map_50"]
            self.assertEqual(delta["fam_wins"], 5)
            self.assertAlmostEqual(delta["summary"]["mean"], 0.02)

    def test_paired_renderer_writes_figure(self):
        rows = []
        for modality in ("vis_ir", "vis", "ir"):
            for configuration in ("additive", "fam"):
                for seed in range(40, 45):
                    rows.append(
                        {
                            "modality": modality,
                            "configuration": configuration,
                            "seed": seed,
                            "metrics": {
                                "map_50": 0.1
                                + (seed - 40) * 0.01
                                + (0.02 if configuration == "fam" else 0.0)
                            },
                        }
                    )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            render_paired_map50({"results": rows}, root)
            self.assertTrue(
                (root / "figures" / "rtdetr_carnation_paired_map50.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
