import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_yolo_final_modality_evaluation import (
    load_compatible_raw,
    load_protocol,
    resolve_run_checkpoint,
    summarize_values,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestYOLOFinalModalityEvaluation(unittest.TestCase):
    def test_frozen_protocol_uses_paired_test_data_and_explicit_masks(self):
        protocol = load_protocol(
            REPO_ROOT
            / "parameters"
            / "YOLO"
            / "yolov10_final_modality_evaluation.yaml"
        )

        self.assertEqual(protocol["checkpoint"], "last")
        self.assertEqual(protocol["split"], "test")
        self.assertEqual(protocol["seeds"], [40, 41, 42, 43, 44])
        self.assertEqual(
            protocol["modalities"],
            {"vis_ir": "fusion", "vis": "rgb", "ir": "ir"},
        )
        self.assertEqual(
            protocol["data_yaml"],
            "parameters/YOLO_datasets/wisards_vis_ir.yaml",
        )

    def test_checkpoint_resolution_uses_recorded_seed_and_complete_training(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / "YOLO-run"
            (run / "weights").mkdir(parents=True)
            (run / "weights" / "last.pt").touch()
            (run / "args.yaml").write_text(
                "seed: 42\n"
                "test_checkpoint: last\n"
                "modal_dropout: true\n"
                "modal_dropout_strategy: feature\n"
                "modal_dropout_probs: [0.2, 0.2, 0.6]\n",
                encoding="utf-8",
            )
            with (run / "results.csv").open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["epoch"])
                writer.writeheader()
                writer.writerows({"epoch": epoch} for epoch in range(1, 201))

            checkpoint, resolved_run = resolve_run_checkpoint(root, 42)

            self.assertEqual(checkpoint, (run / "weights" / "last.pt").resolve())
            self.assertEqual(resolved_run, run.resolve())

    def test_checkpoint_resolution_rejects_incomplete_training(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / "YOLO-run"
            (run / "weights").mkdir(parents=True)
            (run / "weights" / "last.pt").touch()
            (run / "args.yaml").write_text(
                "seed: 42\n"
                "test_checkpoint: last\n"
                "modal_dropout: true\n"
                "modal_dropout_strategy: feature\n"
                "modal_dropout_probs: [0.2, 0.2, 0.6]\n",
                encoding="utf-8",
            )
            (run / "results.csv").write_text("epoch\n1\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "1 epochs, expected 200"):
                resolve_run_checkpoint(root, 42)

    def test_existing_raw_result_must_match_checkpoint_and_protocol(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "raw.json"
            payload = {
                "protocol_id": "protocol-v1",
                "checkpoint_sha256": "abc",
                "metrics": {"map_50": 0.5},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(
                load_compatible_raw(
                    path,
                    {"protocol_id": "protocol-v1", "checkpoint_sha256": "abc"},
                ),
                payload,
            )
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                load_compatible_raw(
                    path,
                    {"protocol_id": "protocol-v1", "checkpoint_sha256": "changed"},
                )

    def test_summary_uses_sample_standard_deviation_across_seeds(self):
        summary = summarize_values([0.1, 0.2, 0.3, 0.4, 0.5])

        self.assertEqual(summary["n"], 5)
        self.assertAlmostEqual(summary["mean"], 0.3)
        self.assertAlmostEqual(summary["median"], 0.3)
        self.assertAlmostEqual(summary["sample_std"], 0.15811388300841897)


if __name__ == "__main__":
    unittest.main()

