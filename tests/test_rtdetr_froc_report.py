import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import rtdetr_froc_report as report_module


def protocol():
    return {
        "id": "rtdetr_recall_fppi_test_v1",
        "seeds": [40, 41, 42, 43, 44],
        "acquisitions": list(report_module.EXPECTED_COUNTS),
        "configurations": list(report_module.CONFIGURATIONS),
        "fppi_budgets": [0.1, 0.5, 1.0],
        "fppi_grid": {"min": 0.01, "max": 10.0, "points": 301},
        "iou_threshold": 0.5,
    }


def records():
    result = []
    for acquisition, counts in report_module.EXPECTED_COUNTS.items():
        for configuration in report_module.CONFIGURATIONS:
            for seed in report_module.EXPECTED_SEEDS:
                delta = (seed - 40) * 20 + (100 if configuration == "historical_fam" else 0)
                tp = np.asarray([0, 100, 200, 300, 400, 500, 700], dtype=np.int64)
                tp[1:] += delta
                fp = np.floor(np.asarray([0, 0.05, 0.1, 0.3, 0.5, 1.0, 20.0]) * counts[0]).astype(np.int64)
                result.append({
                    "acquisition": acquisition, "configuration": configuration, "seed": seed,
                    "n_images": counts[0], "n_gt": counts[1], "n_empty": counts[2],
                    "checkpoint_sha256": f"{configuration}-{seed}",
                    "cache_path": f"cache/{acquisition}/{configuration}/{seed}.npz",
                    "source_sha256": "test-source-hash", "implementation_sha256": "test-implementation-hash",
                    "inventory_sha256": f"{acquisition}-inventory", "content_sha256": f"{acquisition}-content",
                    "torch_version": "test-torch", "transformers_version": "test-transformers",
                    "batch_size": 4, "collection_threshold": 0.0, "native_queries": 300,
                    "max_batches": None,
                    "curve": {
                        "threshold": np.asarray([np.inf, .9, .8, .6, .4, .2, 0]),
                        "recall": tp / counts[1], "fppi": fp / counts[0],
                        "tp": tp, "fp": fp,
                    },
                })
    return result


def fake_plot(grid_rows, output_dir, complete):
    paths = []
    for extension in ("pdf", "png"):
        path = output_dir / f"{report_module.STEM}.{extension}"
        path.write_bytes(b"test figure")
        paths.append(path)
    return paths


class TestRTDETRFROCReport(unittest.TestCase):
    def test_complete_report_has_per_seed_budgets_and_paired_statistics(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(report_module, "_plot", fake_plot):
            output_dir = Path(directory)
            result = report_module.write_report(records(), protocol(), output_dir)
            self.assertTrue(result["complete"])
            self.assertEqual(result["n_jobs"], 30)
            self.assertEqual(len(result["summary"]), 9)
            first = result["summary"][0]
            self.assertAlmostEqual(first["additive_mean"], 240 / 1770)
            self.assertAlmostEqual(first["fam_mean"], 340 / 1770)
            self.assertAlmostEqual(first["paired_delta_mean"], 100 / 1770)
            self.assertAlmostEqual(first["paired_delta_sd"], 0)
            self.assertAlmostEqual(first["additive_sd"], np.std(np.asarray([200, 220, 240, 260, 280]) / 1770, ddof=1))
            self.assertEqual(first["positive_seed_count"], 5)
            self.assertEqual(first["paired_seed_count"], 5)
            for suffix, expected in (("budgets", 90), ("paired", 45), ("summary", 9), ("grid", 1806)):
                with (output_dir / f"{report_module.STEM}_{suffix}.csv").open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), expected)
            saved = json.loads((output_dir / f"{report_module.STEM}.json").read_text())
            self.assertFalse(saved["interpretation"]["model_selection_allowed"])
            self.assertFalse(saved["interpretation"]["deployment_threshold_selection_allowed"])
            self.assertEqual(saved["jobs"][0]["batch_size"], 4)
            self.assertEqual(saved["jobs"][0]["source_sha256"], "test-source-hash")
            self.assertIsNone(saved["jobs"][0]["max_batches"])
            with (output_dir / f"{report_module.STEM}_budgets.csv").open() as handle:
                budget_rows = list(csv.DictReader(handle))
            row = next(row for row in budget_rows if row["acquisition"] == "mterie" and
                       row["configuration"] == "historical_additive" and row["seed"] == "40" and row["fppi_budget"] == "0.1")
            self.assertEqual(int(row["tp"]), 200)
            self.assertEqual(int(row["fp"]), 70)
            self.assertAlmostEqual(float(row["attained_fppi"]), 70 / 708)
            self.assertAlmostEqual(float(row["threshold"]), 0.8)
            self.assertEqual(row["reject_all"], "False")

    def test_publication_requires_exact_thirty_jobs_and_inventory_counts(self):
        full = records()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "30 unique jobs"):
                report_module.write_report(full[:-1], protocol(), directory, publish=True)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                report_module.write_report(full + [full[0]], protocol(), directory, publish=True)
            full[0]["n_images"] += 1
            with self.assertRaisesRegex(ValueError, "sample counts"):
                report_module.write_report(full, protocol(), directory, publish=True)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_publication_rejects_changed_scope_or_budgets_before_writes(self):
        for key, value in (("seeds", [40]), ("iou_threshold", 0.75), ("fppi_budgets", [.2, .5, 1])):
            changed = protocol()
            changed[key] = value
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    report_module.write_report(records(), changed, directory, publish=True)
                self.assertFalse(list(Path(directory).iterdir()))

    def test_publication_copies_assets_but_does_not_change_thesis_tex(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(report_module, "_plot", fake_plot):
            root = Path(directory)
            thesis_dir = root / "notes" / "Search_and_Rescue"
            thesis_dir.mkdir(parents=True)
            thesis = thesis_dir / "main.tex"
            thesis.write_text("unchanged thesis")
            with patch.object(report_module, "REPO_ROOT", root):
                report_module.write_report(records(), protocol(), root / "out", publish=True)
            self.assertEqual(thesis.read_text(), "unchanged thesis")
            self.assertEqual(len(list((thesis_dir / "results").glob("*"))), 5)
            self.assertEqual(len(list((thesis_dir / "images").glob("*"))), 2)
            markdown = (root / "notes" / "rtdetr_recall_fppi.md").read_text()
            self.assertIn("30/30", markdown)
            self.assertIn("post-hoc", markdown)
            self.assertIn("5/5", markdown)
            self.assertIn("non sono", markdown)
            self.assertIn("## Integrazione nella tesi", markdown)
            self.assertIn("`sec:recall-fppi`", markdown)
            self.assertNotIn("I file `.tex` non sono stati modificati", markdown)

    def test_partial_report_does_not_invent_paired_observations(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(report_module, "_plot", fake_plot):
            result = report_module.write_report(records()[:1], protocol(), directory)
            self.assertFalse(result["complete"])
            first = result["summary"][0]
            self.assertEqual(first["additive_n"], 1)
            self.assertIsNone(first["additive_sd"])
            self.assertEqual(first["paired_seed_count"], 0)
            self.assertIsNone(first["paired_delta_mean"])

    def test_no_grid_or_budget_extrapolation_past_collected_endpoint(self):
        full = records()
        # One checkpoint has no native predictions beyond FPPI 0.3.
        for key in full[0]["curve"]:
            full[0]["curve"][key] = full[0]["curve"][key][:4]
        with tempfile.TemporaryDirectory() as directory, patch.object(report_module, "_plot", fake_plot):
            result = report_module.write_report(full, protocol(), directory)
            self.assertAlmostEqual(result["common_endpoint_fppi"]["mterie"], full[0]["curve"]["fppi"][-1])
            with (Path(directory) / f"{report_module.STEM}_grid.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(float(row["fppi_budget"]) <= 0.3 for row in rows if row["acquisition"] == "mterie"))
            at_half = next(row for row in result["summary"] if row["acquisition"] == "mterie" and row["fppi_budget"] == .5)
            self.assertEqual(at_half["additive_n"], 4)
            self.assertEqual(at_half["paired_seed_count"], 4)

    def test_nonmonotonic_or_nonfinite_curves_rejected(self):
        for modified in ([0, .1, .05, .3, .4, .5, .7], [0, .1, .2, .3, .4, .5, np.nan]):
            full = records()
            full[0]["curve"]["recall"] = np.asarray(modified)
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    report_module.write_report(full, protocol(), directory)

    def test_reject_all_operating_point_serializes_without_infinite_threshold(self):
        full = records()
        curve = full[0]["curve"]
        curve["fppi"][1:] = np.asarray([2, 3, 4, 5, 6, 20])
        curve["fp"][1:] = (curve["fppi"][1:] * full[0]["n_images"]).astype(np.int64)
        with tempfile.TemporaryDirectory() as directory, patch.object(report_module, "_plot", fake_plot):
            report_module.write_report(full, protocol(), directory)
            with (Path(directory) / f"{report_module.STEM}_budgets.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            row = next(row for row in rows if row["acquisition"] == full[0]["acquisition"] and
                       row["configuration"] == full[0]["configuration"] and row["seed"] == "40")
            self.assertEqual(row["reject_all"], "True")
            self.assertEqual(row["threshold"], "")
            self.assertEqual(row["tp"], "0")
            self.assertEqual(row["fp"], "0")

    def test_actual_plot_can_render_headlessly(self):
        rows = [
            {"acquisition": acquisition, "configuration": configuration,
             "fppi_budget": budget, "recall_mean": recall, "recall_sd": .05}
            for acquisition in report_module.EXPECTED_COUNTS
            for configuration in report_module.CONFIGURATIONS
            for budget, recall in ((.01, .1), (.1, .3), (1, .6), (10, .8))
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = report_module._plot(rows, Path(directory), True)
            self.assertEqual(paths[0].read_bytes()[:4], b"%PDF")
            self.assertIn(b"/FontFile2", paths[0].read_bytes())
            self.assertNotIn(b"/Subtype /Type3", paths[0].read_bytes())
            self.assertEqual(paths[1].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
