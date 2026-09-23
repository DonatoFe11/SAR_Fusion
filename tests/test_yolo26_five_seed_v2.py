"""Run with sarfusion-yolo26. No GPU training is performed by these tests."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import hashlib
import json
import tempfile
import unittest

import yaml
from scripts import run_yolo26_stage_a_five_seed_v2 as runner

ROOT = Path(__file__).resolve().parents[1]


def load(arm):
    return yaml.safe_load((ROOT / f"parameters/YOLO26/yolo26s_{arm}_stage_a_five_seed_v2.yaml").read_text())


class YOLO26FiveSeedV2Tests(unittest.TestCase):
    def test_matched_five_seed_configs_and_no_screen(self):
        additive, fam = load("additive"), load("fam")
        for config in (additive, fam):
            runner.validate_campaign_config(config)
            self.assertEqual(config["study"]["seeds"], [40, 41, 42, 43, 44])
        fam["study"]["arm"] = "additive"
        fam["model"]["use_fam"] = False
        self.assertEqual(additive, fam)

    def test_repair_recipe_preserved_without_seed_override(self):
        old = yaml.safe_load((ROOT / "parameters/YOLO26/yolo26s_additive_seed40_stage_a_repair_v1.yaml").read_text())
        new = load("additive")
        for key in ("model", "dataset", "selection", "preflight"):
            self.assertEqual(old[key], new[key])
        for key in ("seed", "name", "project"):
            old["training"].pop(key)
            new["training"].pop(key, None)
        self.assertEqual(old["training"], new["training"])

    def test_screen_fields_and_short_budget_rejected(self):
        for field in ("requires_control_audit", "control_vitality_min_map50", "future_seeds_on_pass"):
            config = load("fam")
            config["study"][field] = 1
            with self.assertRaises(ValueError):
                runner.validate_campaign_config(config)
        config = load("fam")
        config["training"]["epochs"] = 1
        with self.assertRaises(ValueError):
            runner.validate_campaign_config(config)

    def test_source_manifest_matches_all_files(self):
        manifest = json.loads((ROOT / load("fam")["study"]["source_manifest"]).read_text())
        self.assertIn("scripts/run_yolo26_stage_a_five_seed_v2.py", [x["path"] for x in manifest["files"]])
        for item in manifest["files"]:
            self.assertEqual(hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest(), item["sha256"], item["path"])

    def test_partial_training_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            csv = folder / "results.csv"
            selection = folder / "selection.jsonl"
            csv.write_text("epoch\n1\n")
            selection.write_text(json.dumps({"epoch": 1, "raw_mAP50": 0.001}) + "\n")
            with self.assertRaises(RuntimeError):
                runner.validate_completed_epochs(csv, selection, 50)

    def test_fam_seed41_runs_without_a_control_audit_even_with_low_map(self):
        config_path = ROOT / "parameters/YOLO26/yolo26s_fam_stage_a_five_seed_v2.yaml"
        config = load("fam")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            split = folder / config["study"]["split_config"]
            split.parent.mkdir(parents=True)
            split.write_text((ROOT / config["study"]["split_config"]).read_text())
            best, last = folder / "best.pt", folder / "last.pt"
            best.write_bytes(b"test checkpoint")
            last.write_bytes(b"test checkpoint")
            csv = folder / "results.csv"
            csv.write_text("epoch\n" + "".join(f"{epoch}\n" for epoch in range(1, 51)))
            selection = folder / "checkpoint_selection.jsonl"
            selection.write_text("".join(json.dumps({"epoch": epoch, "raw_mAP50": 0.001}) + "\n" for epoch in range(1, 51)))
            fake_trainer = SimpleNamespace(
                save_dir=folder, best=best, last=last, csv=csv, selection_path=selection,
                metrics={runner.YOLO26FusionTrainer.MAP50_KEY: 0.001},
                best_fitness=0.001, selection_best_epoch=1, train=Mock(),
            )
            model = Mock()
            model.initialization_report.return_value = {"shared_sha256": "shared", "ir_sha256": "ir", "fam_sha256": "fam"}
            args = SimpleNamespace(config=config_path, seed=41, skip_gpu_preflight=False)
            import ultralytics.engine.trainer as trainer_module
            with (
                patch.object(runner, "REPOSITORY", folder),
                patch.object(runner.os, "chdir"),
                patch.object(runner, "parse_args", return_value=args),
                patch.object(runner, "assert_environment", return_value={}),
                patch.object(runner, "verify_source_manifest", return_value={}),
                patch.object(runner, "sha256_file", return_value=config["model"]["weights_sha256"]),
                patch.object(runner, "build_stage_a_dataset", return_value={"data_yaml": "unused.yaml", "counts": {"val": 896}}),
                patch.object(runner, "load_pretrained_model", return_value=(model, {})),
                patch.object(runner, "build_fusion_model", return_value=model) as build,
                patch.object(runner, "candidate_safe_gpu_preflight", return_value={"status": "passed"}),
                patch.object(runner, "YOLO26FusionTrainer", return_value=fake_trainer) as trainer,
                patch.object(runner, "SETTINGS"),
                patch.object(trainer_module, "check_amp"),
            ):
                runner.YOLO26FusionTrainer.MAP50_KEY = "metrics/mAP50(B)"
                self.assertEqual(runner.main(), 0)
            fake_trainer.train.assert_called_once()
            self.assertEqual(trainer.call_args.kwargs["overrides"]["seed"], 41)
            self.assertTrue(all(call.kwargs["seed"] == 41 for call in build.call_args_list))
            completion = json.loads((folder / "completion.json").read_text())
            self.assertEqual(completion["status"], "completed")
            self.assertEqual(completion["seed"], 41)
            self.assertEqual(completion["best_mAP50"], 0.001)
            self.assertFalse(completion["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
