"""CPU-only tests: no downloads, W&B runs or detector training."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers.models.rt_detr.configuration_rt_detr import RTDetrConfig
from transformers.models.rt_detr.configuration_rt_detr_resnet import RTDetrResNetConfig

from sarfusion.models import MODEL_REGISTRY, build_fusion_rt_detr
from sarfusion.models.rtdetr_fusion import RTDetrFusionForObjectDetection
from sarfusion.utils.reproducibility import model_digests
from scripts import rtdetr_fam_zero_offset as intervention
from scripts import run_rtdetr_fam_zero_offset_stage_a as runner


def tiny_model():
    backbone = RTDetrResNetConfig(num_channels=3, embedding_size=8,
                                 hidden_sizes=[16, 32, 64, 128], depths=[1, 1, 1, 1],
                                 layer_type="bottleneck", downsample_in_first_stage=False,
                                 out_indices=[2, 3, 4])
    config = RTDetrConfig(backbone_config=backbone, encoder_hidden_dim=32,
                         encoder_in_channels=[32, 64, 128], feat_strides=[8, 16, 32],
                         encoder_layers=1, encoder_ffn_dim=64, encoder_attention_heads=4,
                         encode_proj_layers=[2], d_model=32, num_queries=10,
                         decoder_in_channels=[32, 32, 32], decoder_ffn_dim=64,
                         num_feature_levels=3, decoder_n_levels=3, decoder_n_points=2,
                         decoder_layers=1, decoder_attention_heads=4, num_denoising=0,
                         id2label={0: "person"}, label2id={"person": 0}, num_channels=4)
    return RTDetrFusionForObjectDetection(config, use_fam=True)


class OffsetOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_effective_post_init_randomness_and_exact_offset_only_change(self):
        torch.manual_seed(40)
        model = tiny_model()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        for _, fam in intervention.fam_modules(model):
            self.assertGreater(torch.count_nonzero(fam.offset_conv.weight[:18]).item(), 0)
            self.assertGreater(torch.count_nonzero(fam.offset_conv.weight[18:]).item(), 0)
        report = intervention.zero_offsets_only(model)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertNotEqual(report["reference_initialization"], report["candidate_initialization"])
        for name, tensor in model.state_dict().items():
            expected = before[name]
            if "fam_modules" in name and ".offset_conv." in name:
                expected[:18].zero_()
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

    def test_zero_at_first_forward_mask_unchanged_and_offsets_can_learn(self):
        torch.manual_seed(41)
        model = tiny_model()
        module = intervention.fam_modules(model)[0][1]
        channels = module.deform_conv.in_channels
        rgb = torch.randn(2, channels, 8, 7)
        ir = torch.randn_like(rgb, requires_grad=True)
        features = torch.cat((rgb, ir), dim=1)
        mask_before = module.offset_conv(features)[:, 18:].detach().clone()
        intervention.zero_offsets_only(model)
        prediction = module.offset_conv(features)
        self.assertEqual(torch.count_nonzero(prediction[:, :18]).item(), 0)
        torch.testing.assert_close(prediction[:, 18:], mask_before, rtol=0, atol=0)
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.001)
        output = module(rgb, ir)
        self.assertFalse(torch.allclose(output, ir))  # Zero offsets leave the learned DCN filter and masks active.
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(module.offset_conv.weight.grad).all())
        self.assertGreater(module.offset_conv.weight.grad[:18].abs().max().item(), 0)
        optimizer.step()
        self.assertGreater(torch.count_nonzero(module.offset_conv.weight[:18]).item(), 0)

    def test_loading_trained_state_does_not_zero_offsets_again(self):
        torch.manual_seed(42)
        trained = tiny_model()
        with torch.no_grad():
            for _, fam in intervention.fam_modules(trained):
                fam.offset_conv.weight[:18].fill_(0.0123)
        loaded = tiny_model()
        intervention.zero_offsets_only(loaded)
        torch.nn.Module.load_state_dict(loaded, trained.state_dict(), strict=True)
        self.assertEqual(model_digests(loaded), model_digests(trained))

    def test_factory_applies_reset_after_complete_pretrained_build(self):
        params = runner.expand_config(runner.load_yaml(runner.CONFIG))[0]["model"]["params"]
        model = tiny_model()
        before = model_digests(model)
        with patch.object(intervention, "build_fusion_rt_detr", return_value=model) as factory:
            result = intervention.build_zero_offset_model(**params)
        factory.assert_called_once_with(**params)
        self.assertIs(result, model)
        self.assertEqual(result.zero_offset_initialization_report["reference_initialization"], before)

    def test_registration_does_not_replace_the_historical_factory(self):
        intervention.register_experiment()
        intervention.register_experiment()
        self.assertIs(MODEL_REGISTRY["fusion_rtdetr"], build_fusion_rt_detr)
        self.assertIs(MODEL_REGISTRY[intervention.MODEL_NAME], intervention.build_zero_offset_model)

    def test_incompatible_interventions_rejected(self):
        params = runner.expand_config(runner.load_yaml(runner.CONFIG))[0]["model"]["params"]
        for key, value in (("use_fam", False), ("freeze_fam", True),
                           ("fam_variant", "identity_dcnv2"), ("spatial_jitter_std", 0.5),
                           ("ir_dropout_rate", 0.4), ("use_p2", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                intervention.build_zero_offset_model(**{**params, key: value})


class ProtocolTests(unittest.TestCase):
    def test_five_seed_config_and_full_fixed_budget(self):
        runs = runner.validate_config(runner.load_yaml(runner.CONFIG))
        self.assertEqual([r["seed"] for r in runs], runner.SEEDS)
        for run in runs:
            self.assertEqual(run["train"]["max_epochs"], 10)
            self.assertEqual(run["test_checkpoint"], "best")
            self.assertFalse(run["run_test"])

    def test_changed_recipe_or_cartesian_seed_grid_rejected(self):
        for change in ("epochs", "offsets_and_masks", "seed_grid", "resume"):
            settings = runner.load_yaml(runner.CONFIG)
            if change == "epochs":
                settings["parameters"]["train"]["max_epochs"] = [1]
            elif change == "offsets_and_masks":
                settings["parameters"]["model"]["params"]["fam_variant"] = ["identity_dcnv2"]
            elif change == "resume":
                settings["experiment"]["resume"] = True
            else:
                settings["parameters"]["seed"] = [40, 41]
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_config(settings)

    def test_checkpoint_rule_is_not_raw_maximum_or_first_epoch_screen(self):
        history = [{"epoch": i, "map50": 0.1} for i in range(1, 11)]
        history[1]["map50"] = 0.1005
        history[7]["map50"] = 0.13
        history[8]["map50"] = 0.1305
        self.assertEqual(runner.selection_from_history(history), {"epoch": 8, "map50": 0.13})
        for invalid in (history[:1], history[:-1], history[::-1],
                        history[:-1] + [{"epoch": 10, "map50": float("nan")} ]):
            with self.assertRaises(RuntimeError):
                runner.selection_from_history(invalid)

    def test_gate_uses_best_all_seeds_and_both_thresholds(self):
        protocol = runner.read_json(runner.ROOT / intervention.PROTOCOL_PATH)
        def result(deltas):
            rows = [{"seed": ref["seed"], "files": f"candidate{ref['seed']}",
                     "best_map50": ref["best_map50"] + delta, "latest_map50": 0.9}
                    for ref, delta in zip(protocol["controls"], deltas)]
            return runner.aggregate(rows, protocol)
        self.assertFalse(result([0.04, 0.04, 0.04, -0.001, -0.001])["stage_a_performance_gate_passed"])
        self.assertFalse(result([0.005] * 5)["stage_a_performance_gate_passed"])
        report = result([0.02] * 5)
        self.assertTrue(report["stage_a_performance_gate_passed"])
        self.assertFalse(report["stage_b_launched"])
        self.assertFalse(report["test_evaluated"])
        with self.assertRaises(RuntimeError):
            runner.aggregate([], protocol)

    def test_initialization_mismatch_stops_before_training(self):
        report = {"reference_initialization": {"model_sha256": "wrong"}}
        with self.assertRaisesRegex(RuntimeError, "stop before training"):
            runner.check_initialization(report, {"initial_model_sha256": "expected"})

    def test_dry_run_lists_exactly_five_fresh_training_workers(self):
        commands = [runner.child_command(seed, "train") for seed in runner.SEEDS]
        self.assertEqual(len(commands), 5)
        for seed, command in zip(runner.SEEDS, commands):
            self.assertEqual(command[-4:], ["--seed", str(seed), "--worker", "train"])
            self.assertNotIn("main.py", command)

    def test_supervisor_completes_all_seeds_despite_negative_deltas_and_skips_done(self):
        protocol = runner.read_json(runner.ROOT / intervention.PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "campaign"
            def fake_worker(command, **kwargs):
                seed, mode = int(command[-3]), command[-1]
                folder = output / f"seed{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                marker = "training_complete.json" if mode == "train" else "replay.json"
                (folder / marker).write_text("{}")
            def fake_verify(folder, manifest, **kwargs):
                seed = int(folder.name.removeprefix("seed"))
                return {"seed": seed, "files": f"candidate{seed}", "best_map50": 0.0, "latest_map50": 0.0}
            with redirect_stdout(StringIO()), patch.object(runner, "OUTPUT", output), \
                    patch.object(runner.subprocess, "run", side_effect=fake_worker) as child, \
                    patch.object(runner, "verify_completed_artifacts", side_effect=fake_verify):
                runner.supervise({}, [], protocol, {})
                self.assertEqual(child.call_count, 10)  # 5 training + 5 replay, no performance screen
                self.assertFalse(json.loads((output / "decision.json").read_text())["stage_a_performance_gate_passed"])
                child.reset_mock()
                runner.supervise({}, [], protocol, {})
                child.assert_not_called()

    def test_pending_replay_is_resumed_without_retraining_completed_seed(self):
        protocol = runner.read_json(runner.ROOT / intervention.PROTOCOL_PATH)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for seed in runner.SEEDS:
                folder = output / f"seed{seed}"
                folder.mkdir()
                (folder / "training_complete.json").write_text("{}")
            def fake_replay(command, **kwargs):
                self.assertEqual(command[-1], "replay")
                (output / f"seed{command[-3]}" / "replay.json").write_text("{}")
            def fake_verify(folder, manifest, **kwargs):
                seed = int(folder.name.removeprefix("seed"))
                return {"seed": seed, "files": f"candidate{seed}", "best_map50": 0.0, "latest_map50": 0.0}
            with redirect_stdout(StringIO()), patch.object(runner, "OUTPUT", output), \
                    patch.object(runner.subprocess, "run", side_effect=fake_replay) as child, \
                    patch.object(runner, "verify_completed_artifacts", side_effect=fake_verify):
                runner.supervise({}, [], protocol, {})
                self.assertEqual(child.call_count, 5)

    def test_incomplete_training_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            partial = output / "seed40"
            partial.mkdir()
            marker = partial / "started.json"
            marker.write_text("preserve me")
            with patch.object(runner, "OUTPUT", output), \
                    patch.object(runner.subprocess, "run") as child:
                with self.assertRaisesRegex(RuntimeError, "--restart-incomplete"):
                    runner.supervise({}, [], {}, {})
                child.assert_not_called()
            self.assertEqual(marker.read_text(), "preserve me")

    def test_reference_source_manifest_matches_current_revision(self):
        manifest = intervention.repro.build_training_source_manifest()
        self.assertEqual(manifest["sha256"], "d621aef9e8628565631ea4372bed5530dc7ee632e724c08cfe725950b404acc2")

    def test_new_manifest_covers_the_intervention_and_runner(self):
        intervention.register_experiment()
        manifest = intervention.repro.build_training_source_manifest(intervention.MANIFEST_ID)
        for name in ("scripts/rtdetr_fam_zero_offset.py", "scripts/run_rtdetr_fam_zero_offset_stage_a.py",
                     intervention.PROTOCOL_PATH):
            self.assertIn(name, manifest["files"])
        expected = runner.load_yaml(runner.CONFIG)["parameters"]["reproducibility"]["training_source_manifest_sha256"][0]
        self.assertEqual(manifest["sha256"], expected)


if __name__ == "__main__":
    unittest.main()
