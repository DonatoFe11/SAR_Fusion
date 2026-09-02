import unittest
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from transformers.models.rt_detr.configuration_rt_detr import RTDetrConfig
from transformers.models.rt_detr.configuration_rt_detr_resnet import (
    RTDetrResNetConfig,
)

from sarfusion.experiment.box_guided_alignment import (
    box_guided_alignment_epoch_scale,
    box_guided_alignment_loss,
    validate_box_guided_alignment_config,
    validate_box_guided_training_contract,
)
from sarfusion.experiment.run import partition_optimizer_parameters
from sarfusion.utils.grid import make_grid
from sarfusion.utils.utils import load_yaml
from sarfusion.models.rtdetr_fusion import (
    BoxGuidedCommonOffsetFeatureAlignmentModule,
    FeatureAlignmentModule,
    RTDetrFusionBackbone,
    RTDetrFusionModel,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _single_grid_config(filename):
    raw = load_yaml(REPO_ROOT / "parameters" / "RTDETR" / filename)
    configs = make_grid(raw["parameters"])
    if len(configs) != 1:
        raise AssertionError(f"{filename} must expand to exactly one run")
    return raw, configs[0]


def tiny_rtdetr_config():
    backbone = RTDetrResNetConfig(
        num_channels=3,
        embedding_size=8,
        hidden_sizes=[16, 32, 64, 128],
        depths=[1, 1, 1, 1],
        layer_type="bottleneck",
        downsample_in_first_stage=False,
        out_indices=[2, 3, 4],
    )
    return RTDetrConfig(
        backbone_config=backbone,
        encoder_hidden_dim=32,
        encoder_in_channels=[32, 64, 128],
        feat_strides=[8, 16, 32],
        encoder_layers=1,
        encoder_ffn_dim=64,
        encoder_attention_heads=4,
        encode_proj_layers=[2],
        d_model=32,
        num_queries=10,
        decoder_in_channels=[32, 32, 32],
        decoder_ffn_dim=64,
        num_feature_levels=3,
        decoder_n_points=2,
        decoder_layers=1,
        decoder_attention_heads=4,
        num_denoising=0,
        id2label={0: "person"},
        label2id={"person": 0},
    )


class TestBoxGuidedCommonOffsetFAM(unittest.TestCase):
    def test_training_contract_rejects_confounded_or_inert_variants(self):
        guidance = {"enabled": True}
        dataset = {
            "box_alignment_targets": True,
            "modal_dropout": True,
        }
        model = {
            "name": "fusion_rtdetr",
            "params": {
                "use_fam": True,
                "fam_variant": "box_guided_common_offset_p3",
                "freeze_fam": False,
            },
        }
        validated = validate_box_guided_training_contract(
            guidance,
            dataset,
            model,
        )
        self.assertTrue(validated["enabled"])

        mutations = (
            ("use_fam", False, "use_fam=true"),
            ("freeze_fam", True, "freeze_fam=false"),
            ("use_p2", True, "not P2"),
            ("spatial_jitter_std", 0.5, "SSJ"),
            ("ir_dropout_rate", 0.1, "feature IR dropout"),
            ("use_reliability_gating", True, "other alignment gates"),
        )
        for name, value, message in mutations:
            with self.subTest(name=name):
                changed = deepcopy(model)
                changed["params"][name] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_box_guided_training_contract(
                        guidance,
                        dataset,
                        changed,
                    )

        with self.assertRaisesRegex(ValueError, "separate ablations"):
            validate_box_guided_training_contract(
                guidance,
                dataset,
                model,
                modality_consistency_enabled=True,
            )

    def test_seed40_matched_control_differs_only_by_declared_intervention(self):
        candidate_raw, candidate = _single_grid_config(
            "rtdetr_fam_box_guided_sequence_validation_seed40.yaml"
        )
        control_raw, control = _single_grid_config(
            "rtdetr_fam_box_guided_matched_control_seed40.yaml"
        )
        self.assertNotEqual(
            candidate_raw["experiment"]["name"],
            control_raw["experiment"]["name"],
        )

        normalized_candidate = deepcopy(candidate)
        normalized_control = deepcopy(control)
        normalized_candidate["model"]["params"]["fam_variant"] = (
            "current_dcnv2"
        )
        normalized_candidate["train"].pop("box_guidance_lr")
        normalized_candidate["train"].pop("box_guided_alignment")
        normalized_candidate["dataset"].pop("box_alignment_targets")
        normalized_candidate["dataset"].pop("box_alignment_max_distance")
        # Tracker labels identify the two arms but are not part of the training
        # recipe being compared.
        normalized_candidate["tracker"].pop("tags")
        normalized_control["tracker"].pop("tags")
        self.assertEqual(normalized_candidate, normalized_control)

    def test_conditional_seeds_41_44_have_fresh_matched_controls(self):
        candidate_raw = load_yaml(
            REPO_ROOT
            / "parameters"
            / "RTDETR"
            / "rtdetr_fam_box_guided_sequence_validation_five_seed.yaml"
        )
        control_raw = load_yaml(
            REPO_ROOT
            / "parameters"
            / "RTDETR"
            / "rtdetr_fam_box_guided_matched_control_seeds41_44.yaml"
        )
        candidate_runs = {
            run["seed"]: run
            for run in make_grid(candidate_raw["parameters"])
            if run["seed"] != 40
        }
        control_runs = {
            run["seed"]: run for run in make_grid(control_raw["parameters"])
        }
        self.assertEqual(set(candidate_runs), {41, 42, 43, 44})
        self.assertEqual(set(control_runs), set(candidate_runs))

        for seed in sorted(candidate_runs):
            candidate = deepcopy(candidate_runs[seed])
            control = deepcopy(control_runs[seed])
            candidate["model"]["params"]["fam_variant"] = "current_dcnv2"
            candidate["train"].pop("box_guidance_lr")
            candidate["train"].pop("box_guided_alignment")
            candidate["dataset"].pop("box_alignment_targets")
            candidate["dataset"].pop("box_alignment_max_distance")
            candidate["tracker"].pop("tags")
            control["tracker"].pop("tags")
            self.assertEqual(candidate, control, msg=f"seed {seed}")

    def test_candidate_preserves_every_shared_fam_weight_and_global_rng(self):
        torch.manual_seed(19)
        baseline = RTDetrFusionModel(
            tiny_rtdetr_config(),
            use_fam=True,
            fam_variant="current_dcnv2",
        )
        baseline_rng = torch.get_rng_state().clone()
        baseline_fam = {
            name: value.detach().clone()
            for name, value in baseline.backbone.fam_modules.state_dict().items()
        }

        torch.manual_seed(19)
        candidate = RTDetrFusionModel(
            tiny_rtdetr_config(),
            use_fam=True,
            fam_variant="box_guided_common_offset_p3",
        )
        candidate_rng = torch.get_rng_state().clone()
        candidate_fam = candidate.backbone.fam_modules.state_dict()

        shared_names = sorted(set(baseline_fam) & set(candidate_fam))
        self.assertTrue(shared_names)
        for name in shared_names:
            self.assertTrue(
                torch.equal(baseline_fam[name], candidate_fam[name]),
                msg=f"shared FAM initialization differs for {name}",
            )
        torch.testing.assert_close(
            candidate_rng,
            baseline_rng,
            atol=0,
            rtol=0,
        )

    def test_neutral_guidance_is_exact_historical_fam(self):
        torch.manual_seed(3)
        baseline = FeatureAlignmentModule(8)
        candidate = BoxGuidedCommonOffsetFeatureAlignmentModule(8)
        candidate.offset_conv.load_state_dict(baseline.offset_conv.state_dict())
        candidate.deform_conv.load_state_dict(baseline.deform_conv.state_dict())
        rgb = torch.randn(2, 8, 9, 7)
        ir = torch.randn_like(rgb)

        expected = baseline(rgb, ir)
        actual = candidate(rgb, ir, both_present=torch.tensor([True, True]))

        torch.testing.assert_close(
            candidate.last_guidance_flow,
            torch.zeros_like(candidate.last_guidance_flow),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_guidance_is_bounded_and_suppressed_when_modality_is_absent(self):
        module = BoxGuidedCommonOffsetFeatureAlignmentModule(4)
        rgb = torch.randn(2, 4, 6, 5)
        ir = torch.randn_like(rgb)
        with torch.no_grad():
            module.guidance_predictor[-1].bias.fill_(100.0)

        flow = module.predict_guidance(
            rgb,
            ir,
            both_present=torch.tensor([True, False]),
        )

        self.assertLessEqual(flow[0].abs().max().item(), 4.0)
        self.assertGreater(flow[0].abs().min().item(), 3.9)
        torch.testing.assert_close(flow[1], torch.zeros_like(flow[1]))

    def test_guidance_uses_dy_dx_interleaving_for_nine_kernel_points(self):
        flow = torch.tensor([[[[2.0]], [[-3.0]]]])
        repeated = flow.repeat(1, 9, 1, 1).flatten().tolist()
        self.assertEqual(repeated, [2.0, -3.0] * 9)

    def test_backbone_adds_guidance_only_at_p3(self):
        backbone = RTDetrFusionBackbone(
            tiny_rtdetr_config(),
            use_fam=True,
            fam_variant="box_guided_common_offset_p3",
        )
        self.assertIsInstance(
            backbone.fam_modules[0],
            BoxGuidedCommonOffsetFeatureAlignmentModule,
        )
        self.assertIsInstance(backbone.fam_modules[1], FeatureAlignmentModule)
        self.assertNotIsInstance(
            backbone.fam_modules[1],
            BoxGuidedCommonOffsetFeatureAlignmentModule,
        )
        self.assertIsInstance(backbone.fam_modules[2], FeatureAlignmentModule)

    def test_sparse_loss_converts_normalized_dy_dx_to_actual_p3_cells(self):
        class FlowHolder(nn.Module):
            def __init__(self):
                super().__init__()
                self.guided = BoxGuidedCommonOffsetFeatureAlignmentModule(4)

        model = FlowHolder()
        flow = torch.zeros(1, 2, 8, 10, requires_grad=True)
        model.guided.last_guidance_flow = flow
        targets = [torch.tensor([[0.5, 0.5, 0.125, -0.1]])]
        config = {
            "enabled": True,
            "weight": 0.2,
            "start_epoch": 0,
            "warmup_epochs": 1,
            "smooth_l1_beta_cells": 0.25,
        }

        loss = box_guided_alignment_loss(
            model,
            targets,
            config,
            epoch_scale=1.0,
        )
        # Both components target exactly +/-1 cell. Smooth-L1(beta=.25)
        # therefore gives .875 before the configured .2 multiplier.
        self.assertAlmostEqual(loss.components.box_guidance_raw_loss.item(), 0.875)
        self.assertAlmostEqual(loss.value.item(), 0.175)
        self.assertEqual(loss.components.box_guidance_matched_boxes.item(), 1.0)
        loss.value.backward()
        self.assertIsNotNone(flow.grad)
        self.assertGreater(flow.grad.abs().sum().item(), 0.0)

    def test_empty_matches_produce_differentiable_zero(self):
        class FlowHolder(nn.Module):
            def __init__(self):
                super().__init__()
                self.guided = BoxGuidedCommonOffsetFeatureAlignmentModule(4)

        model = FlowHolder()
        flow = torch.randn(2, 2, 8, 8, requires_grad=True)
        model.guided.last_guidance_flow = flow
        config = validate_box_guided_alignment_config({"enabled": True})
        loss = box_guided_alignment_loss(
            model,
            [torch.empty(0, 4), torch.empty(0, 4)],
            config,
            epoch_scale=1.0,
        )
        self.assertEqual(loss.value.item(), 0.0)
        loss.value.backward()
        torch.testing.assert_close(flow.grad, torch.zeros_like(flow.grad))

    def test_config_warmup_and_optimizer_partition(self):
        config = validate_box_guided_alignment_config(
            {
                "enabled": True,
                "weight": 0.2,
                "start_epoch": 1,
                "warmup_epochs": 2,
            }
        )
        self.assertEqual(box_guided_alignment_epoch_scale(config, 0), 0.0)
        self.assertEqual(box_guided_alignment_epoch_scale(config, 1), 0.5)
        self.assertEqual(box_guided_alignment_epoch_scale(config, 2), 1.0)

        model = nn.Module()
        model.backbone = nn.Linear(2, 2)
        model.common_projection = nn.Linear(2, 2)
        model.guidance_predictor = nn.Linear(2, 2)
        groups = partition_optimizer_parameters(model.named_parameters())
        self.assertEqual(len(groups["box_guidance"]), 4)
        self.assertEqual(len(groups["backbone"]), 2)


if __name__ == "__main__":
    unittest.main()
