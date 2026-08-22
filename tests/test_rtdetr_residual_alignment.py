import inspect
from pathlib import Path
import unittest

import torch
from torch import nn
from transformers.models.rt_detr.configuration_rt_detr import RTDetrConfig
from transformers.models.rt_detr.configuration_rt_detr_resnet import (
    RTDetrResNetConfig,
)

from sarfusion.experiment.run import partition_optimizer_parameters
from sarfusion.models import build_fusion_rt_detr
from sarfusion.models.rtdetr_fusion import (
    RTDetrFusionBackbone,
    RTDetrFusionForObjectDetection,
    ReliabilityConditionedResidualAlignment,
)
from sarfusion.utils.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_sequence_validation_fixed10_protocol.yaml"
)
PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_residual_alignment_sequence_validation_five_seed.yaml"
)
PROBE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_residual_alignment_runtime_probe.yaml"
)


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


class TestReliabilityConditionedResidualAlignment(unittest.TestCase):
    def test_neutral_initialization_is_exact_fam_pass_through(self):
        torch.manual_seed(42)
        rgb = torch.randn(2, 8, 11, 9)
        raw = torch.randn_like(rgb)
        aligned = torch.randn_like(rgb)
        gate = ReliabilityConditionedResidualAlignment(hidden_channels=4)

        alpha = gate.compute_alpha(
            rgb,
            raw,
            aligned,
            rgb_present=torch.tensor([True, False]),
            ir_present=torch.tensor([False, True]),
        )
        selected = gate(
            rgb,
            raw,
            aligned,
            rgb_present=torch.tensor([True, False]),
            ir_present=torch.tensor([False, True]),
        )

        torch.testing.assert_close(alpha, torch.ones_like(alpha))
        torch.testing.assert_close(selected, aligned, atol=0.0, rtol=0.0)

    def test_gate_receives_gradient_and_can_change_alignment_residual(self):
        torch.manual_seed(7)
        rgb = torch.randn(2, 4, 7, 6)
        raw = torch.randn_like(rgb)
        aligned = raw + torch.randn_like(raw)
        gate = ReliabilityConditionedResidualAlignment(hidden_channels=3)
        optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-2)

        before = gate(rgb, raw, aligned)
        before.square().mean().backward()

        self.assertIsNotNone(gate.logit_conv.weight.grad)
        self.assertTrue(torch.isfinite(gate.logit_conv.weight.grad).all())
        self.assertGreater(gate.logit_conv.weight.grad.abs().max().item(), 0.0)
        optimizer.step()
        after = gate(rgb, raw, aligned)
        self.assertFalse(torch.allclose(after, before))

    def test_alpha_zero_bypasses_and_alpha_two_amplifies_fam_correction(self):
        gate = ReliabilityConditionedResidualAlignment(hidden_channels=1)
        rgb = torch.randn(1, 3, 4, 5)
        raw = torch.randn_like(rgb)
        aligned = torch.randn_like(rgb)
        residual = aligned - raw

        with torch.no_grad():
            gate.logit_conv.bias.fill_(-100.0)
        bypassed = gate(rgb, raw, aligned)
        torch.testing.assert_close(bypassed, raw)

        with torch.no_grad():
            gate.logit_conv.bias.fill_(100.0)
        amplified = gate(rgb, raw, aligned)
        torch.testing.assert_close(amplified, raw + 2.0 * residual)

    def test_invalid_shapes_and_presence_are_rejected(self):
        gate = ReliabilityConditionedResidualAlignment(hidden_channels=2)
        feature = torch.randn(2, 4, 5, 6)
        with self.assertRaisesRegex(ValueError, "equal RGB"):
            gate(feature, feature[:, :, :-1], feature)
        with self.assertRaisesRegex(ValueError, "presence must have shape"):
            gate.compute_alpha(
                feature,
                feature,
                feature,
                rgb_present=torch.ones(2, 1),
            )

    def test_backbone_starts_as_exact_additive_fam_and_tracks_presence(self):
        backbone = RTDetrFusionBackbone(
            tiny_rtdetr_config(),
            use_fam=True,
            use_residual_alignment_gating=True,
            residual_alignment_hidden_channels=4,
        )
        backbone.eval()
        self.assertEqual(len(backbone.fam_modules), 3)
        self.assertEqual(len(backbone.alignment_gates), 3)

        observed = {}

        def capture_presence(module, args, kwargs, output):
            observed["rgb"] = kwargs["rgb_present"].detach().cpu()
            observed["ir"] = kwargs["ir_present"].detach().cpu()

        handle = backbone.alignment_gates[0].register_forward_hook(
            capture_presence,
            with_kwargs=True,
        )
        pixels = torch.randn(2, 4, 64, 64)
        pixels[0, :3].zero_()
        pixels[1, 3:].zero_()
        mask = torch.ones(2, 64, 64, dtype=torch.bool)
        try:
            with torch.no_grad():
                selected = backbone(pixels, mask)
                rgb_feats = backbone.rgb_backbone(pixels[:, :3], mask)
                ir_feats = backbone.ir_backbone(pixels[:, 3:], mask)
                baseline = [
                    (
                        rgb_feat + backbone.fam_modules[index](rgb_feat, ir_feat),
                        rgb_mask,
                    )
                    for index, ((rgb_feat, rgb_mask), (ir_feat, _)) in enumerate(
                        zip(rgb_feats, ir_feats)
                    )
                ]
        finally:
            handle.remove()

        self.assertEqual(observed["rgb"].tolist(), [False, True])
        self.assertEqual(observed["ir"].tolist(), [True, False])
        for (selected_feat, selected_mask), (base_feat, base_mask) in zip(
            selected, baseline
        ):
            torch.testing.assert_close(
                selected_feat, base_feat, atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(selected_mask, base_mask)

    def test_ablation_constraints_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "requires use_fam=True"):
            RTDetrFusionBackbone(
                tiny_rtdetr_config(),
                use_fam=False,
                use_residual_alignment_gating=True,
            )
        with self.assertRaisesRegex(ValueError, "separate ablations"):
            RTDetrFusionBackbone(
                tiny_rtdetr_config(),
                use_fam=True,
                use_reliability_gating=True,
                use_residual_alignment_gating=True,
            )

    def test_full_detector_and_public_factories_expose_candidate(self):
        model = RTDetrFusionForObjectDetection(
            tiny_rtdetr_config(),
            use_fam=True,
            use_residual_alignment_gating=True,
            residual_alignment_hidden_channels=4,
        )
        model.eval()
        with torch.no_grad():
            output = model(
                torch.randn(1, 4, 64, 64),
                pixel_mask=torch.ones(1, 64, 64, dtype=torch.bool),
            )

        self.assertEqual(tuple(output.logits.shape), (1, 10, 1))
        self.assertTrue(torch.isfinite(output.logits).all())
        for callable_object in (
            build_fusion_rt_detr,
            RTDetrFusionForObjectDetection.from_pretrained,
        ):
            parameters = inspect.signature(callable_object).parameters
            self.assertIn("use_residual_alignment_gating", parameters)
            self.assertIn("residual_alignment_hidden_channels", parameters)

    def test_optimizer_partition_isolates_alignment_gate_parameters(self):
        model = nn.Module()
        model.backbone = nn.Linear(2, 2)
        model.ir_backbone = nn.Linear(2, 2)
        model.class_embed = nn.Linear(2, 2)
        model.alignment_gates = nn.ModuleList([nn.Linear(2, 2)])

        groups = partition_optimizer_parameters(model.named_parameters())
        parameter_ids = {
            name: {id(parameter) for parameter in parameters}
            for name, parameters in groups.items()
        }
        all_ids = set().union(*parameter_ids.values())

        self.assertEqual(len(all_ids), len(list(model.parameters())))
        self.assertTrue(parameter_ids["alignment_gate"])
        for first, first_ids in parameter_ids.items():
            for second, second_ids in parameter_ids.items():
                if first != second:
                    self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_protocol_is_frozen_against_baseline(self):
        baseline = load_yaml(BASELINE_PATH)["parameters"]
        protocol = load_yaml(PROTOCOL_PATH)["parameters"]
        model = protocol["model"]["params"]

        self.assertEqual(protocol["seed"], [40, 41, 42, 43, 44])
        self.assertEqual(protocol["train"]["max_epochs"], [10])
        self.assertNotIn("early_stopping_patience", protocol["train"])
        self.assertEqual(protocol["train"]["alignment_gate_lr"], [0.0002])
        self.assertEqual(protocol["run_test"], [False])
        self.assertEqual(model["use_fam"], [True])
        self.assertEqual(model["use_p2"], [False])
        self.assertEqual(model["use_reliability_gating"], [False])
        self.assertEqual(model["use_residual_alignment_gating"], [True])
        self.assertEqual(model["residual_alignment_hidden_channels"], [16])
        self.assertEqual(protocol["dataset"], baseline["dataset"])
        self.assertEqual(protocol["dataloader"], baseline["dataloader"])

        candidate_model = dict(model)
        candidate_model.pop("use_p2")
        candidate_model.pop("use_reliability_gating")
        candidate_model.pop("use_residual_alignment_gating")
        candidate_model.pop("residual_alignment_hidden_channels")
        self.assertEqual(candidate_model, baseline["model"]["params"])

        candidate_train = dict(protocol["train"])
        candidate_train.pop("alignment_gate_lr")
        self.assertEqual(candidate_train, baseline["train"])

    def test_probe_is_short_checkpoint_free_and_otherwise_identical(self):
        probe = load_yaml(PROBE_PATH)["parameters"]
        protocol = load_yaml(PROTOCOL_PATH)["parameters"]

        self.assertEqual(probe["seed"], [40])
        self.assertEqual(probe["train"]["max_epochs"], [1])
        self.assertEqual(probe["train"]["max_steps_per_epoch"], [20])
        self.assertEqual(probe["train"]["save_checkpoints"], [False])
        self.assertIn("ExcludeFromCampaign", probe["tracker"]["tags"][0])
        self.assertEqual(probe["model"], protocol["model"])
        self.assertEqual(probe["dataset"], protocol["dataset"])
        self.assertEqual(probe["dataloader"], protocol["dataloader"])


if __name__ == "__main__":
    unittest.main()
