import inspect
from pathlib import Path
import unittest

import torch
from torch import nn
from transformers.models.rt_detr.configuration_rt_detr import RTDetrConfig
from transformers.models.rt_detr.configuration_rt_detr_resnet import (
    RTDetrResNetConfig,
)

from sarfusion.models import build_fusion_rt_detr
from sarfusion.experiment.run import partition_optimizer_parameters
from sarfusion.models.rtdetr_fusion import (
    RTDetrFusionBackbone,
    RTDetrFusionForObjectDetection,
    ReliabilityGatedFusion,
)
from sarfusion.utils.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_sequence_validation_fixed10_protocol.yaml"
)
GATE_PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_reliability_gate_sequence_validation_seed40.yaml"
)
GATE_RUNTIME_PROBE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_reliability_gate_runtime_probe.yaml"
)
GATE_FIVE_SEED_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_reliability_gate_sequence_validation_five_seed.yaml"
)
GATE_LR10X_PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_reliability_gate_lr10x_sequence_validation_seed40.yaml"
)
GATE_LR10X_PROBE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_reliability_gate_lr10x_runtime_probe.yaml"
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


class TestReliabilityGatedFusion(unittest.TestCase):
    def test_neutral_initialization_is_exact_additive_fusion(self):
        torch.manual_seed(42)
        rgb = torch.randn(2, 8, 11, 9)
        ir = torch.randn_like(rgb)
        gate = ReliabilityGatedFusion(hidden_channels=4)

        rgb_weight, ir_weight = gate.compute_weights(
            rgb,
            ir,
            rgb_present=torch.tensor([True, False]),
            ir_present=torch.tensor([False, True]),
        )
        fused = gate(
            rgb,
            ir,
            rgb_present=torch.tensor([True, False]),
            ir_present=torch.tensor([False, True]),
        )

        torch.testing.assert_close(rgb_weight, torch.ones_like(rgb_weight))
        torch.testing.assert_close(ir_weight, torch.ones_like(ir_weight))
        torch.testing.assert_close(fused, rgb + ir, atol=0.0, rtol=0.0)

    def test_gate_receives_gradients_and_can_leave_neutral_fusion(self):
        torch.manual_seed(7)
        rgb = torch.randn(2, 4, 7, 6)
        ir = torch.randn_like(rgb)
        gate = ReliabilityGatedFusion(hidden_channels=3)
        optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-2)

        before = gate(rgb, ir)
        loss = before.square().mean()
        loss.backward()

        self.assertIsNotNone(gate.logit_conv.weight.grad)
        self.assertTrue(torch.isfinite(gate.logit_conv.weight.grad).all())
        self.assertGreater(gate.logit_conv.weight.grad.abs().max().item(), 0.0)
        optimizer.step()
        after = gate(rgb, ir)
        self.assertFalse(torch.allclose(after, before))

    def test_presence_descriptors_can_change_modality_reliability(self):
        gate = ReliabilityGatedFusion(hidden_channels=1)
        with torch.no_grad():
            gate.descriptor_conv.weight.zero_()
            gate.descriptor_conv.bias.zero_()
            gate.logit_conv.weight.zero_()
            gate.logit_conv.bias.zero_()
            gate.descriptor_conv.weight[0, 5, 1, 1] = 1.0
            gate.logit_conv.weight[0, 0, 0, 0] = 1.0

        feature = torch.ones(2, 3, 5, 4)
        rgb_weight, ir_weight = gate.compute_weights(
            feature,
            feature,
            rgb_present=torch.tensor([False, True]),
            ir_present=torch.tensor([True, True]),
        )

        torch.testing.assert_close(rgb_weight[0], torch.ones_like(rgb_weight[0]))
        self.assertTrue((rgb_weight[1] > 1.0).all())
        torch.testing.assert_close(ir_weight, torch.ones_like(ir_weight))

    def test_reliability_gating_requires_fam(self):
        with self.assertRaisesRegex(ValueError, "requires use_fam=True"):
            RTDetrFusionBackbone(
                tiny_rtdetr_config(),
                use_fam=False,
                use_reliability_gating=True,
            )

    def test_backbone_has_three_neutral_gates_and_tracks_missing_modalities(self):
        backbone = RTDetrFusionBackbone(
            tiny_rtdetr_config(),
            use_fam=True,
            use_reliability_gating=True,
            reliability_gate_hidden_channels=4,
        )
        backbone.eval()
        self.assertEqual(len(backbone.fam_modules), 3)
        self.assertEqual(len(backbone.reliability_gates), 3)

        observed = {}

        def capture_presence(module, args, kwargs, output):
            observed["rgb"] = kwargs["rgb_present"].detach().cpu()
            observed["ir"] = kwargs["ir_present"].detach().cpu()

        handle = backbone.reliability_gates[0].register_forward_hook(
            capture_presence,
            with_kwargs=True,
        )
        pixels = torch.randn(2, 4, 64, 64)
        pixels[0, :3].zero_()
        pixels[1, 3:].zero_()
        mask = torch.ones(2, 64, 64, dtype=torch.bool)
        try:
            with torch.no_grad():
                gated = backbone(pixels, mask)
                rgb_feats = backbone.rgb_backbone(pixels[:, :3], mask)
                ir_feats = backbone.ir_backbone(pixels[:, 3:], mask)
                additive = [
                    (
                        rgb_feat + backbone.fam_modules[index](rgb_feat, ir_feat),
                        rgb_mask,
                    )
                    for index, (
                        (rgb_feat, rgb_mask),
                        (ir_feat, _),
                    ) in enumerate(zip(rgb_feats, ir_feats))
                ]
        finally:
            handle.remove()

        self.assertEqual(observed["rgb"].tolist(), [False, True])
        self.assertEqual(observed["ir"].tolist(), [True, False])
        for (gated_feat, gated_mask), (additive_feat, additive_mask) in zip(
            gated, additive
        ):
            torch.testing.assert_close(
                gated_feat, additive_feat, atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(gated_mask, additive_mask)

    def test_full_detector_forward_and_public_factory_expose_gating(self):
        model = RTDetrFusionForObjectDetection(
            tiny_rtdetr_config(),
            use_fam=True,
            use_reliability_gating=True,
            reliability_gate_hidden_channels=4,
        )
        model.eval()
        with torch.no_grad():
            output = model(
                torch.randn(1, 4, 64, 64),
                pixel_mask=torch.ones(1, 64, 64, dtype=torch.bool),
            )

        self.assertEqual(tuple(output.logits.shape), (1, 10, 1))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertIn(
            "use_reliability_gating",
            inspect.signature(build_fusion_rt_detr).parameters,
        )
        self.assertIn(
            "use_reliability_gating",
            inspect.signature(
                RTDetrFusionForObjectDetection.from_pretrained
            ).parameters,
        )

    def test_stage_a_protocol_changes_only_the_declared_gate(self):
        baseline = load_yaml(BASELINE_PATH)
        protocol = load_yaml(GATE_PROTOCOL_PATH)
        params = protocol["parameters"]
        model = params["model"]["params"]

        self.assertEqual(params["seed"], [40])
        self.assertEqual(params["train"]["max_epochs"], [10])
        self.assertNotIn("early_stopping_patience", params["train"])
        self.assertEqual(params["dataloader"]["batch_size"], [4])
        self.assertEqual(params["dataloader"]["evaluation_batch_size"], [12])
        self.assertEqual(params["run_test"], [False])
        self.assertEqual(model["use_fam"], [True])
        self.assertEqual(model["use_p2"], [False])
        self.assertEqual(model["use_reliability_gating"], [True])
        self.assertEqual(model["reliability_gate_hidden_channels"], [16])
        self.assertEqual(
            params["dataset"], baseline["parameters"]["dataset"]
        )

        gate_model = dict(model)
        gate_model.pop("use_p2")
        gate_model.pop("use_reliability_gating")
        gate_model.pop("reliability_gate_hidden_channels")
        self.assertEqual(
            gate_model,
            baseline["parameters"]["model"]["params"],
        )

    def test_runtime_probe_is_short_checkpoint_free_and_excluded(self):
        probe = load_yaml(GATE_RUNTIME_PROBE_PATH)
        full = load_yaml(GATE_PROTOCOL_PATH)
        params = probe["parameters"]

        self.assertEqual(params["seed"], [40])
        self.assertEqual(params["train"]["max_epochs"], [1])
        self.assertEqual(params["train"]["max_steps_per_epoch"], [20])
        self.assertEqual(params["train"]["save_checkpoints"], [False])
        self.assertEqual(params["dataloader"]["batch_size"], [4])
        self.assertIn("ExcludeFromCampaign", params["tracker"]["tags"][0])
        self.assertEqual(
            params["model"], full["parameters"]["model"]
        )
        self.assertEqual(
            params["dataset"], full["parameters"]["dataset"]
        )

    def test_five_seed_expansion_changes_only_the_seed_grid(self):
        pilot = load_yaml(GATE_PROTOCOL_PATH)
        campaign = load_yaml(GATE_FIVE_SEED_PATH)

        self.assertEqual(campaign["parameters"]["seed"], [40, 41, 42, 43, 44])
        pilot_without_seed = dict(pilot["parameters"])
        campaign_without_seed = dict(campaign["parameters"])
        pilot_without_seed.pop("seed")
        campaign_without_seed.pop("seed")
        self.assertEqual(campaign_without_seed, pilot_without_seed)
        self.assertEqual(campaign["experiment"], pilot["experiment"])

    def test_optimizer_partition_isolates_reliability_gate_parameters(self):
        model = nn.Module()
        model.backbone = nn.Linear(2, 2)
        model.ir_backbone = nn.Linear(2, 2)
        model.class_embed = nn.Linear(2, 2)
        model.reliability_gates = nn.ModuleList([nn.Linear(2, 2)])

        groups = partition_optimizer_parameters(model.named_parameters())
        parameter_ids = {
            name: {id(parameter) for parameter in parameters}
            for name, parameters in groups.items()
        }
        all_ids = set().union(*parameter_ids.values())

        self.assertEqual(len(all_ids), len(list(model.parameters())))
        self.assertTrue(parameter_ids["reliability_gate"])
        self.assertTrue(parameter_ids["backbone"])
        self.assertTrue(parameter_ids["new_modules"])
        self.assertTrue(parameter_ids["head_and_dino"])
        for first, first_ids in parameter_ids.items():
            for second, second_ids in parameter_ids.items():
                if first != second:
                    self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_lr10x_protocol_changes_only_the_declared_gate_learning_rate(self):
        original = load_yaml(GATE_PROTOCOL_PATH)
        lr10x = load_yaml(GATE_LR10X_PROTOCOL_PATH)
        original_params = original["parameters"]
        lr10x_params = lr10x["parameters"]

        self.assertEqual(lr10x_params["seed"], [40])
        self.assertEqual(lr10x_params["train"]["initial_lr"], [0.00002])
        self.assertEqual(
            lr10x_params["train"]["reliability_gate_lr"], [0.0002]
        )
        self.assertEqual(lr10x_params["model"], original_params["model"])
        self.assertEqual(lr10x_params["dataset"], original_params["dataset"])
        self.assertEqual(
            lr10x_params["dataloader"], original_params["dataloader"]
        )

        lr10x_train = dict(lr10x_params["train"])
        lr10x_train.pop("reliability_gate_lr")
        self.assertEqual(lr10x_train, original_params["train"])

    def test_lr10x_probe_is_short_checkpoint_free_and_excluded(self):
        probe = load_yaml(GATE_LR10X_PROBE_PATH)
        full = load_yaml(GATE_LR10X_PROTOCOL_PATH)
        params = probe["parameters"]

        self.assertEqual(params["train"]["max_epochs"], [1])
        self.assertEqual(params["train"]["max_steps_per_epoch"], [20])
        self.assertEqual(params["train"]["save_checkpoints"], [False])
        self.assertEqual(params["train"]["reliability_gate_lr"], [0.0002])
        self.assertIn("ExcludeFromCampaign", params["tracker"]["tags"][0])
        self.assertEqual(params["model"], full["parameters"]["model"])
        self.assertEqual(params["dataset"], full["parameters"]["dataset"])


if __name__ == "__main__":
    unittest.main()
