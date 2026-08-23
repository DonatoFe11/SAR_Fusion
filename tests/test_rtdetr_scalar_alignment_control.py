import csv
import inspect
from pathlib import Path
import statistics
import unittest

import torch
from transformers.models.rt_detr.configuration_rt_detr import RTDetrConfig
from transformers.models.rt_detr.configuration_rt_detr_resnet import (
    RTDetrResNetConfig,
)

from sarfusion.experiment.run import partition_optimizer_parameters
from sarfusion.models import build_fusion_rt_detr
from sarfusion.models.rtdetr_fusion import (
    RTDetrFusionBackbone,
    RTDetrFusionForObjectDetection,
    ScalarResidualAlignment,
)
from sarfusion.utils.utils import load_yaml
from scripts.run_rtdetr_fam_scalar_alignment_control_audit import (
    find_scalar_alignment_gates,
    scalar_rows,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_sequence_validation_fixed10_protocol.yaml"
)
RCRA_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_residual_alignment_sequence_validation_five_seed.yaml"
)
CONTROL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_scalar_alignment_control_sequence_validation_five_seed.yaml"
)
PROBE_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_scalar_alignment_control_runtime_probe.yaml"
)
AUDIT_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_scalar_alignment_control_audit.yaml"
)
AUDIT_RESULTS_PATH = (
    REPO_ROOT
    / "notes"
    / "Search_and_Rescue"
    / "results"
    / "rtdetr_fam_scalar_alignment_control_audit.csv"
)
PERFORMANCE_RESULTS_PATH = (
    REPO_ROOT
    / "notes"
    / "Search_and_Rescue"
    / "results"
    / "rtdetr_fam_scalar_alignment_control_stage_a_validation.csv"
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


class TestScalarResidualAlignmentControl(unittest.TestCase):
    def test_neutral_initialization_is_exact_aligned_pass_through(self):
        torch.manual_seed(1)
        raw = torch.randn(2, 4, 5, 6)
        aligned = torch.randn_like(raw)
        gate = ScalarResidualAlignment()

        self.assertEqual(gate.compute_alpha().item(), 1.0)
        selected = gate(torch.randn_like(raw), raw, aligned)
        torch.testing.assert_close(selected, aligned, atol=0.0, rtol=0.0)

    def test_one_parameter_receives_gradient_and_leaves_identity(self):
        torch.manual_seed(2)
        raw = torch.randn(2, 4, 5, 6)
        aligned = raw + torch.randn_like(raw)
        gate = ScalarResidualAlignment()
        optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-2)

        before = gate(None, raw, aligned)
        before.square().mean().backward()
        self.assertIsNotNone(gate.logit.grad)
        self.assertTrue(torch.isfinite(gate.logit.grad))
        self.assertNotEqual(gate.logit.grad.item(), 0.0)
        optimizer.step()
        self.assertNotEqual(gate.compute_alpha().item(), 1.0)

    def test_extreme_logits_bypass_or_amplify_residual(self):
        raw = torch.randn(1, 3, 4, 5)
        aligned = torch.randn_like(raw)
        residual = aligned - raw
        gate = ScalarResidualAlignment()

        with torch.no_grad():
            gate.logit.fill_(-100.0)
        torch.testing.assert_close(gate(None, raw, aligned), raw)

        with torch.no_grad():
            gate.logit.fill_(100.0)
        torch.testing.assert_close(
            gate(None, raw, aligned),
            raw + 2.0 * residual,
        )

    def test_backbone_has_exactly_three_scalar_gates_and_starts_as_fam(self):
        backbone = RTDetrFusionBackbone(
            tiny_rtdetr_config(),
            use_fam=True,
            use_scalar_residual_alignment=True,
        )
        backbone.eval()
        self.assertEqual(len(backbone.alignment_gates), 3)
        self.assertTrue(
            all(
                isinstance(gate, ScalarResidualAlignment)
                for gate in backbone.alignment_gates
            )
        )
        self.assertEqual(
            sum(parameter.numel() for gate in backbone.alignment_gates for parameter in gate.parameters()),
            3,
        )

        pixels = torch.randn(2, 4, 64, 64)
        mask = torch.ones(2, 64, 64, dtype=torch.bool)
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

        for (selected_feat, selected_mask), (base_feat, base_mask) in zip(
            selected, baseline
        ):
            torch.testing.assert_close(
                selected_feat, base_feat, atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(selected_mask, base_mask)

    def test_scalar_control_requires_fam_and_is_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "requires use_fam=True"):
            RTDetrFusionBackbone(
                tiny_rtdetr_config(),
                use_scalar_residual_alignment=True,
            )
        with self.assertRaisesRegex(ValueError, "separate ablations"):
            RTDetrFusionBackbone(
                tiny_rtdetr_config(),
                use_fam=True,
                use_scalar_residual_alignment=True,
                use_residual_alignment_gating=True,
            )
        with self.assertRaisesRegex(ValueError, "separate ablations"):
            RTDetrFusionBackbone(
                tiny_rtdetr_config(),
                use_fam=True,
                use_scalar_residual_alignment=True,
                use_reliability_gating=True,
            )

    def test_full_detector_and_public_factories_expose_control(self):
        model = RTDetrFusionForObjectDetection(
            tiny_rtdetr_config(),
            use_fam=True,
            use_scalar_residual_alignment=True,
        )
        model.eval()
        with torch.no_grad():
            output = model(
                torch.randn(1, 4, 64, 64),
                pixel_mask=torch.ones(1, 64, 64, dtype=torch.bool),
            )
        self.assertEqual(tuple(output.logits.shape), (1, 10, 1))

        for callable_object in (
            build_fusion_rt_detr,
            RTDetrFusionForObjectDetection.from_pretrained,
        ):
            self.assertIn(
                "use_scalar_residual_alignment",
                inspect.signature(callable_object).parameters,
            )

    def test_optimizer_partition_keeps_three_scalars_in_alignment_group(self):
        backbone = RTDetrFusionBackbone(
            tiny_rtdetr_config(),
            use_fam=True,
            use_scalar_residual_alignment=True,
        )
        groups = partition_optimizer_parameters(backbone.named_parameters())
        self.assertEqual(
            sum(parameter.numel() for parameter in groups["alignment_gate"]),
            3,
        )
        alignment_ids = {
            id(parameter) for parameter in groups["alignment_gate"]
        }
        other_ids = {
            id(parameter)
            for name, parameters in groups.items()
            if name != "alignment_gate"
            for parameter in parameters
        }
        self.assertTrue(alignment_ids.isdisjoint(other_ids))

    def test_control_protocol_changes_only_declared_architecture_from_rcra(self):
        rcra = load_yaml(RCRA_PATH)["parameters"]
        control = load_yaml(CONTROL_PATH)["parameters"]
        baseline = load_yaml(BASELINE_PATH)["parameters"]

        self.assertEqual(control["seed"], [40, 41, 42, 43, 44])
        self.assertEqual(control["train"], rcra["train"])
        self.assertEqual(control["dataset"], rcra["dataset"])
        self.assertEqual(control["dataloader"], rcra["dataloader"])
        self.assertEqual(control["dataset"], baseline["dataset"])
        model = control["model"]["params"]
        self.assertEqual(model["use_residual_alignment_gating"], [False])
        self.assertEqual(model["use_scalar_residual_alignment"], [True])

        rcra_model = dict(rcra["model"]["params"])
        rcra_model.pop("residual_alignment_hidden_channels")
        rcra_model["use_residual_alignment_gating"] = [False]
        rcra_model["use_scalar_residual_alignment"] = [True]
        self.assertEqual(model, rcra_model)

    def test_probe_is_short_checkpoint_free_and_campaign_equivalent(self):
        probe = load_yaml(PROBE_PATH)["parameters"]
        control = load_yaml(CONTROL_PATH)["parameters"]

        self.assertEqual(probe["seed"], [40])
        self.assertEqual(probe["train"]["max_epochs"], [1])
        self.assertEqual(probe["train"]["max_steps_per_epoch"], [20])
        self.assertEqual(probe["train"]["save_checkpoints"], [False])
        self.assertIn("ExcludeFromCampaign", probe["tracker"]["tags"][0])
        self.assertEqual(probe["model"], control["model"])
        self.assertEqual(probe["dataset"], control["dataset"])
        self.assertEqual(probe["dataloader"], control["dataloader"])

    def test_audit_protocol_and_scalar_extraction_are_frozen(self):
        protocol = load_yaml(AUDIT_PATH)
        validate_protocol(protocol)
        self.assertEqual(protocol["checkpoint"], "best")

        model = torch.nn.Sequential(
            ScalarResidualAlignment(),
            ScalarResidualAlignment(),
            ScalarResidualAlignment(),
        )
        gates = find_scalar_alignment_gates(model)
        rows = scalar_rows(40, gates, protocol["level_labels"])
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["alpha"] for row in rows], [1.0, 1.0, 1.0])

    def test_completed_performance_follows_frozen_selection_rule(self):
        with PERFORMANCE_RESULTS_PATH.open(
            newline="", encoding="utf-8"
        ) as result_file:
            rows = list(csv.DictReader(result_file))

        self.assertEqual(len(rows), 5)
        self.assertEqual([int(row["seed"]) for row in rows], list(range(40, 45)))

        scalar_minus_fam = [
            float(row["scalar_minus_baseline"]) for row in rows
        ]
        rcra_minus_fam = [
            float(row["rcra_best_map50"])
            - float(row["baseline_best_map50"])
            for row in rows
        ]
        rcra_minus_scalar = [
            float(row["rcra_minus_scalar"]) for row in rows
        ]

        self.assertLess(statistics.fmean(scalar_minus_fam), 0.01)
        self.assertEqual(sum(delta > 0.0 for delta in scalar_minus_fam), 2)
        self.assertGreaterEqual(statistics.fmean(rcra_minus_fam), 0.01)
        self.assertEqual(sum(delta > 0.0 for delta in rcra_minus_fam), 4)
        self.assertEqual(sum(delta > 0.0 for delta in rcra_minus_scalar), 3)

    def test_completed_audit_has_three_active_bounded_scalars_per_seed(self):
        with AUDIT_RESULTS_PATH.open(
            newline="", encoding="utf-8"
        ) as result_file:
            rows = list(csv.DictReader(result_file))

        self.assertEqual(len(rows), 5 * 3)
        self.assertEqual(
            sorted({int(row["seed"]) for row in rows}), list(range(40, 45))
        )
        for seed in range(40, 45):
            seed_rows = [row for row in rows if int(row["seed"]) == seed]
            self.assertEqual(
                [row["level_label"] for row in seed_rows], ["P3", "P4", "P5"]
            )
            self.assertTrue(
                all(0.0 < float(row["alpha"]) < 2.0 for row in seed_rows)
            )
            self.assertGreaterEqual(
                float(seed_rows[0]["abs_delta_one"]), 0.02
            )


if __name__ == "__main__":
    unittest.main()
