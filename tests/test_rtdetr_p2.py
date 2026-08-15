import inspect
from pathlib import Path
import unittest

import torch
from sarfusion.models import build_fusion_rt_detr
from transformers.models.rt_detr.configuration_rt_detr import RTDetrConfig
from transformers.models.rt_detr.configuration_rt_detr_resnet import (
    RTDetrResNetConfig,
)

from sarfusion.models.rtdetr_fusion import (
    RTDetrFusionBackbone,
    RTDetrFusionForObjectDetection,
    build_p2_pretrained_state,
    configure_rtdetr_p2,
)
from sarfusion.utils.utils import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
P2_PROTOCOL_PATH = (
    REPO_ROOT
    / "parameters"
    / "RTDETR"
    / "rtdetr_fam_p2_sequence_validation_seed40.yaml"
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


class TestRTDetrP2(unittest.TestCase):
    def test_configuration_exposes_p2_through_p5(self):
        config = configure_rtdetr_p2(tiny_rtdetr_config())

        self.assertEqual(config.backbone_config.out_indices, [1, 2, 3, 4])
        self.assertEqual(config.backbone_config.out_features, [
            "stage1", "stage2", "stage3", "stage4"
        ])
        self.assertEqual(config.encoder_in_channels, [16, 32, 64, 128])
        self.assertEqual(config.feat_strides, [4, 8, 16, 32])
        self.assertEqual(config.encode_proj_layers, [3])
        self.assertEqual(config.decoder_in_channels, [32, 32, 32, 32])
        self.assertEqual(config.num_feature_levels, 4)
        self.assertTrue(config.use_p2)

    def test_existing_three_level_path_is_unchanged_by_default(self):
        config = tiny_rtdetr_config()
        backbone = RTDetrFusionBackbone(config, use_fam=True)

        self.assertEqual(config.backbone_config.out_indices, [2, 3, 4])
        self.assertEqual(config.feat_strides, [8, 16, 32])
        self.assertEqual(config.num_feature_levels, 3)
        self.assertEqual(len(backbone.fam_modules), 3)
        self.assertFalse(backbone.use_p2)

    def test_fusion_backbone_outputs_four_aligned_levels(self):
        config = configure_rtdetr_p2(tiny_rtdetr_config())
        backbone = RTDetrFusionBackbone(config, use_fam=True)
        backbone.eval()
        pixels = torch.randn(1, 4, 64, 64)
        mask = torch.ones(1, 64, 64, dtype=torch.bool)

        with torch.no_grad():
            features = backbone(pixels, mask)

        self.assertEqual(len(backbone.fam_modules), 4)
        self.assertEqual(
            [tuple(feature.shape) for feature, _ in features],
            [
                (1, 16, 16, 16),
                (1, 32, 8, 8),
                (1, 64, 4, 4),
                (1, 128, 2, 2),
            ],
        )
        self.assertTrue(all(tuple(level_mask.shape) == (1, *feature.shape[-2:])
                            for feature, level_mask in features))

    def test_full_detector_forward_accepts_four_channel_p2_input(self):
        config = configure_rtdetr_p2(tiny_rtdetr_config())
        config.num_channels = 4
        model = RTDetrFusionForObjectDetection(
            config,
            use_fam=True,
            use_p2=True,
        )
        model.eval()

        with torch.no_grad():
            output = model(
                torch.randn(1, 4, 64, 64),
                pixel_mask=torch.ones(1, 64, 64, dtype=torch.bool),
            )

        self.assertEqual(tuple(output.logits.shape), (1, 10, 1))
        self.assertEqual(tuple(output.pred_boxes.shape), (1, 10, 4))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.isfinite(output.pred_boxes).all())

    def test_pretrained_levels_are_shifted_instead_of_relabelled(self):
        config = tiny_rtdetr_config()
        config.decoder_attention_heads = 2
        config.decoder_n_points = 2
        configure_rtdetr_p2(config)

        source = {
            "model.encoder_input_proj.0.0.weight": torch.full((2, 3, 1, 1), 10.0),
            "model.encoder_input_proj.1.0.weight": torch.full((2, 4, 1, 1), 11.0),
            "model.decoder_input_proj.0.0.weight": torch.full((2, 2, 1, 1), 20.0),
            "model.encoder.downsample_convs.0.conv.weight": torch.full((2, 2, 3, 3), 30.0),
            "model.encoder.fpn_blocks.1.conv.weight": torch.full((2, 2, 1, 1), 40.0),
            "model.stable.weight": torch.tensor([50.0]),
        }
        sampling_rows_old = 2 * 3 * 2 * 2
        sampling_rows_new = 2 * 4 * 2 * 2
        attention_rows_old = 2 * 3 * 2
        attention_rows_new = 2 * 4 * 2
        sampling_key = "model.decoder.layers.0.encoder_attn.sampling_offsets.weight"
        attention_key = "model.decoder.layers.0.encoder_attn.attention_weights.bias"
        source[sampling_key] = torch.arange(
            sampling_rows_old * 5, dtype=torch.float32
        ).reshape(sampling_rows_old, 5)
        source[attention_key] = torch.arange(attention_rows_old, dtype=torch.float32)

        target = {
            "model.encoder_input_proj.0.0.weight": torch.empty(2, 2, 1, 1),
            "model.encoder_input_proj.1.0.weight": torch.empty(2, 3, 1, 1),
            "model.encoder_input_proj.2.0.weight": torch.empty(2, 4, 1, 1),
            "model.decoder_input_proj.0.0.weight": torch.empty(2, 2, 1, 1),
            "model.decoder_input_proj.1.0.weight": torch.empty(2, 2, 1, 1),
            "model.encoder.downsample_convs.0.conv.weight": torch.empty(2, 2, 3, 3),
            "model.encoder.downsample_convs.1.conv.weight": torch.empty(2, 2, 3, 3),
            "model.encoder.fpn_blocks.1.conv.weight": torch.empty(2, 2, 1, 1),
            "model.encoder.fpn_blocks.2.conv.weight": torch.empty(2, 2, 1, 1),
            "model.stable.weight": torch.empty(1),
            sampling_key: torch.full((sampling_rows_new, 5), -1.0),
            attention_key: torch.full((attention_rows_new,), -1.0),
        }

        remapped = build_p2_pretrained_state(source, target, config)

        torch.testing.assert_close(
            remapped["model.encoder_input_proj.1.0.weight"],
            source["model.encoder_input_proj.0.0.weight"],
        )
        torch.testing.assert_close(
            remapped["model.encoder_input_proj.2.0.weight"],
            source["model.encoder_input_proj.1.0.weight"],
        )
        expected_identity = torch.eye(2).reshape(2, 2, 1, 1)
        torch.testing.assert_close(
            remapped["model.encoder_input_proj.0.0.weight"], expected_identity
        )
        torch.testing.assert_close(
            remapped["model.decoder_input_proj.0.0.weight"],
            source["model.decoder_input_proj.0.0.weight"],
        )
        torch.testing.assert_close(
            remapped["model.decoder_input_proj.1.0.weight"],
            source["model.decoder_input_proj.0.0.weight"],
        )
        torch.testing.assert_close(
            remapped["model.encoder.fpn_blocks.2.conv.weight"],
            source["model.encoder.fpn_blocks.1.conv.weight"],
        )
        self.assertEqual(remapped["model.stable.weight"].item(), 50.0)

        old_sampling = source[sampling_key].reshape(2, 3, 2, 2, 5)
        new_sampling = remapped[sampling_key].reshape(2, 4, 2, 2, 5)
        torch.testing.assert_close(new_sampling[:, 0], old_sampling[:, 0])
        torch.testing.assert_close(new_sampling[:, 1:], old_sampling)
        old_attention = source[attention_key].reshape(2, 3, 2)
        new_attention = remapped[attention_key].reshape(2, 4, 2)
        torch.testing.assert_close(new_attention[:, 0], old_attention[:, 0])
        torch.testing.assert_close(new_attention[:, 1:], old_attention)

    def test_p2_requires_four_backbone_stages(self):
        config = tiny_rtdetr_config()
        config.backbone_config.hidden_sizes = [16, 32, 64]
        with self.assertRaisesRegex(ValueError, "four-stage backbone"):
            configure_rtdetr_p2(config)

    def test_public_factories_expose_p2_flag(self):
        self.assertIn(
            "use_p2",
            inspect.signature(build_fusion_rt_detr).parameters,
        )
        self.assertIn(
            "use_p2",
            inspect.signature(RTDetrFusionForObjectDetection.from_pretrained).parameters,
        )

    def test_stage_a_yaml_is_one_full_seed_with_effective_batch_four(self):
        protocol = load_yaml(P2_PROTOCOL_PATH)
        params = protocol["parameters"]
        train = params["train"]
        model = params["model"]["params"]

        self.assertEqual(
            protocol["experiment"]["name"],
            "RTDETR_FAM_P2_SequenceVal_Seed40",
        )
        self.assertEqual(params["seed"], [40])
        self.assertEqual(train["max_epochs"], [10])
        self.assertEqual(train["run_validation"], [True])
        self.assertNotIn("early_stopping_patience", train)
        self.assertEqual(train["watch_metric"], ["map_50"])
        self.assertEqual(train["gradient_accumulation_steps"], [2])
        self.assertEqual(params["dataloader"]["batch_size"], [2])
        self.assertEqual(params["dataloader"]["evaluation_batch_size"], [12])
        self.assertEqual(params["run_test"], [False])
        self.assertEqual(params["test_checkpoint"], ["best"])
        self.assertEqual(model["use_fam"], [True])
        self.assertEqual(model["fam_variant"], ["current_dcnv2"])
        self.assertEqual(model["use_p2"], [True])
        self.assertEqual(len(params["dataset"]["train_folders"][0]), 2)
        self.assertEqual(len(params["dataset"]["val_folders"][0]), 1)


if __name__ == "__main__":
    unittest.main()
