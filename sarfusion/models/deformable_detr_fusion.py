"""
Deformable DETR Fusion Model for RGB-IR Object Detection.

This implementation uses channel concatenation with optional feature alignment (FAM).
"""

import copy
from typing import Optional

import torch
from torch import nn
from torchvision.ops import DeformConv2d

from transformers import DeformableDetrForObjectDetection
from transformers.models.deformable_detr.configuration_deformable_detr import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrConvEncoder,
    DeformableDetrConvModel,
    DeformableDetrModel,
    build_position_encoding,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class FeatureAlignmentModule(nn.Module):
    """
    Feature Alignment Module using Deformable Convolution.
    RGB features guide the alignment of IR features.
    """

    def __init__(self, in_channels, freeze=False, spatial_jitter_std=0.0):
        super().__init__()
        self.spatial_jitter_std = spatial_jitter_std

        self.offset_conv = nn.Conv2d(
            in_channels * 2,
            27,
            kernel_size=3,
            padding=1,
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )

        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, rgb_feat, ir_feat):
        concat = torch.cat([rgb_feat, ir_feat], dim=1)
        out = self.offset_conv(concat)

        offset = out[:, :18, :, :]
        mask = torch.sigmoid(out[:, 18:, :, :])

        if self.training and self.spatial_jitter_std > 0.0:
            noise = torch.randn_like(offset) * self.spatial_jitter_std
            offset = offset + noise

        ir_aligned = self.deform_conv(ir_feat, offset, mask)
        return ir_aligned


class DeformableDetrFusionBackbone(nn.Module):
    """
    Dual backbone for RGB-IR fusion with optional FAM + channel concatenation.
    """

    def __init__(
        self,
        config: DeformableDetrConfig,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
    ):
        super().__init__()

        rgb_config = copy.deepcopy(config)
        if rgb_config.backbone_kwargs is None:
            rgb_config.backbone_kwargs = {}
        rgb_config.backbone_kwargs["in_chans"] = 3

        rgb_backbone = DeformableDetrConvEncoder(rgb_config)
        position_embeddings = build_position_encoding(rgb_config)
        self.rgb_backbone = DeformableDetrConvModel(rgb_backbone, position_embeddings)

        ir_config = copy.deepcopy(config)
        if ir_config.backbone_kwargs is None:
            ir_config.backbone_kwargs = {}
        ir_config.backbone_kwargs["in_chans"] = 1

        ir_backbone = DeformableDetrConvEncoder(ir_config)
        ir_position_embeddings = build_position_encoding(ir_config)
        self.ir_backbone = DeformableDetrConvModel(ir_backbone, ir_position_embeddings)

        self.intermediate_channel_sizes = rgb_backbone.intermediate_channel_sizes
        self.num_feature_levels = config.num_feature_levels

        self.use_fam = use_fam
        self.freeze_fam = freeze_fam
        self.ir_dropout_rate = ir_dropout_rate
        self.spatial_jitter_std = spatial_jitter_std

        if self.ir_dropout_rate > 0.0:
            self.ir_dropout = nn.Dropout2d(p=self.ir_dropout_rate)
        else:
            self.ir_dropout = None

        if self.use_fam:
            self.fam_modules = nn.ModuleList(
                [
                    FeatureAlignmentModule(
                        channels,
                        freeze=self.freeze_fam,
                        spatial_jitter_std=self.spatial_jitter_std,
                    )
                    for channels in self.intermediate_channel_sizes
                ]
            )
        else:
            self.fam_modules = None

        self.channel_fusion = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels * 2, channels, kernel_size=1),
                    nn.GroupNorm(32, channels),
                    nn.ReLU(inplace=True),
                )
                for channels in self.intermediate_channel_sizes
            ]
        )

        self.position_embedding = position_embeddings

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
    ):
        num_channels = pixel_values.shape[1]

        if num_channels == 1:
            ir_features, ir_pos_embeds = self.ir_backbone(pixel_values, pixel_mask)
            return (
                ir_features[: self.num_feature_levels],
                ir_pos_embeds[: self.num_feature_levels],
            )
        if num_channels == 3:
            rgb_features, rgb_pos_embeds = self.rgb_backbone(pixel_values, pixel_mask)
            return (
                rgb_features[: self.num_feature_levels],
                rgb_pos_embeds[: self.num_feature_levels],
            )
        if num_channels == 4:
            rgb_features, rgb_pos_embeds = self.rgb_backbone(
                pixel_values[:, :3], pixel_mask
            )
            ir_features, ir_pos_embeds = self.ir_backbone(
                pixel_values[:, 3:], pixel_mask
            )

            rgb_features = rgb_features[: self.num_feature_levels]
            rgb_pos_embeds = rgb_pos_embeds[: self.num_feature_levels]
            ir_features = ir_features[: self.num_feature_levels]
            ir_pos_embeds = ir_pos_embeds[: self.num_feature_levels]

            fused_features = []
            fused_pos_embeds = []
            for level_idx, (
                (rgb_feat, rgb_mask),
                (ir_feat, _),
                rgb_pos,
                _,
            ) in enumerate(
                zip(rgb_features, ir_features, rgb_pos_embeds, ir_pos_embeds)
            ):
                if self.use_fam:
                    ir_processed = self.fam_modules[level_idx](rgb_feat, ir_feat)
                else:
                    ir_processed = ir_feat

                if self.ir_dropout is not None:
                    ir_processed = self.ir_dropout(ir_processed)

                concat_feat = torch.cat([rgb_feat, ir_processed], dim=1)
                fused_feat = self.channel_fusion[level_idx](concat_feat)

                fused_features.append((fused_feat, rgb_mask))
                fused_pos_embeds.append(rgb_pos)

            return fused_features, fused_pos_embeds

        raise ValueError(f"Unsupported number of channels: {num_channels}")


class DeformableDetrFusionModel(DeformableDetrModel):
    """
    Deformable DETR model with RGB-IR fusion backbone + optional FAM.
    """

    def __init__(
        self,
        config: DeformableDetrConfig,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
    ):
        super().__init__(config)
        self.backbone = DeformableDetrFusionBackbone(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
        )


class DeformableDetrFusionForObjectDetection(DeformableDetrForObjectDetection):
    """
    Deformable DETR model for object detection with RGB-IR fusion + optional FAM.
    """

    def __init__(
        self,
        config: DeformableDetrConfig,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
    ):
        super().__init__(config)
        self.model = DeformableDetrFusionModel(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
        )
        self.use_fam = use_fam
        self.freeze_fam = freeze_fam
        self.ir_dropout_rate = ir_dropout_rate
        self.spatial_jitter_std = spatial_jitter_std
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        id2label,
        label2id,
        ignore_mismatched_sizes=True,
        num_feature_levels=None,
        use_fam=False,
        freeze_fam=False,
        ir_dropout_rate=0.0,
        spatial_jitter_std=0.0,
        **kwargs,
    ):
        base_model = DeformableDetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            **kwargs,
        )

        config = base_model.config
        if num_feature_levels is not None:
            config.num_feature_levels = num_feature_levels

        fusion_model = cls(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
        )

        fusion_model.load_state_dict(base_model.state_dict(), strict=False)

        position_emb_state = {
            k.replace("model.backbone.", ""): v
            for k, v in base_model.state_dict().items()
            if "backbone.position_embedding" in k
        }
        fusion_model.model.backbone.rgb_backbone.position_embedding.load_state_dict(
            position_emb_state, strict=False
        )
        fusion_model.model.backbone.ir_backbone.position_embedding.load_state_dict(
            position_emb_state, strict=False
        )

        rgb_backbone_state = base_model.model.backbone.conv_encoder.state_dict()
        ir_backbone_state = {}
        for key, value in rgb_backbone_state.items():
            if value.dim() == 4 and value.shape[1] == 3:
                ir_backbone_state[key] = value.mean(dim=1, keepdim=True)
            else:
                ir_backbone_state[key] = value

        fusion_model.model.backbone.rgb_backbone.conv_encoder.load_state_dict(
            rgb_backbone_state, strict=False
        )
        fusion_model.model.backbone.ir_backbone.conv_encoder.load_state_dict(
            ir_backbone_state, strict=False
        )

        for fusion_block in fusion_model.model.backbone.channel_fusion:
            conv = fusion_block[0]
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
            gn = fusion_block[1]
            nn.init.ones_(gn.weight)
            nn.init.zeros_(gn.bias)

        return fusion_model
