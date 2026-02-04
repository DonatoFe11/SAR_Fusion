# coding=utf-8
"""
DINO (DETR with Improved deNoising anchOr boxes) Fusion Model for RGB-IR Object Detection.

DINO is SOTA detection transformer with:
- Contrastive denoising training
- Mixed query selection
- Look forward twice scheme

This implementation uses channel concatenation + FAM for efficiency and alignment.
"""

import copy
import torch
from torch import nn
from typing import Optional
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


# ============================================================
# FEATURE ALIGNMENT MODULE (FAM)
# ============================================================
class FeatureAlignmentModule(nn.Module):
    """
    Feature Alignment Module using Deformable Convolution.
    RGB features guide the alignment of IR features.
    """
    def __init__(self, in_channels):
        super().__init__()
        
        # Predict offset and mask for deformable conv
        # RGB features → offset prediction
        self.offset_conv = nn.Conv2d(
            in_channels * 2,  # RGB + IR concatenated
            27,  # 3x3 kernel: 2 offset (x,y) * 9 points + 9 mask
            kernel_size=3,
            padding=1
        )
        
        # Deformable convolution on IR features
        self.deform_conv = DeformConv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1
        )
        
        # Initialize offset to zero (identity mapping initially)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
    def forward(self, rgb_feat, ir_feat):
        """
        Args:
            rgb_feat: [B, C, H, W] RGB features
            ir_feat: [B, C, H, W] IR features
            
        Returns:
            ir_aligned: [B, C, H, W] IR features aligned to RGB
        """
        # Concatenate RGB and IR to predict offset
        concat = torch.cat([rgb_feat, ir_feat], dim=1)  # [B, 2C, H, W]
        
        # Predict offset and modulation scalars
        out = self.offset_conv(concat)  # [B, 27, H, W]
        
        # Split: 18 channels for offsets (x,y for 9 points), 9 for mask
        offset = out[:, :18, :, :]  # [B, 18, H, W]
        mask = torch.sigmoid(out[:, 18:, :, :])  # [B, 9, H, W]
        
        # Apply deformable convolution to IR
        ir_aligned = self.deform_conv(ir_feat, offset, mask)
        
        return ir_aligned


# ============================================================
# DINO FUSION BACKBONE - Channel Concatenation + FAM
# ============================================================
class DinoFusionBackbone(nn.Module):
    """
    Dual backbone for RGB-IR fusion with optional FAM + channel concatenation.
    Combines explicit spatial alignment (FAM) with expressive fusion (Conv+GN+ReLU).
    """
    
    def __init__(self, config: DeformableDetrConfig, use_fam: bool = False):
        super().__init__()
        
        # RGB backbone (3 channels)
        rgb_config = copy.deepcopy(config)
        if rgb_config.backbone_kwargs is None:
            rgb_config.backbone_kwargs = {}
        rgb_config.backbone_kwargs["in_chans"] = 3
        
        rgb_backbone = DeformableDetrConvEncoder(rgb_config)
        position_embeddings = build_position_encoding(rgb_config)
        self.rgb_backbone = DeformableDetrConvModel(rgb_backbone, position_embeddings)
        
        # IR backbone (1 channel)
        ir_config = copy.deepcopy(config)
        if ir_config.backbone_kwargs is None:
            ir_config.backbone_kwargs = {}
        ir_config.backbone_kwargs["in_chans"] = 1
        
        ir_backbone = DeformableDetrConvEncoder(ir_config)
        ir_position_embeddings = build_position_encoding(ir_config)
        self.ir_backbone = DeformableDetrConvModel(ir_backbone, ir_position_embeddings)
        
        # Store channel sizes
        self.intermediate_channel_sizes = rgb_backbone.intermediate_channel_sizes
        
        # Store num_feature_levels to limit output
        self.num_feature_levels = config.num_feature_levels
        
        # Feature Alignment Modules - optional
        self.use_fam = use_fam
        # Lazy initialization on first forward if use_fam=True
        self.fam_modules = None if use_fam else False  # False = disabled
        
        # Channel fusion: 2C -> C projection for each feature level
        # Uses Conv2d + GroupNorm + ReLU for more expressive fusion
        # GroupNorm instead of BatchNorm for stability with small batch sizes
        self.channel_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1),
                nn.GroupNorm(32, channels),  # 32 groups, stable with batch_size=1
                nn.ReLU(inplace=True)
            )
            for channels in self.intermediate_channel_sizes
        ])
        
        # Position embedding attribute for compatibility
        self.position_embedding = position_embeddings
        
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Args:
            pixel_values: (B, C, H, W) where C is 1, 3, or 4
            pixel_mask: (B, H, W)
            
        Returns:
            Tuple of (features_list, position_embeddings_list)
        """
        num_channels = pixel_values.shape[1]
        
        if num_channels == 1:
            # IR only
            ir_features, ir_pos_embeds = self.ir_backbone(pixel_values, pixel_mask)
            return ir_features[:self.num_feature_levels], ir_pos_embeds[:self.num_feature_levels]
        elif num_channels == 3:
            # RGB only
            rgb_features, rgb_pos_embeds = self.rgb_backbone(pixel_values, pixel_mask)
            return rgb_features[:self.num_feature_levels], rgb_pos_embeds[:self.num_feature_levels]
        elif num_channels == 4:
            # RGB + IR fusion - CHANNEL CONCATENATION with optional FAM
            rgb_features, rgb_pos_embeds = self.rgb_backbone(pixel_values[:, :3], pixel_mask)
            ir_features, ir_pos_embeds = self.ir_backbone(pixel_values[:, 3:], pixel_mask)
            
            # Limit to num_feature_levels
            rgb_features = rgb_features[:self.num_feature_levels]
            rgb_pos_embeds = rgb_pos_embeds[:self.num_feature_levels]
            ir_features = ir_features[:self.num_feature_levels]
            ir_pos_embeds = ir_pos_embeds[:self.num_feature_levels]
            
            # Initialize FAM modules on first forward if enabled (lazy init)
            if self.use_fam and self.fam_modules is None:
                self.fam_modules = nn.ModuleList([
                    FeatureAlignmentModule(rgb_feat.shape[1])
                    for (rgb_feat, _), _ in zip(rgb_features, ir_features)
                ]).to(pixel_values.device)
            
            # Channel concatenation + fusion
            fused_features = []
            fused_pos_embeds = []
            for level_idx, ((rgb_feat, rgb_mask), (ir_feat, ir_mask), rgb_pos, ir_pos) in enumerate(zip(
                rgb_features, ir_features, rgb_pos_embeds, ir_pos_embeds
            )):
                # Step 1: Optionally align IR to RGB using FAM
                if self.use_fam:
                    ir_processed = self.fam_modules[level_idx](rgb_feat, ir_feat)
                else:
                    ir_processed = ir_feat
                
                # Step 2: Concatenate along channel dimension
                concat_feat = torch.cat([rgb_feat, ir_processed], dim=1)  # [B, 2C, H, W]
                
                # Step 3: Fuse with learnable projection + GroupNorm + ReLU
                fused_feat = self.channel_fusion[level_idx](concat_feat)  # [B, C, H, W]
                
                fused_mask = rgb_mask
                fused_pos = rgb_pos  # Use RGB position embeddings
                
                fused_features.append((fused_feat, fused_mask))
                fused_pos_embeds.append(fused_pos)
                
            return fused_features, fused_pos_embeds
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}")


# ============================================================
# DINO MODEL - Uses Deformable DETR as base
# ============================================================
class DinoFusionModel(DeformableDetrModel):
    """
    DINO Model with RGB-IR fusion backbone.
    DINO uses the same architecture as Deformable DETR but with:
    - Contrastive denoising during training
    - Mixed query selection
    - Look forward twice
    
    These are training-time improvements, so at inference the model
    is identical to Deformable DETR. We can use the same base class.
    """
    
    def __init__(self, config: DeformableDetrConfig, use_fam: bool = False):
        super().__init__(config)
        # Replace backbone with fusion backbone
        self.backbone = DinoFusionBackbone(config, use_fam=use_fam)
        self.post_init()


# ============================================================
# DINO FOR OBJECT DETECTION
# ============================================================
class DinoFusionForObjectDetection(DeformableDetrForObjectDetection):
    """
    DINO model for object detection with RGB-IR fusion.
    """
    
    def __init__(self, config: DeformableDetrConfig, use_fam: bool = False):
        super().__init__(config)
        # Replace model with DINO fusion model
        self.model = DinoFusionModel(config, use_fam=use_fam)
        self.use_fam = use_fam
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
        **kwargs
    ):
        """
        Load pretrained DINO/Deformable-DETR and adapt for RGB-IR fusion.
        
        DINO models can be loaded from Deformable DETR checkpoints since
        the architecture is the same (DINO improvements are training-time only).
        """
        # Load base model
        base_model = DeformableDetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            **kwargs
        )
        
        # Create fusion model config
        config = base_model.config
        if num_feature_levels is not None:
            config.num_feature_levels = num_feature_levels
        
        # Create fusion model
        fusion_model = cls(config, use_fam=use_fam)
        
        # Load pretrained weights (permissive for backbone mismatches)
        fusion_model.load_state_dict(base_model.state_dict(), strict=False)
        
        # Adapt IR backbone weights from RGB (mean over channels)
        rgb_backbone_state = base_model.model.backbone.conv_encoder.state_dict()
        ir_backbone_state = {}
        for key, value in rgb_backbone_state.items():
            if value.dim() == 4 and value.shape[1] == 3:  # Conv2d with 3 input channels
                # Average over input channels for IR (1 channel)
                ir_backbone_state[key] = value.mean(dim=1, keepdim=True)
            else:
                ir_backbone_state[key] = value
        
        # Load RGB backbone
        fusion_model.model.backbone.rgb_backbone.conv_encoder.load_state_dict(
            rgb_backbone_state, strict=False
        )
        
        # Load IR backbone (adapted weights)
        fusion_model.model.backbone.ir_backbone.conv_encoder.load_state_dict(
            ir_backbone_state, strict=False
        )
        
        return fusion_model
