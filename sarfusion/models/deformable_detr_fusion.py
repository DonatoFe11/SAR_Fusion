# coding=utf-8
# Copyright 2022 SenseTime and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Deformable DETR Fusion Model for RGB-IR Object Detection.

Simple implementation following the pattern of detr_fusion.py and rtdetr_fusion.py.
"""

import copy
import torch
from torch import nn
from typing import Optional, List, Dict

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
# FUSION BACKBONE - Simple dual backbone for RGB+IR
# ============================================================
class DeformableDetrFusionBackbone(nn.Module):
    """
    Dual backbone for RGB-IR fusion.
    Processes RGB (3ch) and IR (1ch) separately, following detr_fusion.py pattern.
    """
    
    def __init__(self, config: DeformableDetrConfig):
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
        
        # Store channel sizes for projection layers
        self.intermediate_channel_sizes = rgb_backbone.intermediate_channel_sizes
        
        # Store num_feature_levels to limit output
        self.num_feature_levels = config.num_feature_levels
        
        # Projection layers: 2C → C for channel concatenation
        # Simple 1×1 conv to fuse concatenated RGB+IR features
        self.channel_proj = nn.ModuleList([
            nn.Conv2d(channels * 2, channels, kernel_size=1)
            for channels in self.intermediate_channel_sizes
        ])
        
        # Add position_embedding attribute for compatibility with DeformableDetrModel
        # This is called by the parent forward, so we need to expose it
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
            compatible with DeformableDetrModel expectations
        """
        num_channels = pixel_values.shape[1]
        
        rgb_features, ir_features = None, None
        rgb_pos_embeds, ir_pos_embeds = None, None
        
        if num_channels == 1:
            # IR only
            ir_features, ir_pos_embeds = self.ir_backbone(pixel_values, pixel_mask)
            # Limit to num_feature_levels
            return ir_features[:self.num_feature_levels], ir_pos_embeds[:self.num_feature_levels]
        elif num_channels == 3:
            # RGB only
            rgb_features, rgb_pos_embeds = self.rgb_backbone(pixel_values, pixel_mask)
            # Limit to num_feature_levels
            return rgb_features[:self.num_feature_levels], rgb_pos_embeds[:self.num_feature_levels]
        elif num_channels == 4:
            # RGB + IR fusion - CHANNEL CONCATENATION (dim=1)
            rgb_features, rgb_pos_embeds = self.rgb_backbone(pixel_values[:, :3], pixel_mask)
            ir_features, ir_pos_embeds = self.ir_backbone(pixel_values[:, 3:], pixel_mask)
            
            # Limit to num_feature_levels BEFORE concatenation
            rgb_features = rgb_features[:self.num_feature_levels]
            rgb_pos_embeds = rgb_pos_embeds[:self.num_feature_levels]
            ir_features = ir_features[:self.num_feature_levels]
            ir_pos_embeds = ir_pos_embeds[:self.num_feature_levels]
            
            # Weighted fusion: combine RGB and IR with learned weights
            # Ensures both modalities contribute (unlike concatenation which can ignore one)
            fused_features = []
            fused_pos_embeds = []
            for level_idx, ((rgb_feat, rgb_mask), (ir_feat, ir_mask), rgb_pos, ir_pos) in enumerate(zip(
                rgb_features, ir_features, rgb_pos_embeds, ir_pos_embeds
            )):
                # Cat along CHANNEL (dim=1)
                fused_feat = torch.cat([rgb_feat, ir_feat], dim=1)  # [B, 2C, H, W]
                
                # Project back to C channels: [B, 2C, H, W] -> [B, C, H, W]
                fused_feat = self.channel_proj[level_idx](fused_feat)
                
                fused_mask = rgb_mask  # Same spatial dimensions
                # Position embeddings represent spatial locations, not modal-specific info
                # Use RGB position embeddings as they match the fused spatial dimensions
                fused_pos = rgb_pos
                
                fused_features.append((fused_feat, fused_mask))
                fused_pos_embeds.append(fused_pos)
                
            return fused_features, fused_pos_embeds
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}")


# ============================================================
# FUSION MODEL - Simple backbone replacement (no forward override)
# ============================================================
class DeformableDetrFusionModel(DeformableDetrModel):
    """
    Deformable DETR Model with RGB-IR fusion backbone.
    The backbone now returns fused features directly, so no forward override needed.
    """
    
    def __init__(self, config: DeformableDetrConfig):
        super().__init__(config)
        # Replace standard backbone with fusion backbone
        self.backbone = DeformableDetrFusionBackbone(config)
        self.post_init()


# ============================================================
# OBJECT DETECTION HEAD
# ============================================================
class DeformableDetrFusionForObjectDetection(DeformableDetrForObjectDetection):
    """
    Deformable DETR with RGB-IR fusion for object detection.
    Clean implementation following rtdetr_fusion.py pattern.
    """
    
    def __init__(self, config: DeformableDetrConfig):
        # Temporarily set num_channels to 3 for super().__init__
        tmp_cfg = copy.deepcopy(config)
        if not hasattr(tmp_cfg, 'num_channels'):
            tmp_cfg.num_channels = 3
        original_num_channels = getattr(tmp_cfg, 'num_channels', 4)
        tmp_cfg.num_channels = 3
        
        super().__init__(tmp_cfg)
        
        # Replace model with fusion model
        self.model = DeformableDetrFusionModel(config)
        
        # Restore original num_channels
        self.config.num_channels = original_num_channels
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        ignore_mismatched_sizes: bool = True,
        **kwargs,
    ):
        """
        Load pretrained Deformable DETR and adapt for RGB-IR fusion.
        
        Args:
            pretrained_model_name_or_path: HuggingFace model name or path
            id2label: Mapping from class IDs to labels
            label2id: Mapping from labels to class IDs
            ignore_mismatched_sizes: Whether to ignore size mismatches
            
        Returns:
            DeformableDetrFusionForObjectDetection instance with pretrained weights
        """
        # Load base model
        base_model = DeformableDetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            **kwargs,
        )
        
        # Create fusion instance
        config = base_model.config
        config.id2label = id2label
        config.label2id = label2id
        config.num_labels = len(id2label)
        config.num_channels = 4  # RGB + IR
        
        # Reduce multi-scale levels if specified (for speed)
        if 'num_feature_levels' in kwargs:
            config.num_feature_levels = kwargs['num_feature_levels']
            logger.info(f"Using {config.num_feature_levels} feature levels (reduced for speed)")
        
        instance = cls(config)
        
        # Get state dict
        base_state_dict = base_model.state_dict()
        
        # Separate backbone weights
        backbone_weights = {
            k.replace("model.backbone.", ""): v
            for k, v in base_state_dict.items()
            if "model.backbone" in k
        }
        
        # Non-backbone weights (encoder, decoder, heads)
        other_weights = {
            k: v for k, v in base_state_dict.items()
            if "model.backbone" not in k
        }
        
        # Load non-backbone weights
        result = instance.load_state_dict(other_weights, strict=False)
        logger.info(f"Loaded non-backbone weights. Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
        
        # Load RGB backbone
        instance.model.backbone.rgb_backbone.load_state_dict(backbone_weights, strict=False)
        logger.info("Loaded RGB backbone weights")
        
        # Prepare IR backbone weights (adapt from RGB)
        ir_backbone_weights = copy.deepcopy(backbone_weights)
        
        # Adapt first conv: average RGB channels to single IR channel
        for key in list(ir_backbone_weights.keys()):
            weight = ir_backbone_weights[key]
            if weight.dim() == 4 and weight.shape[1] == 3:
                ir_backbone_weights[key] = weight.mean(dim=1, keepdim=True)
                logger.info(f"Adapted {key}: {weight.shape} -> {ir_backbone_weights[key].shape}")
                
        instance.model.backbone.ir_backbone.load_state_dict(ir_backbone_weights, strict=False)
        logger.info("Loaded IR backbone weights (adapted from RGB)")
        
        return instance
