import copy
import torch
from torch import nn
from typing import Optional, List, Tuple, Union

# Import specific paths
from transformers import RTDetrForObjectDetection, RTDetrPreTrainedModel
from transformers.models.rt_detr.configuration_rt_detr import RTDetrConfig
from transformers.models.rt_detr.modeling_rt_detr import (
    RTDetrConvEncoder, 
    RTDetrHybridEncoder, 
    RTDetrDecoder,
    RTDetrModel
)


# 1. FUSION BACKBONE
class RTDetrFusionBackbone(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()
        
        # RGB Config (3 channels)
        rgb_config = copy.deepcopy(config)
        rgb_config.num_channels = 3
        self.rgb_backbone = RTDetrConvEncoder(rgb_config)
        
        # IR Config (1 channel)
        ir_config = copy.deepcopy(config)
        ir_config.num_channels = 1 
        self.ir_backbone = RTDetrConvEncoder(ir_config)
        
        self._adapt_backbone_to_ir()
    
    def _adapt_backbone_to_ir(self):
        """Adapts first IR layer and bypasses Hugging Face checks"""
        for module in self.ir_backbone.modules():
            if hasattr(module, "num_channels"):
                module.num_channels = 1
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                new_conv = nn.Conv2d(1, module.out_channels, module.kernel_size,
                                   module.stride, module.padding, bias=(module.bias is not None))
                with torch.no_grad():
                    new_conv.weight[:] = module.weight.mean(dim=1, keepdim=True)
                    if module.bias is not None: new_conv.bias[:] = module.bias
                module.in_channels = 1
                module.weight = nn.Parameter(new_conv.weight)

    def forward(self, pixel_values: torch.FloatTensor, pixel_mask: Optional[torch.Tensor] = None):
        num_channels = pixel_values.shape[1]
        
        # Case 1: IR only (Modal Dropout)
        if num_channels == 1:
            ir_output = self.ir_backbone(pixel_values, pixel_mask)
            return None, ir_output
        
        # Case 2: RGB only (Modal Dropout)
        elif num_channels == 3:
            rgb_output = self.rgb_backbone(pixel_values, pixel_mask)
            return rgb_output, None
        
        # Case 3: Fusion (4 channels)
        elif num_channels == 4:
            rgb_output = self.rgb_backbone(pixel_values[:, :3], pixel_mask)
            ir_output = self.ir_backbone(pixel_values[:, 3:], pixel_mask)
            return rgb_output, ir_output
        
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}")


# 2. FUSION MODEL
class RTDetrFusionModel(RTDetrModel):
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)
        # Replace standard backbone
        self.backbone = RTDetrFusionBackbone(config)
        
        # Projection layer for fusion (dim=1)
        self.fusion_projections = nn.ModuleList([
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1) 
            for in_ch in [512, 1024, 2048]
        ])
        
        self.post_init()
    
    def forward(self, pixel_values, pixel_mask=None, **kwargs):
        original_forward = self.backbone.forward
        
        def fused_backbone_forward(pv, pm):
            rgb_output, ir_output = original_forward(pv, pm)
            
            # FUSION case
            if rgb_output is not None and ir_output is not None:
                fused_features = []
                for i, ((r_feat, r_mask), (ir_feat, ir_mask)) in enumerate(zip(rgb_output, ir_output)):
                    concat = torch.cat([r_feat, ir_feat], dim=1)
                    fused_feat = self.fusion_projections[i](concat)
                    fused_features.append((fused_feat, r_mask))
                return fused_features
            
            # MODAL DROPOUT case (RGB only or IR only)
            active = rgb_output or ir_output
            fused_features = []
            for i, (feat, mask) in enumerate(active):
                # Simulate concatenation by duplicating the active sensor
                # to avoid breaking the 1x1 projection weights
                concat = torch.cat([feat, feat], dim=1)
                fused_feat = self.fusion_projections[i](concat)
                fused_features.append((fused_feat, mask))
            return fused_features
        
        self.backbone.forward = fused_backbone_forward
        try:
            return super().forward(pixel_values, pixel_mask, **kwargs)
        finally:
            self.backbone.forward = original_forward


# 3. WRAPPER FOR OBJECT DETECTION
class RTDetrFusionForObjectDetection(RTDetrForObjectDetection):
    def __init__(self, config: RTDetrConfig):
        tmp_config = copy.deepcopy(config)
        tmp_config.num_channels = 3
        super().__init__(tmp_config)
        
        # Save original heads
        saved_class_embed = self.class_embed
        saved_bbox_embed = self.bbox_embed
        
        # Swap model
        self.model = RTDetrFusionModel(config)
        
        # Restore heads in the decoder
        self.model.decoder.class_embed = saved_class_embed
        self.model.decoder.bbox_embed = saved_bbox_embed
        self.config = config

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, id2label, label2id, ignore_mismatched_sizes=True):
        standard_model = RTDetrForObjectDetection.from_pretrained(
            pretrained_model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=ignore_mismatched_sizes
        )
        config = standard_model.config
        config.num_channels = 4
        instance = cls(config)
        
        # Load weights with strict=False through our override
        instance.load_state_dict(standard_model.state_dict())
        
        # RGB and IR backbone weights
        state_dict = standard_model.state_dict()
        backbone_dict = {k: v for k, v in state_dict.items() if "model.backbone" in k}
        rgb_w = {k.replace("model.backbone.", ""): v for k, v in backbone_dict.items()}
        instance.model.backbone.rgb_backbone.load_state_dict(rgb_w, strict=False)
        
        ir_w = copy.deepcopy(rgb_w)
        for k in list(ir_w.keys()):
            if "embedder.0" in k and "weight" in k and ir_w[k].dim() == 4 and ir_w[k].shape[1] == 3:
                ir_w[k] = ir_w[k].mean(dim=1, keepdim=True)
        instance.model.backbone.ir_backbone.load_state_dict(ir_w, strict=False)
        
        # Projection initialization (identity mean 0.5+0.5)
        with torch.no_grad():
            for proj in instance.model.fusion_projections:
                out_ch, in_ch_half = proj.weight.shape[0], proj.weight.shape[1] // 2
                identity = torch.eye(out_ch).view(out_ch, out_ch, 1, 1)
                proj.weight[:, :in_ch_half, :, :] = identity * 0.5
                proj.weight[:, in_ch_half:, :, :] = identity * 0.5
        
        return instance