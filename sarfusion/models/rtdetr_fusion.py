import copy
import torch
from torch import nn
from typing import Optional
from torchvision.ops import DeformConv2d

from transformers import RTDetrForObjectDetection
from transformers.models.rt_detr.configuration_rt_detr import RTDetrConfig
from transformers.models.rt_detr.modeling_rt_detr import (
    RTDetrConvEncoder,
    RTDetrModel,
)

# ============================================================
# 1. FEATURE ALIGNMENT MODULE (FAM)
#   - RGB guides spatial offset prediction for IR
#   - Deformable Conv on IR for explicit alignment
#   - Solves RGB-IR misalignment issue
# ============================================================
class FeatureAlignmentModule(nn.Module):
    """
    Feature Alignment Module using Deformable Convolution.
    RGB features guide the spatial alignment of IR features.
    """
    def __init__(self, in_channels, freeze=False):
        super().__init__()
        
        # Predicts offset and mask for deformable conv
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
            in_channels,  # output same as input
            kernel_size=3,
            padding=1
        )
        
        # Initialize offset to zero (identity mapping initially)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        # Se freeze=True, replichiamo formalmente l'effetto "regolarizzatore"
        # congelando i pesi e impedendo l'aggiornamento dei gradienti.
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        
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
# 2. FUSION BACKBONE (RT-DETR + FAM)
#    - RGB and IR processed separately
#    - FAM aligns IR to RGB
#    - Additive fusion on aligned feature maps
#    - Geometry preserved
# ============================================================
class RTDetrFusionBackbone(nn.Module):
    def __init__(self, config: RTDetrConfig, use_fam: bool = False, freeze_fam: bool = False, ir_dropout_rate: float = 0.0):
        super().__init__()

        # RGB backbone (standard)
        rgb_cfg = copy.deepcopy(config)
        rgb_cfg.num_channels = 3
        self.rgb_backbone = RTDetrConvEncoder(rgb_cfg)

        # IR backbone (1 channel)
        ir_cfg = copy.deepcopy(config)
        ir_cfg.num_channels = 1
        self.ir_backbone = RTDetrConvEncoder(ir_cfg)

        self._adapt_ir_backbone()
        
        # Feature Alignment Modules - optional
        self.use_fam = use_fam
        self.freeze_fam = freeze_fam
        self.ir_dropout_rate = ir_dropout_rate
        
        if self.ir_dropout_rate > 0.0:
            # Dropout2d azzera randomicamente interi canali della feature map (Spatial Dropout)
            self.ir_dropout = nn.Dropout2d(p=self.ir_dropout_rate)
        else:
            self.ir_dropout = None

        if self.use_fam:
            # Build FAM eagerly so its parameters are visible to the optimizer
            # before the first training step.
            feature_channels = getattr(config, "encoder_in_channels", None)
            if feature_channels is None:
                raise ValueError(
                    "RTDetrFusionBackbone requires config.encoder_in_channels when use_fam=True"
                )
            self.fam_modules = nn.ModuleList(
                [FeatureAlignmentModule(ch, freeze=self.freeze_fam) for ch in feature_channels]
            )
        else:
            self.fam_modules = None

    def _adapt_ir_backbone(self):
        """
        Adapts RGB weights to IR case (1 channel)
        by averaging across channels.
        """
        for module in self.ir_backbone.modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                module.weight = nn.Parameter(
                    module.weight.mean(dim=1, keepdim=True)
                )
                module.in_channels = 1
            if hasattr(module, "num_channels"):
                module.num_channels = 1

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ):
        c = pixel_values.shape[1]

        # RGB only
        if c == 3:
            return self.rgb_backbone(pixel_values, pixel_mask)

        # IR only
        if c == 1:
            return self.ir_backbone(pixel_values, pixel_mask)

        # RGB + IR (4 channels)
        if c == 4:
            rgb_feats = self.rgb_backbone(pixel_values[:, :3], pixel_mask)
            ir_feats  = self.ir_backbone(pixel_values[:, 3:], pixel_mask)

            # Mode with FAM (Feature Alignment Module)
            if self.use_fam:
                if len(self.fam_modules) != len(rgb_feats):
                    raise ValueError(
                        f"FAM levels mismatch: got {len(self.fam_modules)} modules for {len(rgb_feats)} feature levels"
                    )

                fused_feats = []
                for idx, ((r_feat, r_mask), (i_feat, _)) in enumerate(zip(rgb_feats, ir_feats)):
                    # Apply FAM to align IR to RGB
                    i_aligned = self.fam_modules[idx](r_feat, i_feat)
                    
                    # Apply Spatial Dropout to IR if active
                    if self.ir_dropout is not None:
                        i_aligned = self.ir_dropout(i_aligned)
                    
                    # Additive fusion on aligned features
                    fused_feats.append((r_feat + i_aligned, r_mask))

                return fused_feats
            
            # Base mode: direct fusion without alignment
            else:
                fused_feats = []
                for (r_feat, r_mask), (i_feat, _) in zip(rgb_feats, ir_feats):
                    i_to_fuse = i_feat
                    # Apply Spatial Dropout to IR if active
                    if self.ir_dropout is not None:
                        i_to_fuse = self.ir_dropout(i_to_fuse)
                        
                    # Simple additive fusion
                    fused_feats.append((r_feat + i_to_fuse, r_mask))
                
                return fused_feats

        raise ValueError(f"Unsupported number of channels: {c}")


# ============================================================
# 3. RT-DETR MODEL (NO forward override!)
# ============================================================
class RTDetrFusionModel(RTDetrModel):
    def __init__(self, config: RTDetrConfig, use_fam: bool = False, freeze_fam: bool = False, ir_dropout_rate: float = 0.0):
        super().__init__(config)
        self.backbone = RTDetrFusionBackbone(config, use_fam=use_fam, freeze_fam=freeze_fam, ir_dropout_rate=ir_dropout_rate)
        self.post_init()


# ============================================================
# 4. OBJECT DETECTION WRAPPER
# ============================================================
class RTDetrFusionForObjectDetection(RTDetrForObjectDetection):
    def __init__(self, config: RTDetrConfig, use_fam: bool = False, freeze_fam: bool = False, ir_dropout_rate: float = 0.0):
        # Trick: initialize as standard RGB model
        tmp_cfg = copy.deepcopy(config)
        tmp_cfg.num_channels = 3
        super().__init__(tmp_cfg)

        # Save original heads
        saved_class_embed = self.class_embed
        saved_bbox_embed = self.bbox_embed

        # Replace the model
        self.model = RTDetrFusionModel(config, use_fam=use_fam, freeze_fam=freeze_fam, ir_dropout_rate=ir_dropout_rate)

        # Restore heads in decoder
        self.model.decoder.class_embed = saved_class_embed
        self.model.decoder.bbox_embed = saved_bbox_embed
        self.config = config
        self.use_fam = use_fam
        self.freeze_fam = freeze_fam
        self.ir_dropout_rate = ir_dropout_rate

    def load_state_dict(self, state_dict, strict=True):
        # Permissive loading (necessary for IR)
        return super().load_state_dict(state_dict, strict=False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name,
        id2label,
        label2id,
        ignore_mismatched_sizes=True,
        use_fam=False,
        freeze_fam=False,
        ir_dropout_rate=0.0,
    ):
        # Standard RT-DETR model
        base = RTDetrForObjectDetection.from_pretrained(
            pretrained_model_name,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

        config = base.config
        config.num_channels = 4
        instance = cls(config, use_fam=use_fam, freeze_fam=freeze_fam, ir_dropout_rate=ir_dropout_rate)

        # Load everything (decoder, encoder, etc.)
        instance.load_state_dict(base.state_dict())

        # ---- RGB backbone ----
        sd = base.state_dict()
        rgb_w = {
            k.replace("model.backbone.", ""): v
            for k, v in sd.items()
            if "model.backbone" in k
        }
        instance.model.backbone.rgb_backbone.load_state_dict(
            rgb_w, strict=False
        )

        # ---- IR backbone (average across channels) ----
        ir_w = copy.deepcopy(rgb_w)
        for k in list(ir_w.keys()):
            if ir_w[k].dim() == 4 and ir_w[k].shape[1] == 3:
                ir_w[k] = ir_w[k].mean(dim=1, keepdim=True)

        instance.model.backbone.ir_backbone.load_state_dict(
            ir_w, strict=False
        )

        return instance
