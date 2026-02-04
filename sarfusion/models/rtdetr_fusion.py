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
#    - RGB guida la predizione degli offset spaziali
#    - Deformable Conv su IR per allineamento esplicito
#    - Risolve il problema del misalignment RGB-IR
# ============================================================
class FeatureAlignmentModule(nn.Module):
    """
    Feature Alignment Module usando Deformable Convolution.
    RGB features guidano l'allineamento di IR features.
    """
    def __init__(self, in_channels):
        super().__init__()
        
        # Predice offset e mask per deformable conv
        # RGB features → offset prediction
        self.offset_conv = nn.Conv2d(
            in_channels * 2,  # RGB + IR concatenati
            27,  # 3x3 kernel: 2 offset (x,y) * 9 punti + 9 mask
            kernel_size=3,
            padding=1
        )
        
        # Deformable convolution su IR features
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
            ir_aligned: [B, C, H, W] IR features allineate a RGB
        """
        # Concatena RGB e IR per predire offset
        concat = torch.cat([rgb_feat, ir_feat], dim=1)  # [B, 2C, H, W]
        
        # Predici offset e modulation scalars
        out = self.offset_conv(concat)  # [B, 27, H, W]
        
        # Split: 18 channels for offsets (x,y for 9 points), 9 for mask
        offset = out[:, :18, :, :]  # [B, 18, H, W]
        mask = torch.sigmoid(out[:, 18:, :, :])  # [B, 9, H, W]
        
        # Apply deformable convolution to IR
        ir_aligned = self.deform_conv(ir_feat, offset, mask)
        
        return ir_aligned


# ============================================================
# 2. FUSION BACKBONE (RT-DETR + FAM)
#    - RGB e IR processati separatamente
#    - FAM allinea IR a RGB
#    - Fusione ADDITIVA sulle feature map allineate
#    - Geometria preservata
# ============================================================
class RTDetrFusionBackbone(nn.Module):
    def __init__(self, config: RTDetrConfig, use_fam: bool = False):
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
        
        # Feature Alignment Modules - opzionali
        self.use_fam = use_fam
        # Inizializzati lazy al primo forward se use_fam=True
        # Verranno creati quando conosciamo i canali effettivi
        self.fam_modules = None if use_fam else False  # False = disabled

    def _adapt_ir_backbone(self):
        """
        Adatta i pesi RGB al caso IR (1 canale)
        facendo la media sui canali.
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

            # Modalità con FAM (Feature Alignment Module)
            if self.use_fam:
                # Initialize FAM modules on first forward (lazy init)
                if self.fam_modules is None:
                    self.fam_modules = nn.ModuleList([
                        FeatureAlignmentModule(r_feat.shape[1])
                        for (r_feat, _), _ in zip(rgb_feats, ir_feats)
                    ]).to(pixel_values.device)

                fused_feats = []
                for idx, ((r_feat, r_mask), (i_feat, _)) in enumerate(zip(rgb_feats, ir_feats)):
                    # Apply FAM to align IR to RGB
                    i_aligned = self.fam_modules[idx](r_feat, i_feat)
                    
                    # Additive fusion on aligned features
                    fused_feats.append((r_feat + i_aligned, r_mask))

                return fused_feats
            
            # Modalità base: fusione diretta senza allineamento
            else:
                fused_feats = []
                for (r_feat, r_mask), (i_feat, _) in zip(rgb_feats, ir_feats):
                    # Simple additive fusion
                    fused_feats.append((r_feat + i_feat, r_mask))
                
                return fused_feats

        raise ValueError(f"Unsupported number of channels: {c}")


# ============================================================
# 2. RT-DETR MODEL (NO forward override!)
# ============================================================
class RTDetrFusionModel(RTDetrModel):
    def __init__(self, config: RTDetrConfig, use_fam: bool = False):
        super().__init__(config)
        self.backbone = RTDetrFusionBackbone(config, use_fam=use_fam)
        self.post_init()


# ============================================================
# 3. OBJECT DETECTION WRAPPER
# ============================================================
class RTDetrFusionForObjectDetection(RTDetrForObjectDetection):
    def __init__(self, config: RTDetrConfig, use_fam: bool = False):
        # Trick: inizializziamo come RGB standard
        tmp_cfg = copy.deepcopy(config)
        tmp_cfg.num_channels = 3
        super().__init__(tmp_cfg)

        # Salviamo le teste originali
        saved_class_embed = self.class_embed
        saved_bbox_embed = self.bbox_embed

        # Sostituiamo il modello
        self.model = RTDetrFusionModel(config, use_fam=use_fam)

        # Ripristiniamo le teste nel decoder
        self.model.decoder.class_embed = saved_class_embed
        self.model.decoder.bbox_embed = saved_bbox_embed
        self.config = config
        self.use_fam = use_fam

    def load_state_dict(self, state_dict, strict=True):
        # Caricamento permissivo (necessario per IR)
        return super().load_state_dict(state_dict, strict=False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name,
        id2label,
        label2id,
        ignore_mismatched_sizes=True,
        use_fam=False,
    ):
        # Modello RT-DETR standard
        base = RTDetrForObjectDetection.from_pretrained(
            pretrained_model_name,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

        config = base.config
        config.num_channels = 4
        instance = cls(config, use_fam=use_fam)

        # Carichiamo tutto (decoder, encoder, ecc.)
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

        # ---- IR backbone (media sui canali) ----
        ir_w = copy.deepcopy(rgb_w)
        for k in list(ir_w.keys()):
            if ir_w[k].dim() == 4 and ir_w[k].shape[1] == 3:
                ir_w[k] = ir_w[k].mean(dim=1, keepdim=True)

        instance.model.backbone.ir_backbone.load_state_dict(
            ir_w, strict=False
        )

        return instance
