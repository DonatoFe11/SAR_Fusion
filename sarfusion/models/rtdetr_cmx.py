"""
RT-DETR CMX Model - State-of-the-Art RGB-T Fusion
Logica: Rettifica (CM-FRM) + Cross-Attention (FFM) + Multi-scale (P3, P4, P5).
Vantaggio: Calibra i sensori per eliminare il rumore e usa l'attention per il disallineamento.
"""

import copy
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

from transformers import RTDetrForObjectDetection, RTDetrPreTrainedModel, RTDetrConfig
from transformers.models.rt_detr.modeling_rt_detr import (
    RTDetrConvEncoder, 
    RTDetrHybridEncoder, 
    RTDetrDecoder,
    RTDetrModel
)

# ---------------------------------------------------------
# 1. CM-FRM: Modulo di Rettifica (Calibrazione)
# ---------------------------------------------------------
class CM_FRM(nn.Module):
    """Calibra una modalità usando l'altra (Channel & Spatial wise)"""
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels * 4, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels * 2),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, rgb, ir):
        b, c, h, w = rgb.shape
        # Channel-wise
        rgb_avg = F.adaptive_avg_pool2d(rgb, 1).view(b, -1)
        rgb_max = F.adaptive_max_pool2d(rgb, 1).view(b, -1)
        ir_avg = F.adaptive_avg_pool2d(ir, 1).view(b, -1)
        ir_max = F.adaptive_max_pool2d(ir, 1).view(b, -1)
        
        combined_stats = torch.cat([rgb_avg, rgb_max, ir_avg, ir_max], dim=1)
        channel_weights = self.mlp(combined_stats)
        w_rgb, w_ir = torch.split(channel_weights, c, dim=1)
        
        # Rettifica (Eq. 2 & 5 del paper)
        rgb_c, ir_c = rgb * w_ir.view(b, c, 1, 1), ir * w_rgb.view(b, c, 1, 1)

        # Spatial-wise
        spatial_feat = torch.cat([rgb, ir], dim=1)
        spatial_weights = self.spatial_conv(spatial_feat)
        ws_rgb, ws_ir = torch.split(spatial_weights, 1, dim=1)
        
        # Output rettificato (informazione pulita)
        return rgb + 0.5 * rgb_c + 0.5 * (rgb * ws_ir), ir + 0.5 * ir_c + 0.5 * (ir * ws_rgb)

# ---------------------------------------------------------
# 2. FFM: Modulo di Fusione (Cross-Attention)
# ---------------------------------------------------------
class FFM(nn.Module):
    """Fonde le feature usando Attention ottimizzata (Scaled Dot Product)"""
    def __init__(self, channels):
        super().__init__()
        self.num_heads = 4
        self.head_dim = channels // self.num_heads
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, rgb, ir):
        b, c, h, w = rgb.shape
        # Se la risoluzione è P3 (80x80), l'attenzione è troppo lenta, usiamo fusione rapida
        if h * w > 1600:
            return self.out_conv(torch.cat([rgb, ir], dim=1))
        
        # Multi-head Cross-Attention (RGB interroga IR)
        q = self.q_proj(rgb).view(b, self.num_heads, self.head_dim, -1).transpose(2, 3)
        k = self.k_proj(ir).view(b, self.num_heads, self.head_dim, -1).transpose(2, 3)
        v = self.v_proj(ir).view(b, self.num_heads, self.head_dim, -1).transpose(2, 3)
        
        fused_ir = F.scaled_dot_product_attention(q, k, v)
        fused_ir = fused_ir.transpose(2, 3).contiguous().view(b, c, h, w)
        
        return self.out_conv(torch.cat([rgb, fused_ir], dim=1))

# ---------------------------------------------------------
# 3. CMX BACKBONE
# ---------------------------------------------------------
class RTDetrCMXBackbone(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()
        # Config separate per evitare ValueError canali
        rgb_c = copy.deepcopy(config); rgb_c.num_channels = 3
        if hasattr(rgb_c, "backbone_config") and rgb_c.backbone_config: rgb_c.backbone_config.num_channels = 3
        self.rgb_backbone = RTDetrConvEncoder(rgb_c)
        
        ir_c = copy.deepcopy(config); ir_c.num_channels = 1
        if hasattr(ir_c, "backbone_config") and ir_c.backbone_config: ir_c.backbone_config.num_channels = 1
        self.ir_backbone = RTDetrConvEncoder(ir_c)
        
        self._adapt_ir()
        # Moduli per le 3 scale di RT-DETR (512, 1024, 2048 canali)
        self.rectifiers = nn.ModuleList([CM_FRM(ch) for ch in [512, 1024, 2048]])
        self.fusers = nn.ModuleList([FFM(ch) for ch in [512, 1024, 2048]])

    def _adapt_ir(self):
        for m in self.ir_backbone.modules():
            if hasattr(m, "num_channels"): m.num_channels = 1
            if isinstance(m, nn.Conv2d) and m.in_channels == 3:
                m.weight = nn.Parameter(m.weight.mean(dim=1, keepdim=True))
                m.in_channels = 1

    def forward(self, pixel_values, pixel_mask=None):
        num_ch = pixel_values.shape[1]
        b, _, h, w = pixel_values.shape
        device = pixel_values.device

        # 1. Estrazione delle feature dalle Backbone
        if num_ch == 4:
            rgb_o = self.rgb_backbone(pixel_values[:, :3], pixel_mask)
            ir_o = self.ir_backbone(pixel_values[:, 3:], pixel_mask)
        elif num_ch == 3:
            rgb_o = self.rgb_backbone(pixel_values, pixel_mask)
            # Creiamo un "finto" IR di zeri per passare attraverso i moduli CMX
            ir_o = [(torch.zeros_like(f), m) for f, m in rgb_o]
        elif num_ch == 1:
            ir_o = self.ir_backbone(pixel_values, pixel_mask)
            # Creiamo un "finto" RGB di zeri
            rgb_o = [(torch.zeros_like(f), m) for f, m in ir_o]
        else:
            raise ValueError(f"Canali non supportati: {num_ch}")

        # 2. Passaggio OBBLIGATORIO attraverso i moduli CMX
        # Ora i Rectifiers e i Fusers imparano a gestire anche i sensori mancanti
        fused = []
        for i, ((r_f, r_m), (i_f, _)) in enumerate(zip(rgb_o, ir_o)):
            # Rettifica (Calibrazione reciproca)
            r_rect, i_rect = self.rectifiers[i](r_f, i_f)
            # Fusione (Cross-Attention)
            f_feat = self.fusers[i](r_rect, i_rect)
            fused.append((f_feat, r_m))
            
        return fused

# ---------------------------------------------------------
# 4. WRAPPER DETR.PY
# ---------------------------------------------------------
class RTDetrCMXModel(RTDetrModel):
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)
        self.backbone = RTDetrCMXBackbone(config)
        self.post_init()

class RTDetrCMXForObjectDetection(RTDetrForObjectDetection):
    def __init__(self, config: RTDetrConfig):
        tmp = copy.deepcopy(config); tmp.num_channels = 3
        super().__init__(tmp)
        s_cls, s_box = self.class_embed, self.bbox_embed
        self.model = RTDetrCMXModel(config)
        self.model.decoder.class_embed, self.model.decoder.bbox_embed = s_cls, s_box
        self.config.num_channels = 4
        self.post_init()

    def load_state_dict(self, sd, strict=True): return super().load_state_dict(sd, strict=False)

    @classmethod
    def from_pretrained(cls, name, id2label, label2id, **kwargs):
        std = RTDetrForObjectDetection.from_pretrained(name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
        instance = cls(std.config)
        instance.load_state_dict(std.state_dict())
        # Caricamento Backbone
        rgb_w = {k.replace("model.backbone.", ""): v for k, v in std.state_dict().items() if "model.backbone" in k}
        instance.model.backbone.rgb_backbone.load_state_dict(rgb_w, strict=False)

        # ---- IR backbone (media sui canali per il primo layer) ----
        ir_w = copy.deepcopy(rgb_w)
        for k_pt in list(ir_w.keys()):
            # Se è il primo layer convoluzionale (3 in -> N out), facciamo la media
            if ir_w[k_pt].dim() == 4 and ir_w[k_pt].shape[1] == 3:
                ir_w[k_pt] = ir_w[k_pt].mean(dim=1, keepdim=True)
                print(f"✅ Converted IR stem: {k_pt}")

        instance.model.backbone.ir_backbone.load_state_dict(ir_w, strict=False)
        return instance