"""
RT-DETR Hybrid Model (FAM + CMX + Positional Encoding)
Logica: 
1) Allinea spazialmente (FAM Deformable Convolutions).
2) Calibra sensori per eliminare il rumore (CM-FRM).
3) Fonde i segnali usando Attenzione 2D (FFM) con Bypass protettivo P3 per i target SAR.
"""

import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from transformers import RTDetrForObjectDetection, RTDetrConfig
from transformers.models.rt_detr.modeling_rt_detr import (
    RTDetrConvEncoder, 
    RTDetrModel
)

# ---------------------------------------------------------
# 1. POSITIONAL ENCODING
# ---------------------------------------------------------
class SinePositionalEncoding2D(nn.Module):
    """Genera i token posizionali 2D per dare cognizione spaziale alla Cross-Attention"""
    def __init__(self, temperature=10000):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        b, c, h, w = x.shape
        y_embed = torch.arange(1, h + 1, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(1, w + 1, dtype=torch.float32, device=x.device)
        y_embed = y_embed.unsqueeze(1).expand(h, w)
        x_embed = x_embed.unsqueeze(0).expand(h, w)
        
        eps = 1e-6
        y_embed = y_embed / (y_embed[-1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, -1:] + eps) * 2 * math.pi

        num_pos_feats = c // 2
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed.unsqueeze(2) / dim_t
        pos_y = y_embed.unsqueeze(2) / dim_t

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        pos = pos.to(dtype=x.dtype)
        return pos

# ---------------------------------------------------------
# 2. FEATURE ALIGNMENT MODULE (FAM)
# ---------------------------------------------------------
class FeatureAlignmentModule(nn.Module):
    """Allinea spazialmente l'IR rispetto all'RGB usando Deformable Convolutions"""
    def __init__(self, in_channels):
        super().__init__()
        # Predice offset e mask (2x9 + 9 = 27 canali) a partire dalla concatenazione
        self.offset_conv = nn.Conv2d(in_channels * 2, 27, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Inizializza a 0 (All'inizio l'operazione fa passare i dati identici)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
    def forward(self, rgb_feat, ir_feat):
        rgb_feat = torch.nan_to_num(rgb_feat, nan=0.0, posinf=1e4, neginf=-1e4)
        ir_feat = torch.nan_to_num(ir_feat, nan=0.0, posinf=1e4, neginf=-1e4)
        concat = torch.cat([rgb_feat, ir_feat], dim=1)
        out = self.offset_conv(concat)
        # Bounding offsets helps DeformConv remain stable in fp16 training.
        offset = 4.0 * torch.tanh(out[:, :18, :, :])
        mask = torch.sigmoid(out[:, 18:, :, :]).clamp(1e-4, 1 - 1e-4)
        ir_aligned = self.deform_conv(ir_feat, offset, mask)
        return torch.nan_to_num(ir_aligned, nan=0.0, posinf=1e4, neginf=-1e4)

# ---------------------------------------------------------
# 3. CM-FRM: Modulo di Rettifica (Calibrazione)
# ---------------------------------------------------------
class CM_FRM(nn.Module):
    """Calibra una modalità usando l'altra (Channel & Spatial wise) limitando le interferenze"""
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
        rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1e4, neginf=-1e4)
        ir = torch.nan_to_num(ir, nan=0.0, posinf=1e4, neginf=-1e4)
        b, c, h, w = rgb.shape
        # Channel-wise
        rgb_avg = F.adaptive_avg_pool2d(rgb, 1).view(b, -1)
        rgb_max = F.adaptive_max_pool2d(rgb, 1).view(b, -1)
        ir_avg = F.adaptive_avg_pool2d(ir, 1).view(b, -1)
        ir_max = F.adaptive_max_pool2d(ir, 1).view(b, -1)
        
        combined_stats = torch.cat([rgb_avg, rgb_max, ir_avg, ir_max], dim=1)
        channel_weights = self.mlp(combined_stats)
        w_rgb, w_ir = torch.split(channel_weights, c, dim=1)
        
        rgb_c, ir_c = rgb * w_ir.view(b, c, 1, 1), ir * w_rgb.view(b, c, 1, 1)

        # Spatial-wise
        spatial_feat = torch.cat([rgb, ir], dim=1)
        spatial_weights = self.spatial_conv(spatial_feat)
        ws_rgb, ws_ir = torch.split(spatial_weights, 1, dim=1)
        
        rgb_out = rgb + 0.5 * rgb_c + 0.5 * (rgb * ws_ir)
        ir_out = ir + 0.5 * ir_c + 0.5 * (ir * ws_rgb)
        return (
            torch.nan_to_num(rgb_out, nan=0.0, posinf=1e4, neginf=-1e4),
            torch.nan_to_num(ir_out, nan=0.0, posinf=1e4, neginf=-1e4),
        )

# ---------------------------------------------------------
# 4. FFM: Modulo di Fusione (Cross-Attention Posizionale)
# ---------------------------------------------------------
class FFM(nn.Module):
    """Fonde le feature usando Attention Posizionale e limitazione OOM su P3"""
    def __init__(self, channels):
        super().__init__()
        self.num_heads = 4
        self.head_dim = channels // self.num_heads
        
        self.pos_encoder = SinePositionalEncoding2D()
        self.q_norm = nn.GroupNorm(32, channels)
        self.k_norm = nn.GroupNorm(32, channels)
        self.v_norm = nn.GroupNorm(32, channels)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_conv = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, rgb, ir):
        rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1e4, neginf=-1e4)
        ir = torch.nan_to_num(ir, nan=0.0, posinf=1e4, neginf=-1e4)
        b, c, h, w = rgb.shape
        
        # Bypass P3 OOM: Se i target sono >1600 usa solo convoluzione 
        # (Ora funziona stupendamente perchè il FAM prima ha allineato le feature!)
        if h * w > 1600:
            out = self.out_conv(torch.cat([rgb, ir], dim=1))
            return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Inietto le coordinate spaziali 2D nei tensori
        pos = self.pos_encoder(rgb)
        rgb_pos = self.q_norm(rgb + pos)
        ir_pos = self.k_norm(ir + pos)
        ir_val = self.v_norm(ir)

        # Q (Cerca) basate su RGB+pos. K (Offrono chiave) basate su IR+pos. V (Offrono dati) basate su IR.
        q = self.q_proj(rgb_pos).view(b, self.num_heads, self.head_dim, -1).transpose(2, 3)
        k = self.k_proj(ir_pos).view(b, self.num_heads, self.head_dim, -1).transpose(2, 3)
        v = self.v_proj(ir_val).view(b, self.num_heads, self.head_dim, -1).transpose(2, 3)
        
        # Compute attention in float32 for numerical stability under AMP.
        fused_ir = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        fused_ir = fused_ir.to(dtype=rgb.dtype)
        fused_ir = fused_ir.transpose(2, 3).contiguous().view(b, c, h, w)
        
        out = self.out_conv(torch.cat([rgb, fused_ir], dim=1))
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

# ---------------------------------------------------------
# 5. HYBRID BACKBONE INTERFACE
# ---------------------------------------------------------
class RTDetrCMXHybridBackbone(nn.Module):
    def __init__(self, config: RTDetrConfig):
        super().__init__()
        rgb_c = copy.deepcopy(config); rgb_c.num_channels = 3
        if hasattr(rgb_c, "backbone_config") and rgb_c.backbone_config: rgb_c.backbone_config.num_channels = 3
        self.rgb_backbone = RTDetrConvEncoder(rgb_c)
        
        ir_c = copy.deepcopy(config); ir_c.num_channels = 1
        if hasattr(ir_c, "backbone_config") and ir_c.backbone_config: ir_c.backbone_config.num_channels = 1
        self.ir_backbone = RTDetrConvEncoder(ir_c)
        
        self._adapt_ir()
        
        # Pipeline Ibrida a 3 step: Align -> Rectify -> Fuse
        self.aligners = nn.ModuleList([FeatureAlignmentModule(ch) for ch in [512, 1024, 2048]])
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
        
        if num_ch == 4:
            rgb_o = self.rgb_backbone(pixel_values[:, :3], pixel_mask)
            ir_o = self.ir_backbone(pixel_values[:, 3:], pixel_mask)
        elif num_ch == 3:
            rgb_o = self.rgb_backbone(pixel_values, pixel_mask)
            ir_o = [(torch.zeros_like(f), m) for f, m in rgb_o]
        elif num_ch == 1:
            ir_o = self.ir_backbone(pixel_values, pixel_mask)
            rgb_o = [(torch.zeros_like(f), m) for f, m in ir_o]
        else:
            raise ValueError(f"Canali non supportati: {num_ch}")

        fused = []
        for i, ((r_f, r_m), (i_f, _)) in enumerate(zip(rgb_o, ir_o)):
            r_f = torch.nan_to_num(r_f, nan=0.0, posinf=1e4, neginf=-1e4)
            i_f = torch.nan_to_num(i_f, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # 1. Trazione e allineamento (Supera la Parallasse)
            i_aligned = self.aligners[i](r_f, i_f)
            
            # 2. Rettifica con pixel ora sovrapponibili (Evita interferenza distruttiva)
            r_rect, i_rect = self.rectifiers[i](r_f, i_aligned)
            
            # 3. Fusione via Positional Cross-Attention (Evita Ghosting) & Convolutional Fallback P3 (Evita OOM)
            f_feat = self.fusers[i](r_rect, i_rect)
            f_feat = torch.nan_to_num(f_feat, nan=0.0, posinf=1e4, neginf=-1e4)
            
            fused.append((f_feat, r_m))
            
        return fused

# ---------------------------------------------------------
# 6. WRAPPER DETR HYBRID
# ---------------------------------------------------------
class RTDetrCMXHybridModel(RTDetrModel):
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)
        self.backbone = RTDetrCMXHybridBackbone(config)
        self.post_init()

class RTDetrCMXHybridForObjectDetection(RTDetrForObjectDetection):
    def __init__(self, config: RTDetrConfig):
        tmp = copy.deepcopy(config); tmp.num_channels = 3
        super().__init__(tmp)
        s_cls, s_box = self.class_embed, self.bbox_embed
        self.model = RTDetrCMXHybridModel(config)
        self.model.decoder.class_embed, self.model.decoder.bbox_embed = s_cls, s_box
        self.config.num_channels = 4
        self.post_init()

    def load_state_dict(self, sd, strict=True): return super().load_state_dict(sd, strict=False)

    @classmethod
    def from_pretrained(cls, name, id2label, label2id, **kwargs):
        std = RTDetrForObjectDetection.from_pretrained(name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
        instance = cls(std.config)
        instance.load_state_dict(std.state_dict())
        
        rgb_w = {k.replace("model.backbone.", ""): v for k, v in std.state_dict().items() if "model.backbone" in k}
        instance.model.backbone.rgb_backbone.load_state_dict(rgb_w, strict=False)

        ir_w = copy.deepcopy(rgb_w)
        for k_pt in list(ir_w.keys()):
            if ir_w[k_pt].dim() == 4 and ir_w[k_pt].shape[1] == 3:
                ir_w[k_pt] = ir_w[k_pt].mean(dim=1, keepdim=True)
                print(f"✅ Converted IR stem: {k_pt}")

        instance.model.backbone.ir_backbone.load_state_dict(ir_w, strict=False)
        return instance