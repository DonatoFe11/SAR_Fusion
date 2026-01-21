import copy
import torch
from torch import nn
from typing import Optional

from transformers import RTDetrForObjectDetection
from transformers.models.rt_detr.configuration_rt_detr import RTDetrConfig
from transformers.models.rt_detr.modeling_rt_detr import (
    RTDetrConvEncoder,
    RTDetrModel,
)

# ============================================================
# 1. FUSION BACKBONE (RT-DETR SAFE)
#    - RGB e IR processati separatamente
#    - Fusione ADDITIVA sulle feature map
#    - Geometria preservata
# ============================================================
class RTDetrFusionBackbone(nn.Module):
    def __init__(self, config: RTDetrConfig):
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

            fused_feats = []
            for (r_feat, r_mask), (i_feat, _) in zip(rgb_feats, ir_feats):
                fused_feats.append((r_feat + i_feat, r_mask))

            return fused_feats

        raise ValueError(f"Unsupported number of channels: {c}")


# ============================================================
# 2. RT-DETR MODEL (NO forward override!)
# ============================================================
class RTDetrFusionModel(RTDetrModel):
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)
        self.backbone = RTDetrFusionBackbone(config)
        self.post_init()


# ============================================================
# 3. OBJECT DETECTION WRAPPER
# ============================================================
class RTDetrFusionForObjectDetection(RTDetrForObjectDetection):
    def __init__(self, config: RTDetrConfig):
        # Trick: inizializziamo come RGB standard
        tmp_cfg = copy.deepcopy(config)
        tmp_cfg.num_channels = 3
        super().__init__(tmp_cfg)

        # Salviamo le teste originali
        saved_class_embed = self.class_embed
        saved_bbox_embed = self.bbox_embed

        # Sostituiamo il modello
        self.model = RTDetrFusionModel(config)

        # Ripristiniamo le teste nel decoder
        self.model.decoder.class_embed = saved_class_embed
        self.model.decoder.bbox_embed = saved_bbox_embed
        self.config = config

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
        instance = cls(config)

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
