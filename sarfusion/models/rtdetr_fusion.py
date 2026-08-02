import copy
import torch
from torch import nn
from torch.nn import functional as F
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
    def __init__(self, in_channels, freeze=False, spatial_jitter_std=0.0):
        super().__init__()
        self.spatial_jitter_std = spatial_jitter_std
        
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
        
        # Initialize offset to zero so that the deformable conv starts as a standard conv
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
        
        # Stochastic Spatial Jitter (SSJ): Inietta rumore Gaussiano negli offset spaziali
        if self.training and self.spatial_jitter_std > 0.0:
            noise = torch.randn_like(offset) * self.spatial_jitter_std
            offset = offset + noise
        
        # Apply deformable convolution to IR
        ir_aligned = self.deform_conv(ir_feat, offset, mask)
        
        return ir_aligned


class IdentityInitializedFeatureAlignmentModule(FeatureAlignmentModule):
    """
    DCNv2 FAM whose initial mapping is exactly the identity on the IR feature.

    The offset predictor starts at zero, hence all offsets are zero and all
    modulation masks are sigmoid(0) = 0.5.  The deformable-convolution kernel
    is therefore zero everywhere except for a 2 * identity matrix at its
    centre.  The factor 2 compensates the initial 0.5 mask.
    """

    def __init__(self, in_channels, freeze=False, spatial_jitter_std=0.0):
        if spatial_jitter_std != 0.0:
            raise ValueError(
                "Identity-initialized FAM requires spatial_jitter_std=0.0 "
                "to remain an identity before the first optimizer step."
            )
        super().__init__(
            in_channels,
            freeze=False,
            spatial_jitter_std=spatial_jitter_std,
        )
        self.reset_identity_parameters()

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def reset_identity_parameters(self):
        with torch.no_grad():
            self.offset_conv.weight.zero_()
            self.offset_conv.bias.zero_()
            self.deform_conv.weight.zero_()
            channel_indices = torch.arange(
                self.deform_conv.in_channels,
                device=self.deform_conv.weight.device,
            )
            self.deform_conv.weight[channel_indices, channel_indices, 1, 1] = 2.0
            if self.deform_conv.bias is not None:
                self.deform_conv.bias.zero_()


class GridSampleFeatureAlignmentModule(nn.Module):
    """
    Direct IR warp with one learned (dx, dy) displacement per feature pixel.

    Displacements are expressed in feature-map pixels.  A zero-initialized
    predictor combined with an align_corners=False pixel-centre grid makes the
    initial mapping an identity without an additional convolutional filter.
    """

    def __init__(self, in_channels, freeze=False, spatial_jitter_std=0.0):
        super().__init__()
        if spatial_jitter_std != 0.0:
            raise ValueError(
                "Grid-sample FAM requires spatial_jitter_std=0.0 for this "
                "ablation."
            )
        self.spatial_jitter_std = spatial_jitter_std
        self.offset_conv = nn.Conv2d(
            in_channels * 2,
            2,
            kernel_size=3,
            padding=1,
        )
        self.reset_identity_parameters()

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def reset_identity_parameters(self):
        with torch.no_grad():
            self.offset_conv.weight.zero_()
            self.offset_conv.bias.zero_()

    @staticmethod
    def _identity_grid(ir_feat):
        batch_size, _, height, width = ir_feat.shape
        # align_corners=False maps pixel centre j to (2*j + 1)/size - 1.
        y = (
            (2 * torch.arange(height, device=ir_feat.device, dtype=ir_feat.dtype) + 1)
            / height
            - 1
        )
        x = (
            (2 * torch.arange(width, device=ir_feat.device, dtype=ir_feat.dtype) + 1)
            / width
            - 1
        )
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )

    def forward(self, rgb_feat, ir_feat):
        if rgb_feat.shape != ir_feat.shape:
            raise ValueError(
                "Grid-sample FAM requires RGB and IR features with the same "
                f"shape, got {tuple(rgb_feat.shape)} and {tuple(ir_feat.shape)}."
            )

        _, _, height, width = ir_feat.shape
        offsets_px = self.offset_conv(torch.cat([rgb_feat, ir_feat], dim=1))
        offsets = torch.stack(
            (
                offsets_px[:, 0] * (2.0 / width),
                offsets_px[:, 1] * (2.0 / height),
            ),
            dim=-1,
        )
        identity_grid = self._identity_grid(ir_feat)
        grid = identity_grid + offsets
        warped = F.grid_sample(
            ir_feat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        # Even a pixel-centre identity grid can accumulate a few 1e-5 of
        # interpolation error in float32 for some odd spatial sizes. Subtract
        # the same operation on the undeformed grid so that zero offsets are
        # an exact identity while preserving gradients through the learned
        # warp. Once offsets change, this remains a direct spatial warp plus
        # only its deterministic floating-point identity correction.
        identity_warp = F.grid_sample(
            ir_feat,
            identity_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return ir_feat + (warped - identity_warp)


FAM_VARIANTS = {
    "current_dcnv2": FeatureAlignmentModule,
    "identity_dcnv2": IdentityInitializedFeatureAlignmentModule,
    "grid_sample": GridSampleFeatureAlignmentModule,
}


def copy_matching_pretrained_label_heads(target, source, target_id2label):
    """Reuse checkpoint class rows whose semantic labels match the target.

    Hugging Face otherwise randomly reinitializes every class-related RT-DETR
    tensor when changing from COCO's 80 classes to a one-class detector.
    """
    target_ids = sorted(int(index) for index in target_id2label)
    if target_ids != list(range(len(target_ids))):
        raise ValueError("Target label IDs must be contiguous and start at zero")

    source_label_to_id = {
        str(label).casefold(): int(index)
        for index, label in source.config.id2label.items()
    }
    source_indices = []
    for target_id in target_ids:
        label = str(target_id2label[target_id])
        try:
            source_indices.append(source_label_to_id[label.casefold()])
        except KeyError as exc:
            raise ValueError(
                f"Target label {label!r} is not present in the pretrained labels"
            ) from exc

    with torch.no_grad():
        for target_layer, source_layer in zip(
            target.class_embed, source.class_embed
        ):
            for target_id, source_id in enumerate(source_indices):
                target_layer.weight[target_id].copy_(source_layer.weight[source_id])
                target_layer.bias[target_id].copy_(source_layer.bias[source_id])

        for target_id, source_id in enumerate(source_indices):
            target.model.enc_score_head.weight[target_id].copy_(
                source.model.enc_score_head.weight[source_id]
            )
            target.model.enc_score_head.bias[target_id].copy_(
                source.model.enc_score_head.bias[source_id]
            )
            target.model.denoising_class_embed.weight[target_id].copy_(
                source.model.denoising_class_embed.weight[source_id]
            )

        target_padding = target.model.denoising_class_embed.padding_idx
        source_padding = source.model.denoising_class_embed.padding_idx
        target.model.denoising_class_embed.weight[target_padding].copy_(
            source.model.denoising_class_embed.weight[source_padding]
        )

    return source_indices


def build_feature_alignment_module(
    variant,
    in_channels,
    freeze=False,
    spatial_jitter_std=0.0,
):
    try:
        module_class = FAM_VARIANTS[variant]
    except KeyError as exc:
        choices = ", ".join(sorted(FAM_VARIANTS))
        raise ValueError(
            f"Unknown FAM variant {variant!r}. Expected one of: {choices}."
        ) from exc
    return module_class(
        in_channels,
        freeze=freeze,
        spatial_jitter_std=spatial_jitter_std,
    )


# ============================================================
# 2. FUSION BACKBONE (RT-DETR + FAM)
#    - RGB and IR processed separately
#    - FAM aligns IR to RGB
#    - Additive fusion on aligned feature maps
#    - Geometry preserved
# ============================================================
class RTDetrFusionBackbone(nn.Module):
    def __init__(self, config: RTDetrConfig, use_fam: bool = False, freeze_fam: bool = False, ir_dropout_rate: float = 0.0, spatial_jitter_std: float = 0.0, fam_variant: str = "current_dcnv2"):
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
        self.spatial_jitter_std = spatial_jitter_std
        self.fam_variant = fam_variant
        
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
                [
                    build_feature_alignment_module(
                        self.fam_variant,
                        ch,
                        freeze=self.freeze_fam,
                        spatial_jitter_std=self.spatial_jitter_std,
                    )
                    for ch in feature_channels
                ]
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
    def __init__(self, config: RTDetrConfig, use_fam: bool = False, freeze_fam: bool = False, ir_dropout_rate: float = 0.0, spatial_jitter_std: float = 0.0, fam_variant: str = "current_dcnv2"):
        super().__init__(config)
        self.backbone = RTDetrFusionBackbone(config, use_fam=use_fam, freeze_fam=freeze_fam, ir_dropout_rate=ir_dropout_rate, spatial_jitter_std=spatial_jitter_std, fam_variant=fam_variant)
        self.post_init()
        # Hugging Face post_init may visit newly attached modules. Restore the
        # ablation's defining initialization after that global initialization.
        if fam_variant != "current_dcnv2" and self.backbone.fam_modules is not None:
            for fam_module in self.backbone.fam_modules:
                fam_module.reset_identity_parameters()


# ============================================================
# 4. OBJECT DETECTION WRAPPER
# ============================================================
class RTDetrFusionForObjectDetection(RTDetrForObjectDetection):
    def __init__(self, config: RTDetrConfig, use_fam: bool = False, freeze_fam: bool = False, ir_dropout_rate: float = 0.0, spatial_jitter_std: float = 0.0, fam_variant: str = "current_dcnv2"):
        # Trick: initialize as standard RGB model
        tmp_cfg = copy.deepcopy(config)
        tmp_cfg.num_channels = 3
        super().__init__(tmp_cfg)

        # Save original heads
        saved_class_embed = self.class_embed
        saved_bbox_embed = self.bbox_embed

        # Replace the model
        self.model = RTDetrFusionModel(config, use_fam=use_fam, freeze_fam=freeze_fam, ir_dropout_rate=ir_dropout_rate, spatial_jitter_std=spatial_jitter_std, fam_variant=fam_variant)

        # Restore heads in decoder
        self.model.decoder.class_embed = saved_class_embed
        self.model.decoder.bbox_embed = saved_bbox_embed
        self.config = config
        self.use_fam = use_fam
        self.freeze_fam = freeze_fam
        self.ir_dropout_rate = ir_dropout_rate
        self.spatial_jitter_std = spatial_jitter_std
        self.fam_variant = fam_variant

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
        spatial_jitter_std=0.0,
        fam_variant="current_dcnv2",
        reuse_pretrained_class_head=False,
    ):
        if reuse_pretrained_class_head:
            # Load the original label space so matching semantic rows (notably
            # COCO "person") remain available for transfer.
            base = RTDetrForObjectDetection.from_pretrained(
                pretrained_model_name,
            )
        else:
            base = RTDetrForObjectDetection.from_pretrained(
                pretrained_model_name,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )

        config = copy.deepcopy(base.config)
        config.id2label = {int(index): label for index, label in id2label.items()}
        config.label2id = {label: int(index) for index, label in config.id2label.items()}
        config.num_channels = 4
        instance = cls(config, use_fam=use_fam, freeze_fam=freeze_fam, ir_dropout_rate=ir_dropout_rate, spatial_jitter_std=spatial_jitter_std, fam_variant=fam_variant)

        # Load everything (decoder, encoder, etc.)
        base_state = base.state_dict()
        if reuse_pretrained_class_head:
            instance_state = instance.state_dict()
            compatible_state = {
                key: value
                for key, value in base_state.items()
                if key in instance_state and value.shape == instance_state[key].shape
            }
            instance.load_state_dict(compatible_state)
            copy_matching_pretrained_label_heads(instance, base, config.id2label)
        else:
            instance.load_state_dict(base_state)

        # ---- RGB backbone ----
        sd = base_state
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
