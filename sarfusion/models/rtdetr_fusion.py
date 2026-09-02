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
        
    def forward(self, rgb_feat, ir_feat, both_present=None):
        """
        Args:
            rgb_feat: [B, C, H, W] RGB features
            ir_feat: [B, C, H, W] IR features
            
        Returns:
            ir_aligned: [B, C, H, W] IR features aligned to RGB
        """
        del both_present

        # Concatenate RGB and IR to predict offset
        concat = torch.cat([rgb_feat, ir_feat], dim=1)  # [B, 2C, H, W]
        
        # Predict offset and modulation scalars
        out = self.offset_conv(concat)  # [B, 27, H, W]
        
        # Split: 18 channels for offsets (x,y for 9 points), 9 for mask
        offset = self.transform_offset(out[:, :18, :, :])  # [B, 18, H, W]
        mask = torch.sigmoid(out[:, 18:, :, :])  # [B, 9, H, W]
        
        # Stochastic Spatial Jitter (SSJ): Inietta rumore Gaussiano negli offset spaziali
        if self.training and self.spatial_jitter_std > 0.0:
            noise = torch.randn_like(offset) * self.spatial_jitter_std
            offset = offset + noise
        
        # Apply deformable convolution to IR
        ir_aligned = self.deform_conv(ir_feat, offset, mask)
        
        return ir_aligned

    def transform_offset(self, offset):
        """Map raw predictor outputs to DCNv2 offsets in feature-map cells."""
        return offset


class BoundedFeatureAlignmentModule(FeatureAlignmentModule):
    """DCNv2 FAM with smoothly bounded sampling offsets.

    Four feature-map cells were fixed before training from the completed
    five-checkpoint audit: it lies above the largest per-sample P90 observed
    in the non-pathological P5 checkpoints (2.81 cells), while ruling out the
    hundreds-of-cells failure mode. Dividing by the limit inside tanh keeps a
    unit derivative at zero, so small raw offsets retain the original FAM's
    local parameterization.
    """

    OFFSET_LIMIT_CELLS = 4.0

    def transform_offset(self, offset):
        limit = self.OFFSET_LIMIT_CELLS
        return limit * torch.tanh(offset / limit)


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

    def forward(self, rgb_feat, ir_feat, both_present=None):
        del both_present
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


class BoxGuidedCommonOffsetFeatureAlignmentModule(FeatureAlignmentModule):
    """P3 FAM with a weakly supervised common-offset guidance branch.

    The historical FAM remains the task-adaptive residual: it still predicts
    18 unconstrained DCNv2 offsets and 9 masks from the raw RGB--IR features.
    A compact shared projection additionally predicts one coarse ``(dy, dx)``
    displacement per location.  That displacement is added to every one of
    the nine DCNv2 sampling points::

        offset_k = residual_offset_k + common_offset

    The last guidance convolution is zero-initialized, so the new branch is
    exactly neutral before optimization.  During Stage A it is supervised
    sparsely at mutually matched VIS/IR box centres; the original detection
    loss can still shape the residual offsets without being forced to perform
    literal image registration.

    Guidance is bounded to four feature cells, but the total DCNv2 offset is
    deliberately not bounded: the previous bounded-total-offset ablation was
    already negative.  The branch is suppressed sample-wise when either input
    modality is absent under Modal Dropout.
    """

    COMMON_CHANNELS = 32
    GUIDANCE_LIMIT_CELLS = 4.0

    def __init__(self, in_channels, freeze=False, spatial_jitter_std=0.0):
        if spatial_jitter_std != 0.0:
            raise ValueError(
                "Box-guided common-offset FAM must be evaluated without SSJ"
            )
        super().__init__(
            in_channels,
            freeze=False,
            spatial_jitter_std=spatial_jitter_std,
        )
        common_channels = self.COMMON_CHANNELS
        # Constructing an extra branch must not advance the global RNG relative
        # to the historical FAM.  Otherwise a nominally identical model seed
        # changes the initialization of the shared P4/P5 FAMs and the RT-DETR
        # training RNG, confounding the architectural comparison.  The new
        # layers receive deterministic ordinary PyTorch initialization inside
        # the fork, while the caller's RNG state is restored afterwards.
        with torch.random.fork_rng(devices=[]):
            self.common_projection = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    common_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(8, common_channels),
                nn.SiLU(),
            )
            # The same projection is applied to both modalities. Difference
            # and product terms expose agreement while retaining both inputs.
            self.guidance_predictor = nn.Sequential(
                nn.Conv2d(
                    common_channels * 4,
                    common_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.SiLU(),
                nn.Conv2d(common_channels, 2, kernel_size=1),
            )
            self.reset_guidance_parameters()

        # RTDetrPreTrainedModel.post_init() reinitializes every nn.Conv2d it
        # discovers.  Mark only the newly constructed subtree as initialized;
        # without this, post_init would consume extra random draws and defeat
        # the fork above.  Shared historical FAM layers remain untouched by
        # this marker and follow exactly their original initialization path.
        for new_subtree in (self.common_projection, self.guidance_predictor):
            for module in new_subtree.modules():
                module._is_hf_initialized = True
        self.last_guidance_flow = None

        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def reset_guidance_parameters(self):
        """Restore exact-neutral guidance after Hugging Face ``post_init``."""
        final_conv = self.guidance_predictor[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    @staticmethod
    def _presence_scale(both_present, feature):
        if both_present is None:
            return feature.new_ones((feature.shape[0], 1, 1, 1))
        presence = both_present.to(device=feature.device, dtype=feature.dtype)
        if presence.ndim == 1:
            presence = presence[:, None, None, None]
        if presence.shape != (feature.shape[0], 1, 1, 1):
            raise ValueError(
                "both_present must have shape [B] or [B, 1, 1, 1], got "
                f"{tuple(presence.shape)}"
            )
        return presence

    def predict_guidance(self, rgb_feat, ir_feat, both_present=None):
        rgb_common = self.common_projection(rgb_feat)
        ir_common = self.common_projection(ir_feat)
        descriptors = torch.cat(
            (
                rgb_common,
                ir_common,
                rgb_common - ir_common,
                rgb_common * ir_common,
            ),
            dim=1,
        )
        raw_flow = self.guidance_predictor(descriptors)
        limit = self.GUIDANCE_LIMIT_CELLS
        flow = limit * torch.tanh(raw_flow / limit)
        return flow * self._presence_scale(both_present, flow)

    def forward(self, rgb_feat, ir_feat, both_present=None):
        if rgb_feat.shape != ir_feat.shape:
            raise ValueError(
                "Box-guided FAM requires equal RGB and IR feature shapes, got "
                f"{tuple(rgb_feat.shape)} and {tuple(ir_feat.shape)}"
            )

        out = self.offset_conv(torch.cat([rgb_feat, ir_feat], dim=1))
        residual_offset = self.transform_offset(out[:, :18])
        mask = torch.sigmoid(out[:, 18:])

        # torchvision DeformConv2d interleaves (dy, dx) for every kernel
        # point. Repeating [dy, dx] nine times preserves that convention.
        guidance_flow = self.predict_guidance(
            rgb_feat,
            ir_feat,
            both_present=both_present,
        )
        self.last_guidance_flow = guidance_flow
        offset = residual_offset + guidance_flow.repeat(1, 9, 1, 1)
        return self.deform_conv(ir_feat, offset, mask)


FAM_VARIANTS = {
    "current_dcnv2": FeatureAlignmentModule,
    "bounded_dcnv2_4": BoundedFeatureAlignmentModule,
    "box_guided_common_offset_p3": BoxGuidedCommonOffsetFeatureAlignmentModule,
    "identity_dcnv2": IdentityInitializedFeatureAlignmentModule,
    "grid_sample": GridSampleFeatureAlignmentModule,
}


P2_BACKBONE_OUT_INDICES = [1, 2, 3, 4]
P2_FEATURE_STRIDES = [4, 8, 16, 32]


def configure_rtdetr_p2(config):
    """Extend an RT-DETR ResNet configuration from P3--P5 to P2--P5."""
    backbone_config = getattr(config, "backbone_config", None)
    hidden_sizes = list(getattr(backbone_config, "hidden_sizes", ()))
    if len(hidden_sizes) != 4:
        raise ValueError(
            "RT-DETR P2 requires a four-stage backbone with hidden_sizes, got "
            f"{hidden_sizes!r}"
        )

    backbone_config.out_indices = list(P2_BACKBONE_OUT_INDICES)
    config.encoder_in_channels = hidden_sizes
    config.feat_strides = list(P2_FEATURE_STRIDES)
    config.encode_proj_layers = [len(hidden_sizes) - 1]
    config.decoder_in_channels = [config.encoder_hidden_dim] * len(hidden_sizes)
    config.num_feature_levels = len(hidden_sizes)
    config.use_p2 = True
    return config


def _copy_indexed_group(source, target, output, prefix, index_offset):
    for key, value in source.items():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix):]
        index_text, separator, suffix = remainder.partition(".")
        if not separator or not index_text.isdigit():
            continue
        mapped_key = f"{prefix}{int(index_text) + index_offset}.{suffix}"
        if mapped_key in target and target[mapped_key].shape == value.shape:
            output[mapped_key] = value


def _copy_group_index(source, target, output, prefix, source_index, target_index):
    source_prefix = f"{prefix}{source_index}."
    target_prefix = f"{prefix}{target_index}."
    for key, value in source.items():
        if not key.startswith(source_prefix):
            continue
        mapped_key = target_prefix + key[len(source_prefix):]
        if mapped_key in target and target[mapped_key].shape == value.shape:
            output[mapped_key] = value


def _expand_deformable_attention_levels(source, target, output, config):
    old_levels = 3
    new_levels = int(config.num_feature_levels)
    heads = int(config.decoder_attention_heads)
    points = int(config.decoder_n_points)
    for key, source_tensor in source.items():
        if ".encoder_attn.sampling_offsets." in key:
            coordinate_dim = 2
        elif ".encoder_attn.attention_weights." in key:
            coordinate_dim = None
        else:
            continue
        target_tensor = target.get(key)
        if target_tensor is None or source_tensor.ndim not in (1, 2):
            continue
        tail_shape = tuple(source_tensor.shape[1:])
        if coordinate_dim is None:
            expected_source = heads * old_levels * points
            expected_target = heads * new_levels * points
            if source_tensor.shape[0] != expected_source or target_tensor.shape[0] != expected_target:
                continue
            source_view = source_tensor.reshape(heads, old_levels, points, *tail_shape)
            target_view = target_tensor.clone().reshape(heads, new_levels, points, *tail_shape)
        else:
            expected_source = heads * old_levels * points * coordinate_dim
            expected_target = heads * new_levels * points * coordinate_dim
            if source_tensor.shape[0] != expected_source or target_tensor.shape[0] != expected_target:
                continue
            source_view = source_tensor.reshape(
                heads, old_levels, points, coordinate_dim, *tail_shape
            )
            target_view = target_tensor.clone().reshape(
                heads, new_levels, points, coordinate_dim, *tail_shape
            )

        # Initialise P2 from the closest pretrained scale (old P3), then place
        # the original P3--P5 tensors at their shifted levels 1--3.
        target_view[:, 0].copy_(source_view[:, 0])
        target_view[:, 1:].copy_(source_view)
        output[key] = target_view.reshape_as(target_tensor)


def build_p2_pretrained_state(source_state, target_state, config):
    """Remap a three-level COCO checkpoint into the four-level P2 model.

    Level-indexed tensors cannot be loaded by their unchanged numeric key:
    adding P2 shifts the semantic meaning of P3--P5 by one position.
    """
    shifted_prefixes = (
        "model.encoder_input_proj.",
        "model.decoder_input_proj.",
        "model.encoder.downsample_convs.",
        "model.encoder.pan_blocks.",
    )
    output = {}
    for key, value in source_state.items():
        if key.startswith(shifted_prefixes):
            continue
        if (
            ".encoder_attn.sampling_offsets." in key
            or ".encoder_attn.attention_weights." in key
        ):
            continue
        if key in target_state and target_state[key].shape == value.shape:
            output[key] = value

    for prefix in shifted_prefixes:
        _copy_indexed_group(source_state, target_state, output, prefix, index_offset=1)

    # New P2 decoder/PAN operations use the closest pretrained P3 operation.
    for prefix in (
        "model.decoder_input_proj.",
        "model.encoder.downsample_convs.",
        "model.encoder.pan_blocks.",
    ):
        _copy_group_index(source_state, target_state, output, prefix, 0, 0)

    # The extra top-down P3->P2 operation is initialised from P4->P3.
    for prefix in (
        "model.encoder.lateral_convs.",
        "model.encoder.fpn_blocks.",
    ):
        _copy_group_index(source_state, target_state, output, prefix, 1, 2)

    # P2 has 256 input/output channels in the R50-vd backbone. Start its 1x1
    # projection as an exact channel identity instead of a random remapping.
    p2_projection_key = "model.encoder_input_proj.0.0.weight"
    p2_projection = target_state.get(p2_projection_key)
    if p2_projection is not None:
        identity = torch.zeros_like(p2_projection)
        diagonal = min(identity.shape[0], identity.shape[1])
        indices = torch.arange(diagonal, device=identity.device)
        identity[indices, indices, 0, 0] = 1.0
        output[p2_projection_key] = identity

    _expand_deformable_attention_levels(source_state, target_state, output, config)
    return output


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


class ReliabilityGatedFusion(nn.Module):
    """Spatially modulate aligned RGB and IR features without changing FAM at init.

    The predictor works on channel-compressed descriptors, so its parameter and
    memory cost does not grow quadratically with the backbone width. Independent
    weights allow one modality to be suppressed without forcing the other to be
    amplified. The final convolution is zero-initialized and weights are
    ``2 * sigmoid(logit)``, hence both weights are exactly one initially and the
    first forward is the existing additive FAM fusion.
    """

    NUM_DESCRIPTORS = 7

    def __init__(self, hidden_channels=16):
        super().__init__()
        if int(hidden_channels) < 1:
            raise ValueError("reliability gate hidden_channels must be positive")
        self.hidden_channels = int(hidden_channels)
        self.descriptor_conv = nn.Conv2d(
            self.NUM_DESCRIPTORS,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.activation = nn.SiLU()
        self.logit_conv = nn.Conv2d(
            self.hidden_channels,
            2,
            kernel_size=1,
        )
        self.reset_neutral_parameters()

    def reset_neutral_parameters(self):
        nn.init.zeros_(self.logit_conv.weight)
        nn.init.zeros_(self.logit_conv.bias)

    @staticmethod
    def _presence_map(presence, feature):
        if presence is None:
            return feature.new_ones((feature.shape[0], 1, 1, 1))
        presence = presence.to(device=feature.device, dtype=feature.dtype)
        if presence.ndim == 1:
            presence = presence[:, None, None, None]
        if presence.shape != (feature.shape[0], 1, 1, 1):
            raise ValueError(
                "modality presence must have shape [B] or [B, 1, 1, 1], "
                f"got {tuple(presence.shape)}"
            )
        return presence

    def compute_weights(
        self,
        rgb_feat,
        ir_feat,
        rgb_present=None,
        ir_present=None,
    ):
        if rgb_feat.shape != ir_feat.shape:
            raise ValueError(
                "Reliability gating requires aligned feature shapes, got "
                f"{tuple(rgb_feat.shape)} and {tuple(ir_feat.shape)}"
            )

        eps = torch.finfo(rgb_feat.dtype).eps
        rgb_mean = rgb_feat.mean(dim=1, keepdim=True)
        ir_mean = ir_feat.mean(dim=1, keepdim=True)
        rgb_rms = rgb_feat.square().mean(dim=1, keepdim=True).add(eps).sqrt()
        ir_rms = ir_feat.square().mean(dim=1, keepdim=True).add(eps).sqrt()
        agreement = (
            (rgb_feat * ir_feat).mean(dim=1, keepdim=True)
            / (rgb_rms * ir_rms).clamp_min(eps)
        ).clamp(-1.0, 1.0)

        height, width = rgb_feat.shape[-2:]
        rgb_presence = self._presence_map(rgb_present, rgb_feat).expand(
            -1, -1, height, width
        )
        ir_presence = self._presence_map(ir_present, ir_feat).expand(
            -1, -1, height, width
        )
        descriptors = torch.cat(
            (
                rgb_mean,
                rgb_rms,
                ir_mean,
                ir_rms,
                agreement,
                rgb_presence,
                ir_presence,
            ),
            dim=1,
        )
        logits = self.logit_conv(
            self.activation(self.descriptor_conv(descriptors))
        )
        weights = 2.0 * torch.sigmoid(logits)
        return weights[:, :1], weights[:, 1:]

    def forward(
        self,
        rgb_feat,
        ir_feat,
        rgb_present=None,
        ir_present=None,
    ):
        rgb_weight, ir_weight = self.compute_weights(
            rgb_feat,
            ir_feat,
            rgb_present=rgb_present,
            ir_present=ir_present,
        )
        return rgb_weight * rgb_feat + ir_weight * ir_feat


class ReliabilityConditionedResidualAlignment(nn.Module):
    """Select how much of FAM's IR alignment correction should be retained.

    FAM always replaces the raw IR feature with its aligned counterpart. This
    module instead predicts a spatial coefficient from compact RGB/raw-IR/
    aligned-IR descriptors and applies it only to FAM's residual correction::

        selected = aligned + (alpha - 1) * (aligned - raw)

    ``alpha = 2 * sigmoid(logit)`` is in ``(0, 2)``. The last convolution is
    zero-initialized, hence ``alpha == 1`` and ``selected == aligned`` exactly
    at initialization. The candidate therefore starts from the existing FAM
    baseline, can bypass a harmful correction as ``alpha`` approaches zero,
    and can moderately amplify a useful correction above one.
    """

    NUM_DESCRIPTORS = 12

    def __init__(self, hidden_channels=16):
        super().__init__()
        if int(hidden_channels) < 1:
            raise ValueError(
                "residual alignment gate hidden_channels must be positive"
            )
        self.hidden_channels = int(hidden_channels)
        self.descriptor_conv = nn.Conv2d(
            self.NUM_DESCRIPTORS,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.activation = nn.SiLU()
        self.logit_conv = nn.Conv2d(
            self.hidden_channels,
            1,
            kernel_size=1,
        )
        self.reset_neutral_parameters()

    def reset_neutral_parameters(self):
        nn.init.zeros_(self.logit_conv.weight)
        nn.init.zeros_(self.logit_conv.bias)

    @staticmethod
    def _presence_map(presence, feature):
        if presence is None:
            return feature.new_ones((feature.shape[0], 1, 1, 1))
        presence = presence.to(device=feature.device, dtype=feature.dtype)
        if presence.ndim == 1:
            presence = presence[:, None, None, None]
        if presence.shape != (feature.shape[0], 1, 1, 1):
            raise ValueError(
                "modality presence must have shape [B] or [B, 1, 1, 1], "
                f"got {tuple(presence.shape)}"
            )
        return presence

    @staticmethod
    def _rms(feature, eps):
        return feature.square().mean(dim=1, keepdim=True).add(eps).sqrt()

    @staticmethod
    def _cosine_agreement(first, second, first_rms, second_rms, eps):
        return (
            (first * second).mean(dim=1, keepdim=True)
            / (first_rms * second_rms).clamp_min(eps)
        ).clamp(-1.0, 1.0)

    def compute_alpha(
        self,
        rgb_feat,
        ir_raw,
        ir_aligned,
        rgb_present=None,
        ir_present=None,
    ):
        if rgb_feat.shape != ir_raw.shape or ir_raw.shape != ir_aligned.shape:
            raise ValueError(
                "Residual alignment gating requires equal RGB, raw-IR and "
                "aligned-IR shapes, got "
                f"{tuple(rgb_feat.shape)}, {tuple(ir_raw.shape)} and "
                f"{tuple(ir_aligned.shape)}"
            )

        eps = torch.finfo(rgb_feat.dtype).eps
        rgb_mean = rgb_feat.mean(dim=1, keepdim=True)
        raw_mean = ir_raw.mean(dim=1, keepdim=True)
        aligned_mean = ir_aligned.mean(dim=1, keepdim=True)
        rgb_rms = self._rms(rgb_feat, eps)
        raw_rms = self._rms(ir_raw, eps)
        aligned_rms = self._rms(ir_aligned, eps)
        raw_agreement = self._cosine_agreement(
            rgb_feat, ir_raw, rgb_rms, raw_rms, eps
        )
        aligned_agreement = self._cosine_agreement(
            rgb_feat, ir_aligned, rgb_rms, aligned_rms, eps
        )
        alignment_residual_rms = self._rms(ir_aligned - ir_raw, eps)
        agreement_gain = aligned_agreement - raw_agreement

        height, width = rgb_feat.shape[-2:]
        rgb_presence = self._presence_map(rgb_present, rgb_feat).expand(
            -1, -1, height, width
        )
        ir_presence = self._presence_map(ir_present, ir_raw).expand(
            -1, -1, height, width
        )
        descriptors = torch.cat(
            (
                rgb_mean,
                rgb_rms,
                raw_mean,
                raw_rms,
                aligned_mean,
                aligned_rms,
                raw_agreement,
                aligned_agreement,
                alignment_residual_rms,
                agreement_gain,
                rgb_presence,
                ir_presence,
            ),
            dim=1,
        )
        logits = self.logit_conv(
            self.activation(self.descriptor_conv(descriptors))
        )
        return 2.0 * torch.sigmoid(logits)

    def forward(
        self,
        rgb_feat,
        ir_raw,
        ir_aligned,
        rgb_present=None,
        ir_present=None,
    ):
        alpha = self.compute_alpha(
            rgb_feat,
            ir_raw,
            ir_aligned,
            rgb_present=rgb_present,
            ir_present=ir_present,
        )
        alignment_residual = ir_aligned - ir_raw
        # This form, instead of raw + alpha * residual, makes alpha == 1 an
        # exact pass-through of ir_aligned even in finite-precision arithmetic.
        return ir_aligned + (alpha - 1.0) * alignment_residual


class ScalarResidualAlignment(nn.Module):
    """Per-level scalar control for the RCRA residual formulation.

    This deliberately removes every image-, modality- and location-dependent
    descriptor from RCRA. One scalar logit is learned for each feature level,
    allowing the experiment to distinguish conditional spatial selection from
    simple calibration of the FAM residual. The parameterization and exact
    neutral initialization are otherwise identical to RCRA.
    """

    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(()))

    def reset_neutral_parameters(self):
        nn.init.zeros_(self.logit)

    def compute_alpha(self):
        return 2.0 * torch.sigmoid(self.logit)

    def forward(
        self,
        rgb_feat,
        ir_raw,
        ir_aligned,
        rgb_present=None,
        ir_present=None,
    ):
        del rgb_feat, rgb_present, ir_present
        if ir_raw.shape != ir_aligned.shape:
            raise ValueError(
                "Scalar residual alignment requires equal raw-IR and "
                f"aligned-IR shapes, got {tuple(ir_raw.shape)} and "
                f"{tuple(ir_aligned.shape)}"
            )
        alpha = self.compute_alpha().to(
            device=ir_aligned.device,
            dtype=ir_aligned.dtype,
        )
        alignment_residual = ir_aligned - ir_raw
        return ir_aligned + (alpha - 1.0) * alignment_residual


# ============================================================
# 2. FUSION BACKBONE (RT-DETR + FAM)
#    - RGB and IR processed separately
#    - FAM aligns IR to RGB
#    - Additive fusion on aligned feature maps
#    - Geometry preserved
# ============================================================
class RTDetrFusionBackbone(nn.Module):
    def __init__(
        self,
        config: RTDetrConfig,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
        fam_variant: str = "current_dcnv2",
        use_reliability_gating: bool = False,
        reliability_gate_hidden_channels: int = 16,
        use_residual_alignment_gating: bool = False,
        residual_alignment_hidden_channels: int = 16,
        use_scalar_residual_alignment: bool = False,
    ):
        super().__init__()

        if use_reliability_gating and not use_fam:
            raise ValueError("Reliability gating requires use_fam=True")
        if use_residual_alignment_gating and not use_fam:
            raise ValueError("Residual alignment gating requires use_fam=True")
        if use_scalar_residual_alignment and not use_fam:
            raise ValueError("Scalar residual alignment requires use_fam=True")
        alignment_ablation_count = sum(
            bool(enabled)
            for enabled in (
                use_reliability_gating,
                use_residual_alignment_gating,
                use_scalar_residual_alignment,
            )
        )
        if alignment_ablation_count > 1:
            raise ValueError(
                "Post-fusion reliability gating, RCRA and scalar residual "
                "alignment must be evaluated as separate ablations"
            )

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
        self.use_p2 = bool(getattr(config, "use_p2", False))
        self.use_fam = use_fam
        self.freeze_fam = freeze_fam
        self.ir_dropout_rate = ir_dropout_rate
        self.spatial_jitter_std = spatial_jitter_std
        self.fam_variant = fam_variant
        self.use_reliability_gating = use_reliability_gating
        self.reliability_gate_hidden_channels = int(
            reliability_gate_hidden_channels
        )
        self.use_residual_alignment_gating = bool(
            use_residual_alignment_gating
        )
        self.residual_alignment_hidden_channels = int(
            residual_alignment_hidden_channels
        )
        self.use_scalar_residual_alignment = bool(
            use_scalar_residual_alignment
        )
        
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
            fam_modules = []
            for level_index, channels in enumerate(feature_channels):
                # Box-derived displacement is reliable at the high-resolution
                # P3 map.  P4/P5 targets would be sub-cell and are deliberately
                # left as the historical FAM rather than adding unsupervised
                # copies of the new branch.
                level_variant = self.fam_variant
                if (
                    self.fam_variant == "box_guided_common_offset_p3"
                    and level_index != 0
                ):
                    level_variant = "current_dcnv2"
                fam_modules.append(
                    build_feature_alignment_module(
                        level_variant,
                        channels,
                        freeze=self.freeze_fam,
                        spatial_jitter_std=self.spatial_jitter_std,
                    )
                )
            self.fam_modules = nn.ModuleList(fam_modules)
        else:
            self.fam_modules = None

        if self.use_reliability_gating:
            self.reliability_gates = nn.ModuleList(
                [
                    ReliabilityGatedFusion(
                        hidden_channels=self.reliability_gate_hidden_channels
                    )
                    for _ in feature_channels
                ]
            )
        else:
            self.reliability_gates = None

        if self.use_residual_alignment_gating:
            self.alignment_gates = nn.ModuleList(
                [
                    ReliabilityConditionedResidualAlignment(
                        hidden_channels=self.residual_alignment_hidden_channels
                    )
                    for _ in feature_channels
                ]
            )
        elif self.use_scalar_residual_alignment:
            self.alignment_gates = nn.ModuleList(
                [ScalarResidualAlignment() for _ in feature_channels]
            )
        else:
            self.alignment_gates = None

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
            rgb_present = pixel_values[:, :3].detach().ne(0).flatten(1).any(dim=1)
            ir_present = pixel_values[:, 3:].detach().ne(0).flatten(1).any(dim=1)
            rgb_feats = self.rgb_backbone(pixel_values[:, :3], pixel_mask)
            ir_feats  = self.ir_backbone(pixel_values[:, 3:], pixel_mask)

            # Mode with FAM (Feature Alignment Module)
            if self.use_fam:
                if len(self.fam_modules) != len(rgb_feats):
                    raise ValueError(
                        f"FAM levels mismatch: got {len(self.fam_modules)} modules for {len(rgb_feats)} feature levels"
                    )
                if (
                    self.reliability_gates is not None
                    and len(self.reliability_gates) != len(rgb_feats)
                ):
                    raise ValueError(
                        "Reliability-gate levels mismatch: got "
                        f"{len(self.reliability_gates)} modules for "
                        f"{len(rgb_feats)} feature levels"
                    )
                if (
                    self.alignment_gates is not None
                    and len(self.alignment_gates) != len(rgb_feats)
                ):
                    raise ValueError(
                        "Residual-alignment-gate levels mismatch: got "
                        f"{len(self.alignment_gates)} modules for "
                        f"{len(rgb_feats)} feature levels"
                    )

                fused_feats = []
                for idx, ((r_feat, r_mask), (i_feat, _)) in enumerate(zip(rgb_feats, ir_feats)):
                    # Apply FAM to align IR to RGB
                    i_aligned = self.fam_modules[idx](
                        r_feat,
                        i_feat,
                        both_present=(rgb_present & ir_present),
                    )

                    if self.alignment_gates is not None:
                        i_aligned = self.alignment_gates[idx](
                            r_feat,
                            i_feat,
                            i_aligned,
                            rgb_present=rgb_present,
                            ir_present=ir_present,
                        )
                    
                    # Apply Spatial Dropout to IR if active
                    if self.ir_dropout is not None:
                        i_aligned = self.ir_dropout(i_aligned)
                    
                    if self.reliability_gates is not None:
                        fused = self.reliability_gates[idx](
                            r_feat,
                            i_aligned,
                            rgb_present=rgb_present,
                            ir_present=ir_present,
                        )
                    else:
                        fused = r_feat + i_aligned

                    fused_feats.append((fused, r_mask))

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
    def __init__(
        self,
        config: RTDetrConfig,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
        fam_variant: str = "current_dcnv2",
        use_p2: bool = False,
        use_reliability_gating: bool = False,
        reliability_gate_hidden_channels: int = 16,
        use_residual_alignment_gating: bool = False,
        residual_alignment_hidden_channels: int = 16,
        use_scalar_residual_alignment: bool = False,
    ):
        super().__init__(config)
        self.backbone = RTDetrFusionBackbone(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
            fam_variant=fam_variant,
            use_reliability_gating=use_reliability_gating,
            reliability_gate_hidden_channels=reliability_gate_hidden_channels,
            use_residual_alignment_gating=use_residual_alignment_gating,
            residual_alignment_hidden_channels=residual_alignment_hidden_channels,
            use_scalar_residual_alignment=use_scalar_residual_alignment,
        )
        self.use_p2 = use_p2
        self.use_reliability_gating = use_reliability_gating
        self.use_residual_alignment_gating = use_residual_alignment_gating
        self.use_scalar_residual_alignment = use_scalar_residual_alignment
        self.post_init()
        # Hugging Face post_init may visit newly attached modules. Restore the
        # ablation's defining initialization after that global initialization.
        if self.backbone.fam_modules is not None:
            for fam_module in self.backbone.fam_modules:
                reset_identity = getattr(fam_module, "reset_identity_parameters", None)
                if reset_identity is not None:
                    reset_identity()
                reset_guidance = getattr(
                    fam_module, "reset_guidance_parameters", None
                )
                if reset_guidance is not None:
                    reset_guidance()
        if self.backbone.reliability_gates is not None:
            for reliability_gate in self.backbone.reliability_gates:
                reliability_gate.reset_neutral_parameters()
        if self.backbone.alignment_gates is not None:
            for alignment_gate in self.backbone.alignment_gates:
                alignment_gate.reset_neutral_parameters()


# ============================================================
# 4. OBJECT DETECTION WRAPPER
# ============================================================
class RTDetrFusionForObjectDetection(RTDetrForObjectDetection):
    def __init__(
        self,
        config: RTDetrConfig,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
        fam_variant: str = "current_dcnv2",
        use_p2: bool = False,
        use_reliability_gating: bool = False,
        reliability_gate_hidden_channels: int = 16,
        use_residual_alignment_gating: bool = False,
        residual_alignment_hidden_channels: int = 16,
        use_scalar_residual_alignment: bool = False,
    ):
        # Trick: initialize as standard RGB model
        tmp_cfg = copy.deepcopy(config)
        tmp_cfg.num_channels = 3
        super().__init__(tmp_cfg)

        # Save original heads
        saved_class_embed = self.class_embed
        saved_bbox_embed = self.bbox_embed

        # Replace the model
        self.model = RTDetrFusionModel(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
            fam_variant=fam_variant,
            use_p2=use_p2,
            use_reliability_gating=use_reliability_gating,
            reliability_gate_hidden_channels=reliability_gate_hidden_channels,
            use_residual_alignment_gating=use_residual_alignment_gating,
            residual_alignment_hidden_channels=residual_alignment_hidden_channels,
            use_scalar_residual_alignment=use_scalar_residual_alignment,
        )

        # Restore heads in decoder
        self.model.decoder.class_embed = saved_class_embed
        self.model.decoder.bbox_embed = saved_bbox_embed
        self.config = config
        self.use_fam = use_fam
        self.freeze_fam = freeze_fam
        self.ir_dropout_rate = ir_dropout_rate
        self.spatial_jitter_std = spatial_jitter_std
        self.fam_variant = fam_variant
        self.use_p2 = use_p2
        self.use_reliability_gating = use_reliability_gating
        self.reliability_gate_hidden_channels = int(
            reliability_gate_hidden_channels
        )
        self.use_residual_alignment_gating = bool(
            use_residual_alignment_gating
        )
        self.residual_alignment_hidden_channels = int(
            residual_alignment_hidden_channels
        )
        self.use_scalar_residual_alignment = bool(
            use_scalar_residual_alignment
        )

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
        use_p2=False,
        use_reliability_gating=False,
        reliability_gate_hidden_channels=16,
        use_residual_alignment_gating=False,
        residual_alignment_hidden_channels=16,
        use_scalar_residual_alignment=False,
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
        if use_p2:
            configure_rtdetr_p2(config)
        else:
            config.use_p2 = False
        instance = cls(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
            fam_variant=fam_variant,
            use_p2=use_p2,
            use_reliability_gating=use_reliability_gating,
            reliability_gate_hidden_channels=reliability_gate_hidden_channels,
            use_residual_alignment_gating=use_residual_alignment_gating,
            residual_alignment_hidden_channels=residual_alignment_hidden_channels,
            use_scalar_residual_alignment=use_scalar_residual_alignment,
        )

        # Load everything (decoder, encoder, etc.). P2 changes the semantic
        # level associated with several numeric module indices, so it requires
        # an explicit remap rather than shape-only loading.
        base_state = base.state_dict()
        instance_state = instance.state_dict()
        if use_p2:
            compatible_state = build_p2_pretrained_state(
                base_state,
                instance_state,
                config,
            )
            instance.load_state_dict(compatible_state)
            print(
                "P2 pretrained transfer: "
                f"{len(compatible_state)} compatible tensors staged; RGB and "
                "IR backbones are transferred separately; new P2/FAM tensors "
                "keep their explicit initialization"
            )
            if reuse_pretrained_class_head:
                copy_matching_pretrained_label_heads(instance, base, config.id2label)
        elif reuse_pretrained_class_head:
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
