"""Dual-backbone RGB/IR fusion for Hugging Face RT-DETRv2.

This module deliberately mirrors the established RT-DETR fusion path while
replacing only the detector implementation.  It lives in a separate module so
the historical Transformers 4.43 environment can still import ``sarfusion``;
callers import it lazily only in the dedicated RT-DETRv2 environment.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn

from transformers import RTDetrV2ForObjectDetection
from transformers.models.rt_detr_v2.configuration_rt_detr_v2 import (
    RTDetrV2Config,
)
from transformers.models.rt_detr_v2.modeling_rt_detr_v2 import (
    RTDetrV2ConvEncoder,
    RTDetrV2Model,
)

from sarfusion.models.rtdetr_fusion import (
    build_feature_alignment_module,
    copy_matching_pretrained_label_heads,
)


HISTORICAL_FAM_INITIALIZATION = "historical_hf_post_init"


class RTDetrV2FusionBackbone(nn.Module):
    """Two modality-specific RT-DETRv2 backbones with optional IR alignment."""

    def __init__(
        self,
        config: RTDetrV2Config,
        *,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
        fam_variant: str = "current_dcnv2",
        fam_initialization: str = HISTORICAL_FAM_INITIALIZATION,
    ):
        super().__init__()

        if not 0.0 <= float(ir_dropout_rate) < 1.0:
            raise ValueError("ir_dropout_rate must be in [0, 1)")

        rgb_config = copy.deepcopy(config)
        rgb_config.num_channels = 3
        self.rgb_backbone = RTDetrV2ConvEncoder(rgb_config)

        ir_config = copy.deepcopy(config)
        ir_config.num_channels = 1
        self.ir_backbone = RTDetrV2ConvEncoder(ir_config)
        self._adapt_ir_backbone()

        self.use_fam = bool(use_fam)
        self.freeze_fam = bool(freeze_fam)
        self.ir_dropout_rate = float(ir_dropout_rate)
        self.spatial_jitter_std = float(spatial_jitter_std)
        self.fam_variant = str(fam_variant)
        self.fam_initialization = str(fam_initialization)
        if self.fam_initialization != HISTORICAL_FAM_INITIALIZATION:
            raise ValueError(
                "RT-DETRv2 Stage A supports only the explicitly frozen FAM "
                f"initialization {HISTORICAL_FAM_INITIALIZATION!r}"
            )
        self.intermediate_channel_sizes = list(
            self.rgb_backbone.intermediate_channel_sizes
        )

        expected_channels = list(config.encoder_in_channels)
        if self.intermediate_channel_sizes != expected_channels:
            raise ValueError(
                "RT-DETRv2 backbone outputs do not match encoder_in_channels: "
                f"{self.intermediate_channel_sizes} != {expected_channels}"
            )

        if self.use_fam:
            self.fam_modules = nn.ModuleList(
                [
                    build_feature_alignment_module(
                        self.fam_variant,
                        channels,
                        freeze=self.freeze_fam,
                        spatial_jitter_std=self.spatial_jitter_std,
                    )
                    for channels in self.intermediate_channel_sizes
                ]
            )
        else:
            self.fam_modules = None

        self.ir_dropout = (
            nn.Dropout2d(p=self.ir_dropout_rate)
            if self.ir_dropout_rate > 0.0
            else None
        )

    def _adapt_ir_backbone(self):
        """Convert the first RGB convolution to one-channel IR input."""
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
        channels = pixel_values.shape[1]

        if channels == 3:
            return self.rgb_backbone(pixel_values, pixel_mask)
        if channels == 1:
            return self.ir_backbone(pixel_values, pixel_mask)
        if channels != 4:
            raise ValueError(
                f"Unsupported number of channels for RT-DETRv2 fusion: {channels}"
            )

        rgb_present = pixel_values[:, :3].detach().ne(0).flatten(1).any(dim=1)
        ir_present = pixel_values[:, 3:].detach().ne(0).flatten(1).any(dim=1)
        rgb_features = self.rgb_backbone(pixel_values[:, :3], pixel_mask)
        ir_features = self.ir_backbone(pixel_values[:, 3:], pixel_mask)

        if len(rgb_features) != len(ir_features):
            raise ValueError(
                "RGB and IR RT-DETRv2 backbones returned different level counts"
            )
        if self.fam_modules is not None and len(self.fam_modules) != len(
            rgb_features
        ):
            raise ValueError(
                "FAM level count does not match the RT-DETRv2 feature pyramid"
            )

        fused_features = []
        for level, ((rgb, rgb_mask), (ir, _)) in enumerate(
            zip(rgb_features, ir_features)
        ):
            if self.fam_modules is not None:
                ir = self.fam_modules[level](
                    rgb,
                    ir,
                    both_present=(rgb_present & ir_present),
                )
            if self.ir_dropout is not None:
                ir = self.ir_dropout(ir)
            fused_features.append((rgb + ir, rgb_mask))

        return fused_features


class RTDetrV2FusionModel(RTDetrV2Model):
    def __init__(
        self,
        config: RTDetrV2Config,
        *,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
        fam_variant: str = "current_dcnv2",
        fam_initialization: str = HISTORICAL_FAM_INITIALIZATION,
    ):
        super().__init__(config)
        self.backbone = RTDetrV2FusionBackbone(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
            fam_variant=fam_variant,
            fam_initialization=fam_initialization,
        )
        self.post_init()

        # The established RT-DETR v1 FAM calls Hugging Face ``post_init`` after
        # constructing the backbone.  For ``current_dcnv2`` this replaces the
        # raw module's zero offset predictor with the HF random convolution
        # initialization.  Stage A intentionally reproduces that effective
        # historical behavior; it must not be silently changed to the raw
        # constructor's zero initialization.  Variants with an explicit reset
        # contract still restore their defining initialization below.
        if self.backbone.fam_modules is not None:
            for fam_module in self.backbone.fam_modules:
                reset_identity = getattr(
                    fam_module, "reset_identity_parameters", None
                )
                if reset_identity is not None:
                    reset_identity()
                reset_guidance = getattr(
                    fam_module, "reset_guidance_parameters", None
                )
                if reset_guidance is not None:
                    reset_guidance()


class RTDetrV2FusionForObjectDetection(RTDetrV2ForObjectDetection):
    """RT-DETRv2 detector whose backbone accepts RGB, IR, or RGB+IR."""

    def __init__(
        self,
        config: RTDetrV2Config,
        *,
        use_fam: bool = False,
        freeze_fam: bool = False,
        ir_dropout_rate: float = 0.0,
        spatial_jitter_std: float = 0.0,
        fam_variant: str = "current_dcnv2",
        fam_initialization: str = HISTORICAL_FAM_INITIALIZATION,
    ):
        rgb_config = copy.deepcopy(config)
        rgb_config.num_channels = 3
        super().__init__(rgb_config)

        class_embed = self.class_embed
        bbox_embed = self.bbox_embed
        self.model = RTDetrV2FusionModel(
            config,
            use_fam=use_fam,
            freeze_fam=freeze_fam,
            ir_dropout_rate=ir_dropout_rate,
            spatial_jitter_std=spatial_jitter_std,
            fam_variant=fam_variant,
            fam_initialization=fam_initialization,
        )
        self.model.decoder.class_embed = class_embed
        self.model.decoder.bbox_embed = bbox_embed
        self.config = config
        # PreTrainedModel infers the loss from the concrete class name.  Our
        # ``Fusion`` infix would otherwise fall back to the generic DETR loss,
        # whose config contract is incompatible with RT-DETR(v2).  Bind the
        # exact upstream loss mapping explicitly.
        self.loss_type = "RTDetrV2ForObjectDetection"

        self.use_fam = bool(use_fam)
        self.freeze_fam = bool(freeze_fam)
        self.ir_dropout_rate = float(ir_dropout_rate)
        self.spatial_jitter_std = float(spatial_jitter_std)
        self.fam_variant = str(fam_variant)
        self.fam_initialization = str(fam_initialization)

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
        fam_initialization=HISTORICAL_FAM_INITIALIZATION,
        reuse_pretrained_class_head=False,
        revision=None,
    ):
        load_kwargs = {"revision": revision} if revision is not None else {}
        if reuse_pretrained_class_head:
            base = RTDetrV2ForObjectDetection.from_pretrained(
                pretrained_model_name,
                **load_kwargs,
            )
        else:
            base = RTDetrV2ForObjectDetection.from_pretrained(
                pretrained_model_name,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                **load_kwargs,
            )

        config = copy.deepcopy(base.config)
        config.id2label = {
            int(index): label for index, label in id2label.items()
        }
        config.label2id = {
            label: int(index) for index, label in config.id2label.items()
        }
        config.num_channels = 4
        # The target constructor contains random FAM parameters while the
        # additive control does not.  Preserve the caller's RNG state so
        # denoising and all later stochastic training operators start from the
        # same state in the two matched runs.  The parameters created inside
        # the fork remain deterministic functions of the experiment seed.
        with torch.random.fork_rng(devices=[]):
            instance = cls(
                config,
                use_fam=use_fam,
                freeze_fam=freeze_fam,
                ir_dropout_rate=ir_dropout_rate,
                spatial_jitter_std=spatial_jitter_std,
                fam_variant=fam_variant,
                fam_initialization=fam_initialization,
            )

        base_state = base.state_dict()
        instance_state = instance.state_dict()
        if reuse_pretrained_class_head:
            compatible_state = {
                key: value
                for key, value in base_state.items()
                if key in instance_state
                and value.shape == instance_state[key].shape
            }
            instance.load_state_dict(compatible_state, strict=False)
            copy_matching_pretrained_label_heads(
                instance, base, config.id2label
            )
        else:
            compatible_state = {
                key: value
                for key, value in base_state.items()
                if key in instance_state
                and value.shape == instance_state[key].shape
            }
            instance.load_state_dict(compatible_state, strict=False)

        backbone_prefix = "model.backbone."
        rgb_state = {
            key[len(backbone_prefix) :]: value
            for key, value in base_state.items()
            if key.startswith(backbone_prefix)
        }
        if not rgb_state:
            raise RuntimeError(
                "The RT-DETRv2 checkpoint did not expose model.backbone weights"
            )
        instance.model.backbone.rgb_backbone.load_state_dict(
            rgb_state, strict=True
        )

        ir_state = copy.deepcopy(rgb_state)
        for key, value in ir_state.items():
            if value.ndim == 4 and value.shape[1] == 3:
                ir_state[key] = value.mean(dim=1, keepdim=True)
        instance.model.backbone.ir_backbone.load_state_dict(
            ir_state, strict=True
        )

        source_unmatched = {
            key
            for key, value in base_state.items()
            if key not in compatible_state
        }
        allowed_source_prefixes = (
            "model.backbone.",
            "class_embed.",
            "model.decoder.class_embed.",
            "model.enc_score_head.",
            "model.denoising_class_embed.",
        )
        disallowed_source = sorted(
            key
            for key in source_unmatched
            if not key.startswith(allowed_source_prefixes)
        )
        if disallowed_source:
            raise RuntimeError(
                "Unexpected RT-DETRv2 pretrained tensors were not transferred: "
                + ", ".join(disallowed_source[:10])
            )

        instance.pretrained_transfer_report = {
            "source_tensors": len(base_state),
            "directly_transferred_tensors": len(compatible_state),
            "source_tensors_handled_explicitly": len(source_unmatched),
            "rgb_backbone_tensors": len(rgb_state),
            "ir_backbone_tensors": len(ir_state),
            "reuse_pretrained_class_head": bool(
                reuse_pretrained_class_head
            ),
            "fam_initialization": str(fam_initialization),
            "revision": revision,
        }

        return instance
