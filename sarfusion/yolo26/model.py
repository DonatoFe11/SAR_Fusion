"""Dual-backbone YOLO26s with matched Additive and FAM fusion arms."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Iterable

import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER

from .fam import FeatureAlignmentModule


def _update_tensor_hash(digest: Any, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())


class YOLO26FusionDetectionModel(DetectionModel):
    """Official YOLO26 graph with RGB/IR backbones and P3/P4/P5 fusion.

    Both experimental arms instantiate the same FAM modules.  ``use_fam``
    only controls whether the modules are applied, preserving construction
    order, RNG consumption, common parameters, and optimizer membership.
    """

    BACKBONE_LAYERS = 11
    FUSION_INDICES = (4, 6, 10)

    def __init__(
        self,
        cfg: str | dict = "yolo26s.yaml",
        *,
        nc: int = 1,
        use_fam: bool = False,
        freeze_fam: bool = False,
        spatial_jitter_std: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self._fusion_ready = False
        super().__init__(cfg=cfg, ch=3, nc=nc, verbose=verbose)

        if len(self.model) != 24 or tuple(self.save[:3]) != self.FUSION_INDICES:
            raise RuntimeError(
                "Unexpected YOLO26 graph: integration expects 24 layers and "
                f"saved backbone outputs {self.FUSION_INDICES}, got "
                f"{len(self.model)} layers and save={self.save}."
            )

        self.ir_backbone = nn.ModuleList(
            deepcopy(list(self.model[: self.BACKBONE_LAYERS]))
        )
        self._replace_ir_stem()

        channels = self._discover_fusion_channels()
        self.fam_modules = nn.ModuleList(
            FeatureAlignmentModule(
                channels[index],
                freeze=freeze_fam,
                spatial_jitter_std=spatial_jitter_std,
            )
            for index in self.FUSION_INDICES
        )
        self.use_fam = bool(use_fam)
        self.freeze_fam = bool(freeze_fam)
        self.spatial_jitter_std = float(spatial_jitter_std)
        self._fusion_ready = True

    def _replace_ir_stem(self) -> None:
        stem = getattr(self.ir_backbone[0], "conv", None)
        if not isinstance(stem, nn.Conv2d) or stem.in_channels != 3:
            raise RuntimeError("Could not locate the 3-channel YOLO26 stem convolution.")
        replacement = nn.Conv2d(
            in_channels=1,
            out_channels=stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=stem.stride,
            padding=stem.padding,
            dilation=stem.dilation,
            groups=stem.groups,
            bias=stem.bias is not None,
            padding_mode=stem.padding_mode,
        ).to(device=stem.weight.device, dtype=stem.weight.dtype)
        self.ir_backbone[0].conv = replacement

    def _discover_fusion_channels(self) -> dict[int, int]:
        # C3k2 contains hidden convolutions, so inspecting its final Conv2d is
        # not a reliable way to infer the block output width.  A no-grad eval
        # pass reads the actual saved feature shapes without updating BN.
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self._run_backbone(
                torch.zeros(1, 3, 64, 64),
                self.model[: self.BACKBONE_LAYERS],
            )
        self.model.train(was_training)
        channels = {}
        for index in self.FUSION_INDICES:
            feature = outputs[index]
            if feature is None:
                raise RuntimeError(f"YOLO26 did not save fusion layer {index}.")
            channels[index] = int(feature.shape[1])
        return channels

    @staticmethod
    def _normalize_modality_mask(
        modality_mask: torch.Tensor | Iterable | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if modality_mask is None:
            return torch.ones((batch_size, 2), device=device, dtype=dtype)
        mask = torch.as_tensor(modality_mask, device=device, dtype=dtype)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != (batch_size, 2):
            raise ValueError(
                "modality_mask must have shape [B, 2] in [RGB, IR] order; "
                f"got {tuple(mask.shape)} for batch size {batch_size}."
            )
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("modality_mask values must be binary.")
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("Every sample must contain at least one modality.")
        return mask

    def _run_backbone(self, x: torch.Tensor, modules: Iterable[nn.Module]) -> list:
        outputs: list[torch.Tensor | None] = []
        for module in modules:
            if module.f != -1:
                x = (
                    outputs[module.f]
                    if isinstance(module.f, int)
                    else [x if j == -1 else outputs[j] for j in module.f]
                )
            x = module(x)
            outputs.append(x if module.i in self.save else None)
        return outputs

    @staticmethod
    def _scatter_feature_batch(
        features: list,
        level: int,
        present: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        feature = features[level]
        restored = feature.new_zeros((batch_size, *feature.shape[1:]))
        indices = present.nonzero(as_tuple=False).flatten()
        return restored.index_copy(0, indices, feature)

    def _predict_fusion_once(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor | Iterable | None,
    ):
        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError(
                "YOLO26FusionDetectionModel expects [B, 4, H, W] input, got "
                f"{tuple(x.shape)}."
            )
        mask = self._normalize_modality_mask(
            modality_mask,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        rgb_present = mask[:, 0].bool()
        ir_present = mask[:, 1].bool()
        both_present = rgb_present & ir_present

        rgb_features = (
            self._run_backbone(
                x[rgb_present, :3].contiguous(),
                self.model[: self.BACKBONE_LAYERS],
            )
            if rgb_present.any()
            else None
        )
        ir_features = (
            self._run_backbone(
                x[ir_present, 3:4].contiguous(),
                self.ir_backbone,
            )
            if ir_present.any()
            else None
        )

        fused: dict[int, torch.Tensor] = {}
        for fam_index, level in enumerate(self.FUSION_INDICES):
            if rgb_features is not None:
                rgb_level = self._scatter_feature_batch(
                    rgb_features, level, rgb_present, x.shape[0]
                )
            else:
                template = ir_features[level]
                rgb_level = template.new_zeros((x.shape[0], *template.shape[1:]))

            if ir_features is not None:
                ir_level = self._scatter_feature_batch(
                    ir_features, level, ir_present, x.shape[0]
                )
            else:
                ir_level = rgb_level.new_zeros(rgb_level.shape)

            # Preserve the historical contract: IR-only samples use their raw
            # features; FAM is applied only when both real sensors are present.
            if self.use_fam and both_present.any():
                aligned = self.fam_modules[fam_index](
                    rgb_level[both_present],
                    ir_level[both_present],
                )
                ir_level = ir_level.index_copy(
                    0,
                    both_present.nonzero(as_tuple=False).flatten(),
                    aligned,
                )
            fused[level] = rgb_level + ir_level

        outputs: list[torch.Tensor | None] = [None] * len(self.model)
        for level, feature in fused.items():
            outputs[level] = feature
        current = fused[self.FUSION_INDICES[-1]]
        for module in self.model[self.BACKBONE_LAYERS :]:
            if module.f != -1:
                current = (
                    outputs[module.f]
                    if isinstance(module.f, int)
                    else [current if j == -1 else outputs[j] for j in module.f]
                )
            current = module(current)
            outputs[module.i] = current if module.i in self.save else None
        return current

    def predict(
        self,
        x: torch.Tensor,
        profile: bool = False,
        augment: bool = False,
        embed=None,
        modality_mask=None,
    ):
        # DetectionModel.__init__ computes strides with a 3-channel dummy
        # before the IR branch exists.  The same path is also the RGB-only
        # parity oracle used by the integration audit.
        if not getattr(self, "_fusion_ready", False) or x.shape[1] == 3:
            return super().predict(x, profile=profile, augment=augment, embed=embed)
        if profile or embed is not None:
            raise ValueError("Fusion profiling/embedding extraction is not supported.")
        if augment:
            LOGGER.warning("Fusion YOLO26 does not support augment=True; using one pass.")
        return self._predict_fusion_once(x, modality_mask)

    def loss(self, batch: dict, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.predict(
                batch["img"],
                modality_mask=batch.get("modality_mask"),
            )
        return self.criterion(preds, batch)

    def initialize_ir_from_rgb(self) -> None:
        """Copy the loaded RGB backbone into IR, averaging the RGB stem."""
        rgb_state = self.model[: self.BACKBONE_LAYERS].state_dict()
        ir_state = self.ir_backbone.state_dict()
        copied: dict[str, torch.Tensor] = {}
        for key, target in ir_state.items():
            source = rgb_state[key]
            if key == "0.conv.weight":
                if source.ndim != 4 or source.shape[1] != 3:
                    raise RuntimeError("Unexpected RGB stem weight shape.")
                source = source.mean(dim=1, keepdim=True)
            if source.shape != target.shape:
                raise RuntimeError(
                    f"IR initialization shape mismatch for {key}: "
                    f"{tuple(source.shape)} != {tuple(target.shape)}."
                )
            copied[key] = source.detach().clone()
        self.ir_backbone.load_state_dict(copied, strict=True)

    def load_official_pretrained(self, weights: nn.Module, verbose: bool = True) -> None:
        """Load official YOLO26 weights and deterministically seed IR."""
        super().load(weights, verbose=verbose)
        self.initialize_ir_from_rgb()

    def state_sha256(self, scope: str = "shared") -> str:
        """Hash shared, IR, or FAM initialization tensors for paired audits."""
        if scope == "shared":
            state = self.model.state_dict()
        elif scope == "ir":
            state = self.ir_backbone.state_dict()
        elif scope == "fam":
            state = self.fam_modules.state_dict()
        else:
            raise ValueError("scope must be one of: shared, ir, fam")
        digest = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            _update_tensor_hash(digest, name, tensor)
        return digest.hexdigest()

    def initialization_report(self) -> dict[str, Any]:
        return {
            "use_fam": self.use_fam,
            "shared_sha256": self.state_sha256("shared"),
            "ir_sha256": self.state_sha256("ir"),
            "fam_sha256": self.state_sha256("fam"),
            "fusion_indices": list(self.FUSION_INDICES),
            "backbone_layers": self.BACKBONE_LAYERS,
            "end2end": bool(self.end2end),
            "stride": [float(x) for x in self.stride.cpu()],
        }
