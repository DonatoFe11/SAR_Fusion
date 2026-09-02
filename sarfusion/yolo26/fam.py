"""Historical DCNv2 Feature Alignment Module used by SARFusion.

The implementation is kept local so importing the YOLO26 integration does
not execute the legacy ``sarfusion.models`` package.  Its parameterization
matches ``FeatureAlignmentModule`` in ``sarfusion/models/rtdetr_fusion.py``:
zero initial offsets/masks and a normally initialized deformable convolution.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import DeformConv2d


class FeatureAlignmentModule(nn.Module):
    """Align an IR feature map with RGB-guided modulated deformable conv."""

    def __init__(
        self,
        in_channels: int,
        *,
        freeze: bool = False,
        spatial_jitter_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.spatial_jitter_std = float(spatial_jitter_std)
        self.offset_conv = nn.Conv2d(
            in_channels * 2,
            27,
            kernel_size=3,
            padding=1,
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.shape != ir_feat.shape:
            raise ValueError(
                "FAM requires RGB and IR feature maps with equal shape, got "
                f"{tuple(rgb_feat.shape)} and {tuple(ir_feat.shape)}."
            )
        prediction = self.offset_conv(torch.cat((rgb_feat, ir_feat), dim=1))
        offset = prediction[:, :18]
        mask = torch.sigmoid(prediction[:, 18:])
        if self.training and self.spatial_jitter_std > 0.0:
            offset = offset + torch.randn_like(offset) * self.spatial_jitter_std
        return self.deform_conv(ir_feat, offset, mask)
