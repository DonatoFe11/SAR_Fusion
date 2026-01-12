from ast import Tuple
from typing import Any, Optional, Type
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv

from sarfusion.models.transformer import TwoWayTransformer


def tensor_is_padding(tensor):
    """
    Check if a tensor is a padding tensor.
    Padding tensors are tensors filled with a single value.
    """
    return tensor is None or tensor.unique().numel() == 1


def detect_channels(x):
    channels = []
    for img in x:
        if img.shape[0] == 3:
            channels.append((img, None))
        elif img.shape[0] == 1:
            channels.append((None, img))
        elif img.shape[0] == 4:
            if img[:3, :, :].eq(FusionConv.IGNORE_VALUE).all():
                channels.append((None, img[3:, :, :]))
            elif img[3:, :, :].eq(FusionConv.IGNORE_VALUE).all():
                channels.append((img[:3, :, :], None))
            else:
                channels.append((img[:3, :, :], img[3:, :, :]))
        else:
            raise ValueError("Unsupported number of channels")
    return channels

class OptionalConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        use_embedding=True,
    ):
        super(OptionalConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        if use_embedding:
            self.embedding = nn.Embedding(1, out_channels)
        else:
            self.embedding = None

    def forward(self, x, shape):
        if x is None:
            device = self.conv.weight.device
            B, C, H, W = shape
            # Conv reduction formula
            H = (
                H
                + 2 * self.conv.padding[0]
                - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
                - 1
            ) // self.conv.stride[0] + 1
            W = (
                W
                + 2 * self.conv.padding[1]
                - self.conv.dilation[1] * (self.conv.kernel_size[1] - 1)
                - 1
            ) // self.conv.stride[1] + 1
            if self.embedding is None:
                return torch.zeros(B, self.conv.out_channels, H, W, device=device)
            return (
                self.embedding(torch.tensor([0], device=device))
                .view(1, -1, 1, 1)
                .expand(B, -1, H, W)
            )
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class FusionConv(nn.Module):
    IGNORE_VALUE = 114

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=True,
        use_embedding=True,
    ):
        super(FusionConv, self).__init__()
        self.optional_rgb = OptionalConv(
            3,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            dilation=dilation,
            groups=groups,
            bias=bias,
            use_embedding=use_embedding,
        )
        self.optional_ir = OptionalConv(
            1,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            dilation=dilation,
            groups=groups,
            bias=bias,
            use_embedding=use_embedding,
        )
        assert in_channels == 4, "in_channels must be 4"

        self.conv = Conv(
            out_channels * 2,
            out_channels,
            kernel_size,
            s=stride,
            p=padding,
            d=dilation,
            g=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        channels = detect_channels(x)
        rgb, ir = zip(*channels)
        rgb = [
            item.unsqueeze(0) if not (tensor_is_padding(item)) else None for item in rgb
        ]
        ir = [
            item.unsqueeze(0) if not (tensor_is_padding(item)) else None for item in ir
        ]

        if all(item is not None for item in rgb):
            rgb = torch.cat(rgb)
            rgb = self.optional_rgb(rgb, x.shape)
        else:
            rgb = torch.cat(
                [self.optional_rgb(item, (1,) + x.shape[1:]) for item in rgb]
            )
        if all(item is not None for item in ir):
            ir = torch.cat(ir)
            ir = self.optional_ir(ir, x.shape)
        else:
            ir = torch.cat([self.optional_ir(item, (1,) + x.shape[1:]) for item in ir])
        x = torch.cat([rgb, ir], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
    
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )
        
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = coords.type(
            self.positional_encoding_gaussian_matrix.dtype
        )  # Ensure same type
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W
    
    
class FusionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        depth: int = 1,
        num_heads: int = 4,
        mlp_dim: int = 128,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        attention_pooling_rate: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.attention_downsample_rate = attention_downsample_rate
        self.attention_pooling_rate = attention_pooling_rate
        
        assert in_channels == 4, "in_channels must be 4"
        
        self.optional_rgb = OptionalConv(
            3,
            embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            use_embedding=True,
        )
        self.optional_ir = OptionalConv(
            1,
            embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            use_embedding=True,
        )

        self.transformer = TwoWayTransformer(
            depth=depth,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            activation=activation,
            attention_downsample_rate=attention_downsample_rate,
            dropout=dropout,
        )
        self.pe = PositionEmbeddingRandom(embedding_dim // 2)
        
    def forward(self, x: torch.Tensor):
        channels = detect_channels(x)
        rgb, ir = zip(*channels)
        rgb = [
            item.unsqueeze(0) if not (tensor_is_padding(item)) else None for item in rgb
        ]
        ir = [
            item.unsqueeze(0) if not (tensor_is_padding(item)) else None for item in ir
        ]
        if all(item is not None for item in rgb):
            rgb = torch.cat(rgb)
            rgb = self.optional_rgb(rgb, x.shape)
        else:
            rgb = torch.cat(
                [self.optional_rgb(item, (1,) + x.shape[1:]) for item in rgb]
            )
        if all(item is not None for item in ir):
            ir = torch.cat(ir)
            ir = self.optional_ir(ir, x.shape)
        else:
            ir = torch.cat([self.optional_ir(item, (1,) + x.shape[1:]) for item in ir])
            
        # Downsample according to attention_pooling_rate
        rgb = F.avg_pool2d(rgb, self.attention_pooling_rate)
        ir = F.avg_pool2d(ir, self.attention_pooling_rate)
        
        rgb_pe = repeat(self.pe(rgb.shape[-2:]), "c h w -> b c h w", b=rgb.shape[0])
        ir_pe = repeat(self.pe(ir.shape[-2:]), "c h w -> b c h w", b=ir.shape[0])
        b, c, h, w = rgb.shape
        rgb, ir = self.transformer(rgb, rgb_pe, ir, ir_pe)
        rgb = rearrange(rgb, "b (h w) c -> b c h w", h=h, w=w)
        ir = rearrange(ir, "b (h w) c -> b c h w", h=h, w=w)
        
        # Upsample to original size
        rgb = F.interpolate(rgb, scale_factor=self.attention_pooling_rate, mode="bilinear")
        ir = F.interpolate(ir, scale_factor=self.attention_pooling_rate, mode="bilinear")
        
        return rgb + ir