import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

# Preset configurations for different model sizes
UNET_PRESETS = {
    "tiny": {"base_channels": 16, "channel_mult": [1, 2], "time_emb_dim": 32},       # ~90K
    "small": {"base_channels": 32, "channel_mult": [1, 2], "time_emb_dim": 64},      # ~350K
    "medium": {"base_channels": 32, "channel_mult": [1, 2, 4], "time_emb_dim": 64},  # ~1.4M
    "large": {"base_channels": 64, "channel_mult": [1, 2, 4], "time_emb_dim": 128},  # ~5.5M
}


def get_norm_layer(norm_type: str, num_channels: int, num_groups: int = 8) -> nn.Module:
    """Factory for normalization layers."""
    if norm_type == "group":
        return nn.GroupNorm(min(num_groups, num_channels), num_channels)
    elif norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_channels)
    elif norm_type == "layer":
        return nn.GroupNorm(1, num_channels)
    raise ValueError(f"Unknown norm type: {norm_type}")


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """Residual block with time conditioning using pre-norm (Norm → ReLU → Conv).
    
    Pre-norm is more stable for deep networks (as used in original DDPM).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        norm_type: str = "group",
        num_groups: int = 8,
    ):
        super().__init__()
        self.norm1 = get_norm_layer(norm_type, in_channels, num_groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = get_norm_layer(norm_type, out_channels, num_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Pre-norm: Norm → ReLU → Conv
        h = F.relu(self.norm1(x))
        h = self.conv1(h)
        # Add time embedding
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        # Second pre-norm block
        h = F.relu(self.norm2(h))
        h = self.conv2(h)
        return h + self.residual_conv(x)


class UNet(nn.Module):
    """U-Net architecture for noise prediction in diffusion models.
    
    Supports configurable depth via channel_mult and preset configurations.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mult: Optional[List[int]] = None,
        time_emb_dim: int = 128,
        norm_type: str = "group",
        num_groups: int = 8,
        preset: Optional[str] = None,
    ):
        super().__init__()

        if preset is not None:
            cfg = UNET_PRESETS[preset]
            base_channels = cfg["base_channels"]
            channel_mult = cfg["channel_mult"]
            time_emb_dim = cfg["time_emb_dim"]

        if channel_mult is None:
            channel_mult = [1, 2, 4]

        self.channel_mult = channel_mult

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        encoder_channels = [base_channels]
        ch_in = base_channels
        for i, mult in enumerate(channel_mult):
            ch_out = base_channels * mult
            self.down_blocks.append(
                ResidualBlock(ch_in, ch_out, time_emb_dim, norm_type, num_groups)
            )
            encoder_channels.append(ch_out)
            # Strided conv for downsampling (except last level)
            if i < len(channel_mult) - 1:
                self.downsample.append(nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1))
            ch_in = ch_out

        # Final downsample before bottleneck
        self.downsample_bottleneck = nn.Conv2d(ch_in, ch_in, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = ResidualBlock(ch_in, ch_in, time_emb_dim, norm_type, num_groups)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mult)):
            skip_ch = encoder_channels[-(i + 1)]
            ch_out = base_channels * mult
            self.up_blocks.append(
                ResidualBlock(ch_in + skip_ch, ch_out, time_emb_dim, norm_type, num_groups)
            )
            ch_in = ch_out

        self.conv_out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = self.conv_in(x)

        # Encoder with strided conv downsampling
        skip_connections = []
        for i, block in enumerate(self.down_blocks):
            x = block(x, t_emb)
            skip_connections.append(x)
            if i < len(self.downsample):
                x = self.downsample[i](x)

        # Bottleneck
        x = self.downsample_bottleneck(x)
        x = self.bottleneck(x, t_emb)

        # Decoder
        for i, block in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 1)]
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)

        return self.conv_out(x)


class UNetSegmentation(nn.Module):
    """U-Net for image segmentation (no time conditioning)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mult: Optional[List[int]] = None,
        norm_type: str = "group",
        num_groups: int = 8,
        preset: Optional[str] = None,
    ):
        super().__init__()

        if preset is not None:
            cfg = UNET_PRESETS[preset]
            base_channels = cfg["base_channels"]
            channel_mult = cfg["channel_mult"]

        if channel_mult is None:
            channel_mult = [1, 2, 4]

        self.channel_mult = channel_mult
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder with strided conv downsampling
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        encoder_channels = [base_channels]
        ch_in = base_channels
        for i, mult in enumerate(channel_mult):
            ch_out = base_channels * mult
            self.down_blocks.append(
                nn.Sequential(
                    get_norm_layer(norm_type, ch_in, num_groups),
                    nn.ReLU(),
                    nn.Conv2d(ch_in, ch_out, 3, padding=1),
                    get_norm_layer(norm_type, ch_out, num_groups),
                    nn.ReLU(),
                    nn.Conv2d(ch_out, ch_out, 3, padding=1),
                )
            )
            encoder_channels.append(ch_out)
            if i < len(channel_mult) - 1:
                self.downsample.append(nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1))
            ch_in = ch_out

        # Final downsample before bottleneck
        self.downsample_bottleneck = nn.Conv2d(ch_in, ch_in, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            get_norm_layer(norm_type, ch_in, num_groups),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_in, 3, padding=1),
            get_norm_layer(norm_type, ch_in, num_groups),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_in, 3, padding=1),
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mult)):
            skip_ch = encoder_channels[-(i + 1)]
            ch_out = base_channels * mult
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch_in + skip_ch, ch_out, 3, padding=1),
                    get_norm_layer(norm_type, ch_out, num_groups),
                    nn.ReLU(),
                    nn.Conv2d(ch_out, ch_out, 3, padding=1),
                    get_norm_layer(norm_type, ch_out, num_groups),
                    nn.ReLU(),
                )
            )
            ch_in = ch_out

        self.conv_out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)

        # Encoder with strided conv downsampling
        skip_connections = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            skip_connections.append(x)
            if i < len(self.downsample):
                x = self.downsample[i](x)

        # Bottleneck
        x = self.downsample_bottleneck(x)
        x = self.bottleneck(x)

        # Decoder
        for i, block in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 1)]
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.conv_out(x)
