import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import SinusodialPositionalEmbedding, ResidualBlock, Downsample, Upsample, AttentionBlock


class TimeEmbedding(nn.Module):
    
    def __init__(self, base_channels, time_emb_dim):
        super().__init__()
        
        self.embedding = nn.Sequential(
            SinusodialPositionalEmbedding(dim=base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
    
    
    def forward(self, t):
        return self.embedding(t)


class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8, use_attention=False):
        super().__init__()
        
        self.res1 = ResidualBlock(in_channels, out_channels, time_emb_dim, num_groups=num_groups)
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim, num_groups=num_groups)
        self.attn = AttentionBlock(out_channels, num_groups=num_groups) if use_attention else nn.Identity()
        self.down = Downsample(out_channels)
    
    
    def forward(self, X, time_emb):
        X = self.res1(X, time_emb)
        X = self.res2(X, time_emb)
        X = self.attn(X)
        skip = X
        X = self.down(X)
        return X, skip


class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8, use_attention=False):
        super().__init__()
        
        self.up = Upsample(in_channels)
        self.res1 = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim, num_groups=num_groups)
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim, num_groups=num_groups)
        self.attn = AttentionBlock(out_channels, num_groups=num_groups) if use_attention else nn.Identity()
    
    
    def forward(self, X, skip, time_emb):
        X = self.up(X)
        
        # Pad X to match skip dimensions (handle non-power-of-2 sizes)
        diffY = skip.size()[2] - X.size()[2]
        diffX = skip.size()[3] - X.size()[3]
        X = F.pad(X, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        X = torch.cat([X, skip], dim=1)
        X = self.res1(X, time_emb)
        X = self.res2(X, time_emb)
        X = self.attn(X)
        return X


class UNetDiffusion(nn.Module):
    
    def __init__(self, in_channels, out_channels, base_channels=128, groups=8, time_emb_dim=512, n_levels=3, channel_mults=None, attn_levels=None):
        super().__init__()
        
        if channel_mults is None:
            channel_mults = tuple(2**i for i in range(n_levels))
        
        n_levels = len(channel_mults)
        self.n_levels = n_levels
        
        if attn_levels is None:
            attn_levels = {n_levels - 1}
        else:
            attn_levels = set(attn_levels)
        
        channels = [base_channels * m for m in channel_mults]
        bottleneck_ch = channels[-1] * 2
        
        # Time embedding
        self.time_emb = TimeEmbedding(base_channels, time_emb_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        
        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(n_levels):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]
            use_attn = i in attn_levels
            self.encoders.append(EncoderBlock(in_ch, out_ch, time_emb_dim, num_groups=groups, use_attention=use_attn))
        
        # Bottleneck
        self.bottleneck_res1 = ResidualBlock(channels[-1], bottleneck_ch, time_emb_dim, num_groups=groups)
        self.bottleneck_res2 = ResidualBlock(bottleneck_ch, bottleneck_ch, time_emb_dim, num_groups=groups)
        self.bottleneck_attn = AttentionBlock(bottleneck_ch, num_groups=groups)
        
        # Decoder (reversed)
        self.decoders = nn.ModuleList()
        for i in reversed(range(n_levels)):
            in_ch = bottleneck_ch if i == n_levels - 1 else channels[i + 1]
            out_ch = channels[i]
            use_attn = i in attn_levels
            self.decoders.append(DecoderBlock(in_ch, out_ch, time_emb_dim, num_groups=groups, use_attention=use_attn))
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(groups, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
        )
        
        
    def forward(self, X, t):
        # Time embedding
        time_emb = self.time_emb(t)
        
        # Initial convolution
        X = self.conv_in(X)
        
        # Encoder
        skips = []
        for encoder in self.encoders:
            X, skip = encoder(X, time_emb)
            skips.append(skip)
        
        # Bottleneck
        X = self.bottleneck_res1(X, time_emb)
        X = self.bottleneck_res2(X, time_emb)
        X = self.bottleneck_attn(X)
        
        # Decoder (uses skips in reverse)
        for decoder in self.decoders:
            skip = skips.pop()
            X = decoder(X, skip, time_emb)
        
        # Output
        X = self.conv_out(X)
        
        return X
        
        
if __name__ == "__main__":
    print("Testing UNetDiffusion:")
    
    # Test 1: RGB images (default 3 levels)
    X = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    model = UNetDiffusion(in_channels=3, out_channels=3, base_channels=64)
    out = model(X, t)
    print(f"RGB 3-level Test: {X.size()} -> {out.size()}")
    assert X.size() == out.size()
    
    # Test 2: MNIST-like images
    X = torch.randn(1, 1, 32, 32)
    t = torch.randint(0, 1000, (1,))
    model = UNetDiffusion(in_channels=1, out_channels=1, base_channels=64)
    out = model(X, t)
    print(f"MNIST Test: {X.size()} -> {out.size()}")
    assert X.size() == out.size()
    
    # Test 3: Latent diffusion with 2 levels on 8x8
    X = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    model = UNetDiffusion(in_channels=4, out_channels=4, base_channels=64, n_levels=2)
    out = model(X, t)
    print(f"Latent 2-level Test: {X.size()} -> {out.size()}")
    assert X.size() == out.size()
    
    # Test 4: Single level
    X = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    model = UNetDiffusion(in_channels=4, out_channels=4, base_channels=32, n_levels=1)
    out = model(X, t)
    print(f"Latent 1-level Test: {X.size()} -> {out.size()}")
    assert X.size() == out.size()
    
    print("✓ All tests passed!")
