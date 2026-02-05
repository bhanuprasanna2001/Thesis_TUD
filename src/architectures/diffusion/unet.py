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
    
    def __init__(self, in_channels, out_channels, base_channels=128, groups=8, time_emb_dim=512):
        super().__init__()
        
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        
        # Time embedding
        self.time_emb = TimeEmbedding(base_channels, time_emb_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        
        # Encoder
        self.enc1 = EncoderBlock(c1, c1, time_emb_dim, num_groups=groups, use_attention=False)
        self.enc2 = EncoderBlock(c1, c2, time_emb_dim, num_groups=groups, use_attention=False)
        self.enc3 = EncoderBlock(c2, c3, time_emb_dim, num_groups=groups, use_attention=True)
        
        # Bottleneck
        self.bottleneck_res1 = ResidualBlock(c3, c4, time_emb_dim, num_groups=groups)
        self.bottleneck_res2 = ResidualBlock(c4, c4, time_emb_dim, num_groups=groups)
        self.bottleneck_attn = AttentionBlock(c4, num_groups=groups)
        
        # Decoder
        self.dec3 = DecoderBlock(c4, c3, time_emb_dim, num_groups=groups, use_attention=True)
        self.dec2 = DecoderBlock(c3, c2, time_emb_dim, num_groups=groups, use_attention=False)
        self.dec1 = DecoderBlock(c2, c1, time_emb_dim, num_groups=groups, use_attention=False)
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(groups, c1),
            nn.SiLU(),
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)
        )
        
        
    def forward(self, X, t):
        # Time embedding
        time_emb = self.time_emb(t)
        
        # Initial convolution
        X = self.conv_in(X)
        
        # Encoder
        X, skip1 = self.enc1(X, time_emb)
        X, skip2 = self.enc2(X, time_emb)
        X, skip3 = self.enc3(X, time_emb)
        
        # Bottleneck
        X = self.bottleneck_res1(X, time_emb)
        X = self.bottleneck_res2(X, time_emb)
        X = self.bottleneck_attn(X)
        
        # Decoder
        X = self.dec3(X, skip3, time_emb)
        X = self.dec2(X, skip2, time_emb)
        X = self.dec1(X, skip1, time_emb)
        
        # Output
        X = self.conv_out(X)
        
        return X
        
        
if __name__ == "__main__":
    print("Testing UNetDiffusion:")
    
    # Test 1: RGB images
    X = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    model = UNetDiffusion(in_channels=3, out_channels=3, base_channels=64)
    out = model(X, t)
    print(f"RGB Test: {X.size()} -> {out.size()}")
    assert X.size() == out.size()
    
    # Test 2: MNIST-like images
    X = torch.randn(1, 1, 32, 32)
    t = torch.randint(0, 1000, (1,))
    model = UNetDiffusion(in_channels=1, out_channels=1, base_channels=64)
    out = model(X, t)
    print(f"MNIST Test: {X.size()} -> {out.size()}")
    assert X.size() == out.size()
    
    print("✓ All tests passed!")
