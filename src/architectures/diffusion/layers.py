import torch
import torch.nn as nn
import torch.nn.functional as F

# Ref: https://github.com/mattroz/diffusion-ddpm/blob/main/src/model/layers.py
# Attention is all you need paper - Secion 3.5

class SinusodialPositionalEmbedding(nn.Module):

    def __init__(self, dim, timesteps=1000):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even."

        self.dim = dim
        pe_matrix = torch.zeros(timesteps, dim)

        # Get all the even dimensions across the embedding vector
        even_indices = torch.arange(0, self.dim, 2)

        # Calculating 10000^(2i/d), we use exp(log(10000) * (2i/d)) for efficiency.
        log_term = torch.log(torch.tensor(10000.0)) / self.dim
        div_term = torch.exp(even_indices * -log_term)

        # Precompute positional encoding matrix based on odd/even timesteps
        t = torch.arange(timesteps).unsqueeze(1)
        pe_matrix[:, 0::2] = torch.sin(t * div_term)
        pe_matrix[:, 1::2] = torch.cos(t * div_term)

        self.register_buffer("pe_matrix", pe_matrix, persistent=False)

    
    def forward(self, timestep):
        timestep = timestep.long()
        return self.pe_matrix[timestep]


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, time_emb_dim=None, num_groups=32):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv block
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_emb_proj = None
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Second conv block
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Residual connection
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    
    def forward(self, X, time_emb=None):
        residual = X
        
        # First conv block
        X = self.norm1(X)
        X = F.silu(X)
        X = self.conv1(X)
        
        # Add time embedding
        if self.time_emb_proj is not None and time_emb is not None:
            time_emb = self.time_emb_proj(F.silu(time_emb))
            X = X + time_emb[:, :, None, None]
        
        # Second conv block
        X = self.norm2(X)
        X = F.silu(X)
        X = self.conv2(X)
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        return X + residual


class Downsample(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    
    def forward(self, X):
        X = self.down(X)
        return X


class Upsample(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    
    
    def forward(self, X):
        X = self.up(X)
        return X


class AttentionBlock(nn.Module):
    
    def __init__(self, channels, num_heads=1, num_groups=32):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        # Normalization
        self.norm = nn.GroupNorm(num_groups, channels)
        
        # Q, K, V projections
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        
        # Output projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    
    def forward(self, X):
        residual = X
        B, C, H, W = X.shape
        
        # Normalize
        X = self.norm(X)
        
        # Compute Q, K, V
        qkv = self.qkv(X)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Multi-head attention
        # Reshape: (B, C, H, W) -> (B, num_heads, C//num_heads, H*W)
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, H * W)
        k = k.view(B, self.num_heads, head_dim, H * W)
        v = v.view(B, self.num_heads, head_dim, H * W)
        
        # Scaled dot-product attention per head
        # (B, num_heads, H*W, head_dim) @ (B, num_heads, head_dim, H*W) -> (B, num_heads, H*W, H*W)
        scale = head_dim ** -0.5
        attn = torch.matmul(q.transpose(-2, -1), k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        # (B, num_heads, H*W, H*W) @ (B, num_heads, H*W, head_dim) -> (B, num_heads, H*W, head_dim)
        X = torch.matmul(attn, v.transpose(-2, -1))
        
        # Reshape back: (B, num_heads, H*W, head_dim) -> (B, C, H, W)
        X = X.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        # Output projection
        X = self.proj(X)
        
        # Residual connection
        return X + residual


if __name__ == "__main__":
    # Test SinusodialPositionalEmbedding
    print("Testing SinusodialPositionalEmbedding:")
    dim = 16
    timesteps = 10
    spe = SinusodialPositionalEmbedding(dim=dim, timesteps=timesteps)
    timestep = torch.tensor(3)
    embedding = spe(timestep)
    print(f"Positional Encoding for timestep 3: {embedding.size()}")
    
    # Test ResidualBlock
    print("\nTesting ResidualBlock:")
    X = torch.randn(2, 64, 32, 32)
    time_emb = torch.randn(2, 128)
    res_block = ResidualBlock(64, 128, time_emb_dim=128)
    out = res_block(X, time_emb)
    print(f"Input: {X.size()} -> Output: {out.size()}")
    
    # Test Downsample
    print("\nTesting Downsample:")
    X = torch.randn(2, 64, 32, 32)
    down = Downsample(64)
    out = down(X)
    print(f"Input: {X.size()} -> Output: {out.size()}")
    
    # Test Upsample
    print("\nTesting Upsample:")
    X = torch.randn(2, 64, 16, 16)
    up = Upsample(64)
    out = up(X)
    print(f"Input: {X.size()} -> Output: {out.size()}")
    
    # Test AttentionBlock
    print("\nTesting AttentionBlock:")
    X = torch.randn(2, 64, 16, 16)
    attn = AttentionBlock(64)
    out = attn(X)
    print(f"Input: {X.size()} -> Output: {out.size()}")

