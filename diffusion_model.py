import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import numpy as np
from tqdm import tqdm

device = "mps" if torch.mps.is_available() else "cpu"

transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5], std=[0.5])
])

training_data = tv.datasets.MNIST(
    root="data",
    train=True,
    transform=transforms,
    download=True,
)

testing_data = tv.datasets.MNIST(
    root="data",
    train=False,
    transform=transforms,
    download=True,
)

train_loader = torch.utils.data.DataLoader(training_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=128, shuffle=False)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.relu(x))
        h = self.norm1(h)
        h = h + self.time_mlp(F.relu(t_emb))[:, :, None, None]
        h = F.relu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return h + self.residual_conv(x)


class UNet(nn.Module):
    """U-Net architecture for noise prediction."""
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128, base_channels=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        # Decoder
        self.up1 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.up2 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)
        self.up3 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)
        
        self.conv_out = nn.Conv2d(base_channels, out_channels, 1)
        
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x = self.conv_in(x)
        
        # Encoder with skip connections
        d1 = self.down1(x, t_emb)
        d2 = self.down2(F.max_pool2d(d1, 2), t_emb)
        d3 = self.down3(F.max_pool2d(d2, 2), t_emb)
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(d3, 2), t_emb)
        
        # Decoder with skip connections (upsample to match skip connection sizes)
        u1 = self.up1(torch.cat([F.interpolate(b, size=d3.shape[2:], mode='nearest'), d3], dim=1), t_emb)
        u2 = self.up2(torch.cat([F.interpolate(u1, size=d2.shape[2:], mode='nearest'), d2], dim=1), t_emb)
        u3 = self.up3(torch.cat([F.interpolate(u2, size=d1.shape[2:], mode='nearest'), d1], dim=1), t_emb)
        
        return self.conv_out(u3)


class DiffusionModel(nn.Module):
    """DDPM-style diffusion model for image generation."""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        self.network = UNet(in_channels=1, out_channels=1, time_emb_dim=128, base_channels=64)
        
        # Variance schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x, c=None):
        """Compute training loss for given data x."""
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x.device).long()
        
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        
        predicted_noise = self.network(x_noisy, t)
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss

    def p_sample(self, x, t, t_index):
        """Single reverse diffusion step: sample x_{t-1} from p(x_{t-1} | x_t)."""
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.network(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def integrate_forward(self, x, steps=None):
        """Move samples from data space to latent space (forward/noising)."""
        if steps is None:
            steps = self.timesteps
        t = torch.full((x.shape[0],), steps - 1, device=x.device, dtype=torch.long)
        return self.q_sample(x, t)

    def integrate_inverse(self, z, steps=None):
        """Move samples from latent space to data space (reverse/denoising)."""
        if steps is None:
            steps = self.timesteps
        
        x = z
        for i in reversed(range(0, steps)):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            x = self.p_sample(x, t, i)
        
        return x

    def integrate(self, xz, steps=None, inverse=False):
        """Generic integrator: forward if inverse=False else inverse."""
        if inverse:
            return self.integrate_inverse(xz, steps)
        else:
            return self.integrate_forward(xz, steps)

    @torch.no_grad()
    def sample(self, n, c=None, steps=None, img_shape=(1, 28, 28)):
        """Draw n random samples from the model."""
        if steps is None:
            steps = self.timesteps
        
        self.eval()
        z = torch.randn(n, *img_shape, device=next(self.parameters()).device)
        samples = self.integrate_inverse(z, steps)
        return samples


def train_model(model, train_loader, epochs=10, lr=1e-3):
    """Training loop for the diffusion model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(model)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(x)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    return model


if __name__ == "__main__":
    # Initialize model
    model = DiffusionModel(timesteps=200)
    print(f"Model initialized on device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model = train_model(model, train_loader, epochs=10, lr=1e-3)
    
    # Generate samples
    print("\nGenerating samples...")
    samples = model.sample(n=16)
    samples = (samples.clamp(-1, 1) + 1) / 2  # Denormalize to [0, 1]
    
    # Save samples
    tv.utils.save_image(samples, "generated_samples.png", nrow=4)
    print("Samples saved to generated_samples.png")
    
    # Save model
    torch.save(model.state_dict(), "diffusion_model.pth")
    print("Model saved to diffusion_model.pth")