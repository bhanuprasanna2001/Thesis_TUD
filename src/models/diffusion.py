import torch
import torch.nn.functional as F
import lightning as L
from typing import Optional, Tuple, Dict, Any, List

from .unet import UNet


class DiffusionModel(L.LightningModule):
    """DDPM implementation as a Lightning module.
    
    Implements the forward and reverse diffusion process from Ho et al. (2020).
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        in_channels: int = 1,
        norm_type: str = "group",
        num_groups: int = 8,
        preset: Optional[str] = None,
        lr: float = 1e-3,
        img_shape: Tuple[int, ...] = (1, 28, 28),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.timesteps = timesteps
        self.lr = lr
        self.img_shape = img_shape
        
        # Type annotations for registered buffers
        self.betas: torch.Tensor
        self.alphas: torch.Tensor
        self.alphas_cumprod: torch.Tensor
        self.sqrt_alphas_cumprod: torch.Tensor
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor
        self.sqrt_recip_alphas: torch.Tensor
        self.posterior_variance: torch.Tensor

        self.network = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            norm_type=norm_type,
            num_groups=num_groups,
            preset=preset,
        )

        self._log_model_info()

        # Precompute diffusion schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def _log_model_info(self) -> None:
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f"UNet parameters: {total_params:,} (trainable: {trainable_params:,})")

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract coefficients at timestep t."""
        batch_size = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise."""
        return self.network(x, t)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        batch_size = x.shape[0]

        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        predicted_noise = self.network(x_noisy, t)

        loss = F.mse_loss(predicted_noise, noise)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        batch_size = x.shape[0]

        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        predicted_noise = self.network(x_noisy, t)

        loss = F.mse_loss(predicted_noise, noise)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs or 9
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """Reverse diffusion step: p(x_{t-1} | x_t)."""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.network(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean

        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, n: int, steps: Optional[int] = None) -> torch.Tensor:
        """Generate samples from noise."""
        steps = steps or self.timesteps
        self.eval()

        x = torch.randn(n, *self.img_shape, device=self.device)
        for i in reversed(range(steps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, i)

        return x
