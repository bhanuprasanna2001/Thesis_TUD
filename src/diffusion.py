import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F

from .architectures import UNetDiffusion

from .utils import get_scheduler, get_unet_preset, EMA


class Diffusion(L.LightningModule):

    def __init__(self, in_channels=1, out_channels=1, lr=0.0002, preset="tiny", start=0.0001, end=0.02, timesteps=1000, scheduler_type="linear", img_shape=(1, 28, 28), groups=8, time_emb_dim=512, use_ema=True, ema_decay=0.9999):
        super().__init__()

        self.lr = lr
        self.img_shape = img_shape
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        self.start = start
        self.end = end
        self.timesteps = timesteps
        self.scheduler_type = scheduler_type

        beta = get_scheduler(start, end, timesteps, scheduler_type)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
        recp_sqrt_alpha_bar = 1 / sqrt_alpha_bar
        beta_recp_sqrt_one_minus_alpha_bar = beta / sqrt_one_minus_alpha_bar

        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
        posterior_variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar)
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_bar)
        self.register_buffer("recp_sqrt_alpha_bar", recp_sqrt_alpha_bar)
        self.register_buffer("beta_recp_sqrt_one_minus_alpha_bar", beta_recp_sqrt_one_minus_alpha_bar)
        self.register_buffer("posterior_variance", posterior_variance)

        base_channels = get_unet_preset(preset)

        self.network = UNetDiffusion(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels, groups=groups, time_emb_dim=time_emb_dim)

        # Initialize EMA (will be properly set up in on_fit_start)
        self.ema = None


    def on_fit_start(self):
        """Initialize EMA after model is moved to the correct device."""
        if self.use_ema:
            self.ema = EMA(self.network, decay=self.ema_decay)


    def on_before_optimizer_step(self, optimizer):
        """Update EMA parameters after each training step."""
        if self.use_ema and self.ema is not None:
            self.ema.update(self.network)


    def q_sample(self, batch):
        batch_size = batch.size(0)
        rand_t = torch.randint(low=0, high=self.timesteps, size=(batch_size,), device=batch.device, dtype=torch.long)

        ndims = batch.ndim - 1
        shape = (batch_size,) + (1,) * ndims

        sqrt_alpha_bar = self.sqrt_alpha_bar[rand_t].view(shape)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[rand_t].view(shape)

        eps = torch.randn_like(batch)
        x_t = sqrt_alpha_bar * batch + sqrt_one_minus_alpha_bar * eps

        # print(rand_t.size(), sqrt_alpha_bar.size(), sqrt_one_minus_alpha_bar.size(), eps.size())
        # print(batch.size())

        return x_t, rand_t, eps


    def training_step(self, batch, batch_idx):
        x, _ = batch
        noise, t, eps = self.q_sample(x)
        pred_noise = self.network(noise, t)

        loss = F.mse_loss(pred_noise, eps)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, _ = batch
        noise, t, eps = self.q_sample(x)
        pred_noise = self.network(noise, t)

        loss = F.mse_loss(pred_noise, eps)
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        ndims = x.ndim - 1
        shape = (x.size(0),) + (1,) * ndims

        beta_t = self.beta[t].view(shape)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(shape)
        recp_sqrt_alpha_t = (1 / torch.sqrt(self.alpha[t])).view(shape)

        eps_theta = self.network(x, t)

        mean = recp_sqrt_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta)

        if t_index == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            var = self.posterior_variance[t].view(shape)
            return mean + torch.sqrt(var) * noise


    @torch.no_grad()
    def sample(self, n):
        self.eval()
        device = self.device
        
        # Use EMA weights for sampling if available
        if self.use_ema and self.ema is not None:
            self.ema.apply(self.network)
        
        x = torch.randn(n, *self.img_shape, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i)
        
        # Restore original weights after sampling
        if self.use_ema and self.ema is not None:
            self.ema.restore(self.network)
            
        return x
    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs or 300
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


if __name__ == "__main__":
    model = Diffusion()

    batch_size = 4
    data_dim = 10
    dummy_batch = torch.randn(batch_size, 1, 28, 28)

    # 1. Update unpacking here
    x_t, rand_t, actual_eps = model.q_sample(dummy_batch)

    assert x_t.shape == dummy_batch.shape, "x_t shape mismatch"
    assert rand_t.shape == (batch_size,), "rand_t shape mismatch"
    assert actual_eps.shape == dummy_batch.shape, "eps shape mismatch"

    # 2. Fix the check: Reconstruct eps from x_t and compare to actual_eps
    ndims = dummy_batch.ndim - 1
    shape = (batch_size,) + (1,) * ndims

    sqrt_alpha_bar = model.sqrt_alpha_bar[rand_t].view(shape)
    sqrt_one_minus_alpha_bar = model.sqrt_one_minus_alpha_bar[rand_t].view(shape)
    
    # Algebra: eps = (x_t - sqrt_alpha_bar * x_0) / sqrt_one_minus_alpha_bar
    reconstructed_eps = (x_t - sqrt_alpha_bar * dummy_batch) / sqrt_one_minus_alpha_bar
    
    assert torch.allclose(actual_eps, reconstructed_eps, atol=1e-5), "x_t math is inconsistent"
    print("Test passed! Forward diffusion math is correct.")

    