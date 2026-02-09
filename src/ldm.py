import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F

from .architectures import AE
from .diffusion import Diffusion


class LDM(L.LightningModule):

    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        latent_channels=64,
        out_channels=1,
        lr=0.0002,
        preset="tiny",
        start=0.0001,
        end=0.02,
        timesteps=1000,
        scheduler_type="linear",
        img_shape=(1, 28, 28),
        groups=8,
        time_emb_dim=512,
        use_ema=True,
        ema_decay=0.9999,
        recon_weight=1.0,
        diff_weight=0.1,
        warmup_steps=0,
        detach_latent_for_diff=True,
    ):
        super().__init__()

        self.lr = lr

        self.recon_weight = recon_weight
        self.diff_weight = diff_weight
        self.warmup_steps = warmup_steps
        self.detach_latent_for_diff = detach_latent_for_diff

        self.klrae = KLRAE(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )

        _, H, W = img_shape
        H_lat, W_lat = H // 4, W // 4

        self.diffusion = Diffusion(
            in_channels=latent_channels,
            out_channels=latent_channels,
            lr=lr,
            preset=preset,
            start=start,
            end=end,
            timesteps=timesteps,
            scheduler_type=scheduler_type,
            img_shape=(latent_channels, H_lat, W_lat),
            groups=groups,
            time_emb_dim=time_emb_dim,
            use_ema=use_ema,
            ema_decay=ema_decay
        )


    def training_step(self, batch, batch_idx):
        x, _ = batch

        # ---- autoencoder ----
        z = self.klrae.encoder(x)
        x_hat = self.klrae.decoder(z)
        recon_loss = F.mse_loss(x_hat, x)

        # ---- diffusion in latent ----
        diff_loss = torch.tensor(0.0, device=x.device)

        if self.global_step >= self.warmup_steps:
            z_for_diff = z.detach() if self.detach_latent_for_diff else z

            noise, t, eps = self.diffusion.q_sample(z_for_diff)
            pred_noise = self.diffusion.network(noise, t)
            diff_loss = F.mse_loss(pred_noise, eps)

        loss = (self.recon_weight * recon_loss) + (self.diff_weight * diff_loss)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/recon_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/diff_loss", diff_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # ---- autoencoder ----
        z = self.klrae.encoder(x)
        x_hat = self.klrae.decoder(z)
        recon_loss = F.mse_loss(x_hat, x)

        # ---- diffusion in latent ----
        noise, t, eps = self.diffusion.q_sample(z.detach())
        pred_noise = self.diffusion.network(noise, t)
        diff_loss = F.mse_loss(pred_noise, eps)

        loss = (self.recon_weight * recon_loss) + (self.diff_weight * diff_loss)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/recon_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/diff_loss", diff_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss


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