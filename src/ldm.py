import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F

from .architectures import AE, KLAE, VQAE, PatchDiscriminator
from .diffusion import Diffusion


class LDM(L.LightningModule):

    def __init__(
        self,
        ae_type="ae",
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
        n_levels=3,
        channel_mults=None,
        use_ema=True,
        ema_decay=0.9999,
        recon_weight=1.0,
        reg_weight=0.0,
        diff_weight=1.0,
        warmup_steps=0,
        detach_latent_for_diff=True,
        num_embeddings=512,
        commitment_cost=0.25,
        use_vq_gan_loss=False,
        adv_weight=0.1,
        fm_weight=1.0,
        disc_start_step=0,
        disc_channels=64,
        disc_num_layers=3,
        disc_lr_mult=1.0,
        manual_gradient_clip_val=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.ae_type = ae_type
        self.use_ema = use_ema

        self.recon_weight = recon_weight
        self.reg_weight = reg_weight
        self.diff_weight = diff_weight
        self.warmup_steps = warmup_steps
        self.detach_latent_for_diff = detach_latent_for_diff
        self.use_vq_gan_loss = (ae_type == "vq") and use_vq_gan_loss
        self.adv_weight = adv_weight
        self.fm_weight = fm_weight
        self.disc_start_step = disc_start_step
        self.disc_lr_mult = disc_lr_mult
        self.manual_gradient_clip_val = manual_gradient_clip_val

        # ---- autoencoder ----
        if ae_type == "ae":
            self.ae = AE(
                groups=groups,
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
                latent_channels=latent_channels,
            )
        elif ae_type == "kl":
            self.ae = KLAE(
                groups=groups,
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
                latent_channels=latent_channels,
            )
        elif ae_type == "vq":
            self.ae = VQAE(
                groups=groups,
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels,
                latent_channels=latent_channels,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
            )
        else:
            raise ValueError(f"Unknown ae_type: {ae_type}. Choose 'ae', 'kl', or 'vq'.")

        # ---- latent shape ----
        _, H, W = img_shape
        H_lat, W_lat = H // 4, W // 4
        self.latent_shape = (latent_channels, H_lat, W_lat)

        # ---- diffusion in latent space ----
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
            ema_decay=ema_decay,
            n_levels=n_levels,
            channel_mults=channel_mults,
        )

        self.discriminator = None
        if self.use_vq_gan_loss:
            self.discriminator = PatchDiscriminator(
                in_channels=out_channels,
                base_channels=disc_channels,
                n_layers=disc_num_layers,
            )


    @property
    def automatic_optimization(self):
        return not self.use_vq_gan_loss


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        if self.use_vq_gan_loss:
            return

        # Let Lightning do the actual stepping/clipping/etc
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)

        # Diffusion.optimizer_step won't fire as a sub-module, so handle EMA here
        if self.use_ema and self.diffusion.ema is not None:
            self.diffusion.ema.update(self.diffusion.network)


    def _set_discriminator_grad(self, requires_grad: bool):
        if self.discriminator is None:
            return

        for p in self.discriminator.parameters():
            p.requires_grad_(requires_grad)


    def _generator_adv_losses(self, x, x_hat):
        if self.discriminator is None:
            zero = torch.tensor(0.0, device=x.device)
            return zero, zero

        fake_logits, fake_features = self.discriminator(x_hat, return_features=True)
        with torch.no_grad():
            _, real_features = self.discriminator(x, return_features=True)

        fm_loss = torch.tensor(0.0, device=x.device)
        for real_f, fake_f in zip(real_features, fake_features):
            fm_loss = fm_loss + F.l1_loss(fake_f, real_f)

        if len(fake_features) > 0:
            fm_loss = fm_loss / len(fake_features)

        gan_g_loss = -fake_logits.mean()
        return gan_g_loss, fm_loss


    def _discriminator_loss(self, x, x_hat):
        if self.discriminator is None:
            return torch.tensor(0.0, device=x.device)

        real_logits = self.discriminator(x)
        fake_logits = self.discriminator(x_hat.detach())

        loss_real = F.relu(1.0 - real_logits).mean()
        loss_fake = F.relu(1.0 + fake_logits).mean()
        return 0.5 * (loss_real + loss_fake)


    def _maybe_clip_gradients(self, optimizer):
        if self.use_vq_gan_loss:
            clip = self.manual_gradient_clip_val
        else:
            clip = self.trainer.gradient_clip_val

        if clip is None or clip <= 0:
            return

        clip_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
        self.clip_gradients(clip_optimizer, gradient_clip_val=clip, gradient_clip_algorithm="norm")


    def _encode(self, x):
        """Encode input to latent z and compute AE regularization loss.
        
        Returns:
            z: latent tensor for diffusion
            x_hat: reconstruction
            recon_loss: MSE reconstruction loss
            reg_loss: regularization loss (KL / VQ / zero for plain AE)
        """
        if self.ae_type == "ae":
            z = self.ae.encoder(x)
            x_hat = self.ae.decoder(z)
            recon_loss = F.mse_loss(x_hat, x)
            reg_loss = torch.tensor(0.0, device=x.device)

        elif self.ae_type == "kl":
            mu, logvar = self.ae.encoder(x)
            z = self.ae.reparameterize(mu, logvar)
            x_hat = self.ae.decoder(z)
            recon_loss = F.mse_loss(x_hat, x)
            reg_loss = self.ae.kl_divergence(mu, logvar)

        elif self.ae_type == "vq":
            z_e = self.ae.encoder(x)
            z, vq_loss, _ = self.ae.vq(z_e)
            x_hat = self.ae.decoder(z)
            if self.use_vq_gan_loss:
                recon_loss = F.l1_loss(x_hat, x)
            else:
                recon_loss = F.mse_loss(x_hat, x)
            reg_loss = vq_loss

        return z, x_hat, recon_loss, reg_loss


    def reconstruct(self, x):
        """Encode and decode for deterministic reconstruction."""
        with torch.no_grad():
            if self.ae_type == "ae":
                z = self.ae.encoder(x)
            elif self.ae_type == "kl":
                mu, _ = self.ae.encoder(x)
                z = mu  # use mean for deterministic reconstruction
            elif self.ae_type == "vq":
                z_e = self.ae.encoder(x)
                z, _, _ = self.ae.vq(z_e)
            return self.ae.decoder(z)


    def training_step(self, batch, batch_idx):
        x, _ = batch

        # ---- autoencoder ----
        z, x_hat, recon_loss, reg_loss = self._encode(x)

        # ---- diffusion in latent ----
        diff_loss = torch.tensor(0.0, device=x.device)

        if self.global_step >= self.warmup_steps:
            z_for_diff = z.detach() if self.detach_latent_for_diff else z

            noise, t, eps = self.diffusion.q_sample(z_for_diff)
            pred_noise = self.diffusion.network(noise, t)
            diff_loss = F.mse_loss(pred_noise, eps)

        if not self.use_vq_gan_loss:
            loss = (self.recon_weight * recon_loss) + (self.reg_weight * reg_loss) + (self.diff_weight * diff_loss)

            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/recon_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/reg_loss", reg_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/diff_loss", diff_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

            return loss

        optimizer_g, optimizer_d = self.optimizers()

        gan_active = self.global_step >= self.disc_start_step
        gan_g_loss = torch.tensor(0.0, device=x.device)
        fm_loss = torch.tensor(0.0, device=x.device)

        if gan_active:
            self._set_discriminator_grad(False)
            gan_g_loss, fm_loss = self._generator_adv_losses(x, x_hat)

        g_loss = (
            (self.recon_weight * recon_loss)
            + (self.reg_weight * reg_loss)
            + (self.diff_weight * diff_loss)
            + (self.adv_weight * gan_g_loss)
            + (self.fm_weight * fm_loss)
        )

        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        self._maybe_clip_gradients(optimizer_g)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        if self.use_ema and self.diffusion.ema is not None:
            self.diffusion.ema.update(self.diffusion.network)

        gan_d_loss = torch.tensor(0.0, device=x.device)
        if gan_active:
            self._set_discriminator_grad(True)
            gan_d_loss = self._discriminator_loss(x, x_hat)
            self.toggle_optimizer(optimizer_d)
            optimizer_d.zero_grad()
            self.manual_backward(gan_d_loss)
            self._maybe_clip_gradients(optimizer_d)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

        self.log("train/loss", g_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/recon_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/reg_loss", reg_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/diff_loss", diff_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/gan_g_loss", gan_g_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/gan_d_loss", gan_d_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/fm_loss", fm_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        return g_loss


    def validation_step(self, batch, batch_idx):
        x, _ = batch

        # ---- autoencoder ----
        z, x_hat, recon_loss, reg_loss = self._encode(x)

        # ---- diffusion in latent ----
        noise, t, eps = self.diffusion.q_sample(z.detach())
        pred_noise = self.diffusion.network(noise, t)
        diff_loss = F.mse_loss(pred_noise, eps)

        gan_g_loss = torch.tensor(0.0, device=x.device)
        gan_d_loss = torch.tensor(0.0, device=x.device)
        fm_loss = torch.tensor(0.0, device=x.device)

        if self.use_vq_gan_loss and self.global_step >= self.disc_start_step:
            gan_g_loss, fm_loss = self._generator_adv_losses(x, x_hat)
            gan_d_loss = self._discriminator_loss(x, x_hat)

        loss = (
            (self.recon_weight * recon_loss)
            + (self.reg_weight * reg_loss)
            + (self.diff_weight * diff_loss)
            + (self.adv_weight * gan_g_loss)
            + (self.fm_weight * fm_loss)
        )

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/recon_loss", recon_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/reg_loss", reg_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/diff_loss", diff_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/gan_g_loss", gan_g_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/gan_d_loss", gan_d_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/fm_loss", fm_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss


    def on_train_epoch_end(self):
        if not self.use_vq_gan_loss:
            return

        schedulers = self.lr_schedulers()
        if schedulers is None:
            return

        if isinstance(schedulers, list):
            for scheduler in schedulers:
                scheduler.step()
        else:
            schedulers.step()


    @torch.no_grad()
    def sample(self, n):
        """Generate images: sample latents from diffusion, decode to pixel space."""
        was_training = self.training
        self.eval()
        device = self.device

        # Use EMA weights for the diffusion network if available
        if self.use_ema and self.diffusion.ema is not None:
            self.diffusion.ema.apply(self.diffusion.network)

        try:
            # Sample latents via the diffusion reverse process
            z = torch.randn(n, *self.latent_shape, device=device)

            for i in reversed(range(self.diffusion.timesteps)):
                t = torch.full((n,), i, device=device, dtype=torch.long)
                z = self.diffusion.p_sample(z, t, i)

            # For VQ, snap to nearest codebook before decoding
            if self.ae_type == "vq":
                z, _, _ = self.ae.vq(z)

            # Decode latents to pixel space
            x = self.ae.decoder(z)
        finally:
            # Restore original weights after sampling
            if self.use_ema and self.diffusion.ema is not None:
                self.diffusion.ema.restore(self.diffusion.network)

            if was_training:
                self.train()

        return x


    def configure_optimizers(self):
        if self.use_vq_gan_loss:
            generator_params = list(self.ae.parameters()) + list(self.diffusion.parameters())

            optimizer_g = torch.optim.AdamW(generator_params, lr=self.lr, weight_decay=0.0, betas=(0.9, 0.99))
            optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(),  # type: ignore[union-attr]
                lr=self.lr * self.disc_lr_mult,
                weight_decay=0.0,
                betas=(0.9, 0.99),
            )

            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_g, T_max=self.trainer.max_epochs or 300
            )

            return [optimizer_g, optimizer_d], [scheduler_g]

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
