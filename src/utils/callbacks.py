import torch
import lightning as L
import torchvision as tv
from pathlib import Path
from lightning.pytorch.callbacks import Callback

class GradientNormCallback(Callback):
    """Log gradient norm for monitoring training stability."""

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_norm_sq = 0.0

        for p in pl_module.parameters():
            if p.grad is None:
                continue

            grad = p.grad.detach()
            param_norm = grad.norm(2)
            total_norm_sq = total_norm_sq + (param_norm.item() ** 2)

        total_norm = total_norm_sq ** 0.5

        pl_module.log("train/grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        
        
class SegmentationVisualizationCallback(Callback):
    """Log segmentation predictions during training."""

    def __init__(self, every_n_epochs: int = 5, n_samples: int = 4):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return

        # Get a batch from validation
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        batch = next(iter(val_dataloader))
        images, masks = batch[0][: self.n_samples], batch[1][: self.n_samples]
        images, masks = images.to(pl_module.device), masks.to(pl_module.device)

        pl_module.eval()
        with torch.no_grad():
            preds = torch.sigmoid(pl_module(images))
            preds_binary = (preds > 0.5).float()

        # Denormalize images (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images_denorm = images * std + mean

        # Create visualization: [image, ground truth, prediction]
        vis_rows = []
        for i in range(self.n_samples):
            img = images_denorm[i]  # 3xHxW
            gt = masks[i].repeat(3, 1, 1)  # 3xHxW (grayscale to RGB)
            pred = preds_binary[i].repeat(3, 1, 1)  # 3xHxW
            vis_rows.append(torch.cat([img, gt, pred], dim=2))  # Horizontal concat

        grid = torch.cat(vis_rows, dim=1)  # Vertical concat

        # Log to W&B
        if trainer.logger and hasattr(trainer.logger, "log_image"):
            trainer.logger.log_image(  # type: ignore[union-attr]
                key="segmentation/predictions",
                images=[grid.permute(1, 2, 0).cpu().numpy()],
                step=trainer.global_step,
                caption=["Image | Ground Truth | Prediction"],
            )


class DiffusionSampleGenerationCallback(Callback):
    """Generate and save Diffusion model samples during training."""

    def __init__(
        self,
        every_n_epochs: int = 5,
        n_samples: int = 16,
        nrow: int = 4,
        output_dir: str = "samples",
        log_key: str = "diffusion/samples",
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples
        self.nrow = nrow
        self.output_dir = Path(output_dir)
        self.log_key = log_key

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return

        if not hasattr(pl_module, "sample"):
            return

        pl_module.eval()
        with torch.no_grad():
            samples = pl_module.sample(self.n_samples)  # type: ignore[operator]

        # Map from [-1, 1] -> [0, 1] for visualization / saving
        samples = (samples.clamp(-1, 1) + 1) / 2

        save_path = self.output_dir / f"epoch_{epoch:04d}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tv.utils.save_image(samples, save_path, nrow=self.nrow)

        if trainer.logger and hasattr(trainer.logger, "log_image"):
            grid = tv.utils.make_grid(samples, nrow=self.nrow)
            trainer.logger.log_image(  # type: ignore[union-attr]
                key=self.log_key,
                images=[grid.permute(1, 2, 0).cpu().numpy()],
                step=trainer.global_step,
                caption=[f"Epoch {epoch} | n={self.n_samples}"],
            )


class LDMVisualizationCallback(Callback):
    """Generate LDM samples and reconstructions during training."""

    def __init__(
        self,
        every_n_epochs: int = 5,
        n_samples: int = 16,
        nrow: int = 4,
        output_dir: str = "samples",
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples
        self.nrow = nrow
        self.output_dir = Path(output_dir)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return

        if not hasattr(pl_module, "sample"):
            return

        # ---- Generated samples from diffusion ----
        pl_module.eval()
        with torch.no_grad():
            samples = pl_module.sample(self.n_samples)

        samples = (samples.clamp(-1, 1) + 1) / 2

        save_path = self.output_dir / f"generated_epoch_{epoch:04d}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tv.utils.save_image(samples, save_path, nrow=self.nrow)

        if trainer.logger and hasattr(trainer.logger, "log_image"):
            grid = tv.utils.make_grid(samples, nrow=self.nrow)
            trainer.logger.log_image(  # type: ignore[union-attr]
                key="ldm/generated",
                images=[grid.permute(1, 2, 0).cpu().numpy()],
                step=trainer.global_step,
                caption=[f"Epoch {epoch} | Generated n={self.n_samples}"],
            )

        # ---- Reconstructions ----
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        batch = next(iter(val_dataloader))
        images = batch[0][: self.n_samples].to(pl_module.device)

        with torch.no_grad():
            recons = pl_module.reconstruct(images)

        images_vis = (images.clamp(-1, 1) + 1) / 2
        recons_vis = (recons.clamp(-1, 1) + 1) / 2

        # Interleave: original, reconstruction, original, reconstruction, ...
        comparison = torch.stack([images_vis, recons_vis], dim=1).reshape(-1, *images_vis.shape[1:])

        save_path = self.output_dir / f"recon_epoch_{epoch:04d}.png"
        tv.utils.save_image(comparison, save_path, nrow=self.nrow * 2)

        if trainer.logger and hasattr(trainer.logger, "log_image"):
            grid = tv.utils.make_grid(comparison, nrow=self.nrow * 2)
            trainer.logger.log_image(  # type: ignore[union-attr]
                key="ldm/reconstructions",
                images=[grid.permute(1, 2, 0).cpu().numpy()],
                step=trainer.global_step,
                caption=[f"Epoch {epoch} | Input <> Reconstruction"],
            )