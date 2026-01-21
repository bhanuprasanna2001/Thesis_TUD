from typing import TYPE_CHECKING

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from pathlib import Path
import torchvision as tv

if TYPE_CHECKING:
    from src.models.diffusion import DiffusionModel


class SampleGenerationCallback(Callback):
    """Generate and save samples during training."""

    def __init__(
        self,
        every_n_epochs: int = 5,
        n_samples: int = 16,
        output_dir: str = "samples",
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples
        self.output_dir = Path(output_dir)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs != 0:
            return

        if not hasattr(pl_module, "sample"):
            return

        pl_module.eval()
        with torch.no_grad():
            samples = pl_module.sample(self.n_samples)  # type: ignore[operator]

        samples = (samples.clamp(-1, 1) + 1) / 2
        save_path = self.output_dir / f"epoch_{epoch:04d}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tv.utils.save_image(samples, save_path, nrow=4)

        # Log to W&B or other logger that supports log_image
        if trainer.logger and hasattr(trainer.logger, "log_image"):
            grid = tv.utils.make_grid(samples, nrow=4)
            trainer.logger.log_image(  # type: ignore[union-attr]
                key="samples",
                images=[grid.permute(1, 2, 0).cpu().numpy()],
                step=trainer.global_step,
            )


class GradientNormCallback(Callback):
    """Log gradient norm for monitoring training stability."""

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer
    ) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        pl_module.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)
