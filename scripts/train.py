"""Train diffusion model with PyTorch Lightning."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import yaml
from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from src.models import DiffusionModel
from src.data import MNISTDataModule
from src.training import SampleGenerationCallback, GradientNormCallback
from src.utils import set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.name or f"{config['logging']['experiment_name']}_{config['model']['preset']}_{timestamp}"
    output_dir = Path("experiments") / experiment_name

    set_seed(config["seed"])

    # Data
    data_cfg = config["data"]
    datamodule = MNISTDataModule(
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_split=data_cfg.get("val_split", 0.0),
    )

    # Model
    model_cfg = config["model"]
    model = DiffusionModel(
        timesteps=model_cfg["timesteps"],
        beta_start=model_cfg["beta_start"],
        beta_end=model_cfg["beta_end"],
        in_channels=model_cfg["in_channels"],
        norm_type=model_cfg.get("norm_type", "group"),
        num_groups=model_cfg.get("num_groups", 8),
        preset=model_cfg.get("preset"),
        lr=config["training"]["lr"],
        img_shape=tuple(model_cfg["img_shape"]),
    )

    # Callbacks
    train_cfg = config["training"]
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="{epoch:02d}-{train/loss_epoch:.4f}",
            save_top_k=3,
            monitor="train/loss_epoch",
            mode="min",
            save_last=True,
        ),
        SampleGenerationCallback(
            every_n_epochs=train_cfg["sample_every"],
            n_samples=train_cfg["n_samples"],
            output_dir=str(output_dir / "samples"),
        ),
        GradientNormCallback(log_every_n_steps=50),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Logger
    log_cfg = config["logging"]
    if log_cfg["use_wandb"]:
        logger = WandbLogger(
            project=log_cfg["project"],
            name=experiment_name,
            save_dir=output_dir,
            config=config,
        )
    else:
        logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Trainer
    trainer = L.Trainer(
        max_epochs=train_cfg["epochs"],
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
