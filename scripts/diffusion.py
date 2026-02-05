"""Train Diffusion on MNIST Dataset for Generation."""

import sys
import yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from src.data import MNIST, CIFAR10
from src.diffusion import Diffusion
from src.utils import set_seed, count_parameters
from src.utils import GradientNormCallback, DiffusionSampleGenerationCallback


def main():
    config = {
        # Dataset
        "dataset": "CIFAR10",  # Options: "MNIST", "CIFAR10"
        "num_workers": 4,
        "val_split": 0.15,
        "seed": 42,

        # Scheduler
        "start": 0.0001,
        "end": 0.02,
        "timesteps": 1000,
        "scheduler_type": "linear",

        # U-Net Model - xtiny(0.8M), tiny(2M), mini(4M), small(6M), medium(13M), base(23M), large(51M), xlarge(89M), xxlarge(198M)
        "in_channels": 1,
        "out_channels": 1,
        "img_shape": (1, 28, 28),
        "groups": 8,
        "preset": "medium",  
        "time_emb_dim": 512,

        # Training
        "lr": 0.0003,
        "epochs": 300,
        "batch_size": 128,
        "gradient_clip_val": 1.0,

        # EMA
        "use_ema": True,
        "ema_decay": 0.9999,

        # Checkpointing and Early Stopping
        "save_top_k": 3,
        "log_every_n_steps": 5,
        "monitor": "val/loss",
        "mode": "min",
        "patience": 295,

        # Diffusion Visualization Callback
        "every_n_epochs": 1,
        "n_samples": 4,

        # Logging
        "use_wandb": True,
        "wandb_project": "thesis-diffusion"
    }

    set_seed(config["seed"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_name = f"{config['dataset']}_{config['preset']}_{timestamp}"
        
    output_dir = Path("experiments") / experiment_name

    # Select dataset based on config
    if config["dataset"] == "MNIST":
        datamodule = MNIST(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            val_split=config["val_split"],
            seed=config["seed"]
        )
        config["in_channels"] = 1
        config["out_channels"] = 1
        config["img_shape"] = (1, 28, 28)
    elif config["dataset"] == "CIFAR10":
        datamodule = CIFAR10(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            val_split=config["val_split"],
            seed=config["seed"]
        )
        config["in_channels"] = 3
        config["out_channels"] = 3
        config["img_shape"] = (3, 32, 32)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}. Choose 'MNIST' or 'CIFAR10'.")

    model = Diffusion(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        img_shape=config["img_shape"],
        lr=config["lr"],
        preset=config["preset"],
        start=config["start"],
        end=config["end"],
        timesteps=config["timesteps"],
        scheduler_type=config["scheduler_type"],
        groups=config["groups"],
        time_emb_dim=config["time_emb_dim"],
        use_ema=config["use_ema"],
        ema_decay=config["ema_decay"]
    )

    print(f"Parameters: {count_parameters(model):,}")
    
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="{epoch:02d}-{val/loss:.4f}",
            save_top_k=config["save_top_k"],
            monitor=config["monitor"],
            mode=config["mode"],
            save_last=True,
        ),
        GradientNormCallback(log_every_n_steps=config["log_every_n_steps"]),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor=config["monitor"], patience=config["patience"], mode=config["mode"]),
        DiffusionSampleGenerationCallback(every_n_epochs=config["every_n_epochs"], output_dir=str(output_dir / "samples"))
    ]

    if config["use_wandb"]:
        logger = WandbLogger(
            project=config["wandb_project"],
            name=experiment_name,
            save_dir=output_dir,
            config=config,
        )
    else:
        logger = TensorBoardLogger(save_dir=output_dir, name="logs")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)


    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config["log_every_n_steps"],
        enable_progress_bar=True,
        gradient_clip_val=config.get("gradient_clip_val", 1.0),
    )
    
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()