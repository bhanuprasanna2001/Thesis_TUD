"""Train Diffusion on MNIST Dataset for Generation."""

import sys
import yaml
import argparse
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
from src.utils import set_seed, count_parameters, get_device
from src.utils import GradientNormCallback, DiffusionSampleGenerationCallback
from src.utils import generate_sample_report


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
        "n_levels": 3,

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
        "n_samples": 16,

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
        ema_decay=config["ema_decay"],
        n_levels=config["n_levels"],
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
        DiffusionSampleGenerationCallback(every_n_epochs=config["every_n_epochs"], n_samples=config["n_samples"], output_dir=str(output_dir / "samples"))
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


def sample_mode(checkpoint_path, n_samples=16, n_grids=4):
    """Generate visualization report from trained checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        n_samples: Number of samples per visualization
        n_grids: Number of sample grids to generate
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load config to get model hyperparameters
    config_path = checkpoint_path.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}\nCannot determine model architecture.")
    
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print(f"Config: {config['dataset']} | {config['preset']} | {config.get('in_channels', 3)}ch")
    
    # Load model from checkpoint with correct hyperparameters
    model = Diffusion.load_from_checkpoint(
        checkpoint_path,
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        img_shape=tuple(config["img_shape"]),
        lr=config["lr"],
        preset=config["preset"],
        start=config["start"],
        end=config["end"],
        timesteps=config["timesteps"],
        scheduler_type=config["scheduler_type"],
        groups=config["groups"],
        time_emb_dim=config["time_emb_dim"],
        use_ema=config.get("use_ema", True),
        ema_decay=config.get("ema_decay", 0.9999),
        n_levels=config.get("n_levels", 3),
    )
    model.eval()
    
    device = get_device()
    model.to(device)
    print(f"Using device: {device}")
    
    # Determine output directory (experiment folder / visualizations)
    if "experiments" in str(checkpoint_path):
        experiment_dir = checkpoint_path.parent.parent
        output_dir = experiment_dir / "visualizations"
    else:
        output_dir = checkpoint_path.parent / "visualizations"
    
    # Generate comprehensive report
    generate_sample_report(
        model=model,
        output_dir=output_dir,
        n_samples=n_samples,
        n_grids=n_grids
    )
    
    print(f"\n✓ Visualization report complete: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or sample from Diffusion model")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "sample"],
        help="Mode: train a new model or sample from checkpoint (default: train)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (required for sample mode)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=16,
        help="Number of samples per visualization (default: 16)"
    )
    parser.add_argument(
        "--n_grids",
        type=int,
        default=4,
        help="Number of sample grids to generate (default: 4)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        main()
    elif args.mode == "sample":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for sample mode")
        sample_mode(
            checkpoint_path=args.checkpoint,
            n_samples=args.n_samples,
            n_grids=args.n_grids
        )
