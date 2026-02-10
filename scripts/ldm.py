"""Train Latent Diffusion Model (LDM) on MNIST/CIFAR10 for Generation."""

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
from src.ldm import LDM
from src.utils import set_seed, count_parameters, get_device
from src.utils import GradientNormCallback, LDMVisualizationCallback


def main():
    config = {
        # Dataset
        "dataset": "CIFAR10",  # Options: "MNIST", "CIFAR10"
        "num_workers": 4,
        "val_split": 0.15,
        "seed": 42,

        # Autoencoder — "ae" (plain), "kl" (KL-regularized VAE), "vq" (VQ-VAE)
        "ae_type": "vq",
        "in_channels": 1,
        "out_channels": 1,
        "img_shape": (1, 28, 28),
        "base_channels": 32,
        "latent_channels": 4,

        # AE Regularization
        # ae → 0.0 (unused), kl → 1e-6 (small to avoid posterior collapse), vq → 1.0
        "reg_weight": 1.0,

        # VQ-specific (only used when ae_type="vq")
        "num_embeddings": 512,
        "commitment_cost": 0.2,

        # Scheduler
        "start": 0.0001,
        "end": 0.02,
        "timesteps": 1000,
        "scheduler_type": "linear",

        # U-Net Model (operates in latent space) - xtiny(0.8M), tiny(2M), mini(4M), small(6M), medium(13M), base(23M), large(51M), xlarge(89M), xxlarge(198M)
        "groups": 8,
        "preset": "base",
        "time_emb_dim": 512,
        "n_levels": 2,

        # Training
        "lr": 0.0002,
        "epochs": 300,
        "batch_size": 128,
        "gradient_clip_val": 1.0,

        # Loss Weights
        "recon_weight": 1.0,
        "diff_weight": 1.0,
        "warmup_steps": 5000,
        "detach_latent_for_diff": True,

        # EMA
        "use_ema": True,
        "ema_decay": 0.9999,

        # Checkpointing and Early Stopping
        "save_top_k": 3,
        "log_every_n_steps": 5,
        "monitor": "val/loss",
        "mode": "min",
        "patience": 295,

        # LDM Visualization Callback
        "every_n_epochs": 1,
        "n_samples": 16,

        # Logging
        "use_wandb": True,
        "wandb_project": "thesis-ldm"
    }

    set_seed(config["seed"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_name = f"{config['dataset']}_{config['ae_type']}_{config['preset']}_{timestamp}"
        
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

    model = LDM(
        ae_type=config["ae_type"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        img_shape=tuple(config["img_shape"]),
        base_channels=config["base_channels"],
        latent_channels=config["latent_channels"],
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
        recon_weight=config["recon_weight"],
        reg_weight=config["reg_weight"],
        diff_weight=config["diff_weight"],
        warmup_steps=config["warmup_steps"],
        detach_latent_for_diff=config["detach_latent_for_diff"],
        num_embeddings=config["num_embeddings"],
        commitment_cost=config["commitment_cost"],
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
        LDMVisualizationCallback(every_n_epochs=config["every_n_epochs"], n_samples=config["n_samples"], output_dir=str(output_dir / "samples"))
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
    """Generate visualization report from trained LDM checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        n_samples: Number of samples per visualization
        n_grids: Number of sample grids to generate
    """
    import torchvision as tv
    
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
    
    print(f"Config: {config['dataset']} | {config.get('ae_type', 'ae')} | {config['preset']} | {config.get('in_channels', 1)}ch")
    
    # Load model from checkpoint with correct hyperparameters
    model = LDM.load_from_checkpoint(
        checkpoint_path,
        ae_type=config.get("ae_type", "ae"),
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        img_shape=tuple(config["img_shape"]),
        base_channels=config["base_channels"],
        latent_channels=config["latent_channels"],
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
        recon_weight=config.get("recon_weight", 1.0),
        reg_weight=config.get("reg_weight", 0.0),
        diff_weight=config.get("diff_weight", 1.0),
        warmup_steps=config.get("warmup_steps", 0),
        detach_latent_for_diff=config.get("detach_latent_for_diff", True),
        num_embeddings=config.get("num_embeddings", 512),
        commitment_cost=config.get("commitment_cost", 0.25),
        n_levels=config.get("n_levels", 3),
    )
    model.eval()
    
    device = get_device()
    model.to(device)
    print(f"Using device: {device}")
    
    # Determine output directory
    if "experiments" in str(checkpoint_path):
        experiment_dir = checkpoint_path.parent.parent
        output_dir = experiment_dir / "visualizations"
    else:
        output_dir = checkpoint_path.parent / "visualizations"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating sample report in {output_dir}")
    print("=" * 60)
    
    nrow = int(n_samples ** 0.5)
    
    # 1. Generate sample grids
    print(f"\n1. Generating {n_grids} sample grids...")
    for i in range(n_grids):
        print(f"   Grid {i+1}/{n_grids}...")
        samples = model.sample(n_samples)
        samples = (samples.clamp(-1, 1) + 1) / 2
        
        save_path = output_dir / f"samples_grid_{i+1:02d}.png"
        tv.utils.save_image(samples, save_path, nrow=nrow, padding=2, pad_value=1.0)
    
    # 2. Generate reconstructions from dataset
    print(f"\n2. Generating reconstructions...")
    if config["dataset"] == "MNIST":
        datamodule = MNIST(batch_size=n_samples, num_workers=0, val_split=0.15, seed=42)
    elif config["dataset"] == "CIFAR10":
        datamodule = CIFAR10(batch_size=n_samples, num_workers=0, val_split=0.15, seed=42)
    
    datamodule.setup("fit")
    batch = next(iter(datamodule.val_dataloader()))
    images = batch[0][: n_samples].to(device)
    
    with torch.no_grad():
        recons = model.reconstruct(images)
    
    images_vis = (images.clamp(-1, 1) + 1) / 2
    recons_vis = (recons.clamp(-1, 1) + 1) / 2
    comparison = torch.stack([images_vis, recons_vis], dim=1).reshape(-1, *images_vis.shape[1:])
    
    save_path = output_dir / "reconstructions.png"
    tv.utils.save_image(comparison, save_path, nrow=nrow * 2, padding=2, pad_value=1.0)
    
    print("\n" + "=" * 60)
    print("Sample report complete!")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or sample from Latent Diffusion Model")
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
