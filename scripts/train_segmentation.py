"""Train U-Net on segmentation tasks."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import yaml
from datetime import datetime

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from src.models import UNetSegmentation, UNET_PRESETS
from src.models.segmentation import SegmentationModel
from src.data.oxford_pet import OxfordPetDataModule
from src.training import GradientNormCallback, SegmentationVisualizationCallback
from src.utils import set_seed, count_parameters


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_torch_hub_unet(
    in_channels: int = 3, out_channels: int = 1, init_features: int = 32
) -> torch.nn.Module:
    """Load U-Net from PyTorch Hub.
    
    Args:
        init_features: Base channels. 8 (~487K), 16 (~1.9M), 32 (~7.8M), 64 (~31M)
    """
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        pretrained=False,
        trust_repo=True,
    )
    return model  # type: ignore[return-value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/segmentation.yaml")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config['model']['type'] == "ours":
        experiment_name = args.name or f"{config['data']['dataset']}_{config['model']['preset']}_{timestamp}"
    else:
        experiment_name = args.name or f"{config['data']['dataset']}_{config['model']['hub_init_features']}_{timestamp}"
        
    output_dir = Path("experiments") / experiment_name

    # Data
    data_cfg = config["data"]
    if data_cfg["dataset"] == "oxford_pet":
        datamodule = OxfordPetDataModule(
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            img_size=data_cfg["img_size"],
            val_split=data_cfg.get("val_split", 0.1),
            seed=config["seed"],
        )

    # Model
    model_cfg = config["model"]
    if model_cfg["type"] == "ours":
        network = UNetSegmentation(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            preset=model_cfg.get("preset"),
            norm_type=model_cfg.get("norm_type", "group"),
            num_groups=model_cfg.get("num_groups", 8),
        )
    else:
        network = get_torch_hub_unet(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            init_features=model_cfg.get("hub_init_features", 32),
        )

    model = SegmentationModel(network=network, lr=config["training"]["lr"])

    print(f"Dataset: {data_cfg['dataset']}")
    print(f"Model: {model_cfg['type']}")
    if model_cfg["type"] == "ours":
        print(f"Preset: {model_cfg.get('preset')}")
    else:
        print(f"Hub init_features: {model_cfg.get('hub_init_features', 32)}")
    print(f"Parameters: {count_parameters(network):,}")
    print("-" * 50)

    # Callbacks
    train_cfg = config["training"]
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="{epoch:02d}-{val/iou:.4f}",
            save_top_k=3,
            monitor="val/iou",
            mode="max",
            save_last=True,
        ),
        GradientNormCallback(log_every_n_steps=50),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val/iou", patience=5, mode="max"),
        SegmentationVisualizationCallback(
            every_n_epochs=train_cfg["sample_every"],
            n_samples=train_cfg["n_samples"],
        ),
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
