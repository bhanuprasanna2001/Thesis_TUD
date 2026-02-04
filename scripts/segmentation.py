"""Train U-Net on Oxford Pet Dataset for Segmentation."""

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

from src.data import OxfordPetIII
from src.segmentation import SegmentationModel
from src.utils import set_seed, count_parameters
from src.utils import GradientNormCallback, SegmentationVisualizationCallback


def main():
    config = {
        # Oxford
        "dataset": "OxfordPetIII",
        "img_size": 128,
        "num_workers": 4,
        "val_split": 0.15,
        "seed": 42,
        
        # Segmentation Model
        "in_channels": 3,
        "out_channels": 1,
        "preset": "small",
        
        # Training
        "lr": 0.0002,
        "epochs": 10,
        "batch_size": 128,
        "gradient_clip_val": 1.0,
        
        # Segmentation Visualization Callback
        "every_n_epochs": 1,
        "n_samples": 4,
        
        
        # Logging
        "use_wandb": True,
        "wandb_project": "thesis-segmentation"
    }
    
    set_seed(config["seed"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_name = f"{config['dataset']}_{config['preset']}_{timestamp}"
        
    output_dir = Path("experiments") / experiment_name
    
    datamodule = OxfordPetIII(
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        val_split=config["val_split"],
        seed=config["seed"]
    )
    
    model = SegmentationModel(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        lr=config["lr"],
        preset=config["preset"]
    )
    
    print(f"Parameters: {count_parameters(model):,}")
    
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
            every_n_epochs=config["every_n_epochs"],
            n_samples=config["n_samples"]
        )
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
        log_every_n_steps=50,
        enable_progress_bar=True,
        gradient_clip_val=config.get("gradient_clip_val", 1.0),
    )
    
    trainer.fit(model, datamodule)
    
    trainer.test(model, datamodule)
    

if __name__ == "__main__":
    main()
