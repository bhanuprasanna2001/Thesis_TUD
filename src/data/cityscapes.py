"""CityScapes dataset for segmentation."""

import lightning as L
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import Cityscapes
from pathlib import Path


class CityscapesDataModule(L.LightningDataModule):
    """Lightning DataModule for Cityscapes segmentation dataset.
    
    Note: CityScapes requires manual download and registration at
    https://www.cityscapes-dataset.com/
    
    Expected directory structure:
        data/cityscapes/
            leftImg8bit/
                train/
                val/
            gtFine/
                train/
                val/
    """

    def __init__(
        self,
        data_dir: str = "data/cityscapes",
        batch_size: int = 8,
        num_workers: int = 4,
        img_size: int = 256,
        mode: str = "fine",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.mode = mode
        self.pin_memory = torch.cuda.is_available()

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.target_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.PILToTensor(),
        ])

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = CityscapesSegmentation(
                root=self.data_dir,
                split="train",
                mode=self.mode,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.val_dataset = CityscapesSegmentation(
                root=self.data_dir,
                split="val",
                mode=self.mode,
                transform=self.transform,
                target_transform=self.target_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class CityscapesSegmentation(Cityscapes):
    """Cityscapes with binary or multi-class segmentation masks."""

    def __init__(
        self,
        root,
        split,
        mode="fine",
        transform=None,
        target_transform=None,
        num_classes: int = 19,
    ):
        super().__init__(
            root=root,
            split=split,
            mode=mode,
            target_type="semantic",
            transform=transform,
            target_transform=target_transform,
        )
        self.num_classes = num_classes

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)
        # Normalize mask to [0, num_classes-1], treating 255 as ignore
        mask = mask.squeeze(0).long()
        mask[mask == 255] = 0
        return image, mask.unsqueeze(0).float()
