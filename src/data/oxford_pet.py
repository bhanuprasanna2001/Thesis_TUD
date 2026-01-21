"""Oxford-IIIT Pet dataset for segmentation."""

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet


class OxfordPetDataModule(L.LightningDataModule):
    """Lightning DataModule for Oxford-IIIT Pet segmentation dataset."""

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 16,
        num_workers: int = 4,
        img_size: int = 128,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.val_split = val_split
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

    def prepare_data(self) -> None:
        OxfordIIITPet(root=self.data_dir, split="trainval", target_types="segmentation", download=True)

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            full_dataset = OxfordIIITPetSegmentation(
                root=self.data_dir,
                split="trainval",
                transform=self.transform,
                target_transform=self.target_transform,
            )

            val_size = int(len(full_dataset) * self.val_split)
            train_size = len(full_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
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


class OxfordIIITPetSegmentation(OxfordIIITPet):
    """Oxford-IIIT Pet with binary segmentation masks."""

    def __init__(self, root, split, transform=None, target_transform=None):
        super().__init__(
            root=root,
            split=split,
            target_types="segmentation",
            transform=transform,
            target_transform=target_transform,
            download=False,
        )

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        # Convert trimap (1=foreground, 2=background, 3=boundary) to binary
        # Foreground (pet) = 1, Background = 0
        mask = (mask == 1).float()
        return image, mask
