import lightning as L
import torchvision as tv
import torch
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(L.LightningDataModule):
    """Lightning DataModule for MNIST with configurable train/val split."""

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 128,
        num_workers: int = 0,
        val_split: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.pin_memory = torch.cuda.is_available()

        self.transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def prepare_data(self) -> None:
        tv.datasets.MNIST(root=self.data_dir, train=True, download=True)
        tv.datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            full_train = tv.datasets.MNIST(
                root=self.data_dir, train=True, transform=self.transform
            )
            if self.val_split > 0:
                val_size = int(len(full_train) * self.val_split)
                train_size = len(full_train) - val_size
                self.train_dataset, self.val_dataset = random_split(
                    full_train, [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
            else:
                self.train_dataset = full_train
                self.val_dataset = tv.datasets.MNIST(
                    root=self.data_dir, train=False, transform=self.transform
                )

        if stage == "test" or stage is None:
            self.test_dataset = tv.datasets.MNIST(
                root=self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
