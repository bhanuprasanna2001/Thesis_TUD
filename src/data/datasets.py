from typing import Tuple
import torch
import torchvision as tv
from torch.utils.data import DataLoader


def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = "data",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create MNIST train/test dataloaders."""

    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = tv.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = tv.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
