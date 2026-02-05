import torch
import lightning as L
import torchvision as tv
from torch.utils.data import DataLoader, random_split

class CIFAR10(L.LightningDataModule):
    """Lightning DataModule for CIFAR-10."""
    
    def __init__(self, data_dir="data", batch_size=32, num_workers=4, val_split=0.15, seed=42):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.pin_memory = torch.cuda.is_available()
        
        self.transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            # output = (input - mean) / std
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def prepare_data(self) -> None:
        tv.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        tv.datasets.CIFAR10(root=self.data_dir, train=False, download=True)
        
    def setup(self, stage: str) -> None:
        if stage == "fit":
            full_train = tv.datasets.CIFAR10(
                root=self.data_dir, train=True, transform=self.transform
            )
            
            if self.val_split == 0.0:
                print(f"Val Split cannot be initialized to 0.0. Setting to 0.15.")
                self.val_split = 0.15
            
            val_size = int(len(full_train) * self.val_split)
            train_size = len(full_train) - val_size
            
            self.train_dataset, self.val_dataset = random_split(
                full_train, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )
            
        if stage == "test":
            self.test_dataset = tv.datasets.CIFAR10(
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
