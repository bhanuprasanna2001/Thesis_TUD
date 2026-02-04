import torch
import lightning as L
import torchvision as tv
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, random_split

class OxfordPetIII(L.LightningDataModule):
    """Lightning DataModule for Oxford-IIIT Pet segmentation dataset."""
    
    def __init__(self, data_dir="data", img_size=128, batch_size=32, num_workers=4, val_split=0.15, seed=42):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.pin_memory = torch.cuda.is_available()
        
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize((img_size, img_size)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.target_transform = tv.transforms.Compose([
            tv.transforms.Resize((img_size, img_size), interpolation=tv.transforms.InterpolationMode.NEAREST),
            tv.transforms.PILToTensor(),
        ])
        
    def prepare_data(self) -> None:
        OxfordIIITPet(root=self.data_dir, split="trainval", target_types="segmentation", download=True)
        OxfordIIITPet(root=self.data_dir, split="test", target_types="segmentation", download=True)
        
    def setup(self, stage: str) -> None:
        if stage == "fit":
            full_train = OxfordIIITPetSegmentation(
                root=self.data_dir, split="trainval", 
                transform=self.transform, target_transform=self.target_transform
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
            self.test_dataset = OxfordIIITPetSegmentation(
                root=self.data_dir, split="test", 
                transform=self.transform, target_transform=self.target_transform
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
