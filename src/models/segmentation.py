"""Segmentation model for U-Net architecture verification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Dict, Any, Tuple


class SegmentationModel(L.LightningModule):
    """Lightning module for semantic segmentation with IoU and Dice metrics."""

    def __init__(
        self,
        network: nn.Module,
        lr: float = 1e-3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["network"])
        self.network = network
        self.lr = lr
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - (2 * intersection + 1) / (union + 1)
        return bce + dice.mean()

    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        intersection = (pred_binary * target).sum(dim=(2, 3))
        union = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        return ((intersection + 1e-6) / (union + 1e-6)).mean()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, masks = batch
        pred = self.network(images)
        loss = self._compute_loss(pred, masks)
        iou = self._compute_iou(pred, masks)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/iou", iou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, masks = batch
        pred = self.network(images)
        loss = self._compute_loss(pred, masks)
        iou = self._compute_iou(pred, masks)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/iou", iou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs or 100
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
