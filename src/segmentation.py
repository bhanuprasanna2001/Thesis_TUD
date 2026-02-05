import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F

from .architectures import UNet

from .utils import get_unet_preset

class SegmentationModel(L.LightningModule):
    """Segmentation Model for Oxford Pet Dataset."""
    
    def __init__(self, in_channels=3, out_channels=1, lr=0.0002, preset=None, base_channels=64):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        base_channels = get_unet_preset(preset)
        
        self.network = UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
        
    
    def forward(self, X):
        return self.network(X)

        
    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: logits, target: {0,1} mask
        target = target.float()
        eps = 1e-6

        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred_prob = torch.sigmoid(pred)

        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice_score = (2.0 * intersection + eps) / (union + eps)
        dice_loss = 1.0 - dice_score

        loss = bce + dice_loss.mean()
        return loss


    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        eps = 1e-6

        pred_prob = torch.sigmoid(pred)
        pred_binary = (pred_prob > 0.5).float()

        intersection = (pred_binary * target).sum(dim=(2, 3))
        union = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

        iou = (intersection + eps) / (union + eps)
        iou = iou.mean()
        return iou

    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        
        pred = self.network(images)
        
        loss = self._compute_loss(pred, masks)
        iou = self._compute_iou(pred, masks)
        
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/iou", iou, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        pred = self.network(images)
        
        loss = self._compute_loss(pred, masks)
        iou = self._compute_iou(pred, masks)
        
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/iou", iou, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        images, masks = batch
        
        pred = self.network(images)
        
        loss = self._compute_loss(pred, masks)
        iou = self._compute_iou(pred, masks)
        
        self.log("test/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("test/iou", iou, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs or 100
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    
        
    
    
        
