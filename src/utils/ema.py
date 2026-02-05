import torch
import torch.nn as nn


class EMA(nn.Module):
    """Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of model parameters that are updated with:
        ema_param = decay * ema_param + (1 - decay) * model_param
    
    This smooths the model weights over training steps, often leading to
    better generalization and sample quality in diffusion models.
    
    Reference: https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
    """
    
    def __init__(self, model, decay=0.9999):
        super().__init__()
        
        self.decay = decay
        
        # Register shadow parameters as buffers so they:
        # 1) move to device automatically, 2) are saved/loaded in checkpoints
        self.name_map = {}
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                buf_name = f"shadow_{idx}"
                self.name_map[name] = buf_name
                # Keep EMA weights in fp32 for stability (esp. with AMP/bf16)
                self.register_buffer(buf_name, param.detach().float().clone())
                idx += 1
                
        self.backup = {}
                
                
    @torch.no_grad()
    def update(self, model):
        """Update the EMA parameters with the current model parameters.
        
        IMPORTANT: call this AFTER optimizer.step().
        """
        d = float(self.decay)
        for name, param in model.named_parameters():
            if name not in self.name_map:
                continue
            shadow = getattr(self, self.name_map[name])
            shadow.mul_(d).add_(param.detach().float(), alpha=(1.0 - d))
                
                
    @torch.no_grad()
    def apply(self, model):
        """Apply EMA parameters to the model (for sampling/evaluation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name not in self.name_map:
                continue
            self.backup[name] = param.detach().clone()
            param.copy_(getattr(self, self.name_map[name]).to(dtype=param.dtype))
                
                
    @torch.no_grad()
    def restore(self, model):
        """Restore original parameters to the model (after sampling/evaluation)."""
        if not self.backup:
            return
        for name, param in model.named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name])
        self.backup = {}