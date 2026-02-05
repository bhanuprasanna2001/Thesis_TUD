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
        
        # Create shadow parameters (deep copy of model parameters)
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
                
    @torch.no_grad()
    def update(self, model):
        """Update the EMA parameters with the current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                
                
    def apply(self, model):
        """Apply EMA parameters to the model (for sampling/evaluation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
                
                
    def restore(self, model):
        """Restore original parameters to the model (after sampling/evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
        
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return state dict for saving."""
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }
        
        
    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict for restoring."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
