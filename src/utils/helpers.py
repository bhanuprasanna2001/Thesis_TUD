import torch
import random
import numpy as np
import torchvision as tv
from pathlib import Path

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_samples(samples, path, nrow):
    """Save generated samples as image grid."""
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples.clamp(-1, 1) + 1) / 2
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tv.utils.save_image(samples, path, nrow=nrow)


def get_unet_preset(preset):
    """Get base channels for UNet based on preset."""
    presets = {
        "tiny": 16,
        "small": 24,
        "medium": 28,
        "large": 32,
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}")
    
    return presets[preset]