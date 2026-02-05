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
    """Get base channels for UNet based on preset.
    
    Note: base_channels must be divisible by groups (default 8) for GroupNorm.
    Parameters scale roughly with base_channels^2.
    
    Preset      base_channels   ~Parameters (3ch input)
    -------     -------------   -----------------------
    xtiny       8               0.8M
    tiny        16              2M
    mini        24              4M
    small       32              6M
    medium      48              13M
    base        64              23M
    large       96              51M
    xlarge      128             89M
    xxlarge     192             198M
    """
    presets = {
        "xtiny": 8,
        "tiny": 16,
        "mini": 24,
        "small": 32,
        "medium": 48,
        "base": 64,
        "large": 96,
        "xlarge": 128,
        "xxlarge": 192,
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")
    
    return presets[preset]