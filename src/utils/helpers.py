import random
import numpy as np
import torch
import torchvision as tv
from pathlib import Path


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_samples(samples: torch.Tensor, path: str, nrow: int = 4) -> None:
    """Save generated samples as image grid."""
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples.clamp(-1, 1) + 1) / 2
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tv.utils.save_image(samples, path, nrow=nrow)
