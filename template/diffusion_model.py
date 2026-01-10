import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    """Abstract template for (conditional) diffusion models."""
    def __init__(self):
        super().__init__()

    def compute_loss(self, x, c=None, steps=100):
        """Compute training loss for given data x and optional conditioning c."""
        raise NotImplementedError

    def integrate_forward(self, x, c=None, steps=100, ...):
        """Move samples from data space to latent space (forward/noising)."""
        raise NotImplementedError

    def integrate_inverse(self, z, c=None, steps=100, ...):
        """Move samples from latent space to data space (reverse/denoising)."""
        raise NotImplementedError

    def integrate(self, xz, c=None, steps=100, inverse=False, ...):
        """Generic integrator: forward if inverse=False else inverse."""
        raise NotImplementedError

    def sample(self, n, c=None, steps=100, ...):
        """Draw n random samples from the model, optionally conditioned on c."""
        raise NotImplementedError
