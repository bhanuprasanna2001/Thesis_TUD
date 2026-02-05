import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F

from .architectures import UNetDiffusion

from .utils import get_scheduler


class Diffusion(L.LightningModule):

    def __init__(self, start=0.0001, end=0.02, timesteps=1000, scheduler_type="linear"):
        super().__init__()

        self.betas = get_scheduler(start, end, timesteps, scheduler_type)

