import torch
import lightning as L

def get_scheduler(start=0.0001, end=0.02, timesteps=1000, type="linear"):
    if type == "linear":
        return torch.linspace(start, end, timesteps)
    else:
        raise NotImplementedError(f"{type} has not been implemented.")
    
