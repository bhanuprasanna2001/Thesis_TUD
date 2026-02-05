import torch
import torch.nn as nn
import torch.nn.functional as F

c


class UNetDiffusion(nn.Module):
    
    def __init__(self, in_channels, out_channels, base_channels=64):
        super().__init__()
        
        
        
    def forward(self, X, t):
        pass
        
        
if __name__ == "__main__":
    X = torch.randn((1, 1, 572, 572))
    unet = UNetDiffusion(1, 1)
    out = unet(X)
    print(X.size(), out.size())
    assert X.size() == out.size()
