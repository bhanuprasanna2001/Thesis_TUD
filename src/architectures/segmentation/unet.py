import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        
    def forward(self, X):
        X = self.double_conv(X)
        return X


class Down(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
        
    def forward(self, X):
        X = self.down(X)
        return X


class Up(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        
        
    def forward(self, X):
        X = self.out(X)
        return X


class UNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, base_channels=64):
        super().__init__()
        
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2
        
        # Contracting Path (left)
        self.double_conv = DoubleConv(in_channels, out_channels=c1)
        self.down1 = Down(in_channels=c1, out_channels=c2)
        self.down2 = Down(in_channels=c2, out_channels=c3)
        self.down3 = Down(in_channels=c3, out_channels=c4)
        self.down4 = Down(in_channels=c4, out_channels=c5)
        
        
        # Expansive Path (right)
        self.up1 = Up(in_channels=c5, out_channels=c4)
        self.up2 = Up(in_channels=c4, out_channels=c3)
        self.up3 = Up(in_channels=c3, out_channels=c2)
        self.up4 = Up(in_channels=c2, out_channels=c1)
        
        # Output
        self.out = OutConv(in_channels=c1, out_channels=out_channels)
        
        
    def forward(self, X):
        # Contracting Path
        X1 = self.double_conv(X)
        X2 = self.down1(X1)
        X3 = self.down2(X2)
        X4 = self.down3(X3)
        X5 = self.down4(X4)
        
        # Expansive Path
        X6 = self.up1(X5, X4)
        X7 = self.up2(X6, X3)
        X8 = self.up3(X7, X2)
        X9 = self.up4(X8, X1)
        
        # Output (input size == output size)
        X10 = self.out(X9)
        
        return X10
        
        
if __name__ == "__main__":
    X = torch.randn((1, 1, 572, 572))
    unet = UNet(1, 1)
    out = unet(X)
    print(X.size(), out.size())
    assert X.size() == out.size()
