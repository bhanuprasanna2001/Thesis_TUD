import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, groups=8, in_channels=1, base_channels=32, latent_channels=64):
        super().__init__()

        # Downsample twice (H/4, W/4) then project to latent channels without further size change
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups, base_channels),
            nn.ReLU(inplace=True)
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups, base_channels * 2),
            nn.ReLU(inplace=True)
        )
        # No activation on final encoder layer — lets latent space be unconstrained
        self.e3 = nn.Conv2d(base_channels * 2, latent_channels, kernel_size=3, padding=1)


    def forward(self, X):
        X = self.e1(X)
        X = self.e2(X)
        X = self.e3(X)
        return X
    

class Decoder(nn.Module):

    def __init__(self, groups=8, out_channels=1, base_channels=32, latent_channels=64):
        super().__init__()

        # Mirror encoder: keep spatial size, then upsample twice back to input res
        self.d1 = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(groups, base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.d2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, base_channels),
            nn.ReLU(inplace=True)
        )
        self.d3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )


    def forward(self, Z):
        Z = self.d1(Z)
        Z = self.d2(Z)
        Z = self.d3(Z)
        return Z


class AE(nn.Module):

    def __init__(self, groups=8, in_channels=1, out_channels=None, base_channels=32, latent_channels=64):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.encoder = Encoder(groups=groups, in_channels=in_channels, base_channels=base_channels, latent_channels=latent_channels)
        self.decoder = Decoder(groups=groups, out_channels=out_channels, base_channels=base_channels, latent_channels=latent_channels)


    def forward(self, X):
        Z = self.encoder(X)
        X_hat = self.decoder(Z)
        return X_hat


if __name__ == "__main__":
    # Quick sanity checks for common image sizes
    for shape in [(1, 1, 28, 28), (1, 3, 32, 32), (1, 3, 64, 64)]:
        X = torch.randn(*shape)
        model = AE(in_channels=shape[1], out_channels=shape[1])
        out = model(X)
        assert X.shape == out.shape, f"Shape mismatch: {X.shape} -> {out.shape}"
    print("✓ AE autoencoder preserves input shapes.")
