import torch
import torch.nn as nn

from .ae import Decoder


class KLEncoder(nn.Module):

    def __init__(self, groups=8, in_channels=1, base_channels=32, latent_channels=64):
        super().__init__()

        # Downsample twice (H/4, W/4), then project to mu and logvar
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
        # Output 2x latent_channels: first half = mu, second half = logvar
        self.proj = nn.Conv2d(base_channels * 2, 2 * latent_channels, kernel_size=3, padding=1)


    def forward(self, X):
        X = self.e1(X)
        X = self.e2(X)
        X = self.proj(X)
        mu, logvar = X.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mu, logvar


class KLAE(nn.Module):

    def __init__(self, groups=8, in_channels=1, out_channels=None, base_channels=32, latent_channels=64):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.encoder = KLEncoder(groups=groups, in_channels=in_channels, base_channels=base_channels, latent_channels=latent_channels)
        self.decoder = Decoder(groups=groups, out_channels=out_channels, base_channels=base_channels, latent_channels=latent_channels)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


    @staticmethod
    def kl_divergence(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


    def forward(self, X):
        mu, logvar = self.encoder(X)
        Z = self.reparameterize(mu, logvar)
        X_hat = self.decoder(Z)
        return X_hat, mu, logvar


if __name__ == "__main__":
    for shape in [(1, 1, 28, 28), (1, 3, 32, 32), (1, 3, 64, 64)]:
        X = torch.randn(*shape)
        model = KLAE(in_channels=shape[1], out_channels=shape[1])
        X_hat, mu, logvar = model(X)
        assert X.shape == X_hat.shape, f"Shape mismatch: {X.shape} -> {X_hat.shape}"
        assert mu.shape[1] == 64, f"mu channels: {mu.shape[1]}"
        assert logvar.shape[1] == 64, f"logvar channels: {logvar.shape[1]}"
        kl = KLAE.kl_divergence(mu, logvar)
        assert kl.ndim == 0, f"KL should be scalar, got shape {kl.shape}"
    print("✓ KLAE autoencoder preserves input shapes.")
