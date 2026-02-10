import torch
import torch.nn as nn
import torch.nn.functional as F

from .ae import Encoder, Decoder


class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)


    def forward(self, z_e):
        # z_e: (B, C, H, W) where C = embedding_dim
        B, C, H, W = z_e.shape

        # Flatten spatial dims for distance computation: (B*H*W, C)
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)

        # Squared distances: ||z - e||^2 = ||z||^2 - 2*z·e + ||e||^2
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )

        # Nearest codebook entry
        indices = distances.argmin(dim=1)
        z_q = self.embedding(indices).view(B, H, W, C).permute(0, 3, 1, 2)

        # VQ losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: forward uses z_q, backward passes gradients to z_e
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, vq_loss, indices.view(B, H, W)


class VQAE(nn.Module):

    def __init__(self, groups=8, in_channels=1, out_channels=None, base_channels=32, latent_channels=64, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.encoder = Encoder(groups=groups, in_channels=in_channels, base_channels=base_channels, latent_channels=latent_channels)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_channels, commitment_cost=commitment_cost)
        self.decoder = Decoder(groups=groups, out_channels=out_channels, base_channels=base_channels, latent_channels=latent_channels)


    def forward(self, X):
        Z_e = self.encoder(X)
        Z_q, vq_loss, indices = self.vq(Z_e)
        X_hat = self.decoder(Z_q)
        return X_hat, vq_loss, indices


if __name__ == "__main__":
    for shape in [(1, 1, 28, 28), (1, 3, 32, 32), (1, 3, 64, 64)]:
        X = torch.randn(*shape)
        model = VQAE(in_channels=shape[1], out_channels=shape[1])
        X_hat, vq_loss, indices = model(X)
        assert X.shape == X_hat.shape, f"Shape mismatch: {X.shape} -> {X_hat.shape}"
        assert vq_loss.ndim == 0, f"VQ loss should be scalar, got shape {vq_loss.shape}"
        B, H_lat, W_lat = indices.shape
        assert B == shape[0], f"Batch mismatch"
        assert H_lat == shape[2] // 4 and W_lat == shape[3] // 4, f"Spatial mismatch"
    print("✓ VQAE autoencoder preserves input shapes.")
