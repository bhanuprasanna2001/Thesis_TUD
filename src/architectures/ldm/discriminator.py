import torch.nn as nn


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, X):
        return self.block(X)


class PatchDiscriminator(nn.Module):

    def __init__(self, in_channels=3, base_channels=64, n_layers=3):
        super().__init__()

        channels = [base_channels * min(2 ** i, 8) for i in range(max(1, n_layers))]

        layers = []
        prev = in_channels
        for i, ch in enumerate(channels):
            stride = 2 if i < len(channels) - 1 else 1
            layers.append(DiscriminatorBlock(prev, ch, stride=stride))
            prev = ch

        self.blocks = nn.ModuleList(layers)
        self.head = nn.Conv2d(prev, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, X, return_features=False):
        features = []

        for block in self.blocks:
            X = block(X)
            if return_features:
                features.append(X)

        logits = self.head(X)

        if return_features:
            return logits, features

        return logits
