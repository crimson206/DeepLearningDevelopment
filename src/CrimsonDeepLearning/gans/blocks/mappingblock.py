
import torch
import torch.nn as nn

from CrimsonDeepLearning.gans.layers.equalized_layers import EqualizedLinear

class MappingNetwork(nn.Module):
    def __init__(self, n_z_latent: int = 256, n_w_latent: int = 256, hidden_sizes: list[int] = [256] * 8, activation: nn.Module = nn.LeakyReLU(negative_slope=0.2, inplace=True)):
        super().__init__()

        layers = []
        sizes = [n_z_latent] + hidden_sizes + [n_w_latent]
        for i in range(len(sizes) - 1):
            layers.append(EqualizedLinear(sizes[i], sizes[i + 1]))
            layers.append(activation)

        self.mapping = nn.Sequential(*layers)

    def forward(self, z_latent):
        z_latent = self._normalize_pixel(z_latent)
        w_latent = self.mapping(z_latent)
        return w_latent

    def _normalize_pixel(self, x, eps=1e-8):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
