import torch
import torch.nn as nn
from typing import Optional

from CrimsonDeepLearning.gans.layers.equalized_layers import EqualizedLinear
from CrimsonDeepLearning.gans.layers.modulations import Conv2dWeightModulate
from CrimsonDeepLearning.gans.layers.normalarizations import AdaIn

class StyleBlock(nn.Module):
    """
    Shape Summary:
    x.shape: (n_batch, hidden_sizes[0], height, width).

    output.shape: (n_batch, hidden_sizes[-1], height, width).
    """
    def __init__(self, n_w_latent: int, input_channel: int, output_channel: int, kernel_size: int = 3, style_apply_mechanism="modulation", demodulate=True):
        super().__init__()

        self.to_style = EqualizedLinear(n_w_latent, input_channel, bias=1.0)
        if style_apply_mechanism=="modulatioin":
            self.style_apply = Conv2dWeightModulate(input_channel, output_channel, kernel_size=kernel_size, demodulate=demodulate)
        elif style_apply_mechanism=="adain":
            self.style_apply = AdaIn(n_channel=input_channel)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(output_channel))

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, feature_map: torch.Tensor, w_latent: torch.Tensor, add_noise=True) -> torch.Tensor:
        """
        If add_noise is True, noise is broadcasted and added to feature_map. 
        The noise tensor initially has shape (n_batch, 1, height, width) and is broadcasted 
        to match the feature_map's shape (n_batch, n_channel, height, width) across the channel dimension.

        Shape Summary:
            feature_map.shape: (n_batch, input_channels, height, width).
            w_latent.shape: (n_batch, n_w_latent).

            output.shape:(n_batch, output_channels, height, width).
        """

        style = self.to_style.forward(w_latent)
        feature_map = self.style_apply.forward(feature_map, style)
        if add_noise:
            noise = self._get_noise(feature_map)
            feature_map = feature_map + self.scale_noise[None, :, None, None] * noise
        return self.activation(feature_map + self.bias[None, :, None, None])

    def _get_noise(self, target_tensor):
            device = target_tensor.device
            n_batch, _, height, width = target_tensor.shape
            noise = torch.randn(n_batch, 1, height, width, device=device)
            return noise


class To_RGB(StyleBlock):
    def __init__(self, n_w_latent: int, input_channel: int, output_channel: int=3, style_apply_mechanism="modulation"):
        super().__init__(n_w_latent, input_channel, output_channel=output_channel, kernel_size=1, style_apply_mechanism=style_apply_mechanism, demodulate=False)

    def forward(self, feature_map: torch.Tensor, w_latent: torch.Tensor) -> torch.Tensor:
        """
        Shape Summary:
            feature_map.shape: (n_batch, input_channels, height, width).
            w_latent.shape: (n_batch, n_w_latent).

            output.shape:(n_batch, output_channels, height, width).
        """
        return super().forward(feature_map, w_latent, add_noise=False)
        