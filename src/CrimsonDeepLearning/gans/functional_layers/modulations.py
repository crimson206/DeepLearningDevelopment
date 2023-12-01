import torch
import torch.nn as nn
import torch.nn.functional as F

from CrimsonDeepLearning.gans.functional_layers.equalized_layers import EqualizedWeight
    
class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, demodulate=True, eps=1e-8):
        super().__init__()
        self.out_features = out_feature
        self.demodulate = demodulate
        self.weight = EqualizedWeight([out_feature, in_feature, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, feature_map, style):
        """
        Shape:
            - feature_map.shape: (n_batch, in_feature, height, width)
            - style: (n_batch, in_feature)

            - output.shape: (n_batch, out_feature, height, width)
        """
        weights = self._modulate(self.weight(), style)
        if self.demodulate:
            weights = self._demodulate(weights, self.eps)
        return self._conv2d_batchwise(feature_map, weights)

    def _modulate(self, weights, style):
        """
        Shape:
            - weights.shape: (out_feature, in_feature, kernel_size, kernel_size)
            - style: (n_batch, in_feature)

            - output.shape: (n_batch, out_feature, in_feature, kernel_size, kernel_size)
        """
        # Expand and rearrange weights
        style = style[:, None, :, None, None]
        weights = weights[None, :, :, :, :]
        weights = weights * style
        return weights

    def _demodulate(self, weights, eps):
        """
        Shape:
            - weights.shape: (n_batch, out_feature, in_feature, kernel_size, kernel_size)

            - output.shape: (n_batch, out_feature, in_feature, kernel_size, kernel_size)
        """
        # Normalize the magnitude of weights
        sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + eps)
        return weights * sigma_inv

    def _conv2d_batchwise(self, feature_map, weights):
        """
        Shape:
            - feature_map.shape: (n_batch, in_feature, height, width)
            - weights.shape: (n_batch, out_feature, in_feature, kernel_size, kernel_size)

            - output.shape: (n_batch, out_feature, height, width)
        """
        # Prepare input for grouped convolution
        n_batch, _, height, width = feature_map.shape
        feature_map = feature_map.reshape(1, -1, height, width)
        _, out_feature, *style_kernel_dims = weights.shape
        weights = weights.reshape(n_batch * out_feature, *style_kernel_dims)
        # Perform grouped convolution
        feature_map = F.conv2d(feature_map, weights, padding="same", groups=n_batch)
        return feature_map.reshape(-1, out_feature, height, width)
