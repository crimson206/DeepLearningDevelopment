import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class EqualizedWeight(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        """
        Shape:
            - output.shape: shape (input of constructor)
        """
        return self.weight * self.c

class EqualizedLinear(nn.Module):
    def __init__(self, in_feature, out_feature, bias = 0.0):
        super().__init__()
        self.weight = EqualizedWeight([out_feature, in_feature])
        self.bias = nn.Parameter(torch.ones(out_feature) * bias)

    def forward(self, input_tensor: torch.Tensor):
        """
        Shape:
            - input_tensor.shape: (n_batch, in_feature)

            - output.shape: (n_batch, out_feature)
        """
        return F.linear(input_tensor, self.weight(), bias=self.bias)
    
class EqualizedConv2d(nn.Module):
    def __init__(self, in_feature, out_feature,
                 kernel_size, padding = 0):

        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_feature, in_feature, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_feature))

    def forward(self, input_tensor: torch.Tensor):
        """
        Shape:
            - input_tensor.shape: (n_batch, in_feature, height, width)

            - output.shape: (n_batch, out_feature, height, width)
        """
        return F.conv2d(input_tensor, self.weight(), bias=self.bias, padding=self.padding)

class EqualizedAffineTransformLayer(nn.Module):
    def __init__(self, n_w_latent, n_channel, bias):
        super().__init__()
        self.affine_transform = EqualizedLinear(n_w_latent, 2*n_channel, bias=bias)

    def forward(self, feature_map, w_latent):
        """
        Shape:
            - feature_map.shape: (n_batch, n_channel, height, width)
            - w_latent: (n_batch, n_w_latent)

            - output.shape: (n_batch, n_channel, height, width)
        """
        style_params = self.affine_transform(w_latent)
        scale, bias = style_params.chunk(2, 1)
    
        feature_map = feature_map * (scale + 1) + bias
        
        return feature_map

