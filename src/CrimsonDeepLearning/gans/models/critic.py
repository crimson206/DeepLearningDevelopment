import math

import torch
import torch.nn as nn
import copy
import sys
# Append the directory of your project to sys.path

from CrimsonDeepLearning.gans.layers.equalized_layers import EqualizedConv2d, EqualizedLinear




class CriticBlock(nn.Module):
    def __init__(self, in_features, out_features, down_sample_mechanism="smooth"):
        super().__init__()

        if down_sample_mechanism=="smooth":
            down_sample = DownSample()
        elif down_sample_mechanism=="average":
            down_sample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.residual_down_sample = nn.Sequential(
            copy.deepcopy(down_sample),
            EqualizedConv2d(in_features, out_features, kernel_size=1)
        )

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = copy.deepcopy(down_sample)
        self.scale = 1 / math.sqrt(2)

    def forward(self, input_tensor):
        """
        Shape Summary:
            input_tensor: (n_batch, input_channels, height, width).

            output.shape:(n_batch, output_channels, height//2, width//2).
        """
        residual = self.residual_down_sample(input_tensor)
        input_tensor = self.block(input_tensor)
        input_tensor = self.down_sample(input_tensor)
        return (input_tensor + residual) * self.scale

class Critic(nn.Module):
    def __init__(self, log_resolution, n_feature = 64, max_feature = 256):

        super().__init__()

        features = [min(max_feature, n_feature * (2 ** i)) for i in range(log_resolution - 1)]
        print(features)
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(in_feature=3, out_feature=features[0], kernel_size=1),
            nn.LeakyReLU(0.2, True),
        )
        n_blocks = len(features) - 1
        blocks = [CriticBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_feature = features[-1] + 1
        self.conv = EqualizedConv2d(in_feature=final_feature, out_feature=final_feature, kernel_size=3, padding=1)

        self.final = None

    def forward(self, input_tensor):
        """
        Shape Summary:
            input_tensor: (n_batch, input_channels, height, width).

            output.shape:(n_batch, 1).
        """
        input_tensor = self.from_rgb(input_tensor)
        input_tensor = self.blocks(input_tensor)

        minibatch_std_feature = self._minibatch_std(input_tensor)
        input_tensor = torch.cat([input_tensor, minibatch_std_feature], dim=1)
        input_tensor = self.conv(input_tensor)

        if self.final is None:
            device = input_tensor.device
            self.final = EqualizedLinear(in_feature=math.prod(input_tensor.shape[1:]), out_feature=1)
            self.final.to(device)

        input_tensor = input_tensor.reshape(input_tensor.shape[0], -1)
        return self.final(input_tensor)

    def _minibatch_std(self, input_tensor):
        """
        This function is to make a diversity feature

        Shape Summary:
            input_tensor: (n_batch, input_channels, height, width).

            output.shape:(n_batch, 1, height, width).
        """
        batch_statistics = torch.std(input_tensor, dim=0).mean().repeat(input_tensor.shape[0], 1, input_tensor.shape[2], input_tensor.shape[3])
        return batch_statistics