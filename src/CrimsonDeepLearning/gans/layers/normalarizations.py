import torch
import torch.nn as nn
from CrimsonDeepLearning.gans.layers.equalized_layers import EqualizedLinear, EqualizedConv2d

def _compute_mean_std(
    feats: torch.Tensor, eps:float=1e-8
) -> torch.Tensor:
    """
    Shape:
        - features: (n_batch, n_channel, height, weight) or (n_batch, n_channel)

        - output: (n_batch, n_channel, 1, 1)
    """

    n_batch, n_channel = feats.shape[:2]

    feats = feats.view(n_batch, n_channel, -1)
    mean = torch.mean(feats, dim=-1).view(n_batch, n_channel, 1, 1)
    std = torch.std(feats, dim=-1).view(n_batch, n_channel, 1, 1) + eps
    return mean, std

def adaptive_instance_normalize(
    feature_map: torch.Tensor,
    style: torch.Tensor,
) -> torch.Tensor:
    """
    Shape:
        - content_feature: (n_batch, n_channel, height, weight)
        - style: (n_batch, n_channel)
        - output: (n_batch, n_channel, height, weight)
    """

    feature_mean, feature_std = _compute_mean_std(feature_map)
    style_mean, style_std = _compute_mean_std(style)

    normalized = (style_std * (feature_map - feature_mean) / feature_std) + style_mean

    return normalized


class AdaIn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.scale_fc = EqualizedLinear(in_feature=in_channel, out_feature=out_channel)
        self.bias_fc = EqualizedLinear(in_feature=in_channel, out_feature=out_channel)
        self.conv = EqualizedConv2d(in_feature=in_channel, out_feature=out_channel, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channel)
        self._out_channel = out_channel
        
    def forward(self, feature_map, style):
        """
        Shape:
            - feature_map.shape: (n_batch, in_feature, height, width)
            - style: (n_batch, in_feature)

            - output.shape: (n_batch, out_feature, height, width)
        """

        n_batch = feature_map.shape[0]

        scale = self.scale_fc(style).view(n_batch, self._out_channel, 1, 1)
        bias = self.bias_fc(style).view(n_batch, self._out_channel, 1, 1)

        feature_map = self.conv.forward(feature_map)
        feature_map = scale * feature_map + bias

        feature_map = self.norm(feature_map)

        return feature_map