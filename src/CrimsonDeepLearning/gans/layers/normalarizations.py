import torch
import torch.nn as nn

def _compute_mean_std(
    feats: torch.Tensor, eps:float=1e-8
) -> torch.Tensor:
    n_batch, n_channel, _, _ = feats.shape

    feats = feats.view(n_batch, n_channel, -1)
    mean = torch.mean(feats, dim=-1).view(n_batch, n_channel, 1, 1)
    std = torch.std(feats, dim=-1).view(n_batch, n_channel, 1, 1) + eps
    return mean, std

def adaptive_instance_normalize(
    content_feature: torch.Tensor,
    style_feature: torch.Tensor,
) -> torch.Tensor:
    """
    Shape:
        - content_feature: (n_batch, n_channel, height, weight)
        - style_feature: (n_batch, n_channel, height, weight)
        - output: (n_batch, n_channel, height, weight)
    """

    c_mean, c_std = _compute_mean_std(content_feature)
    s_mean, s_std = _compute_mean_std(style_feature)

    normalized = (s_std * (content_feature - c_mean) / c_std) + s_mean

    return normalized

class AdaIn(nn.Module):
    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)
        
    def forward(self, image, style):
        factor, bias = style[:,:,None,None].chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias  
        return result