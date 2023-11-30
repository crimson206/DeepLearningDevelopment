import torch
import torch.nn as nn
from typing import List

class SizePreservingBlock(nn.Module):
    """
    Shape Summary:
    Input tensor shape:
    (n_batch, hidden_sizes[0], height, width)

    Output tensor shape:
    (n_batch, hidden_sizes[-1], height, width)
    """
    def __init__(self, in_channel: int, out_channel: int, dropout_rate: float = 0.1, activation: nn.Module = nn.ReLU(), use_residual: bool = True) -> None:
        super(SizePreservingBlock, self).__init__()
        self.use_residual: bool = use_residual
        self.conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channel),
            activation,
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channel),
            nn.Dropout(dropout_rate)
        )
        if self.use_residual and in_channel == out_channel:
            self.residual: nn.Module = nn.Identity()
        elif self.use_residual:
            self.residual: nn.Module = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shape Summary:
        Input tensor shape:
        (n_batch, hidden_sizes[0], height, width)

        Output tensor shape:
        (n_batch, hidden_sizes[-1], height, width)
        """
        if self.use_residual:
            return self.residual(x) + self.conv(x)
        else:
            return self.conv(x)

class SizePreservingBlocks(nn.Module):
    """
    Shape Summary:
    Input tensor shape:
    (n_batch, hidden_sizes[0], height, width)

    Output tensor shape:
    (n_batch, hidden_sizes[-1], height, width)
    """
    def __init__(self, hidden_channels: List[int], activation: nn.Module, dropout_rate: float = 0.1, use_residual: bool = False) -> None:
        super(SizePreservingBlocks, self).__init__()
        layers: List[nn.Module] = []
        for i in range(len(hidden_channels) - 1):
            layers.append(SizePreservingBlock(
                in_channel=hidden_channels[i],
                out_channel=hidden_channels[i+1],
                activation=activation,
                dropout_rate=dropout_rate,
                use_residual=use_residual))
        self.res_blocks: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shape Summary:
        Input tensor shape:
        (n_batch, hidden_sizes[0], height, width)

        Output tensor shape:
        (n_batch, hidden_sizes[-1], height, width)
        """
        return self.res_blocks(x)
