import torch
import torch.nn as nn
from typing import List

class ConvDownsampleBlocks(nn.Module):
    """
    Shape Summary:
    Input tensor shape:
    (n_batch, hidden_sizes[0], height, width)

    Output tensor shape:
    (n_batch, hidden_sizes[-1], height / size_down, width / size_down)
    where / size_down = (/_up 2) ** (len(hidden_sizes) - 1)
    """
    def __init__(self, hidden_sizes: List[int]) -> None:
        super(ConvDownsampleBlocks, self).__init__()
        layers: List[nn.Module] = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(ConvDownsampleBlock(hidden_sizes[i], hidden_sizes[i+1]))

        self.downsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shape Summary:
        x.shape: (n_batch, hidden_sizes[0], height, width).

        output.shape:
        (n_batch, hidden_sizes[-1], height / size_down, width / size_down)
        where / size_down = (/_up 2) ** (len(hidden_sizes) - 1)
        """
        return self.downsample(x)

class ConvDownsampleBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, activation=nn.ReLU(inplace=True)) -> None:
        super(ConvDownsampleBlock, self).__init__()
        layers: List[nn.Module] = []

        layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(out_channel))
        layers.append(activation)

        self.downsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)

class DownsampleBlock(nn.Module):
    def __init__(self, down_channel, down_module=nn.MaxPool2d(kernel_size=2)) -> None:
        super(DownsampleBlock, self).__init__()
        layers: List[nn.Module] = []

        layers.append(down_module)
        layers.append(nn.InstanceNorm2d(down_channel))
        layers.append(nn.ReLU(inplace=True))

        self.downsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)
