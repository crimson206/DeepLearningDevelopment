import torch
import torch.nn as nn
from typing import List

class UpsampleBlocks(nn.Module):
    """
    Shape Summary:
    Input tensor shape:
    (n_batch, hidden_sizes[0], height, width)

    Output tensor shape:
    (n_batch, hidden_sizes[-1], height * size_up, width * size_up)
    where size_up = 2 ** (len(hidden_sizes) - 1)
    """
    def __init__(self, hidden_sizes: List[int]) -> None:
        super(UpsampleBlocks, self).__init__()
        layers: List[nn.Module] = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.ConvTranspose2d(hidden_sizes[i], hidden_sizes[i+1], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(hidden_sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))

        self.upsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shape Summary:
        x.shape: (n_batch, hidden_sizes[0], height, width).

        output.shape: (n_batch, hidden_sizes[-1], height * size_up, width * size_up).
        """
        return self.upsample(x)

class UpsampleBlock(nn.Module):
    def __init__(self, up_channel:int, up_module) -> None:
        super(UpsampleBlocks, self).__init__()
        layers: List[nn.Module] = []
        layers.append(up_module)
        layers.append(nn.InstanceNorm2d(up_channel[i+1]))
        layers.append(nn.ReLU(inplace=True))

        self.upsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shape Summary:
        x.shape: (n_batch, hidden_sizes[0], height, width).

        output.shape: (n_batch, hidden_sizes[-1], height * size_up, width * size_up).
        """
        return self.upsample(x)


