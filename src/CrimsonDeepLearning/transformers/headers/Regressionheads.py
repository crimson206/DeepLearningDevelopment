import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class RegressionHead(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: Optional[List[int]] = None, 
        output_size: int = 1,
        pool: bool = False,
        squeeze: bool = True,
    ):
        super(RegressionHead, self).__init__()
        # Initialize hidden sizes if not provided
        if hidden_sizes is None:
            hidden_sizes = [input_size // 2, input_size // 4]

        # Calculate the sizes for each layer
        sizes = [input_size] + hidden_sizes + [output_size]
        # Create a list of linear layers based on the sizes
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )

        self.pool = pool
        self.squeeze = squeeze
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mean pooling if enabled
        if self.pool:
            x = x.mean(dim=1)

        # Pass input through each layer and apply ReLU activation for non-final layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)

        # Squeeze the last dimension if required
        if self.squeeze:
            x = x.squeeze(-1)

        return x