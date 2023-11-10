import torch
import torch.nn as nn
from torch import Tensor

class ContinuousFeatureEmbedder(nn.Module):
    """
    A neural network module for embedding continuous features.
    
    Attributes:
        linear (nn.Linear): A linear layer for transforming features into embeddings.
        layer_norm (nn.LayerNorm): A layer normalization module applied post-embedding.
    
    Args:
        input_dim (int): The number of dimensions of the input features.
        output_dim (int): The number of dimensions of the output embeddings.
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(ContinuousFeatureEmbedder, self).__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the embedder.

        Processes the input tensor through a linear layer and then applies layer normalization
        to produce embedded features.

        Args:
            x (Tensor): The input tensor containing continuous features.
                        Shape: (n_batch, n_seq, input_dim).

        Returns:
            Tensor: The output tensor containing the embedded features after layer normalization.
                    Shape: (n_batch, n_seq, output_dim).
        """
        x = x.float()
        embedded_x = self.linear(x)
        normalized_embedded_x = self.layer_norm(embedded_x)
        return normalized_embedded_x
