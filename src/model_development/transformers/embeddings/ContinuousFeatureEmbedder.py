import torch
import torch.nn as nn
from torch import Tensor

class ContinuousFeatureEmbedder(nn.Module):
    """
    A neural network module for embedding continuous features.
    
    Attributes:
        mean (Tensor): A buffer for the mean used to normalize features.
        std (Tensor): A buffer for the standard deviation used to normalize features.
        linear (nn.Linear): A linear layer for transforming normalized features.
    
    Args:
        input_dim (int): The number of dimensions of the input features.
        output_dim (int): The number of dimensions of the output embeddings.
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(ContinuousFeatureEmbedder, self).__init__()
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('std', torch.ones(input_dim))
        
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the embedder.

        Processes the input tensor through a normalization layer followed by a linear layer to produce embedded features.

        Args:
            x (Tensor): The input tensor containing continuous features.
                        Shape: (n_batch, n_seq, input_dim), where:
                        - n_batch is the batch size,
                        - n_seq is the sequence length,
                        - input_dim is the number of continuous features per sequence element.

        Returns:
            Tensor: The output tensor containing the embedded features.
                    Shape: (n_batch, n_seq, output_dim), where:
                    - n_batch is the batch size,
                    - n_seq is the sequence length,
                    - output_dim is the number of output dimensions as defined in the model.
        """
        x = x.float()
        normalized_x = (x - self.mean) / (self.std + 1e-6)
        embedded_x = self.linear(normalized_x)
        return embedded_x
    
    def fit_normalization(self, data: Tensor) -> None:
        """
        Calculates and sets the mean and standard deviation for normalization.

        This should be called with the training data to set the mean and std buffers used for normalizing inputs during the forward pass.

        Args:
            data (Tensor): The input tensor containing the training data
                           from which to calculate the mean and standard deviation.
                           Shape: (n_batch, n_seq, input_dim), where:
                           - n_batch is the batch size,
                           - n_seq is the sequence length,
                           - input_dim is the number of continuous features per sequence element.

        Returns:
            None
        """
        # Calculate the mean and std along the batch and sequence dimensions
        self.mean = data.mean(dim=(0, 1))
        self.std = data.std(dim=(0, 1)) + 1e-6