import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

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
        
    def forward(self, continuous_feature: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the embedder.

        Processes the input tensor through a linear layer and then applies layer normalization
        to produce embedded features.

        Args:
            x (Tensor): The input tensor containing continuous features.
                        Shape: (n_batch, n_seq, input_dim).
            attention_mask (Optional[torch.Tensor]): An optional mask tensor indicating which positions should be
                                                     attended to and which should not. 
                                                     The shape is (n_batch, seq_len)

        Returns:
            Tensor: The output tensor containing the embedded features after layer normalization.
                    Shape: (n_batch, n_seq, output_dim).
        """
        continuous_feature = continuous_feature.float()
        embedded_x = self.linear(continuous_feature)
        normalized_embedded_x = self.layer_norm(embedded_x)

        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(normalized_embedded_x)
            normalized_embedded_x = normalized_embedded_x * expanded_mask

        return normalized_embedded_x

import torch
import torch.nn as nn

class ContinuousToCategoryEmbedder(nn.Module):
    def __init__(self, d_emb=2, embedding_size=200, buffer=5):
        super(ContinuousToCategoryEmbedder, self).__init__()
        self.bn = nn.BatchNorm1d(1)
        self.buffer = buffer
        self.embedder = nn.Embedding(num_embeddings=embedding_size + 1, embedding_dim=d_emb)
        self.scale = embedding_size / (2 * buffer)
        self.nan_padding = embedding_size

        self.total_batches_processed = 0
        self.cumulative_out_of_range_rate = 0

    def forward(self, input_tensor: torch.Tensor, return_norm=False) -> Tensor:
        shape = input_tensor.shape

        input_tensor = input_tensor.flatten(0)

        batch_size = input_tensor.shape[0]
        nan_mask = torch.isnan(input_tensor)
        input_tensor = input_tensor[~nan_mask]
        input_tensor = input_tensor.unsqueeze(-1)
        normalized_tensor = self.bn(input_tensor)

        transformed_tensor = (normalized_tensor + self.buffer) * self.scale

        # track out range
        out_of_range_mask = (normalized_tensor < -self.buffer) | (normalized_tensor > self.buffer)
        out_of_range_rate = out_of_range_mask.sum().item() / normalized_tensor.numel()
        self.total_batches_processed += 1
        self.cumulative_out_of_range_rate += (out_of_range_rate - self.cumulative_out_of_range_rate) / self.total_batches_processed

        output_tensor = torch.full((batch_size, 1), self.nan_padding, dtype=torch.long)

        output_tensor[~nan_mask, :] = transformed_tensor.clamp(0, self.nan_padding-1).long()

        output_tensor = output_tensor.view(shape)

        embedding = self.embedder(output_tensor)
        if return_norm:
            return embedding, normalized_tensor
        else:
            return embedding

class MultiContinuousToCategoryEmbedder(nn.Module):
    def __init__(self, n_feature, d_emb=2, embedding_size=200, buffer=5):
        super(MultiContinuousToCategoryEmbedder, self).__init__()
        # Initialize n_feature number of ContinuousToCategoryEmbedders
        self.embedders = nn.ModuleList([ContinuousToCategoryEmbedder(d_emb=d_emb, embedding_size=embedding_size, buffer=buffer) for _ in range(n_feature)])

    def forward(self, input_tensor):
        # Assuming input_tensor is a 2D tensor with shape (n_batch, n_feature)
        output_tensors = []

        # Apply each embedder to the corresponding feature column
        for i, embedder in enumerate(self.embedders):
            transformed_column = embedder.forward(input_tensor[:, :, i])
            output_tensors.append(transformed_column)

        # Assuming you want to concatenate the outputs along the last dimension
        # Convert the list of tensors to a single tensor
        output_tensor = torch.cat(output_tensors, dim=-1)

        return output_tensor