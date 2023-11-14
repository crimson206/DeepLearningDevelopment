import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

class ConcatCategoricalFeatureEmbedder(nn.Module):
    """
    A PyTorch module for embedding multiple categorical features and concatenating
    them into a single tensor. 

    Attributes:
        embeddings (nn.ModuleList): A list of embedding layers, one for each categorical feature.
    
    Args:
        categorical_sizes (List[int]): A list containing the number of unique values for each categorical feature.
        embedding_sizes (List[int]): A list containing the dimension of the embedding vector for each categorical feature.
    """

    def __init__(self, categorical_sizes: List[int], embedding_sizes: List[int]):
        """
        Inits ConcatCategoricalFeatureEmbedder with the given categorical sizes and embedding sizes.
        """
        super(ConcatCategoricalFeatureEmbedder, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=dim) 
            for size, dim in zip(categorical_sizes, embedding_sizes)
        ])

        total_embedding_size = sum(embedding_sizes)
        self.layer_norm = nn.LayerNorm(total_embedding_size)

    def forward(self, categorical_inputs: List[torch.Tensor], attention_mask: Optional[Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the ConcatCategoricalFeatureEmbedder. Embeds the input indices for each 
        categorical feature and concatenates them into a single tensor.

        Args:
            categorical_inputs (List[torch.Tensor]): A list of tensors where each tensor contains
                                                    integer indices of the categorical features in a batch.
            attention_mask (Optional[torch.Tensor]): An optional mask tensor indicating which positions should be
                                                     attended to and which should not. 
                                                     The shape is (n_batch, seq_len)
        Returns:
            torch.Tensor: A tensor containing the concatenated embeddings of the input features.
        """
        embedded_features = [emb(input) for emb, input in zip(self.embeddings, categorical_inputs)]
        concatenated_embeddings = torch.cat(embedded_features, dim=-1)
        normalized_embeddings = self.layer_norm(concatenated_embeddings)

        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(normalized_embeddings)
            normalized_embeddings = normalized_embeddings * expanded_mask

        return normalized_embeddings


class SumCategoricalFeatureEmbedder(nn.Module):
    def __init__(self, categorical_sizes, embedding_dim):
        super(SumCategoricalFeatureEmbedder, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim) 
            for size in categorical_sizes
        ])

    def forward(self, categorical_inputs):
        embedded_features = [emb(input) for emb, input in zip(self.embeddings, categorical_inputs)]
        summed_embeddings = sum(embedded_features)
        return summed_embeddings
