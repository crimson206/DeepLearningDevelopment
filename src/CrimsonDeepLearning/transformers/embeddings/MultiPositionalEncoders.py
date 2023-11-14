from .PositionalEncoders import ArbitraryPositionalEncoder, PositionalEncoder

import torch
import torch.nn as nn
from typing import List, Optional

class MultiPositionalEncoder(nn.Module):
    def __init__(self, max_lengths, embedding_dims):
        super(MultiPositionalEncoder, self).__init__()
        self.encodings = nn.ModuleList([
            PositionalEncoder(d_emb, max_len) for max_len, d_emb in zip(max_lengths, embedding_dims)
        ])

    def forward(self, *positional_ids):
        encoded_features = [encoding(pos_id) for encoding, pos_id in zip(self.encodings, positional_ids)]
        combined_encoding = torch.cat(encoded_features, dim=-1)
        return combined_encoding
    
class MultiArbitraryPositionalEncoder(nn.Module):
    """
    A module that encapsulates multiple arbitrary positional encoders for different feature dimensions.
    
    This module is useful when you want to have separate positional encodings for different parts of your model
    that may have different sequence lengths or feature dimensions.
    
    Attributes:
        encodings (nn.ModuleList): A list of arbitrary positional encoders.
    
    Args:
        max_lengths (List[int]): A list of maximum lengths for each positional encoding dimension.
        embedding_dims (List[int]): A list of embedding dimensions for each positional encoding.
    """
    def __init__(self, max_lengths: List[int], embedding_dims: List[int]) -> torch.Tensor:
        """
        Inits MultiArbitraryPositionalEncoder with specified max lengths and embedding dimensions for each encoding.
        """
        super(MultiArbitraryPositionalEncoder, self).__init__()
        self.encodings = nn.ModuleList([
            ArbitraryPositionalEncoder(max_pos_len, d_emb) for max_pos_len, d_emb in zip(max_lengths, embedding_dims)
        ])

    def forward(self, positional_ids: List[torch.Tensor], attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the MultiArbitraryPositionalEncoder. It encodes each tensor in positional_ids with the
        corresponding positional encoder and then concatenates them.
        
        Args:
            positional_ids (List[torch.Tensor]): A list of tensors where each tensor is a batch of a unique positional
                                                information.
                                                The shape is [(n_batch, seq_len)]
            attention_mask (Optional[torch.Tensor]): An optional mask tensor indicating which positions should be
                                                     attended to and which should not. 
                                                     The shape is (n_batch, seq_len)
        
        Returns:
            torch.Tensor: A tensor containing the concatenated positional encodings for all tensors in positional_ids.
        """
        device = positional_ids[0].device
        encoded_features = [encoding(pos_id.to(device).long()) for encoding, pos_id in zip(self.encodings, positional_ids)]
        combined_encoding = torch.cat(encoded_features, dim=-1).to(device)

        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(combined_encoding)
            combined_encoding = combined_encoding * expanded_mask

        return combined_encoding



    
