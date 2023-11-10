from PositionalEncoders import ArbitraryPositionalEncoder, PositionalEncoder

import torch
import torch.nn as nn

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
    def __init__(self, max_lengths, embedding_dims):
        super(MultiArbitraryPositionalEncoder, self).__init__()
        self.encodings = nn.ModuleList([
            ArbitraryPositionalEncoder(max_pos_len, d_emb) for max_pos_len, d_emb in zip(max_lengths, embedding_dims)
        ])

    def forward(self, *positional_ids, attention_mask=None):
        encoded_features = [encoding(pos_id.long()) for encoding, pos_id in zip(self.encodings, positional_ids)]
        combined_encoding = torch.cat(encoded_features, dim=-1)

        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(combined_encoding)
            combined_encoding = combined_encoding * expanded_mask

        return combined_encoding
