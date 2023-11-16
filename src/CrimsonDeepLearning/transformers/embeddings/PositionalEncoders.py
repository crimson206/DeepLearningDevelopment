import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoder, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class ArbitraryPositionalEncoder(nn.Module):
    def __init__(self, max_pos_len, d_emb):
        super(ArbitraryPositionalEncoder, self).__init__()
        self.d_emb = d_emb
        self.max_pos_len = max_pos_len
        self.encodings = get_sinusoidal_encoding(self.max_pos_len + 1, self.d_emb)

    def forward(self, input_seqs):
        """
        Apply sinusoidal positional encoding to the input batch of sequences using custom positions.

        Args:
        input_seqs: A 2D torch.Tensor of shape (n_batch, n_seq) containing sequences of positions.

        Returns:
        A 3D torch.Tensor of shape (n_batch, n_seq, d_emb) containing the positional encodings for the batch.
        """
        # Get the positional encodings for the batch
        batch_encodings = torch.stack([self.encodings[idx] for idx in input_seqs])

        return batch_encodings

def get_sinusoidal_encoding(max_seq_len, d_emb):
    """
    Generate sinusoidal positional encodings for a given sequence length and dimensionality.

    Args:
    max_seq_len: The maximum length of the input sequences for which to generate encodings.
    d_emb: The dimensionality of the embeddings.

    Returns:
    A 2D torch.Tensor of shape (max_seq_len, d_emb) containing the positional encodings.
    """
    position = torch.arange(max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_emb, 2).float() * -(math.log(10000.0) / d_emb))
    encoding = torch.zeros(max_seq_len, d_emb)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding