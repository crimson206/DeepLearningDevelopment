import torch
import torch.nn as nn
from typing import List
from ..embeddings import AssembledEmbedder

class EncoderTransformer(nn.Module):
    """
    A wrapper class for a pre_defined transformer encoder that uses custom embeddings.

    This class is designed to take a pre_defined transformer and apply it to custom embeddings,
    bypassing the transformer model's usual embedding layer.

    Attributes:
        pre defined transformer.

    Args:
        pre defined transformer.
    """

    def __init__(self, transformer):
        super(EncoderTransformer, self).__init__()
        self.transformer = transformer

    def forward(self, embedding: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EncoderTransformer using custom embeddings.

        Args:
            embedding (torch.Tensor): The custom embedding tensor to be processed by a pre_defined transformer.
                                     Shape: (batch_size, seq_length, embedding_dim)
            attention_mask (torch.Tensor): The attention mask for a pre_defined transformer.
                                           Shape: (batch_size, seq_length)

        Returns:
            torch.Tensor: The output from a pre_defined transformer encoder representing the hidden states.
                          Shape: (batch_size, seq_length, hidden_size)
        """
        # In Hugging Face's implementation, 'inputs_embeds' is used to pass custom embeddings.
        # 'attention_mask' is used as usual.

        #if attention_mask is not None:
        #    expanded_mask = attention_mask.unsqueeze(-1).expand_as(embedding)
        #    embedding = embedding * expanded_mask

        hidden_states = self.transformer(
            inputs_embeds=embedding,
            attention_mask=attention_mask,
        )

        return hidden_states
    
class MultiEmbeddingTransformer(nn.Module):
    def __init__(
        self,
        enc_transformer: EncoderTransformer,
        pos_max_lengths: List[int],
        pos_embedding_dims: List[int],
        categorical_sizes: List[int],
        categorical_emb_dims: List[int],
        continuous_feature_in: int=None,
        continuous_feature_out: int=None,
    ):
        super(MultiEmbeddingTransformer, self).__init__()
        self.enc_transformer = enc_transformer

        self.assembled_embedder = AssembledEmbedder(
            pos_max_lengths=pos_max_lengths,
            pos_embedding_dims=pos_embedding_dims,
            categorical_sizes=categorical_sizes,
            categorical_emb_dims=categorical_emb_dims,
            continuous_feature_in=continuous_feature_in,
            continuous_feature_out=continuous_feature_out,
        )

        self.d_emb = sum(pos_embedding_dims)

        self.sum_embedder = nn.Embedding(2, self.d_emb)

    def forward(
        self,
        split_position_ids: List[torch.Tensor],
        split_categorical_ids: List[torch.Tensor]=None,
        continuous_feature: torch.Tensor=None,
        skip_embedding: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
    ):

        assembled_embedding = self.assembled_embedder.forward(
            split_position_ids, 
            split_categorical_ids, 
            continuous_feature, 
            skip_embedding, 
            attention_mask
        )

        hidden_states = self.enc_transformer(assembled_embedding, attention_mask)

        return hidden_states