import torch
import torch.nn as nn
from typing import List

from transformers import AutoConfig, AutoModel
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

class MultiEmbeddingTransformerBuilder():
    def __init__(self):
        self.model_name_list = ["microsoft/mpnet-base", "bert-base-uncased", "roberta-base", "microsoft/deberta-base"]
        self.default_config = {
            "hidden_size": 256,
            "num_hidden_layers": 9,
            "num_attention_heads": 4,
            "intermediate_size": 756,
            "attention_probs_dropout_prob": 0.1,
            "hidden_dropout_prob": 0.1,
            "output_hidden_states": True,
        }
        self.multi_emb_transformer_input_frame = {
            "pos_max_lengths":None,
            "pos_embedding_dims":None,
            "categorical_sizes":None,
            "categorical_emb_dims":None,
            "skip_emb_size":None,
            "continuous_feature_in":None,
            "continuous_feature_out":None,
        }

    def build(
            self,
            model_name,
            config,
            multi_emb_transformer_input_frame,
            autoconfig_from_pretrained,
            automodel_from_config,
            ):
        auto_config = AutoConfig.from_pretrained(model_name)
        auto_config.update(config)

        transformer = AutoModel.from_config(auto_config)

        enc_transformer = EncoderTransformer(transformer)
        multi_emb_transformer = MultiEmbeddingTransformer(
            enc_transformer=enc_transformer,
            **multi_emb_transformer_input_frame,
        )

        return multi_emb_transformer

class MultiEmbeddingTransformer(nn.Module):
    """
    A transformer model that combines various embedding representations before passing them through
    a transformer encoder. It handles positional, categorical, and continuous features, along with optional
    skip connections.

    Attributes:
        enc_transformer (EncoderTransformer): An encoder transformer to which the combined embeddings are passed.
        assembled_embedder (AssembledEmbedder): A module that assembles different embeddings into a single tensor.
        d_emb (int): The total dimension of the assembled embeddings.
        skip_emb_size (int): The size of skip embeddings, calculated based on the input feature dimensions.
        sum_embedder (nn.Embedding): An embedding layer for summing categorical embeddings if provided.

    Args:
        enc_transformer (EncoderTransformer): The transformer encoder for processing embedded inputs.
        pos_max_lengths (List[int]): Maximum lengths for positional embeddings.
        pos_embedding_dims (List[int]): Embedding dimensions for positional embeddings.
        categorical_sizes (List[int]): Sizes of each categorical feature for embeddings.
        categorical_emb_dims (List[int]): Embedding dimensions for each categorical feature.
        skip_emb_size (int): Expected size of skip embeddings.
        continuous_feature_in (int, optional): Input dimension for continuous features.
        continuous_feature_out (int, optional): Output dimension for continuous features after embedding.

    Raises:
        ValueError: If the calculated skip embedding size does not match the provided skip embedding size.

    Forward method args:
        split_position_ids (List[torch.Tensor]): A list of tensors containing positional IDs for embeddings.
                                                  Each tensor shape: (n_batch, n_seq)
        split_categorical_ids (List[torch.Tensor], optional): A list of tensors containing categorical IDs for embeddings.
                                                              Each tensor shape: (n_batch, n_seq)
        sum_categorical_ids (torch.Tensor, optional): A tensor containing categorical IDs for summed embeddings.
                                                      Tensor shape: (n_batch, n_seq)
        continuous_feature (torch.Tensor, optional): A tensor containing continuous features for embedding.
                                                     Tensor shape: (n_batch, n_seq, continuous_feature_in)
        skip_embedding (torch.Tensor, optional): A tensor containing precomputed skip embeddings.
                                                 Tensor shape: (n_batch, n_seq, skip_emb_size)
        attention_mask (torch.Tensor, optional): A tensor to apply attention masks in the transformer encoder.
                                                 Tensor shape: (n_batch, n_seq)

    Returns:
        torch.Tensor: The output tensor from the encoder transformer representing the hidden states.
                      Tensor shape: (n_batch, n_seq, d_emb)
    """
    def __init__(
        self,
        enc_transformer: EncoderTransformer,
        pos_max_lengths: List[int],
        pos_embedding_dims: List[int],
        categorical_sizes: List[int],
        categorical_emb_dims: List[int],
        skip_emb_size: int,
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

        self.skip_emb_size = self.d_emb - sum(categorical_emb_dims)
        if continuous_feature_in is not None:
            self.skip_emb_size -= continuous_feature_out

        # Check if the calculated skip embedding size matches the provided skip embedding size
        if self.skip_emb_size != skip_emb_size:
            raise ValueError(f"Calculated skip embedding size ({self.skip_emb_size}) does not match the provided skip embedding size ({skip_emb_size}).")
 
        self.sum_embedder = nn.Embedding(2, self.d_emb)

    def forward(
        self,
        split_position_ids: List[torch.Tensor],
        split_categorical_ids: List[torch.Tensor]=None,
        sum_categorical_ids: torch.Tensor=None,
        skip_embedding: torch.Tensor=None,
        continuous_feature: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
    ):
        # split_position_ids shape (n_batch, n_seq) * len(pos_max_lengths)
        # split_categorical_ids shape (n_batch, n_seq) * len(categorical_sizes)
        # sum_categorical_ids shape (n_batch, n_seq)
        # continuous_feature shape (n_batch, n_seq, continuous_feature_in)
        # skip_embedding shape (n_batch, n_seq, skip_emb_size),
        # attention mask shape (n_batch, n_seq)

        assembled_embedding = self.assembled_embedder.forward(
            split_position_ids=split_position_ids, 
            split_categorical_ids=split_categorical_ids, 
            continuous_feature=continuous_feature,
            skip_embedding=skip_embedding, 
            attention_mask=attention_mask
        )

        if sum_categorical_ids is not None:
            # (n_batch, n_seq, d_emb)
            sum_categorical_embedding = self.sum_embedder(sum_categorical_ids)
            assembled_embedding = assembled_embedding + sum_categorical_embedding

        transformer_output = self.enc_transformer(embedding=assembled_embedding, attention_mask=attention_mask)

        return transformer_output