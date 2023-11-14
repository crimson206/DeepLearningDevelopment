import torch
import torch.nn as nn
from .CategoricalFeatureEmbedders import ConcatCategoricalFeatureEmbedder
from .ContinuousFeatureEmbedder import ContinuousFeatureEmbedder
from .MultiPositionalEncoders import MultiArbitraryPositionalEncoder
from typing import List

class AssembledEmbedder(nn.Module):
    def __init__(
        self,
        pos_max_lengths: List[int],
        pos_embedding_dims: List[int],
        categorical_sizes: List[int]=None,
        categorical_emb_dims: List[int]=None,
        continuous_feature_in: int=None,
        continuous_feature_out: int=None,
    ):
        super(AssembledEmbedder, self).__init__()

        self.pos_emb = MultiArbitraryPositionalEncoder(pos_max_lengths, pos_embedding_dims)

        self.categorical_embedder = None
        self.continuous_embedder = None

        if categorical_sizes is not None and categorical_emb_dims is not None:
            self.categorical_embedder = ConcatCategoricalFeatureEmbedder(categorical_sizes, categorical_emb_dims)
            
        if continuous_feature_in is not None and continuous_feature_out is not None:
            self.continuous_embedder = ContinuousFeatureEmbedder(continuous_feature_in, continuous_feature_out)
            

    def forward(
        self,
        split_position_ids: List[torch.Tensor],
        split_categorical_ids: List[torch.Tensor]=None,
        continuous_feature: torch.Tensor=None,
        skip_embedding: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
    ) -> torch.Tensor:

        pos_embedding = self.pos_emb.forward(split_position_ids, attention_mask)

        embeddings = []

        if split_categorical_ids is not None:
            embeddings.append(self.categorical_embedder.forward(split_categorical_ids, attention_mask))
        if continuous_feature is not None:
            embeddings.append(self.continuous_embedder.forward(continuous_feature, attention_mask))
        if skip_embedding is not None:
            embeddings.append(skip_embedding)

        combined_embedding = torch.cat(embeddings, dim=2)

        final_embedding = combined_embedding + pos_embedding

        return final_embedding
