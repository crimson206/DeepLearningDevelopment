import torch
import torch.nn as nn
from typing import List
from ..headers.Regressionheads import RegressionHead
from .MultiEmbeddingTransformers import MultiEmbeddingTransformer

class Crimsonformer(nn.Module):
    def __init__(
        self,
        multi_embedding_transformer: MultiEmbeddingTransformer,
        regression_heads: nn.ModuleList(list[RegressionHead]),
    ):
        super(Crimsonformer, self).__init__()
        self.multi_embedding_transformer = multi_embedding_transformer
        self.base_transformer_config = multi_embedding_transformer.enc_transformer.transformer.config

        self.regression_heads: list[RegressionHead] = regression_heads

    def forward(
        self,
        split_position_ids: List[torch.Tensor],
        split_categorical_ids: List[torch.Tensor]=None,
        sum_categorical_ids: torch.Tensor=None,
        skip_embedding: torch.Tensor=None,
        continuous_feature: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
    ):

        transformer_output = self.multi_embedding_transformer(
            split_position_ids,
            split_categorical_ids,
            sum_categorical_ids,
            continuous_feature,
            skip_embedding,
            attention_mask,
        )

        hidden_states = transformer_output.hidden_states

        n_regression_heads = len(self.regression_heads)
        n_layers = self.base_transformer_config.num_hidden_layers

        outputs = [
            regression_head(hidden_states[(i * n_layers) // n_regression_heads])
            for i, regression_head in enumerate(self.regression_heads)
        ]

        return outputs
