import torch
import torch.nn as nn

class ConcatCategoricalFeatureEmbedder(nn.Module):
    def __init__(self, categorical_sizes, embedding_sizes):
        super(ConcatCategoricalFeatureEmbedder, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=dim) 
            for size, dim in zip(categorical_sizes, embedding_sizes)
        ])

    def forward(self, *categorical_inputs):
        embedded_features = [emb(input) for emb, input in zip(self.embeddings, categorical_inputs)]
        concatenated_embeddings = torch.cat(embedded_features, dim=-1)
        return concatenated_embeddings

class SumCategoricalFeatureEmbedder(nn.Module):
    def __init__(self, categorical_sizes, embedding_dim):
        super(SumCategoricalFeatureEmbedder, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim) 
            for size in categorical_sizes
        ])

    def forward(self, *categorical_inputs):
        embedded_features = [emb(input) for emb, input in zip(self.embeddings, categorical_inputs)]
        summed_embeddings = sum(embedded_features)
        return summed_embeddings
