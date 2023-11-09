import torch
import torch.nn as nn
import math

from transformers import AutoModel

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class CustomEmbeddingLayer(nn.Module):
    def __init__(self, num_continuous, n_emb, num_embeddings, dropout_rate=0.1, freeze_continuous_emb=True, freeze_categorical_emb=True):
        super(CustomEmbeddingLayer, self).__init__()

        self.fc = nn.Linear(num_continuous, n_emb)
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_emb, n_emb) for num_emb in num_embeddings
        ])

        self.pos_emb = SinusoidalPosEmb(n_emb)

        self.layer_norm = nn.LayerNorm(n_emb, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

        if freeze_continuous_emb:
            for param in self.fc.parameters():
                param.requires_grad = False

        if freeze_categorical_emb:
            for layer in self.embedding_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
    def forward(self, continuous_data, categorical_data):
        batch_size = continuous_data.size()[0]
        device = continuous_data.device

        continuous_emb = self.fc(continuous_data)

        cat_embs = [emb_layer(categorical_data[:, :, i]) for i, emb_layer in enumerate(self.embedding_layers)]
        cat_emb_summed = torch.stack(cat_embs, dim=-1).sum(dim=-1)

        sequence_length = continuous_data.size()[1]
        position_ids = torch.arange(sequence_length).unsqueeze(0).repeat(batch_size, 1).to(device)
        pos_emb = self.pos_emb(position_ids)

        combined_emb = continuous_emb + cat_emb_summed + pos_emb
        combined_emb = self.layer_norm(combined_emb)
        combined_emb = self.dropout(combined_emb)
        return combined_emb
    
class BothDTypeTransformer(nn.Module):
    def __init__(self, num_continuous, n_emb, num_embeddings, transformer_config, dropout_rate=0.1, freeze_continuous_emb=True, freeze_categorical_emb=True):
        super(BothDTypeTransformer, self).__init__()
        
        self.embeddings = CustomEmbeddingLayer(
            num_continuous,
            n_emb,
            num_embeddings,
            dropout_rate=dropout_rate,
            freeze_continuous_emb=freeze_continuous_emb,
            freeze_categorical_emb=freeze_categorical_emb,
        )
        self.transformer = AutoModel.from_config(transformer_config)

        self.fc1 = nn.Linear(n_emb, n_emb // 2)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(n_emb // 2, n_emb // 4)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(n_emb // 4, 1)

    def forward(self, continuous_data, categorical_data, attention_mask=None):

        embeddings = self.embeddings(continuous_data, categorical_data)

        transformer_output = self.transformer(inputs_embeds=embeddings, attention_mask=attention_mask)

        output = self.fc1(transformer_output.last_hidden_state)
        output = self.activation1(output)
        output = self.fc2(output)
        output = self.activation2(output)
        output = self.fc3(output)

        return output.squeeze(-1)
    