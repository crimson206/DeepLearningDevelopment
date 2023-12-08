import torch
import torch.nn as nn

class CategoryEmbeddingBlock(nn.Module):
    """
    Shape Summary:
    Input conditions shape:
    n_domain = len(domain_sizes)
    (n_batch, n_domain)

    Output tensor shape:
    (n_batch, n_domain, d_emb)
    """
    def __init__(self, domain_sizes, d_emb):
        super(CategoryEmbeddingBlock, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=domain_size, embedding_dim=d_emb) 
            for domain_size in domain_sizes
        ])
        self._d_emb = d_emb

    def forward(self, conditions):
        """
        Shape Summary:
        Input conditions shape:
        n_domain = len(domain_sizes)
        (n_batch, n_domain)

        Output tensor shape:
        (n_batch, n_domain, d_emb)
        """
        conditions = torch.stack([embedding(conditions[:, i]) for i, embedding in enumerate(self.embeddings)], dim=1)
        conditions = conditions.view(conditions.shape[0], -1, self._d_emb)
        return conditions
