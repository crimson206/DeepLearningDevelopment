import torch
import torch.nn as nn

def _expand_tensor(tensor, target_shape):
    # Ensure that the first three dimensions of the target shape match the tensor
    if tensor.shape != target_shape[:len(tensor.shape)]:
        raise ValueError("The first three dimensions of the target shape must match the tensor shape.")

    # Creating a view with singleton dimensions added
    expanded_shape = tensor.shape + (1,) * (len(target_shape) - len(tensor.shape))
    tensor = tensor.view(expanded_shape)

    # Expanding the tensor to the target shape
    return tensor.expand(target_shape)


class DomainEmbeddingBlock(nn.Module):
    """
    Shape Summary:
    Input conditions shape:
    n_domain = len(domain_sizes)
    (n_batch, 1)

    Output tensor shape:
    (n_batch, n_channel)
    """
    def __init__(self, domain_sizes, d_emb, out_dim_per_domain):
        super(DomainEmbeddingBlock, self).__init__()
        self.embedders = nn.ModuleList([
            nn.Embedding(num_embeddings=domain_size, embedding_dim=d_emb) 
            for domain_size in domain_sizes
        ])

        self.domain_fc = nn.Linear(d_emb, out_dim_per_domain)

    def forward(self, domains, image_shape):
        """
        Shape Summary:
        Input conditions shape:
            n_domain = len(domain_sizes)
            (n_batch, n_domain)

        Output tensor shape:
            (n_batch, n_domain*d_emb, height, width)
        """
        n_batch = domains.shape[0]
        domains = torch.stack([embedder(domains[:, i]) for i, embedder in enumerate(self.embedders)], dim=1)
        converted_domains = self.domain_fc.forward(domains).view(n_batch, -1)

        n_batch, d_domain = converted_domains.shape
        height, width = image_shape
        output_shape = (n_batch, d_domain,height, width)

        expanded_domains = _expand_tensor(converted_domains, output_shape)

        return expanded_domains