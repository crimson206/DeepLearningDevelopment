import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class VAE(nn.Module):
    """
    A Variational Autoencoder (VAE) implemented in PyTorch. It encodes input data into 
    a latent space representation and then reparameterizes the encoding for stochasticity 
    in the latent space.

    Parameters
    ----------
    input_size : int
        The size of each input sample.
    hidden_size : List[int]
        A list specifying the number of neurons in each hidden layer of the encoder.
    latent_size : int
        The size of the latent space representation.

    Attributes
    ----------
    encoder_layers : nn.ModuleList
        A ModuleList containing the sequence of linear layers in the encoder.
    layer_norms : nn.ModuleList
        A ModuleList containing layer normalization modules corresponding to each encoder layer.
    fc_mu : nn.Linear
        A fully connected linear layer that outputs the mean (mu) vector of the latent space.
    fc_logvar : nn.Linear
        A fully connected linear layer that outputs the log-variance (logvar) vector of the latent space.

    Methods
    -------
    encode(input_tensor)
        Encodes the input tensor into the latent space representation, returning the mean and log-variance.
    reparameterize(mu, logvar)
        Reparameterizes the encoded mean and log-variance to introduce stochasticity into the latent space representation.

    Examples
    --------
    >>> vae = VAE(input_size=784, hidden_size=[512, 256], latent_size=20)
    >>> input_tensor = torch.randn(64, 512, 784)
    >>> mu, logvar = vae.encode(input_tensor)
    >>> z = vae.reparameterize(mu, logvar)
    """
    
    def __init__(self, input_size: int, hidden_size: List[int], latent_size: int):
        super(VAE, self).__init__()
        
        # Initialize the encoder layers and their corresponding layer normalizations
        self.encoder_layers: nn.ModuleList = nn.ModuleList()
        self.layer_norms: nn.ModuleList = nn.ModuleList()
        previous_dim: int = input_size
        for hidden_dim in hidden_size:
            self.encoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            previous_dim = hidden_dim
        
        # Initialize the fully connected layers for mean and log-variance of the latent space
        self.fc_mu: nn.Linear = nn.Linear(hidden_size[-1], latent_size)
        self.fc_logvar: nn.Linear = nn.Linear(hidden_size[-1], latent_size)

    def encode(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input tensor into latent space representation by passing it through a series 
        of encoder layers with ReLU activations and layer normalization.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor with shape (n_batch, n_seq, input_size).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the mean and log-variance tensors of the latent space representation, 
            both with shape (n_batch, n_seq, latent_size).
        """
        for layer, norm in zip(self.encoder_layers, self.layer_norms):
            input_tensor = F.relu(norm(layer(input_tensor)))
        mu: torch.Tensor = self.fc_mu(input_tensor)
        logvar: torch.Tensor = self.fc_logvar(input_tensor)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterizes the encoded input by combining the mean and log-variance with a random 
        noise tensor to introduce stochasticity, necessary for the VAE to learn a continuous latent 
        distribution.

        Parameters
        ----------
        mu : torch.Tensor
            The mean vector tensor from the encoder with shape (n_batch, n_seq, latent_size).
        logvar : torch.Tensor
            The log-variance vector tensor from the encoder with shape (n_batch, n_seq, latent_size).

        Returns
        -------
        torch.Tensor
            The reparameterized latent vector tensor with shape (n_batch, n_seq, latent_size).
        """
        std: torch.Tensor = torch.exp(0.5 * logvar)
        eps: torch.Tensor = torch.randn_like(std)
        return mu + eps * std

class VAEformer(nn.Module):
    def __init__(
        self,
        multi_embedding_transformer,
        regression_heads,
        vae_model,
    ):
        super(VAEformer, self).__init__()
        self.multi_embedding_transformer = multi_embedding_transformer
        self.base_transformer_config = multi_embedding_transformer.enc_transformer.transformer.config
        self.regression_heads = regression_heads
        self.vae :VAE= vae_model

    def forward(
        self,
        split_position_ids: List[torch.Tensor],
        split_categorical_ids: List[torch.Tensor]=None,
        sum_categorical_ids: torch.Tensor=None,
        continuous_feature: torch.Tensor=None,
        vae_feature: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
    ):

        n_seq = split_position_ids[0].shape[1]
        mu, logvar = self.vae.encode(vae_feature)
        skip_embedding = self.vae.reparameterize(mu, logvar)

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
    
        return outputs, mu, logvar
