import torch
import torch.nn as nn

from typing import Tuple
from CrimsonDeepLearning.cnns.blocks.coder_builders import EncoderSetupHolder

class VAEEncoder(nn.Module):
    def __init__(self, input_channel, n_latent, size="small", mechanism="conv", activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()

        encoder_setup_holder = EncoderSetupHolder(input_channel=input_channel)

        self.encoder = encoder_setup_holder.generate_encoder(size=size, mechanism=mechanism, activation=activation)

        if size=="small":
            encoder_output_channel = encoder_setup_holder.small_hidden_channels_list[-1][-1]
        elif size=="medium":
            encoder_output_channel = encoder_setup_holder.medium_hidden_channels_list[-1][-1]

        print(encoder_output_channel)

        self.pooler = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc_mu: nn.Linear = nn.Linear(encoder_output_channel, n_latent)
        self.fc_logvar: nn.Linear = nn.Linear(encoder_output_channel, n_latent)

    def encode(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder.forward(input_tensor=input_tensor)
        pooled = self.pooler.forward(input=encoded).view(encoded.shape[0], encoded.shape[1])

        mu: torch.Tensor = self.fc_mu(pooled)
        logvar: torch.Tensor = self.fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std: torch.Tensor = torch.exp(0.5 * logvar)
        eps: torch.Tensor = torch.randn_like(std)
        return mu + eps * std