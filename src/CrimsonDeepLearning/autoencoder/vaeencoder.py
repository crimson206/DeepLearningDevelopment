import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Tuple

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        layers = [nn.Conv2d(in_channels, hidden_channels[0], kernel_size=4, stride=2, padding=1)]
        layers += [nn.Conv2d(hidden_channels[i], hidden_channels[i+1], kernel_size=4, stride=2, padding=1)
                   for i in range(len(hidden_channels)-1)]
        self.conv_layers = nn.ModuleList(layers)
        self.fc_mu = nn.Conv2d(hidden_channels[-1], latent_dim, kernel_size=1, stride=1, padding=0)
        self.fc_logvar = nn.Conv2d(hidden_channels[-1], latent_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        for layer in self.conv_layers:
            x = F.relu(layer(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        hidden_channels = [latent_dim] + hidden_channels
        layers = [nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i+1], kernel_size=4, stride=2, padding=1)
                  for i in range(len(hidden_channels) -1)]
        layers.append(nn.ConvTranspose2d(hidden_channels[-1], out_channels, kernel_size=1, stride=1, padding=0))
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, z):
        for layer in self.conv_layers[:-1]:
            z = F.relu(layer(z))
        reconstruction = torch.sigmoid(self.conv_layers[-1](z))
        return reconstruction


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_channels, latent_dim, output_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(input_dim, hidden_channels, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_channels[::-1], output_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        print(mu.shape)
        print(z.shape)
        return self.decoder(z), mu, log_var


def kl_loss(mu, log_var):
    loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return loss