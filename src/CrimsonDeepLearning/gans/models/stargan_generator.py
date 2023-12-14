
from CrimsonDeepLearning.cnns.blocks.coder_builders import EncoderSetupHolder, DecoderSetupHolder, CoderBuilder
from CrimsonDeepLearning.cnns.blocks.sizepreservingblocks import SizePreservingBlocks
from CrimsonDeepLearning.gans.blocks.domainblocks import DomainEmbeddingBlock

import torch
import torch.nn as nn

import math
import numpy as np

class StarGanGenerator(nn.Module):
    def __init__(self, in_channel, out_channel, domain_sizes, out_dim_per_domain, down_hidden_channels_list, up_hidden_channels_list, final_hidden_channels):
        super(StarGanGenerator, self).__init__()
        encoderr_setup_holder = EncoderSetupHolder(input_channel=in_channel)
        decoder_setup_holder = DecoderSetupHolder(output_channel=out_channel)

        conv_down_modules = encoderr_setup_holder.generate_conv_down_modules(
            down_hidden_channels_list,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        conv_up_moduels = decoder_setup_holder.generate_conv_up_modules(
            up_hidden_channels_list,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        builder = CoderBuilder()

        self.encoder = builder.build_coder(
            hidden_channels_list=down_hidden_channels_list,
            modules=conv_down_modules, 
            activation=nn.LeakyReLU(negative_slope=0.2),
        )

        self.decoder = builder.build_coder(
            hidden_channels_list=up_hidden_channels_list,
            modules=conv_up_moduels,
            activation=nn.LeakyReLU(negative_slope=0.2),
        )

        self.domain_embedder_blocks = [
            DomainEmbeddingBlock(
                domain_sizes=domain_sizes,
                d_emb=256,
                out_dim_per_domain=out_dim_per_domain
            )
            for _ in range(len(down_hidden_channels_list)+1)
        ]

        self.final_block = SizePreservingBlocks(
            hidden_channels=final_hidden_channels,
            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            use_residual=True
        )

    def forward(self, input_tensor, domains):
        _, _, height, width = input_tensor.shape

        output = input_tensor
        residuals = []

        for encoder_layer, domain_block in zip(self.encoder.coder_layers, self.domain_embedder_blocks):
            domain_embedding = domain_block.forward(domains, image_shape=output.shape[-2:])

            output = torch.cat([output, domain_embedding], dim=1)
            output, residual = encoder_layer.forward(output, return_intermediate=True, log2_adjust=True)
            residuals.append(residual)

        for i, decoder_layer in enumerate(self.decoder.coder_layers):
            if i == 0:
                addition = self.domain_embedder_blocks[-1].forward(
                    domains=domains, 
                    image_shape=output.shape[-2:]
                )
            else:
                addition = residuals[-i]

            output = torch.cat([output, addition], dim=1)
            output = decoder_layer(output)

        output = torch.cat([output, residuals[0]], dim=1)
        output = F.interpolate(output, size=(height, width), mode='bilinear', align_corners=False)
        output = self.final_block.forward(output)

        return output
    

def generate_recommended_hidden_channels(in_channel, out_channel, domain_dim, height, width, start_hidden_dim, end_hidden_dim):
    down_hidden_channels_list, up_hidden_channels_list = [], []

    log_resolution = math.ceil(np.min([np.log2(height), np.log2(width)]))

    feature_dims = []

    multi = np.log2(end_hidden_dim / start_hidden_dim)
    for i in range(log_resolution):
        base_log = multi / (log_resolution - 1) * i
        feature_dims.append(start_hidden_dim * 2 ** math.ceil(base_log))
    feature_dims

    for i in range(log_resolution-1):
        if i==0:
            down_hidden_channels_list.append([in_channel+domain_dim, feature_dims[i], feature_dims[i+1]])
        elif i+2!=log_resolution:
            down_hidden_channels_list.append([feature_dims[i]+domain_dim, feature_dims[i], feature_dims[i+1]])

    for i in range(log_resolution-1):
        if i == 0:
            up_hidden_channels_list.append([feature_dims[-i-2]+domain_dim, feature_dims[-i-2], feature_dims[-i-1]])
        elif i+2!=log_resolution:
            up_hidden_channels_list.append([feature_dims[-i-1]+feature_dims[-i-2], feature_dims[-i-1], feature_dims[-i-2]])

    final_hidden_channels = [feature_dims[1]+feature_dims[1], feature_dims[1], out_channel]

    return down_hidden_channels_list, up_hidden_channels_list, final_hidden_channels