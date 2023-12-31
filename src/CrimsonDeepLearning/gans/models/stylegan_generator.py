import torch
import torch.nn as nn
import torch.nn.functional as F

from CrimsonDeepLearning.gans.blocks.styleblocks import StyleBlock, To_RGB

class GeneratorBlock(nn.Module):
    def __init__(self, n_w_latent, input_channel, output_channel, hidden_channels=None, style_apply_mechanism="modulation"):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [output_channel]

        channels = [input_channel] + hidden_channels + [output_channel]

        self.style_blocks:list[StyleBlock] = nn.ModuleList(
            [
                StyleBlock(
                    n_w_latent=n_w_latent,
                    input_channel=channels[i],
                    output_channel=channels[i+1],
                    kernel_size=3,
                    style_apply_mechanism=style_apply_mechanism,
                )
                for i in range(len(channels)-1)
            ]
        )

        self.to_rgb = To_RGB(n_w_latent=n_w_latent, input_channel=output_channel)

    def forward(self, feature_map, w_latent):
        """
        Shape Summary:
            feature_map.shape: (n_batch, input_channels, height, width).
            w_latent.shape: (n_batch, n_w_latent).

            out_feature_map.shape:(n_batch, output_channels, height, width).
            out_rgb_image:(n_batch, output_channels, height, width).
        """
        for style_block in self.style_blocks:
            feature_map = style_block.forward(feature_map=feature_map, w_latent=w_latent, add_noise=True)
        rgb_image = self.to_rgb(feature_map, w_latent)
        return feature_map, rgb_image

class Generator(nn.Module):
    def __init__(self, log_resolution, n_w_latent, start_feature = 32, max_feature = 256, style_apply_mechanism="modulation"):
        super().__init__()

        self.features = [min(max_feature, start_feature * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(self.features) - 1

        self.initial_constant = nn.Parameter(torch.randn((1, self.features[0], 4, 4)))

        self.style_block = StyleBlock(
            n_w_latent=n_w_latent,
            input_channel=self.features[0],
            output_channel=self.features[0],
            kernel_size=3,
            style_apply_mechanism=style_apply_mechanism
        )
        
        self.to_rgb = To_RGB(n_w_latent=n_w_latent, input_channel=self.features[0])
        self.blocks:list[GeneratorBlock] = nn.ModuleList([GeneratorBlock(n_w_latent=n_w_latent, input_channel=self.features[i], output_channel=self.features[i+1], style_apply_mechanism=style_apply_mechanism) for i in range(self.n_blocks)])

    def forward(self, w_latent1, w_latent2, return_whole_w_letent=False):
        initial_w_latent = w_latent1[None,:,:].repeat(2, 1, 1)
        whole_w_latent = w_latent2[None,:,:].repeat(self.n_blocks, 1, 1)
        batch_size = whole_w_latent[0].shape[0]
        feature_map = self.initial_constant.expand(batch_size, -1, -1, -1)
        feature_map = self.style_block(feature_map, initial_w_latent[0, ...])
        rgb = self.to_rgb(feature_map, initial_w_latent[1, ...])

        for i, generator_block in enumerate(self.blocks):
            feature_map = F.interpolate(feature_map, scale_factor=2, mode="bilinear")
            feature_map, rgb_new = generator_block.forward(feature_map, whole_w_latent[i, ...])
            pre_rgb_upsampled = F.interpolate(rgb, scale_factor=2, mode="bilinear")
            rgb = pre_rgb_upsampled + rgb_new

        if return_whole_w_letent:
            return torch.tanh(rgb), whole_w_latent
        else:
            return torch.tanh(rgb)