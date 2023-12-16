import torch
import torch.nn as nn


class DCGanGeneratorBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(DCGanGeneratorBlock, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d( in_channel, out_channel, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channel),
        )
    
    def forward(self, input_tensor):
        return self.main(input_tensor)

class DCGanGenerator(nn.Module):
    def __init__(self, n_latent, channel_sizes, out_channel, activation=nn.LeakyReLU(0.2, True)):
        super(DCGanGenerator, self).__init__()
        self.blocks = nn.ModuleList([DCGanGeneratorBlock(n_latent, channel_sizes[0], 4, 1, 0)])
        self.blocks.extend([DCGanGeneratorBlock(channel_sizes[i], channel_sizes[i+1], 4, 2, 1) for i in range(len(channel_sizes)-1)])
        self.output_conv = nn.ConvTranspose2d(channel_sizes[-1], out_channel, kernel_size=4, stride=2, padding=1, bias=False)

        self.activation = activation
        self.tanh = nn.Tanh()

    def forward(self, input_tensor):

        intermediate = input_tensor

        for block in self.blocks:
            intermediate = self.activation(block(intermediate))
        output = self.tanh(self.output_conv(intermediate))
        return output