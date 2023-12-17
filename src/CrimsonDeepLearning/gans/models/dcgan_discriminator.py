import torch
import torch.nn as nn

class DCGanDiscriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DCGanDiscriminatorBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
    
    def forward(self, input_tensor):
        return self.main(input_tensor)
    
class DCGanDiscriminator(nn.Module):
    def __init__(self, in_channel, channel_sizes, activation=nn.LeakyReLU(0.2, True)):
        super(DCGanDiscriminator, self).__init__()

        self.blocks = nn.ModuleList([DCGanDiscriminatorBlock(in_channel, channel_sizes[0])])
        self.blocks.extend([DCGanDiscriminatorBlock(channel_sizes[i], channel_sizes[i+1]) for i in range(len(channel_sizes)-1)])
        self.activation = activation

        self.out_conv = nn.Conv2d(channel_sizes[-1], 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        intermediate = input_tensor

        for block in self.blocks:
            intermediate = self.activation(block(intermediate))

        output = self.sigmoid(self.out_conv(intermediate)).flatten(start_dim=1)
        return output
