import torch.nn as nn

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel, negative_slope):
        super(DiscriminatorBlock, self).__init__()

        layers = [
            nn.Linear(in_channel, out_channel),
            nn.LeakyReLU(negative_slope, inplace=True)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self.layers(input_tensor)

class Discriminator(nn.Module):
    def __init__(self, hidden_channels, negative_slope, out_size=1, sigmoid=True):
        super(Discriminator, self).__init__()

        layers = [
            DiscriminatorBlock(hidden_channels[i], hidden_channels[i+1], negative_slope)
            for i in range(len(hidden_channels)-1)
        ]

        self.layers = nn.ModuleList(layers)
        self.out_fc = nn.Linear(hidden_channels[-1], out_size)
        self.sigmoid = sigmoid

    def forward(self, input_tensor):

        intermediate = input_tensor.flatten(1, -1)
        for layer in self.layers:
            intermediate = layer.forward(intermediate)

        if self.sigmoid:
            output = nn.functional.sigmoid(self.out_fc.forward(intermediate))
        else:
            output = self.out_fc.forward(intermediate)
        return output