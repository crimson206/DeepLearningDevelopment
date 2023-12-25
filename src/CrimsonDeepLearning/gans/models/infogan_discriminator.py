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
    def __init__(self, hidden_channels, out_size=1, sigmoid=True):
        super(Discriminator, self).__init__()

        layers = [
            DiscriminatorBlock(hidden_channels[i], hidden_channels[i+1], negative_slope=0.2)
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
        return output, intermediate

class QHead(nn.Module):
    def __init__(self, hidden_channels, c_sizes):
        super(QHead, self).__init__()

        layers = [
            DiscriminatorBlock(hidden_channels[i], hidden_channels[i+1], negative_slope=0.2)
            for i in range(len(hidden_channels)-1)
        ]

        self.layers = nn.ModuleList(layers)

        self.out_heads = nn.ModuleList([nn.Linear(hidden_channels[-1], c_size) for c_size in c_sizes])

    def forward(self, feature):
        
        feature = self.layers(feature)

        outputs = []

        for out_head in self.out_heads:
            outputs.append(out_head(feature))
        
        return outputs