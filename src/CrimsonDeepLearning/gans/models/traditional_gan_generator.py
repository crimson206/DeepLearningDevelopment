import torch.nn as nn

class GeneratorBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm=True, negative_slope=0.2):
        super(GeneratorBlock, self).__init__()

        layers = []
        layers.append(nn.Linear(in_channel, out_channel))
        if norm:
            layers.append(nn.BatchNorm1d(out_channel))
        layers.append(nn.LeakyReLU(negative_slope, inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self.layers(input_tensor)

class Generator(nn.Module):
    def __init__(self, image_shape, hidden_channels, negative_slope, tanh=True):
        super(Generator, self).__init__()

        layers = [
            GeneratorBlock(hidden_channels[i], hidden_channels[i+1], norm=False if i==0 else True, negative_slope=negative_slope)
            for i in range(len(hidden_channels)-1)
        ]

        self.layers = nn.Sequential(*layers)
        self.out_fc = nn.Linear(hidden_channels[-1], np.prod(image_shape))
        self.tanh = tanh
        self._image_shape = image_shape

    def forward(self, z_latent):
        output_tensor = z_latent

        output_tensor = self.layers(output_tensor)
        output_tensor = self.out_fc(output_tensor)
        output_tensor = output_tensor.view(output_tensor.size(0), *self._image_shape)
        if self.tanh:
            output_tensor = nn.functional.tanh(output_tensor)
        return output_tensor

