import torch
import torch.nn as nn
from CrimsonDeepLearning.cnns.blocks.commonblocks import CommoncoderLayer, Commoncoder

class CoderBuilder(nn.Module):
    def __init__(self):
        super(CoderBuilder, self).__init__()
        pass

    def build_coder(self, hidden_channels_list, modules, activation=nn.ReLU()):
        coder_layers = []
        for hidden_channels, up_module in zip(hidden_channels_list, modules):
            coder_layers.append(
                CommoncoderLayer(
                    hidden_channels=hidden_channels,
                    output_channel=hidden_channels[-1],
                    module=up_module,
                    activation=activation,
                    )
                )

        return Commoncoder(coder_layers=coder_layers)

class EncoderSetupHolder():
    def __init__(self, input_channel):
        self.small_hidden_channels_list = [
            [input_channel, 64, 128],
            [128, 128, 256] 
        ]

        self.medium_hidden_channels_list = [
            [input_channel, 64, 128],
            [128, 128, 256],
            [256, 256, 512],
            [512, 512, 512],
        ]

    def generate_conv_down_modules(self, hidden_channels_list, kernel_size=3, stride=2, padding=1):
        up_modules = []
        for hidden_channels in hidden_channels_list:
            up_modules.append(
                nn.Conv2d(
                    in_channels=hidden_channels[-1],
                    out_channels=hidden_channels[-1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
        return up_modules

    def generate_pooler_donw_modules(self, hidden_channels_list, mechanism="max"):
        down_modules = []
        constructor = nn.MaxPool2d if mechanism=="max" else nn.AvgPool2d
        for _ in hidden_channels_list:
            down_modules.append(
                constructor(kernel_size=2, stride=2)
            )
        return down_modules

def generate_encoder(input_channel, size="small", mechanism="conv", activation=nn.ReLU()):

    encoder_setup_holder = EncoderSetupHolder(input_channel=input_channel)
    hidden_channels_list = encoder_setup_holder.small_hidden_channels_list if size=="small" else encoder_setup_holder.medium_hidden_channels_list

    if mechanism=="conv":
        modules = encoder_setup_holder.generate_conv_down_modules(
            hidden_channels_list=hidden_channels_list,
            kernel_size=3,
            stride=2,
            padding=1
        )
    elif mechanism=="max_pool":
        modules = encoder_setup_holder.generate_pooler_donw_modules(
            hidden_channels_list=hidden_channels_list,
            mechanism="max",
        )
    elif mechanism=="avg_pool":
        modules = encoder_setup_holder.generate_pooler_donw_modules(
            hidden_channels_list=hidden_channels_list,
            mechanism="avg", 
        )
    
    coder_builder = CoderBuilder()

    encoder = coder_builder.build_coder(
        hidden_channels_list=hidden_channels_list,
        modules=modules, 
        activation=activation
    )

    return encoder

class DecoderSetupHolder():
    def __init__(self, output_channel):
        self.small_hidden_channels_list = [
            [256, 256, 128],
            [128, 64, output_channel] 
        ]
        self.medium_hidden_channels_list = [
            [512, 512, 512],
            [512, 512, 256],
            [256, 256, 128],
            [128, 64, output_channel] 
        ]

    def generate_conv_up_modules(self, hidden_channels_list, kernel_size=3, stride=2, padding=1):
        up_modules = []
        for hidden_channels in hidden_channels_list:
            up_modules.append(
                nn.ConvTranspose2d(
                    in_channels=hidden_channels[-1],
                    out_channels=hidden_channels[-1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=1,
                )
            )
        return up_modules

    def generate_pooler_up_modules(self, hidden_channels_list):
        up_modules = []
        for _ in hidden_channels_list:
            up_modules.append(
                nn.UpsamplingNearest2d(scale_factor=2)
            )
        return up_modules


def generate_decoder(output_channel, size="small", mechanism="conv", activation=nn.ReLU()):

    decoder_setup_holder = DecoderSetupHolder(output_channel=output_channel)
    hidden_channels_list = decoder_setup_holder.small_hidden_channels_list if size=="small" else decoder_setup_holder.medium_hidden_channels_list

    if mechanism=="conv":
        modules = decoder_setup_holder.generate_conv_up_modules(
            hidden_channels_list=hidden_channels_list,
            kernel_size=3,
            stride=2,
            padding=1,
        )
    elif mechanism=="pool":
        modules = decoder_setup_holder.generate_pooler_up_modules(
            hidden_channels_list=hidden_channels_list,
        )
    
    coder_builder = CoderBuilder()

    encoder = coder_builder.build_coder(
        hidden_channels_list=hidden_channels_list,
        modules=modules, 
        activation=activation
    )

    return encoder
