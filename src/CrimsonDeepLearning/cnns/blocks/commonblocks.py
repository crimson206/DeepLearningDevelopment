import torch
import torch.nn as nn
from typing import List
from CrimsonDeepLearning.cnns.blocks.sizepreservingblocks import SizePreservingBlocks

class CommonBlock(nn.Module):
    def __init__(self, output_channel:int, module:nn.Module, activation=nn.ReLU(), dropout_rate: float = 0.1) -> None:
        """
        CommonBlock is a utility class to create a neural network module that combines a given module with
        instance normalization, an activation function, and optional dropout. This class simplifies the construction
        of commonly used sequences in neural networks.

        Args:
        output_channel (int): Number of output channels for the instance normalization.
        module (nn.Module): The main module (e.g., a convolutional layer).
        activation (callable, optional): The activation function to use. Defaults to nn.ReLU().
        dropout_rate (float, optional): Dropout rate between 0 and 1. Default is 0, which means no dropout.

        The forward pass applies the sequence of module, normalization, activation, and dropout (if dropout_rate > 0) to the input.
        """
        super(CommonBlock, self).__init__()
        layers: List[nn.Module] = []
        layers.append(module)
        layers.append(nn.InstanceNorm2d(output_channel))
        layers.append(activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Shape Summary:
        input_tensor.shape: (n_batch, input_channels, height, width).

        output.shape: (n_batch, output_channels, new_height, new_width).
        It depends on the module input
        """
        return self.layers(input_tensor)
    
class CommoncoderLayer(nn.Module):
    """
    The CommoncoderLayer includes 
    - SizePreservingBlocks : Size of image is preserved. Mainly for feature extraction.
    - CommonBlock : Custom Block. Mainly, image size modification is expected. 

    Args:
    hidden_channels: Number of hidden channels for the SizePreservingBlocks.
    output_channel (int): Number of output channels for the final operation in CommonBlock.
    module (nn.Module): The main module for the CommonBlock, typically a convolutional layer.
    dropout_rate (float): Dropout rate for both SizePreservingBlocks and CommonBlock.
    activation (callable): Activation function for both SizePreservingBlocks and CommonBlock.

    The forward pass applies the sequence of intermediate operations followed by the final CommonBlock operation. 
    It can optionally return intermediate results for further processing or analysis.
    """
    def __init__(self, hidden_channels:List[int], output_channel:int, module:nn.Module, activation:nn.Module=nn.ReLU(), dropout_rate:float=0.1) -> None:
        super(CommoncoderLayer, self).__init__()
        self.flatsample = SizePreservingBlocks(hidden_channels=hidden_channels, activation=activation, dropout_rate=dropout_rate, use_residual=True)
        self.block = CommonBlock(output_channel=output_channel, module=module, activation=activation, dropout_rate=dropout_rate)

    def forward(self, input_tensor: torch.Tensor, return_intermediate=False) -> torch.Tensor:
        """
        Shape Summary:
        input_tensor.shape: (n_batch, input_channels, height, width).

        if return_intermediate is False
            return output
        elif return_intermediate is True
            return output, intermediate

        output.shape: (n_batch, output_channel, new_height, new_width).
        intermediate.shape: (n_batch, input_channel, height, width).
        """
        intermediate = self.flatsample.forward(input_tensor)
        output = self.block.forward(intermediate)
        if return_intermediate:
            return output, intermediate
        else:
            return output

class Commoncoder(nn.Module):
    """
    The Commoncoder class is designed to encapsulate multiple CommoncoderLayer instances.

    Args:
    coder_layers (List[CommoncoderLayer]): A list of CommoncoderLayer instances. Each layer in the list is 
    applied in sequence to process the input tensor.

    The Commoncoder class simplifies the management and sequential application of multiple CommoncoderLayer 
    instances. Its forward pass method applies each layer in the coder_layers sequence to the input tensor, 
    optionally collecting and returning intermediate outputs from each layer.

    Methods:
    forward(input_tensor: torch.Tensor, return_intermediate: bool = False):
        Returns:
        If return_intermediate is False:
            return output
        If return_intermediate is True:
            return output, intermediates
    """
    def __init__(self, coder_layers) -> None:
        super(Commoncoder, self).__init__()
        self.coder_layers = nn.ModuleList(coder_layers)

    def forward(self, input_tensor: torch.Tensor, return_intermediate: bool = False):
        """
        Processes the input tensor through each layer in the coder_layers sequence.

        Args:
        input_tensor (torch.Tensor): The input tensor with shape (n_batch, input_channels, height, width).
        return_intermediate (bool, optional): Flag to determine whether to return intermediate outputs. 
        Default is False.

        Returns:
        If return_intermediate is False:
            return output
        If return_intermediate is True:
            return output, intermediates

        output.shape: (n_batch, output_channels, final_height, final_width).
        intermediates: [(n_batch, output_channels, various_height, various_width)] * len(coder_layers)
        """
        intermediates = []
        output = input_tensor

        for layer in self.coder_layers:
            if return_intermediate:
                output, intermediate = layer(output, return_intermediate=True)
                intermediates.append(intermediate)
            else:
                output = layer(output)

        if return_intermediate:
            return output, intermediates
        else:
            return output