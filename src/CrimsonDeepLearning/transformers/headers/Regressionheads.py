import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class RegressionHead(nn.Module):
    """
    A regression head neural network module suitable for attaching to the end of any feature extractor 
    to perform regression tasks.

    The module consists of a sequence of fully connected layers that regress to a single value or a 
    multi-dimensional output.

    Parameters
    ----------
    input_size : int
        The size of each input sample.
    hidden_sizes : Optional[List[int]], optional
        A list specifying the number of neurons in each hidden layer. If not specified, two hidden layers 
        are created with sizes `input_size // 2` and `input_size // 4`, by default.
    output_size : int, optional
        The size of each output sample. For regression tasks, this is typically 1, by default.

    Attributes
    ----------
    layers : nn.ModuleList
        A ModuleList containing the sequence of linear layers in the network.

    Methods
    -------
    forward(x, pool=False, squeeze=True):
        Defines the computation performed at every call. Optionally applies mean pooling and squeezes 
        the output dimension if `output_size` is 1.

    Examples
    --------
    >>> input_tensor = torch.randn(10, 5, 32)
    >>> reg_head = RegressionHead(input_size=32)
    >>> output_tensor = reg_head(input_tensor, pool=True)
    >>> output_tensor.shape
    torch.Size([10, 1])
    """

    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: Optional[List[int]] = None, 
        output_size: int = 1
    ):
        super(RegressionHead, self).__init__()
        # Initialize hidden sizes if not provided
        if hidden_sizes is None:
            hidden_sizes = [input_size // 2, input_size // 4]

        # Calculate the sizes for each layer
        sizes = [input_size] + hidden_sizes + [output_size]
        # Create a list of linear layers based on the sizes
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
    
    def forward(self, x: torch.Tensor, pool: bool = False, squeeze: bool = True) -> torch.Tensor:
        """
        Forward pass of the regression head. If pooling is enabled, a mean pooling operation is 
        applied to the input tensor. If the `output_size` is 1 and `squeeze` is True, the output 
        tensor's last dimension is squeezed.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape (n_batch, n_seq, d_emb).
        pool : bool, optional
            If True, mean pooling is applied along the sequence dimension (dim=1), by default False.
        squeeze : bool, optional
            If True and the output size is 1, the output tensor's last dimension is squeezed, 
            by default True.

        Returns
        -------
        torch.Tensor
            The output tensor. Possible shapes are:
            - If `pool` is True, (n_batch, output_size) or, if `squeeze` is also True, (n_batch).
            - If `pool` is False, (n_batch, n_seq, output_size) or, if `squeeze` is also True and
              `output_size` is 1, (n_batch, n_seq).

        Examples
        --------
        >>> input_tensor = torch.randn(10, 5, 32)
        >>> reg_head = RegressionHead(input_size=32)
        >>> output_tensor = reg_head(input_tensor, pool=True)
        >>> output_tensor.shape
        torch.Size([10])
        """
        # Apply mean pooling if enabled
        if pool:
            x = x.mean(dim=1)

        # Pass input through each layer and apply ReLU activation for non-final layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)

        # Squeeze the last dimension if required
        if squeeze:
            x = x.squeeze(-1)

        return x
