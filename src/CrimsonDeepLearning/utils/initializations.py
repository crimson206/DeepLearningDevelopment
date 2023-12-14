import torch
import torch.nn as nn

def inject_initialization(initialization_fn, model: nn.Module, target=nn.Conv2d):
    """
    Apply a specified initialization function to all layers of a given type in a model.

    Parameters:
    initialization_fn (callable): The initialization function to apply (e.g., nn.init.kaiming_normal_).
    model (nn.Module): The model to initialize.
    target (type): The type of layer to apply the initialization to. Default is nn.Conv2d.
                   Can be set to other types like nn.ConvTranspose2d, nn.Linear, etc.
    """
    for m in model.modules():
        if isinstance(m, target):
            initialization_fn(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

