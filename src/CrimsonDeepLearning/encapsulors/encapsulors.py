import torch.nn as nn

class ActivationCapsule(nn.Module):
    def __init__(self, model, activation):
        super().__init__()
        self.model = model
        self.activation = activation

    def forward(self, *input_args):
        output = self.model(*input_args)
        return self.activation(output)