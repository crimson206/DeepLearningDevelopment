import torch

def softmax_tau(x, tau=1.0):
    """Compute softmax values for each set of scores in x using PyTorch tensors."""
    e_x = torch.exp((x - torch.max(x)) / tau)
    return e_x / e_x.sum(axis=0)
