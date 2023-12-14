
import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    """
    Seed everything to make results reproducible.
    
    Args:
    seed (int): The seed number.
    """
    random.seed(seed)       # Python's built-in random library
    np.random.seed(seed)    # Numpy library
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set PYTHONHASHSEED environment variable

    # For PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False