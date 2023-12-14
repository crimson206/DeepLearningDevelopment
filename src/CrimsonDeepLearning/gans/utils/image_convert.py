import numpy as np
import copy
import torch

def expand_channel(mnist_images):
    image_rgb = np.repeat(copy.deepcopy(mnist_images), 3, axis=1)
    return image_rgb

def give_coler_to_mnist(image_rgb, idx_to_remove):
    image_rgb = copy.deepcopy(image_rgb)
    black_mask = image_rgb == -1
    # Convert to numpy and create an RGB version
    full_indexes = [0, 1, 2]

    remained_indexes = [idx for idx in full_indexes if idx != idx_to_remove]

    image_rgb[:, remained_indexes, :, :] = 0
    image_rgb[black_mask] = -1
    return torch.tensor(image_rgb)

def add_striped_pattern_to_mnist(image_rgb, angle):
    image_rgb = copy.deepcopy(image_rgb)
    # Convert to numpy and repeat along the color channel to make it RGB
    black_mask = image_rgb == -1
    # Add stripes
    for i in range(0, image_rgb.shape[2], 4):
        if angle == 0:
            image_rgb[:, :,i:i+2,:] = 0.4 * image_rgb[:, :,i:i+2,:] # Gray stripes
        else:
            image_rgb[:, :, :, i:i+2] = 0.4 * image_rgb[:, :, :, i:i+2] # Gray stripes

    image_rgb[black_mask] = -1
    return torch.tensor(image_rgb)