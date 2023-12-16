import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random


class DictionaryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        first_key = next(iter(self.data))
        return len(self.data[first_key])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}

def dict_to_device(data, device, dtype=torch.float16):
    new_data = {}
    for key, tensor in data.items():
        tensor = tensor.to(device)
        if dtype and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        new_data[key] = tensor
    return new_data


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]




class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class RandomRotationTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)  # Choose a random angle
        return transforms.functional.rotate(x, angle)