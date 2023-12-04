import torch
from torch.utils.data import Dataset

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