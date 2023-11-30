from torch.utils.data import Dataset

class DictionaryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        first_key = next(iter(self.data))
        return len(self.data[first_key])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}