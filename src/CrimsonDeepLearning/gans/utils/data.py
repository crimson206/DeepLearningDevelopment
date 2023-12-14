from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

def get_loader(data_dir, log_resolution, batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize((2 ** log_resolution, 2 ** log_resolution)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return loader