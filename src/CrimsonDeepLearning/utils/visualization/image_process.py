import torch
import torch.nn as nn
import torch.nn.functional as F

class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(b, c, h, w)

class Upsample(nn.Module):
    def __init__(self, smooth=True):
        super().__init__()
        if smooth:
            self.smooth = Smooth()
        else:
            self.smooth = None

    def forward(self, x: torch.Tensor):
        if self.smooth is not None:
            x = self.smooth(x)
        # Double the spatial dimensions
        return F.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2), mode='bilinear', align_corners=False)

def merge_list_images(images, n_per_image, upsample):

    largest_size = 0
    for image in images:
        if image.shape[-1] > largest_size:
            largest_size = image.shape[-1]

    new_images = []
    for image in images:
        for _ in range(10):
            if image.shape[-1] == largest_size:
                break
            image = upsample(image)
        new_images.append(image[:n_per_image,...])

    return torch.cat(new_images)
