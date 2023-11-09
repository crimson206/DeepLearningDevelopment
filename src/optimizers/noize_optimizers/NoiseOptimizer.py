from torch.optim import Optimizer
import torch

class NoiseOptimizer(Optimizer):
    def __init__(self, optimizer, noise_stddev=0.01, noise_decay=0.995):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("optimizer should be an instance of torch.optim.Optimizer")
        self.optimizer = optimizer
        self.noise_stddev = noise_stddev
        self.noise_decay = noise_decay
        self.param_groups = self.optimizer.param_groups
    
    def introduce_noise(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                noise = torch.randn_like(p.grad.data) * self.noise_stddev
                p.grad.data.add_(noise)

    def apply_noise_decay(self):
        self.noise_stddev *= self.noise_decay

    def step(self, closure=None):
        self.introduce_noise()
        return self.optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def __getattr__(self, name):
        return getattr(self.optimizer, name)