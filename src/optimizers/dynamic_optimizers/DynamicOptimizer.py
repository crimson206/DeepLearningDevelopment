import torch

class DynamicOptimizer:
    def __init__(self, model_parameters, optim_configs, switch_frequency, lr_decay):
        self.model_parameters = list(model_parameters)
        self.optim_configs = optim_configs
        self.switch_frequency = switch_frequency
        self.lr_decay = lr_decay
        self.current_epoch = 0
        self.current_optimizer_index = 0
        self.current_optimizer = self._create_optimizer(self.current_optimizer_index)

    def _create_optimizer(self, idx):
        optim_class, kwargs = self.optim_configs[idx]
        return optim_class(self.model_parameters, **kwargs)

    def step(self):
        self.current_optimizer.step()

    def zero_grad(self):
        self.current_optimizer.zero_grad()
    
    def epoch_step(self):
        self.current_epoch += 1
        for param_group in self.current_optimizer.param_groups:
            param_group['lr'] *= self.lr_decay

        if self.current_epoch % self.switch_frequency == 0:
            lr = self.current_optimizer.param_groups[0]['lr']
            self.current_optimizer_index = (self.current_optimizer_index + 1) % len(self.optim_configs)
            self.current_optimizer = self._create_optimizer(self.current_optimizer_index)

            for param_group in self.current_optimizer.param_groups:
                param_group['lr'] = lr
                
    def introduce_noise(self):
        for group in self.current_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                noise = torch.randn_like(p.grad.data) * self.noise_stddev
                p.grad.data.add_(noise)
                
    def __getattr__(self, name):
        return getattr(self.current_optimizer, name)