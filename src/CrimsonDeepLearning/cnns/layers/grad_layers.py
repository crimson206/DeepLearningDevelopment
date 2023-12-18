import torch.nn as nn
import torch.nn.functional as F

class Conv2dGradFix(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.grad_weight = None
        self._input_tensor = None
        self.register_full_backward_hook(custom_backward_hook)

    def forward(self, input_tensor):
        self._input_tensor = input_tensor
        return F.conv2d(input_tensor, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def custom_backward_hook(module, _, grad_output):
    input_tensor = module._input_tensor
    weight = module.weight

    # Calculate grad_weight
    grad_weight = F.grad.conv2d_weight(input_tensor, weight.size(), grad_output[0], stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
    grad_weight.retain_grad()
    module.grad_weight = grad_weight
