import torch
import torch.nn as nn
import math
from CrimsonDeepLearning.gans.models.critic import Critic 

class PathLengthPenalty(nn.Module):
    def __init__(self, beta:float=0.99):
        """
        Parameters:
        - beta: The beta coefficient for the exponential moving average of gradient norms.
        """
        super().__init__()
        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, feature_map: torch.Tensor, whole_w_latent: torch.Tensor) -> torch.Tensor:
        """
        Shape Summary:
            whole_w_latent.shape: (n_resolution, n_batch, n_w_latent).
            feature_map.shape: (n_batch, n_channel, height, width).

            loss.shape: single value
        """
        device = feature_map.device
        image_size = feature_map.shape[2] * feature_map.shape[3]
        y = torch.randn(feature_map.shape, device=device)

        output = (feature_map * y).sum() / math.sqrt(image_size)

        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=whole_w_latent,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:

            a = self.exp_sum_a / (1 - self.beta ** self.steps)

            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss
    
def gradient_penalty(critic: Critic, real_image: torch.Tensor, fake_image: torch.Tensor) -> torch.Tensor:
    n_batch, n_channel, height, width = real_image.shape
    device = fake_image.device

    beta = torch.rand((n_batch, 1, 1, 1)).repeat(1, n_channel, height, width).to(device)
    interpolated_images = real_image * beta + fake_image.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def wgangp_critic_loss(critic: Critic, real_image: torch.Tensor, fake_image: torch.Tensor, lambda_gp: float=10.0, critic_output_reg_coeff: float = 0.001, return_metric=False) -> torch.Tensor:
    """
    Calculate the critic loss for WGAN-GP.

    Parameters:
    - critic: The critic (or discriminator) model.
    - real_image.shape: (n_batch, n_channel, height, width).
    - fake_image.shape: (n_batch, n_channel, height, width).
    - lambda_gp: The coefficient for the gradient penalty.
    - critic_output_reg_coeff: Coefficient for the critic output regularization.

    Returns:
    - loss.shape: single value
    """
    critic_fake = critic(fake_image.detach())

    if real_image is None:
        return torch.mean(critic_fake)

    critic_real = critic(real_image)

    gp = gradient_penalty(critic, real_image, fake_image)

    wasserstein_loss = torch.mean(critic_real) - torch.mean(critic_fake)

    output_reg_loss = critic_output_reg_coeff * torch.mean(critic_real ** 2)

    loss = - wasserstein_loss + lambda_gp * gp + output_reg_loss

    metric = {
        "wasserstein_loss":wasserstein_loss.item(),
        "gradient_penalty":gp.item(),
        "output_reg_loss":output_reg_loss.item(),
    }

    if return_metric:
        return loss, metric
    else:
        return loss