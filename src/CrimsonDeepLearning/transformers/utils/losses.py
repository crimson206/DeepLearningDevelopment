import torch
import numpy as np

def combined_loss(criterions, preds, labels, loss_weights):
    losses_item = []
    loss_sum = None
    for criterion, pred, label, weight in zip(criterions, preds, labels, loss_weights):

        loss = criterion(pred, label)
        losses_item.append(loss.item())
        if loss_sum is None:
            loss_sum = weight * loss
        else:
            loss_sum += weight * loss

    return loss_sum, np.array(losses_item)

def vae_loss(mu_logvar, dummy_label=None):
    mu = mu_logvar[0]
    logvar = mu_logvar[1]
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss



class CustomLoss(nn.Module):
    def __init__(self, criterion, platten=False):
        super(CustomLoss, self).__init__()
        self.criterion = criterion
        self.criterion.reduction = "none"
        self.platten = platten
    def forward(self, pred, label):
        if self.platten:
            pred = pred.flatten(0, 1)
            label = label.flatten(0, 1)
        # Create a mask for non-NaN values
        pad_mask = ~torch.isnan(label)

        # Calculate loss using the provided loss function
        loss = self.criterion(pred, label)

        # Calculate the mean loss only for non-masked (non-padded) values
        # Avoid division by zero in case there are only NaN values in the label
        final_loss = loss[pad_mask].mean()

        return final_loss