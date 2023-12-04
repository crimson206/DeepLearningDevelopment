import torch
import torch.nn as nn

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

def vae_loss(mu, logvar):
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss

class NanProofLoss(nn.Module):
    def __init__(self, criterion):
        super(NanProofLoss, self).__init__()
        self.criterion = criterion

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten(start_dim=0)
        label = label.flatten(start_dim=0)

        pad_mask = ~torch.isnan(label)
        loss = self.criterion(pred[pad_mask], label[pad_mask])
        final_loss = loss.mean()

        return final_loss

class FlattenLoss(nn.Module):
    def __init__(self, criterion):
        super(FlattenLoss, self).__init__()
        self.criterion = criterion

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pred = pred.flatten(0, 1)
        label = label.flatten(0, 1)

        return self.criterion(pred, label)