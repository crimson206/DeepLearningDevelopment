import torch
import torch.nn as nn
import torch.nn.functional as F

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

def info_nce_loss(feats, temperature=0.7):
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    return nll

def neg_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = neg_loss(similarity, dim=0)
    image_loss = neg_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

