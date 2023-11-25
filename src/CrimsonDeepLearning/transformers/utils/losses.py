import torch

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

    return loss_sum, losses_item

def vae_loss(mu_logvar, dummy_label=None):
    mu = mu_logvar[0]
    logvar = mu_logvar[1]
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss
