import numpy as np

class LossLogger():
    def __init__(self):
        self.epoch_losses_dict = {}
        self.batch_losses_dict = {}

    @property
    def batch_loss_dict(self):
        loss_dict = {}
        for key, losses in self.batch_losses_dict.items():
            loss_dict[key] = losses[-1] if losses else None
        return loss_dict

    def batch_step(self, losses, labels):
        for loss, label in zip(losses, labels):
            if label in self.batch_losses_dict.keys():
                self.batch_losses_dict[label].append(loss)
            else:
                self.batch_losses_dict[label] = [loss]
    
    def epoch_step(self):
        for label, losses in self.batch_losses_dict.items():
            avg_loss = np.mean(losses)
            if label in self.epoch_losses_dict.keys():
                self.epoch_losses_dict[label].append(avg_loss)
            else:
                self.epoch_losses_dict[label] = [avg_loss]
    
        self.batch_losses_dict = {}