import numpy as np
import torch

class EarlyStop:
    def __init__(self, patience, threshold, save_dir=None):
        self.patience = patience
        self.threshold = threshold
        self.save_dir = save_dir
        self.best_loss = np.inf
        self.wait = 0
        self.stopped = False

    def step(self, loss, model):
        self.save_last_model(model)
        if loss + self.threshold < self.best_loss:
            self.best_loss = loss
            self.wait = 0
            if self.save_dir:
                self.save_best_model(model)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True

    def save_best_model(self, model):
        if self.save_dir:
            torch.save(model.state_dict(), f"{self.save_dir}/best_model.pth")

    def save_last_model(self, model):
        if self.save_dir:
            torch.save(model.state_dict(), f"{self.save_dir}/last_model.pth")

    def should_stop(self):
        return self.stopped
