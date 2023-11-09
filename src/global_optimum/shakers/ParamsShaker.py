import torch
import copy

class ParamsShaker:
    def __init__(self, model, patience, threshold, shake_std=0.1):
        self.model = model
        self.patience = patience
        self.threshold = threshold
        self.shake_std = shake_std
        self.best_val_loss = float('inf')
        self.temp_best_val_loss = float('inf')
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.patience_counter = 0
        self.epoch = 0
        self.shaken_epochs = []

    def step(self, val_loss):
        self.epoch += 1

        # Improved over the best ever loss
        if val_loss < self.best_val_loss - self.threshold:
            self.best_val_loss = val_loss
            self.best_model_state = copy.deepcopy(self.model.state_dict())

        # Improved over the temp best loss
        if val_loss < self.temp_best_val_loss - self.threshold:
            self.temp_best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

            if self.patience_counter >= self.patience:
                # Reset the model to the best saved state
                self.model.load_state_dict(self.best_model_state)
                self.shake_parameters()
                # Reset the temp best val loss
                self.temp_best_val_loss = float('inf')
                self.shaken_epochs.append(self.epoch)

    def shake_parameters(self):
        with torch.no_grad():
            for param in self.model.parameters():
                noise = torch.randn_like(param) * self.shake_std
                param.add_(noise)
