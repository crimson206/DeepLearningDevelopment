import numpy as np

class Logger():
    def __init__(self):
        self.epoch_values_dict = {}
        self.batch_values_dict = {}

    @property
    def batch_dict(self):
        value_dict = {}
        for key, losses in self.batch_values_dict.items():
            value_dict[key] = losses[-1] if losses else None
        return value_dict

    def batch_step(self, values, labels):
        for loss, label in zip(values, labels):
            if label in self.batch_values_dict.keys():
                self.batch_values_dict[label].append(loss)
            else:
                self.batch_values_dict[label] = [loss]
    
    def epoch_step(self):
        for label, values in self.batch_values_dict.items():
            avg_value = np.mean(values)
            if label in self.epoch_values_dict.keys():
                self.epoch_values_dict[label].append(avg_value)
            else:
                self.epoch_values_dict[label] = [avg_value]
    
        self.batch_values_dict = {}