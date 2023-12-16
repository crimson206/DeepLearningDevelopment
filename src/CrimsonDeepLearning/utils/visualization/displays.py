from IPython import display
from .plots import plot_losses

def display_progress(train_logger, val_logger, second_axis_keys, clear_output=True, fig_size=(600, 400)):
    if clear_output:
        display.clear_output(True)
    plot_losses(train_logger.epoch_losses_dict, second_axis_keys=second_axis_keys , title="Train Losses", figsize=fig_size)
    plot_losses(val_logger.epoch_losses_dict, second_axis_keys=second_axis_keys, title="Validation Losses", figsize=fig_size)