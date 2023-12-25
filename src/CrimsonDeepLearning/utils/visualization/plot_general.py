import matplotlib.pyplot as plt
import io
import itertools
import torch

def plot_dictionary(data_dict, primary_axis_keys=[], second_axis_keys=[], title=None, 
                    figsize=(6, 4), xlabel='X-axis', primary_ylabel='Primary Y-axis', 
                    secondary_ylabel='Secondary Y-axis', primary_linestyle='-', 
                    secondary_linestyle='--', primary_colors=None, secondary_colors=None):
    """
    Plots curves from a dictionary of data using Matplotlib, with options for dual y-axes.
    ...
    """
    # Create a figure and a single subplot
    fig, ax1 = plt.subplots(figsize=figsize)

    if len(primary_axis_keys) == 0:
        primary_axis_keys = data_dict.keys()

    # Default colors for primary and secondary y-axis plots
    if not primary_colors:
        primary_colors = plt.cm.tab10.colors  # Default to using 10 categorical colors
    if not secondary_colors:
        secondary_colors = plt.cm.tab20c.colors  # Default to using 20 categorical colors

    primary_color_cycle = itertools.cycle(primary_colors)
    secondary_color_cycle = itertools.cycle(secondary_colors)

    # Plotting data on the primary y-axis
    for key, values in data_dict.items():
        if key in primary_axis_keys and key not in second_axis_keys:
            ax1.plot(values, label=key, color=next(primary_color_cycle), linestyle=primary_linestyle)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(primary_ylabel, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Handling the second y-axis
    if second_axis_keys:
        ax2 = ax1.twinx()
        for key in second_axis_keys:
            if key in data_dict:
                ax2.plot(data_dict[key], label=key, color=next(secondary_color_cycle), linestyle=secondary_linestyle)
        ax2.set_ylabel(secondary_ylabel, color='black')
        ax2.tick_params(axis='y', labelcolor='black')

    # Title and legend
    if title:
        plt.title(title)
    fig.tight_layout()
    ax1.legend(loc='upper left')
    if second_axis_keys:
        ax2.legend(loc='upper right')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.show()
    plt.close()
    return buffer.getvalue()



def plot_images(images, titles=None, rows=2, cols=4):
    num_images = min(len(images), rows * cols)

    plt.figure(figsize=(2 * cols, 2 * rows))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)

        if 1 < images.shape[1]:
            img = images[i].permute(1, 2, 0)
        elif 1 == images.shape[1]:
            img = images[i].squeeze()
        #img = img.clamp(0, 1)
        plt.imshow(img)
        plt.axis('off')

        # Set the title if provided
        if titles is not None and i < len(titles):
            plt.title(titles[i])
        else:
            plt.title('Image')

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.show()
    plt.close()
    return buffer.getvalue()


go = "donotuseplotpy"

#import plotly.graph_objects as go

def _plot_losses_with_plotly(losses_dict, second_axis_keys=[], title=None, figsize=(600, 400)):
    """
    Plots the loss curves from a dictionary of losses using Plotly, with hover functionality
    and the option to plot some losses on a second y-axis.
    
    Parameters:
    losses_dict (dict): A dictionary where keys are the names of the losses 
                        and values are lists of 
                        loss values.
    second_axis_keys (list): List of keys to be plotted on a second y-axis.
    title (str, optional): Title of the plot.
    figsize (tuple): Size of the figure (width, height) in pixels.
    """
    # Create figure with secondary y-axis
    fig = go.Figure()

    # Adding traces for the first y-axis
    for loss_name, loss_values in losses_dict.items():
        if loss_name not in second_axis_keys:
            fig.add_trace(go.Scatter(x=list(range(len(loss_values))), y=loss_values, mode='lines', name=loss_name))

    # Adding traces for the second y-axis
    if second_axis_keys:
        for key in second_axis_keys:
            if key in losses_dict:
                fig.add_trace(go.Scatter(x=list(range(len(losses_dict[key]))), y=losses_dict[key], mode='lines', name=key, yaxis="y2", line=dict(width=2, dash='dash')))

        # Create axis objects
        fig.update_layout(
            yaxis2=dict(
                title="Second Metric",
                overlaying="y",
                side="right"
            )
        )

    # Update layout with figure size
    fig.update_layout(
        title=title or "Loss Curves",
        xaxis_title="Epochs",
        yaxis_title="Loss",
        legend_title="Legend",
        hovermode="closest",
        width=figsize[0],  # Width of the figure
        height=figsize[1]  # Height of the figure
    )

    fig.show()
