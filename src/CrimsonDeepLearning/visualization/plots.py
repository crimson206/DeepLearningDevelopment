import plotly.graph_objects as go

def plot_losses(losses_dict, second_axis_keys=[], title=None, figsize=(600, 400)):
    """
    Plots the loss curves from a dictionary of losses using Plotly, with hover functionality
    and the option to plot some losses on a second y-axis.
    
    Parameters:
    losses_dict (dict): A dictionary where keys are the names of the losses 
                        and values are lists of loss values.
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
