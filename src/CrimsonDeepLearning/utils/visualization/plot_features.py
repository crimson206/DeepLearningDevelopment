import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

def get_activation_visualization(model, layer, input, channel):
    """
    Modify the specified layer of the model to activate only the specified channel.

    Args:
    model (torch.nn.Module): The neural network model.
    layer_name (str): the layer to modify.
    channel (int): The channel number to activate.

    Returns:
    torch.Tensor: The output of the model with modified activation.
    """
    def set_activation(module, input, output):
        new_output = torch.zeros_like(output)
        new_output[:, channel] = output[:, channel]

        return new_output

    hook = layer.register_forward_hook(set_activation)

    with torch.no_grad():
        model.eval()
        output = model(input)

    hook.remove()
    return output

def plot_gcam_2d(conv_layer, index, input_tensor, model):
    outputs, gradients = None, None

    def save_gradients(*args):
        nonlocal gradients
        gradients = args[2]

    def save_outputs(*args):
        nonlocal outputs
        outputs = args[2]

    forward_hook = conv_layer.register_forward_hook(save_outputs)
    backward_hook = conv_layer.register_backward_hook(save_gradients)

    model.zero_grad()
    model.eval()

    # Forward pass
    output = model(input_tensor)
    target_class = output.argmax(dim=1)[index].item()

    # Zero gradients
    model.zero_grad()
    # Backward pass
    one_hot_target = torch.zeros_like(output)
    one_hot_target[index][target_class] = 1
    output.backward(gradient=one_hot_target)

    weights = F.adaptive_avg_pool2d(gradients[0], (1, 1)).squeeze()[index]

    feature_maps = outputs

    # Apply the weights to the feature maps
    gcam = torch.zeros_like(feature_maps[0][0])
    for i, w in enumerate(weights):
        gcam += w * feature_maps[index][i]

    # ReLU to only keep positive influences
    gcam = F.relu(gcam)

    # Remove hooks
    forward_hook.remove()
    backward_hook.remove()

    # Normalize and visualize
    gcam = gcam.squeeze().cpu().detach().numpy()
    gcam = (gcam - gcam.min()) / (gcam.max() - gcam.min())

    plt.imshow(gcam)
    plt.show()
