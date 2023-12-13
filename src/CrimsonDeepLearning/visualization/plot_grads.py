import matplotlib.pyplot as plt

def get_gradient_magnitudes(model):
    grad_magnitudes = {}
    for name, parameters in model.named_parameters():
        if parameters.grad is not None:
            grad_magnitudes[name] = parameters.grad.data.norm(2).item()
    return grad_magnitudes

def plot_gradient_magnitudes(grad_magnitudes, title=None):
    layer_names = list(grad_magnitudes.keys())
    magnitudes = list(grad_magnitudes.values())

    plt.bar(layer_names, magnitudes)
    plt.xlabel('Layers')
    plt.ylabel('Gradient Magnitude')
    if title is None:
        title = 'Gradient Magnitudes per Layer'
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()

