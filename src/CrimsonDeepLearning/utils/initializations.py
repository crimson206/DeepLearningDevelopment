import torch
import torch.nn as nn

def get_layer_statistics(layer, target_param_str, target_fn):
    stastics = None
    target_param = getattr(layer, target_param_str, None)

    if target_param is not None:
        stastics = target_fn(target_param.data)

    return stastics

def collect_statistics(model, target_instance, target_param_str, target_fn=torch.mean):
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, target_instance):
            stats[name] = get_layer_statistics(module, target_param_str, target_fn)
    return stats

def initialize_weights(layer, target_param_str, init_fn=torch.nn.init.xavier_uniform_):
    target_param = getattr(layer, target_param_str, None)
    if target_param is not None:
        init_fn(target_param)

def initialize_custom_weights(layer, target_param_str, init_fn):
    target_param = getattr(layer, target_param_str, None)
    if target_param is not None:
        # Apply the custom initialization function and create a new Parameter
        initialized_param = init_fn(target_param.shape)
        setattr(layer, target_param_str, torch.nn.Parameter(initialized_param))

def apply_initialization(model, target_instance, target_param_str, init_fn=torch.nn.init.xavier_uniform_, custom=False):
    for name, module in model.named_modules():
        if isinstance(module, target_instance):
            if custom:
                initialize_custom_weights(module, target_param_str, init_fn)
            else:
                initialize_weights(module, target_param_str, init_fn)

def generate_log_resolution_channels(n_feature, max_feature, min_feature=64, reverse=False):

    channel_sizes = []

    for i in range(n_feature):
       feature = max_feature // 2**i
       if feature < min_feature:
           feature = min_feature
       channel_sizes.append(feature)

    if reverse:
        channel_sizes = list(reversed(channel_sizes))

    return channel_sizes

