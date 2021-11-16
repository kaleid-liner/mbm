import torch.nn as nn


def copy_weights_from_identical_module(target_module: nn.Module, module: nn.Module):
    for target_param, param in zip(target_module.parameters(), module.parameters()):
        target_param.data.copy_(param.data)
