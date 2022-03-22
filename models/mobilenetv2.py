import torch
from torch import nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
from typing import Callable, Any, Optional, List, OrderedDict

from ir.tree import Tree
from .tree_module import TreeModule
from .layers import ConvBNReLU, InvertedResidual
from .utils import copy_weights_from_identical_module
from .base_mbmodel import BaseMBModel


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}




def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(BaseMBModel):
    state_dict_map = {
        'features.0.0.weight': 'conv_first.0.weight', 'features.0.1.weight': 'conv_first.1.weight', 'features.0.1.bias': 'conv_first.1.bias', 'features.0.1.running_mean': 'conv_first.1.running_mean', 'features.0.1.running_var': 'conv_first.1.running_var', 'features.0.1.num_batches_tracked': 'conv_first.1.num_batches_tracked', 'features.1.conv.0.0.weight': 'blocks.0.branches.0.0.conv.0.0.weight', 'features.1.conv.0.1.weight': 'blocks.0.branches.0.0.conv.0.1.weight', 'features.1.conv.0.1.bias': 'blocks.0.branches.0.0.conv.0.1.bias', 'features.1.conv.0.1.running_mean': 'blocks.0.branches.0.0.conv.0.1.running_mean', 'features.1.conv.0.1.running_var': 'blocks.0.branches.0.0.conv.0.1.running_var', 'features.1.conv.0.1.num_batches_tracked': 'blocks.0.branches.0.0.conv.0.1.num_batches_tracked', 'features.1.conv.1.weight': 'blocks.0.branches.0.0.conv.1.weight', 'features.1.conv.2.weight': 'blocks.0.branches.0.0.conv.2.weight', 'features.1.conv.2.bias': 'blocks.0.branches.0.0.conv.2.bias', 'features.1.conv.2.running_mean': 'blocks.0.branches.0.0.conv.2.running_mean', 'features.1.conv.2.running_var': 'blocks.0.branches.0.0.conv.2.running_var', 'features.1.conv.2.num_batches_tracked': 'blocks.0.branches.0.0.conv.2.num_batches_tracked', 'features.2.conv.0.0.weight': 'blocks.1.branches.0.0.conv.0.0.weight', 'features.2.conv.0.1.weight': 'blocks.1.branches.0.0.conv.0.1.weight', 'features.2.conv.0.1.bias': 'blocks.1.branches.0.0.conv.0.1.bias', 'features.2.conv.0.1.running_mean': 'blocks.1.branches.0.0.conv.0.1.running_mean', 'features.2.conv.0.1.running_var': 'blocks.1.branches.0.0.conv.0.1.running_var', 'features.2.conv.0.1.num_batches_tracked': 'blocks.1.branches.0.0.conv.0.1.num_batches_tracked', 'features.2.conv.1.0.weight': 'blocks.1.branches.0.0.conv.1.0.weight', 'features.2.conv.1.1.weight': 'blocks.1.branches.0.0.conv.1.1.weight', 'features.2.conv.1.1.bias': 'blocks.1.branches.0.0.conv.1.1.bias', 'features.2.conv.1.1.running_mean': 'blocks.1.branches.0.0.conv.1.1.running_mean', 'features.2.conv.1.1.running_var': 'blocks.1.branches.0.0.conv.1.1.running_var', 'features.2.conv.1.1.num_batches_tracked': 'blocks.1.branches.0.0.conv.1.1.num_batches_tracked', 'features.2.conv.2.weight': 'blocks.1.branches.0.0.conv.2.weight', 'features.2.conv.3.weight': 'blocks.1.branches.0.0.conv.3.weight', 'features.2.conv.3.bias': 'blocks.1.branches.0.0.conv.3.bias', 'features.2.conv.3.running_mean': 'blocks.1.branches.0.0.conv.3.running_mean', 'features.2.conv.3.running_var': 'blocks.1.branches.0.0.conv.3.running_var', 'features.2.conv.3.num_batches_tracked': 'blocks.1.branches.0.0.conv.3.num_batches_tracked', 'features.3.conv.0.0.weight': 'blocks.1.branches.0.1.conv.0.0.weight', 'features.3.conv.0.1.weight': 'blocks.1.branches.0.1.conv.0.1.weight', 'features.3.conv.0.1.bias': 'blocks.1.branches.0.1.conv.0.1.bias', 'features.3.conv.0.1.running_mean': 'blocks.1.branches.0.1.conv.0.1.running_mean', 'features.3.conv.0.1.running_var': 'blocks.1.branches.0.1.conv.0.1.running_var', 'features.3.conv.0.1.num_batches_tracked': 'blocks.1.branches.0.1.conv.0.1.num_batches_tracked', 'features.3.conv.1.0.weight': 'blocks.1.branches.0.1.conv.1.0.weight', 'features.3.conv.1.1.weight': 'blocks.1.branches.0.1.conv.1.1.weight', 'features.3.conv.1.1.bias': 'blocks.1.branches.0.1.conv.1.1.bias', 'features.3.conv.1.1.running_mean': 'blocks.1.branches.0.1.conv.1.1.running_mean', 'features.3.conv.1.1.running_var': 'blocks.1.branches.0.1.conv.1.1.running_var', 'features.3.conv.1.1.num_batches_tracked': 'blocks.1.branches.0.1.conv.1.1.num_batches_tracked', 'features.3.conv.2.weight': 'blocks.1.branches.0.1.conv.2.weight', 'features.3.conv.3.weight': 'blocks.1.branches.0.1.conv.3.weight', 'features.3.conv.3.bias': 'blocks.1.branches.0.1.conv.3.bias', 'features.3.conv.3.running_mean': 'blocks.1.branches.0.1.conv.3.running_mean', 'features.3.conv.3.running_var': 'blocks.1.branches.0.1.conv.3.running_var', 'features.3.conv.3.num_batches_tracked': 'blocks.1.branches.0.1.conv.3.num_batches_tracked', 'features.4.conv.0.0.weight': 'blocks.2.branches.0.0.conv.0.0.weight', 'features.4.conv.0.1.weight': 'blocks.2.branches.0.0.conv.0.1.weight', 'features.4.conv.0.1.bias': 'blocks.2.branches.0.0.conv.0.1.bias', 'features.4.conv.0.1.running_mean': 'blocks.2.branches.0.0.conv.0.1.running_mean', 'features.4.conv.0.1.running_var': 'blocks.2.branches.0.0.conv.0.1.running_var', 'features.4.conv.0.1.num_batches_tracked': 'blocks.2.branches.0.0.conv.0.1.num_batches_tracked', 'features.4.conv.1.0.weight': 'blocks.2.branches.0.0.conv.1.0.weight', 'features.4.conv.1.1.weight': 'blocks.2.branches.0.0.conv.1.1.weight', 'features.4.conv.1.1.bias': 'blocks.2.branches.0.0.conv.1.1.bias', 'features.4.conv.1.1.running_mean': 'blocks.2.branches.0.0.conv.1.1.running_mean', 'features.4.conv.1.1.running_var': 'blocks.2.branches.0.0.conv.1.1.running_var', 'features.4.conv.1.1.num_batches_tracked': 'blocks.2.branches.0.0.conv.1.1.num_batches_tracked', 'features.4.conv.2.weight': 'blocks.2.branches.0.0.conv.2.weight', 'features.4.conv.3.weight': 'blocks.2.branches.0.0.conv.3.weight', 'features.4.conv.3.bias': 'blocks.2.branches.0.0.conv.3.bias', 'features.4.conv.3.running_mean': 'blocks.2.branches.0.0.conv.3.running_mean', 'features.4.conv.3.running_var': 'blocks.2.branches.0.0.conv.3.running_var', 'features.4.conv.3.num_batches_tracked': 'blocks.2.branches.0.0.conv.3.num_batches_tracked', 'features.5.conv.0.0.weight': 'blocks.2.branches.0.1.conv.0.0.weight', 'features.5.conv.0.1.weight': 'blocks.2.branches.0.1.conv.0.1.weight', 'features.5.conv.0.1.bias': 'blocks.2.branches.0.1.conv.0.1.bias', 'features.5.conv.0.1.running_mean': 'blocks.2.branches.0.1.conv.0.1.running_mean', 'features.5.conv.0.1.running_var': 'blocks.2.branches.0.1.conv.0.1.running_var', 'features.5.conv.0.1.num_batches_tracked': 'blocks.2.branches.0.1.conv.0.1.num_batches_tracked', 'features.5.conv.1.0.weight': 'blocks.2.branches.0.1.conv.1.0.weight', 'features.5.conv.1.1.weight': 'blocks.2.branches.0.1.conv.1.1.weight', 'features.5.conv.1.1.bias': 'blocks.2.branches.0.1.conv.1.1.bias', 'features.5.conv.1.1.running_mean': 'blocks.2.branches.0.1.conv.1.1.running_mean', 'features.5.conv.1.1.running_var': 'blocks.2.branches.0.1.conv.1.1.running_var', 'features.5.conv.1.1.num_batches_tracked': 'blocks.2.branches.0.1.conv.1.1.num_batches_tracked', 'features.5.conv.2.weight': 'blocks.2.branches.0.1.conv.2.weight', 'features.5.conv.3.weight': 'blocks.2.branches.0.1.conv.3.weight', 'features.5.conv.3.bias': 'blocks.2.branches.0.1.conv.3.bias', 'features.5.conv.3.running_mean': 'blocks.2.branches.0.1.conv.3.running_mean', 'features.5.conv.3.running_var': 'blocks.2.branches.0.1.conv.3.running_var', 'features.5.conv.3.num_batches_tracked': 'blocks.2.branches.0.1.conv.3.num_batches_tracked', 'features.6.conv.0.0.weight': 'blocks.2.branches.0.2.conv.0.0.weight', 'features.6.conv.0.1.weight': 'blocks.2.branches.0.2.conv.0.1.weight', 'features.6.conv.0.1.bias': 'blocks.2.branches.0.2.conv.0.1.bias', 'features.6.conv.0.1.running_mean': 'blocks.2.branches.0.2.conv.0.1.running_mean', 'features.6.conv.0.1.running_var': 'blocks.2.branches.0.2.conv.0.1.running_var', 'features.6.conv.0.1.num_batches_tracked': 'blocks.2.branches.0.2.conv.0.1.num_batches_tracked', 'features.6.conv.1.0.weight': 'blocks.2.branches.0.2.conv.1.0.weight', 'features.6.conv.1.1.weight': 'blocks.2.branches.0.2.conv.1.1.weight', 'features.6.conv.1.1.bias': 'blocks.2.branches.0.2.conv.1.1.bias', 'features.6.conv.1.1.running_mean': 'blocks.2.branches.0.2.conv.1.1.running_mean', 'features.6.conv.1.1.running_var': 'blocks.2.branches.0.2.conv.1.1.running_var', 'features.6.conv.1.1.num_batches_tracked': 'blocks.2.branches.0.2.conv.1.1.num_batches_tracked', 'features.6.conv.2.weight': 'blocks.2.branches.0.2.conv.2.weight', 'features.6.conv.3.weight': 'blocks.2.branches.0.2.conv.3.weight', 'features.6.conv.3.bias': 'blocks.2.branches.0.2.conv.3.bias', 'features.6.conv.3.running_mean': 'blocks.2.branches.0.2.conv.3.running_mean', 'features.6.conv.3.running_var': 'blocks.2.branches.0.2.conv.3.running_var', 'features.6.conv.3.num_batches_tracked': 'blocks.2.branches.0.2.conv.3.num_batches_tracked', 'features.7.conv.0.0.weight': 'blocks.3.branches.0.0.conv.0.0.weight', 'features.7.conv.0.1.weight': 'blocks.3.branches.0.0.conv.0.1.weight', 'features.7.conv.0.1.bias': 'blocks.3.branches.0.0.conv.0.1.bias', 'features.7.conv.0.1.running_mean': 'blocks.3.branches.0.0.conv.0.1.running_mean', 'features.7.conv.0.1.running_var': 'blocks.3.branches.0.0.conv.0.1.running_var', 'features.7.conv.0.1.num_batches_tracked': 'blocks.3.branches.0.0.conv.0.1.num_batches_tracked', 'features.7.conv.1.0.weight': 'blocks.3.branches.0.0.conv.1.0.weight', 'features.7.conv.1.1.weight': 'blocks.3.branches.0.0.conv.1.1.weight', 'features.7.conv.1.1.bias': 'blocks.3.branches.0.0.conv.1.1.bias', 'features.7.conv.1.1.running_mean': 'blocks.3.branches.0.0.conv.1.1.running_mean', 'features.7.conv.1.1.running_var': 'blocks.3.branches.0.0.conv.1.1.running_var', 'features.7.conv.1.1.num_batches_tracked': 'blocks.3.branches.0.0.conv.1.1.num_batches_tracked', 'features.7.conv.2.weight': 'blocks.3.branches.0.0.conv.2.weight', 'features.7.conv.3.weight': 'blocks.3.branches.0.0.conv.3.weight', 'features.7.conv.3.bias': 'blocks.3.branches.0.0.conv.3.bias', 'features.7.conv.3.running_mean': 'blocks.3.branches.0.0.conv.3.running_mean', 'features.7.conv.3.running_var': 'blocks.3.branches.0.0.conv.3.running_var', 'features.7.conv.3.num_batches_tracked': 'blocks.3.branches.0.0.conv.3.num_batches_tracked', 'features.8.conv.0.0.weight': 'blocks.3.branches.0.1.conv.0.0.weight', 'features.8.conv.0.1.weight': 'blocks.3.branches.0.1.conv.0.1.weight', 'features.8.conv.0.1.bias': 'blocks.3.branches.0.1.conv.0.1.bias', 'features.8.conv.0.1.running_mean': 'blocks.3.branches.0.1.conv.0.1.running_mean', 'features.8.conv.0.1.running_var': 'blocks.3.branches.0.1.conv.0.1.running_var', 'features.8.conv.0.1.num_batches_tracked': 'blocks.3.branches.0.1.conv.0.1.num_batches_tracked', 'features.8.conv.1.0.weight': 'blocks.3.branches.0.1.conv.1.0.weight', 'features.8.conv.1.1.weight': 'blocks.3.branches.0.1.conv.1.1.weight', 'features.8.conv.1.1.bias': 'blocks.3.branches.0.1.conv.1.1.bias', 'features.8.conv.1.1.running_mean': 'blocks.3.branches.0.1.conv.1.1.running_mean', 'features.8.conv.1.1.running_var': 'blocks.3.branches.0.1.conv.1.1.running_var', 'features.8.conv.1.1.num_batches_tracked': 'blocks.3.branches.0.1.conv.1.1.num_batches_tracked', 'features.8.conv.2.weight': 'blocks.3.branches.0.1.conv.2.weight', 'features.8.conv.3.weight': 'blocks.3.branches.0.1.conv.3.weight', 'features.8.conv.3.bias': 'blocks.3.branches.0.1.conv.3.bias', 'features.8.conv.3.running_mean': 'blocks.3.branches.0.1.conv.3.running_mean', 'features.8.conv.3.running_var': 'blocks.3.branches.0.1.conv.3.running_var', 'features.8.conv.3.num_batches_tracked': 'blocks.3.branches.0.1.conv.3.num_batches_tracked', 'features.9.conv.0.0.weight': 'blocks.3.branches.0.2.conv.0.0.weight', 'features.9.conv.0.1.weight': 'blocks.3.branches.0.2.conv.0.1.weight', 'features.9.conv.0.1.bias': 'blocks.3.branches.0.2.conv.0.1.bias', 'features.9.conv.0.1.running_mean': 'blocks.3.branches.0.2.conv.0.1.running_mean', 'features.9.conv.0.1.running_var': 'blocks.3.branches.0.2.conv.0.1.running_var', 'features.9.conv.0.1.num_batches_tracked': 'blocks.3.branches.0.2.conv.0.1.num_batches_tracked', 'features.9.conv.1.0.weight': 'blocks.3.branches.0.2.conv.1.0.weight', 'features.9.conv.1.1.weight': 'blocks.3.branches.0.2.conv.1.1.weight', 'features.9.conv.1.1.bias': 'blocks.3.branches.0.2.conv.1.1.bias', 'features.9.conv.1.1.running_mean': 'blocks.3.branches.0.2.conv.1.1.running_mean', 'features.9.conv.1.1.running_var': 'blocks.3.branches.0.2.conv.1.1.running_var', 'features.9.conv.1.1.num_batches_tracked': 'blocks.3.branches.0.2.conv.1.1.num_batches_tracked', 'features.9.conv.2.weight': 'blocks.3.branches.0.2.conv.2.weight', 'features.9.conv.3.weight': 'blocks.3.branches.0.2.conv.3.weight', 'features.9.conv.3.bias': 'blocks.3.branches.0.2.conv.3.bias', 'features.9.conv.3.running_mean': 'blocks.3.branches.0.2.conv.3.running_mean', 'features.9.conv.3.running_var': 'blocks.3.branches.0.2.conv.3.running_var', 'features.9.conv.3.num_batches_tracked': 'blocks.3.branches.0.2.conv.3.num_batches_tracked', 'features.10.conv.0.0.weight': 'blocks.3.branches.0.3.conv.0.0.weight', 'features.10.conv.0.1.weight': 'blocks.3.branches.0.3.conv.0.1.weight', 'features.10.conv.0.1.bias': 'blocks.3.branches.0.3.conv.0.1.bias', 'features.10.conv.0.1.running_mean': 'blocks.3.branches.0.3.conv.0.1.running_mean', 'features.10.conv.0.1.running_var': 'blocks.3.branches.0.3.conv.0.1.running_var', 'features.10.conv.0.1.num_batches_tracked': 'blocks.3.branches.0.3.conv.0.1.num_batches_tracked', 'features.10.conv.1.0.weight': 'blocks.3.branches.0.3.conv.1.0.weight', 'features.10.conv.1.1.weight': 'blocks.3.branches.0.3.conv.1.1.weight', 'features.10.conv.1.1.bias': 'blocks.3.branches.0.3.conv.1.1.bias', 'features.10.conv.1.1.running_mean': 'blocks.3.branches.0.3.conv.1.1.running_mean', 'features.10.conv.1.1.running_var': 'blocks.3.branches.0.3.conv.1.1.running_var', 'features.10.conv.1.1.num_batches_tracked': 'blocks.3.branches.0.3.conv.1.1.num_batches_tracked', 'features.10.conv.2.weight': 'blocks.3.branches.0.3.conv.2.weight', 'features.10.conv.3.weight': 'blocks.3.branches.0.3.conv.3.weight', 'features.10.conv.3.bias': 'blocks.3.branches.0.3.conv.3.bias', 'features.10.conv.3.running_mean': 'blocks.3.branches.0.3.conv.3.running_mean', 'features.10.conv.3.running_var': 'blocks.3.branches.0.3.conv.3.running_var', 'features.10.conv.3.num_batches_tracked': 'blocks.3.branches.0.3.conv.3.num_batches_tracked', 'features.11.conv.0.0.weight': 'blocks.4.branches.0.0.conv.0.0.weight', 'features.11.conv.0.1.weight': 'blocks.4.branches.0.0.conv.0.1.weight', 'features.11.conv.0.1.bias': 'blocks.4.branches.0.0.conv.0.1.bias', 'features.11.conv.0.1.running_mean': 'blocks.4.branches.0.0.conv.0.1.running_mean', 'features.11.conv.0.1.running_var': 'blocks.4.branches.0.0.conv.0.1.running_var', 'features.11.conv.0.1.num_batches_tracked': 'blocks.4.branches.0.0.conv.0.1.num_batches_tracked', 'features.11.conv.1.0.weight': 'blocks.4.branches.0.0.conv.1.0.weight', 'features.11.conv.1.1.weight': 'blocks.4.branches.0.0.conv.1.1.weight', 'features.11.conv.1.1.bias': 'blocks.4.branches.0.0.conv.1.1.bias', 'features.11.conv.1.1.running_mean': 'blocks.4.branches.0.0.conv.1.1.running_mean', 'features.11.conv.1.1.running_var': 'blocks.4.branches.0.0.conv.1.1.running_var', 'features.11.conv.1.1.num_batches_tracked': 'blocks.4.branches.0.0.conv.1.1.num_batches_tracked', 'features.11.conv.2.weight': 'blocks.4.branches.0.0.conv.2.weight', 'features.11.conv.3.weight': 'blocks.4.branches.0.0.conv.3.weight', 'features.11.conv.3.bias': 'blocks.4.branches.0.0.conv.3.bias', 'features.11.conv.3.running_mean': 'blocks.4.branches.0.0.conv.3.running_mean', 'features.11.conv.3.running_var': 'blocks.4.branches.0.0.conv.3.running_var', 'features.11.conv.3.num_batches_tracked': 'blocks.4.branches.0.0.conv.3.num_batches_tracked', 'features.12.conv.0.0.weight': 'blocks.4.branches.0.1.conv.0.0.weight', 'features.12.conv.0.1.weight': 'blocks.4.branches.0.1.conv.0.1.weight', 'features.12.conv.0.1.bias': 'blocks.4.branches.0.1.conv.0.1.bias', 'features.12.conv.0.1.running_mean': 'blocks.4.branches.0.1.conv.0.1.running_mean', 'features.12.conv.0.1.running_var': 'blocks.4.branches.0.1.conv.0.1.running_var', 'features.12.conv.0.1.num_batches_tracked': 'blocks.4.branches.0.1.conv.0.1.num_batches_tracked', 'features.12.conv.1.0.weight': 'blocks.4.branches.0.1.conv.1.0.weight', 'features.12.conv.1.1.weight': 'blocks.4.branches.0.1.conv.1.1.weight', 'features.12.conv.1.1.bias': 'blocks.4.branches.0.1.conv.1.1.bias', 'features.12.conv.1.1.running_mean': 'blocks.4.branches.0.1.conv.1.1.running_mean', 'features.12.conv.1.1.running_var': 'blocks.4.branches.0.1.conv.1.1.running_var', 'features.12.conv.1.1.num_batches_tracked': 'blocks.4.branches.0.1.conv.1.1.num_batches_tracked', 'features.12.conv.2.weight': 'blocks.4.branches.0.1.conv.2.weight', 'features.12.conv.3.weight': 'blocks.4.branches.0.1.conv.3.weight', 'features.12.conv.3.bias': 'blocks.4.branches.0.1.conv.3.bias', 'features.12.conv.3.running_mean': 'blocks.4.branches.0.1.conv.3.running_mean', 'features.12.conv.3.running_var': 'blocks.4.branches.0.1.conv.3.running_var', 'features.12.conv.3.num_batches_tracked': 'blocks.4.branches.0.1.conv.3.num_batches_tracked', 'features.13.conv.0.0.weight': 'blocks.4.branches.0.2.conv.0.0.weight', 'features.13.conv.0.1.weight': 'blocks.4.branches.0.2.conv.0.1.weight', 'features.13.conv.0.1.bias': 'blocks.4.branches.0.2.conv.0.1.bias', 'features.13.conv.0.1.running_mean': 'blocks.4.branches.0.2.conv.0.1.running_mean', 'features.13.conv.0.1.running_var': 'blocks.4.branches.0.2.conv.0.1.running_var', 'features.13.conv.0.1.num_batches_tracked': 'blocks.4.branches.0.2.conv.0.1.num_batches_tracked', 'features.13.conv.1.0.weight': 'blocks.4.branches.0.2.conv.1.0.weight', 'features.13.conv.1.1.weight': 'blocks.4.branches.0.2.conv.1.1.weight', 'features.13.conv.1.1.bias': 'blocks.4.branches.0.2.conv.1.1.bias', 'features.13.conv.1.1.running_mean': 'blocks.4.branches.0.2.conv.1.1.running_mean', 'features.13.conv.1.1.running_var': 'blocks.4.branches.0.2.conv.1.1.running_var', 'features.13.conv.1.1.num_batches_tracked': 'blocks.4.branches.0.2.conv.1.1.num_batches_tracked', 'features.13.conv.2.weight': 'blocks.4.branches.0.2.conv.2.weight', 'features.13.conv.3.weight': 'blocks.4.branches.0.2.conv.3.weight', 'features.13.conv.3.bias': 'blocks.4.branches.0.2.conv.3.bias', 'features.13.conv.3.running_mean': 'blocks.4.branches.0.2.conv.3.running_mean', 'features.13.conv.3.running_var': 'blocks.4.branches.0.2.conv.3.running_var', 'features.13.conv.3.num_batches_tracked': 'blocks.4.branches.0.2.conv.3.num_batches_tracked', 'features.14.conv.0.0.weight': 'blocks.5.branches.0.0.conv.0.0.weight', 'features.14.conv.0.1.weight': 'blocks.5.branches.0.0.conv.0.1.weight', 'features.14.conv.0.1.bias': 'blocks.5.branches.0.0.conv.0.1.bias', 'features.14.conv.0.1.running_mean': 'blocks.5.branches.0.0.conv.0.1.running_mean', 'features.14.conv.0.1.running_var': 'blocks.5.branches.0.0.conv.0.1.running_var', 'features.14.conv.0.1.num_batches_tracked': 'blocks.5.branches.0.0.conv.0.1.num_batches_tracked', 'features.14.conv.1.0.weight': 'blocks.5.branches.0.0.conv.1.0.weight', 'features.14.conv.1.1.weight': 'blocks.5.branches.0.0.conv.1.1.weight', 'features.14.conv.1.1.bias': 'blocks.5.branches.0.0.conv.1.1.bias', 'features.14.conv.1.1.running_mean': 'blocks.5.branches.0.0.conv.1.1.running_mean', 'features.14.conv.1.1.running_var': 'blocks.5.branches.0.0.conv.1.1.running_var', 'features.14.conv.1.1.num_batches_tracked': 'blocks.5.branches.0.0.conv.1.1.num_batches_tracked', 'features.14.conv.2.weight': 'blocks.5.branches.0.0.conv.2.weight', 'features.14.conv.3.weight': 'blocks.5.branches.0.0.conv.3.weight', 'features.14.conv.3.bias': 'blocks.5.branches.0.0.conv.3.bias', 'features.14.conv.3.running_mean': 'blocks.5.branches.0.0.conv.3.running_mean', 'features.14.conv.3.running_var': 'blocks.5.branches.0.0.conv.3.running_var', 'features.14.conv.3.num_batches_tracked': 'blocks.5.branches.0.0.conv.3.num_batches_tracked', 'features.15.conv.0.0.weight': 'blocks.5.branches.0.1.conv.0.0.weight', 'features.15.conv.0.1.weight': 'blocks.5.branches.0.1.conv.0.1.weight', 'features.15.conv.0.1.bias': 'blocks.5.branches.0.1.conv.0.1.bias', 'features.15.conv.0.1.running_mean': 'blocks.5.branches.0.1.conv.0.1.running_mean', 'features.15.conv.0.1.running_var': 'blocks.5.branches.0.1.conv.0.1.running_var', 'features.15.conv.0.1.num_batches_tracked': 'blocks.5.branches.0.1.conv.0.1.num_batches_tracked', 'features.15.conv.1.0.weight': 'blocks.5.branches.0.1.conv.1.0.weight', 'features.15.conv.1.1.weight': 'blocks.5.branches.0.1.conv.1.1.weight', 'features.15.conv.1.1.bias': 'blocks.5.branches.0.1.conv.1.1.bias', 'features.15.conv.1.1.running_mean': 'blocks.5.branches.0.1.conv.1.1.running_mean', 'features.15.conv.1.1.running_var': 'blocks.5.branches.0.1.conv.1.1.running_var', 'features.15.conv.1.1.num_batches_tracked': 'blocks.5.branches.0.1.conv.1.1.num_batches_tracked', 'features.15.conv.2.weight': 'blocks.5.branches.0.1.conv.2.weight', 'features.15.conv.3.weight': 'blocks.5.branches.0.1.conv.3.weight', 'features.15.conv.3.bias': 'blocks.5.branches.0.1.conv.3.bias', 'features.15.conv.3.running_mean': 'blocks.5.branches.0.1.conv.3.running_mean', 'features.15.conv.3.running_var': 'blocks.5.branches.0.1.conv.3.running_var', 'features.15.conv.3.num_batches_tracked': 'blocks.5.branches.0.1.conv.3.num_batches_tracked', 'features.16.conv.0.0.weight': 'blocks.5.branches.0.2.conv.0.0.weight', 'features.16.conv.0.1.weight': 'blocks.5.branches.0.2.conv.0.1.weight', 'features.16.conv.0.1.bias': 'blocks.5.branches.0.2.conv.0.1.bias', 'features.16.conv.0.1.running_mean': 'blocks.5.branches.0.2.conv.0.1.running_mean', 'features.16.conv.0.1.running_var': 'blocks.5.branches.0.2.conv.0.1.running_var', 'features.16.conv.0.1.num_batches_tracked': 'blocks.5.branches.0.2.conv.0.1.num_batches_tracked', 'features.16.conv.1.0.weight': 'blocks.5.branches.0.2.conv.1.0.weight', 'features.16.conv.1.1.weight': 'blocks.5.branches.0.2.conv.1.1.weight', 'features.16.conv.1.1.bias': 'blocks.5.branches.0.2.conv.1.1.bias', 'features.16.conv.1.1.running_mean': 'blocks.5.branches.0.2.conv.1.1.running_mean', 'features.16.conv.1.1.running_var': 'blocks.5.branches.0.2.conv.1.1.running_var', 'features.16.conv.1.1.num_batches_tracked': 'blocks.5.branches.0.2.conv.1.1.num_batches_tracked', 'features.16.conv.2.weight': 'blocks.5.branches.0.2.conv.2.weight', 'features.16.conv.3.weight': 'blocks.5.branches.0.2.conv.3.weight', 'features.16.conv.3.bias': 'blocks.5.branches.0.2.conv.3.bias', 'features.16.conv.3.running_mean': 'blocks.5.branches.0.2.conv.3.running_mean', 'features.16.conv.3.running_var': 'blocks.5.branches.0.2.conv.3.running_var', 'features.16.conv.3.num_batches_tracked': 'blocks.5.branches.0.2.conv.3.num_batches_tracked', 'features.17.conv.0.0.weight': 'blocks.6.branches.0.0.conv.0.0.weight', 'features.17.conv.0.1.weight': 'blocks.6.branches.0.0.conv.0.1.weight', 'features.17.conv.0.1.bias': 'blocks.6.branches.0.0.conv.0.1.bias', 'features.17.conv.0.1.running_mean': 'blocks.6.branches.0.0.conv.0.1.running_mean', 'features.17.conv.0.1.running_var': 'blocks.6.branches.0.0.conv.0.1.running_var', 'features.17.conv.0.1.num_batches_tracked': 'blocks.6.branches.0.0.conv.0.1.num_batches_tracked', 'features.17.conv.1.0.weight': 'blocks.6.branches.0.0.conv.1.0.weight', 'features.17.conv.1.1.weight': 'blocks.6.branches.0.0.conv.1.1.weight', 'features.17.conv.1.1.bias': 'blocks.6.branches.0.0.conv.1.1.bias', 'features.17.conv.1.1.running_mean': 'blocks.6.branches.0.0.conv.1.1.running_mean', 'features.17.conv.1.1.running_var': 'blocks.6.branches.0.0.conv.1.1.running_var', 'features.17.conv.1.1.num_batches_tracked': 'blocks.6.branches.0.0.conv.1.1.num_batches_tracked', 'features.17.conv.2.weight': 'blocks.6.branches.0.0.conv.2.weight', 'features.17.conv.3.weight': 'blocks.6.branches.0.0.conv.3.weight', 'features.17.conv.3.bias': 'blocks.6.branches.0.0.conv.3.bias', 'features.17.conv.3.running_mean': 'blocks.6.branches.0.0.conv.3.running_mean', 'features.17.conv.3.running_var': 'blocks.6.branches.0.0.conv.3.running_var', 'features.17.conv.3.num_batches_tracked': 'blocks.6.branches.0.0.conv.3.num_batches_tracked', 'features.18.0.weight': 'conv_last.0.weight', 'features.18.1.weight': 'conv_last.1.weight', 'features.18.1.bias': 'conv_last.1.bias', 'features.18.1.running_mean': 'conv_last.1.running_mean', 'features.18.1.running_var': 'conv_last.1.running_var', 'features.18.1.num_batches_tracked': 'conv_last.1.num_batches_tracked', 'classifier.1.weight': 'classifier.1.weight', 'classifier.1.bias': 'classifier.1.bias'
    }

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        stem_stride: int = 2,
        start_trees: Optional[List[Tree]] = None,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, f
                [1, 16, 1, 1, 1],
                [6, 24, 2, 2, 1],
                [6, 32, 3, 2, 1],
                [6, 64, 4, 2, 1],
                [6, 96, 3, 1, 1],
                [6, 160, 3, 2, 1],
                [6, 320, 1, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 5-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.conv_first = ConvBNReLU(3, input_channel, stride=stem_stride, norm_layer=norm_layer)

        self.blocks = nn.ModuleList([])
        self.idx2layer = {}
        self.downsamples1 = nn.ModuleList([])
        self.downsamples2 = nn.ModuleList([])

        self.path = 0

        if start_trees is None:
            self.trees: List[Tree] = []
            # building inverted residual blocks
            for t, c, n, s, f in inverted_residual_setting:
                if f > 1:
                    root = Tree(None, [], 'copy', {}, input_channel)
                else:
                    root = Tree(None, [], None, {}, input_channel)
                for _ in range(f):
                    output_channel = _make_divisible(c * width_mult, round_nearest)
                    parent = root
                    inp = input_channel
                    for i in range(n):
                        stride = s if i == 0 else 1
                        tree = Tree(parent, [], 'InvertedResidual', {
                            'inp': inp,
                            'oup': output_channel,
                            'stride': stride,
                            'expand_ratio': t,
                            'norm_layer': norm_layer,
                        }, output_channel)
                        parent.children.append(tree)
                        parent = tree
                        inp = output_channel
                input_channel = output_channel
                self.trees.append(root)
                self.downsamples1.append(nn.Sequential(
                    nn.Conv2d(input_channel, input_channel // 2, 1),
                    nn.BatchNorm2d(input_channel // 2),
                ))
                self.downsamples2.append(nn.Sequential(
                    nn.Conv2d(input_channel, input_channel // 2, 1),
                    nn.BatchNorm2d(input_channel // 2),
                ))
        else:
            self.trees = start_trees

        for tree in self.trees:
            block = TreeModule(tree)
            self.idx2layer.update(block.idx2layer)
            self.blocks.append(block)
            input_channel = tree.get_leaves()[0].output_channel

        # building last several layers
        self.conv_last = ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def copy_weights_from_sequential(self, net: 'MobileNetV2'):
        super().copy_weights_from_sequential(net)

        self.conv_first.load_state_dict(net.conv_first.state_dict())
        self.conv_last.load_state_dict(net.conv_last.state_dict())
        self.classifier.load_state_dict(net.classifier.state_dict())

    def copy_weights_from_original(self, net: 'MobileNetV2'):
        super().copy_weights_from_original(net)

        self.conv_first.load_state_dict(net.conv_first.state_dict())
        self.conv_last.load_state_dict(net.conv_last.state_dict())
        self.classifier.load_state_dict(net.classifier.state_dict())

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_first(x)

        for block, downsample1, downsample2 in zip(self.blocks, self.downsamples1, self.downsamples2):
            if self.path == 0 or len(block.branches) == 1:
                out = block(out)
            else:
                x1 = block.branches[0](out)
                x1 = downsample1(x1)
                x2 = block.branches[1](out)
                x2 = downsample2(x2)
                out = torch.cat([x1, x2], 1)

        out = self.conv_last(out)

        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict, False)
    return model
