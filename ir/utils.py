import torch.nn as nn

from .tree import Tree
from models.layers import InvertedResidual, get_identity_layer, ResidualBlock, ConvBNActivation


def get_layer_from_tree(tree: Tree) -> nn.Module:
    if tree.node_type == 'InvertedResidual':
        return InvertedResidual(**tree.params)
    elif tree.node_type == 'ResidualBlock':
        return ResidualBlock(**tree.params)
    elif tree.node_type == 'ConvBNActivation':
        return ConvBNActivation(**tree.params)
    else:
        return get_identity_layer(tree.node_type, tree.output_channel, tree.output_channel)
