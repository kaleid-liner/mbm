import torch.nn as nn

from .tree import Tree
from models.layers import InvertedResidual


def get_layer_from_tree(tree: Tree) -> nn.Module:
    if tree.node_type == 'InvertedResidual':
        return InvertedResidual(**tree.params)
