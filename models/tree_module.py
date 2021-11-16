from functools import reduce
from operator import iadd

import torch.nn as nn

from ir.tree import Tree
from ir.utils import get_layer_from_tree


class TreeModule(nn.Module):
    def __init__(self, root: Tree):
        super().__init__()

        assert root.is_root
        branches = []
        self.idx2layer = {}

        for child in root.children:
            branch = []
            cur = child
            while True:
                layer = get_layer_from_tree(cur)
                branch.append(layer)
                self.idx2layer[cur.idx] = layer
                if cur.is_leaf:
                    break
                cur = cur.children[0]
            branches.append(branch)

        self.branches = nn.ModuleList([nn.Sequential(*branch) for branch in branches])
        self.branch_num = len(branches)

    def forward(self, x):
        if self.branch_num > 1:
            branch_outs = [branch(x) for branch in self.branches]
            out = reduce(iadd, branch_outs) * (1 / len(self.branches))
        else:
            out = self.branches[0](x)
        return out
