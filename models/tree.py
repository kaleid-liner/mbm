from copy import deepcopy
import math
from functools import reduce
from torch import nn
from operator import iadd


class TreeModule(nn.Module):
    def __init__(self, nodes, actions, layer_mapping=None, prefix='', prefix_mapping='', lid=0):
        super().__init__()

        split_action = next((a for a in actions if a['type'] == 'split'), None)

        self.branches = nn.ModuleList([])
        if split_action and len(nodes) >= 2:
            mode = split_action['args']['mode']
            if mode == 'auto':
                end = math.ceil(len(nodes) / 2)
            elif mode == 'full':
                end = len(nodes)
            for i in range(split_action['args']['num']):
                branch = []
                for j in range(end):
                    name = prefix + '.branches.{}.{}'.format(i, j)
                    name_mapping = prefix_mapping + '.{}'.format(lid + j)
                    if layer_mapping is not None:
                        layer_mapping[name] = name_mapping
                    branch.append(deepcopy(nodes[j]))
                self.branches.append(nn.Sequential(*branch))
        else:
            for j in range(len(nodes)):
                name = prefix + '.branches.{}.{}'.format(0, j)
                name_mapping = prefix_mapping + '.{}'.format(lid + j)
                if layer_mapping is not None:
                    layer_mapping[name] = name_mapping
            self.branches.append(nn.Sequential(*nodes))


    def forward(self, x):
        branch_outs = [branch(x) for branch in self.branches]
        out = reduce(iadd, branch_outs) * (1 / len(self.branches))
        return out
