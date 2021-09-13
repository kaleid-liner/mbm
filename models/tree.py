from copy import deepcopy
import math
from functools import reduce
from torch import nn
from operator import iadd


class TreeModule(nn.Module):
    def __init__(self, nodes, actions):
        super().__init__()

        split_action = next((a for a in actions if a['type'] == 'split'), None)

        self.branches = nn.ModuleList([])
        if split_action and len(nodes) >= 2:
            mode = split_action['args']['mode']
            if mode == 'auto':
                end = math.ceil(len(nodes) / 2)
            elif mode == 'full':
                end = len(nodes)
            for _ in range(split_action['args']['num']):
                branch = []
                for node in nodes[:end]:
                    branch.append(deepcopy(node))
                self.branches.append(nn.Sequential(*branch))
        else:
            self.branches.append(nn.Sequential(*nodes))


    def forward(self, x):
        branch_outs = [branch(x) for branch in self.branches]
        out = reduce(iadd, branch_outs) * (1 / len(self.branches))
        return out
