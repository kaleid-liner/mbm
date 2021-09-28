from copy import deepcopy
import math
from functools import reduce
from torch import nn
from operator import iadd

from .layers import get_identity_layer


def predict_index(branch_id, layer_id, actions):
    n_inserted_before = len([
        action for action in actions
        if (
            action['type'] == 'insert'
            and action['args']['index'][0] == branch_id
            and action['args']['index'][1] <= layer_id
        )
    ])
    return layer_id + n_inserted_before


class TreeModule(nn.Module):
    def __init__(self, nodes, actions, layer_mapping=None, prefix='', prefix_mapping='', lid=0):
        super().__init__()

        split_action = next((a for a in actions if a['type'] == 'split'), None)

        # split branches
        branches = []
        if split_action and len(nodes) >= 2:
            mode = split_action['args']['mode']
            if mode == 'floor':
                end = math.floor(len(nodes) / 2)
            elif mode == 'ceil':
                end = math.ceil(len(nodes) / 2)
            elif mode == 'full':
                end = len(nodes)
            num_branch = split_action['args']['num']
        else:
            end = len(nodes)
            num_branch = 1

        for i in range(num_branch):
            branch = []
            for j in range(end):
                name = prefix + '.branches.{}.{}'.format(i, predict_index(i, j, actions))
                name_mapping = prefix_mapping + '.{}'.format(lid + j)
                if layer_mapping is not None:
                    layer_mapping[name] = name_mapping
                branch.append(deepcopy(nodes[j]))
            branches.append(branch)

        for action in actions:
            if action['type'] == 'insert':
                branch_id, depth = action['args']['index']
                if branch_id >= len(branches):
                    continue
                if depth >= len(branches[branch_id]):
                    in_channels = out_channels = branches[branch_id][len(branches[branch_id]) - 1].out_channels
                else:
                    in_channels = out_channels = branches[branch_id][depth].in_channels
                branches[branch_id].insert(depth, get_identity_layer(action['args']['node'], in_channels, out_channels))

        self.branches = nn.ModuleList([nn.Sequential(*branch) for branch in branches])
                
    def forward(self, x):
        branch_outs = [branch(x) for branch in self.branches]
        out = reduce(iadd, branch_outs) * (1 / len(self.branches))
        return out
