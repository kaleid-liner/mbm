from typing import Optional, List, Dict
import torch


class Tree:
    global_idx = 0

    def __init__(self, parent: Optional['Tree'], children: List['Tree'], node_type: str, params: Dict):
        self.parent = parent
        self.children = children
        self.node_type = node_type
        self.idx = self.global_idx
        self.global_idx += 1
        self.state = None
        self.params = params

    @property
    def is_merge_type(self):
        return self.node_type == 'copy'

    @property
    def child_num(self):
        return len(self.children)

    @property
    def is_leaf(self):
        return self.child_num == 0

    @property
    def is_root(self):
        return self.parent is None

    def get_state(self):
        if self.state is not None:
            bottom_up_state, top_down_state = self.state
            if top_down_state is None or self.is_root:
                return bottom_up_state  # [1, hidden_size]
            else:
                h = torch.cat([bottom_up_state[0], top_down_state[0]], dim=1)  # [1, 2 * hidden_size]
                c = torch.cat([bottom_up_state[1], top_down_state[1]], dim=1)  # [1, 2 * hidden_size]
                return h, c
        else:
            return None
    
    def get_output(self):
        state = self.get_state()
        if state is None:
            return None
        else:
            return state[0]
        