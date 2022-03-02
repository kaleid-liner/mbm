import torch.nn as nn


class BaseMBModel(nn.Module):
    def copy_weights_from_sequential(self, net):
        for target_root, root in zip(self.trees, net.trees):
            for child in target_root.children:
                target_cur = child
                cur = root.children[0]
                while True:
                    self.idx2layer[target_cur.idx].load_state_dict(net.idx2layer[cur.idx].state_dict())
                    if target_cur.is_leaf:
                        break
                    target_cur = target_cur.children[0]
                    cur = cur.children[0]

    def copy_weights_from_original(self, net):
        for idx, layer in self.idx2layer.items():
            if idx in net.idx2layer:
                layer.load_state_dict(net.idx2layer[idx].state_dict())

    def load_state_dict_from_pretrained(self, state_dict):
        new_state_dict = {
            value: state_dict[key]
            for key, value in self.state_dict_map.items()
        }
        self.load_state_dict(new_state_dict)