from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

from ir.tree import Tree


def build_linear_seq(in_dim, out_dim, config_list):
    layer_seq = OrderedDict()
    dim = in_dim
    for _i, layer_config in enumerate(config_list):
        layer_seq['linear_%d' % _i] = nn.Linear(dim, layer_config['units'], bias=layer_config['use_bias'])
        if layer_config['activation'] == 'sigmoid':
            layer_seq['sigmoid_%d' % _i] = nn.Sigmoid()
        elif layer_config['activation'] == 'tanh':
            layer_seq['tanh_%d' % _i] = nn.Tanh()
        elif layer_config['activation'] == 'relu':
            layer_seq['relu_%d' % _i] = nn.ReLU(inplace=True)
        dim = layer_config['units']
    layer_seq['final_linear'] = nn.Linear(dim, out_dim)
    return nn.Sequential(layer_seq)


def init_linear_seq(linear_seq: nn.Sequential, scheme: dict):
    for module in linear_seq.modules():
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.zero_()
            type = scheme.get('type', 'default')
            if type == 'uniform':
                _min, _max = scheme.get('min', -0.1), scheme.get('max', 0.1)
                module.weight.data.uniform_(_min, _max)
            elif type == 'normal':
                _mean, _std = scheme.get('mean', 0), scheme.get('std', 0.1)
                module.weight.data.normal_(_mean, _std)
            elif type == 'default':
                pass
            else:
                raise NotImplementedError


class RemoveActor(nn.Module):
    def __init__(self, encoder_hidden_size, actor_config):
        super().__init__()
        
        self.encoder_hidden_size = encoder_hidden_size
        self.actor_config = actor_config
        
        # define parameters
        self.remove_decider = build_linear_seq(self.encoder_hidden_size, 1, self.actor_config)
    
    def init_actor(self, scheme: dict):
        init_linear_seq(self.expand_decider, scheme)
    
    def forward(self, tree: Tree, decision=None):
        if decision:
            return self._known_decision_forward(tree, decision)
        
        state = tree.get_output()  # [1, hidden_size]
        remove_logits = self.remove_decider(state)  # [1, 1]
        remove_probs = F.sigmoid(remove_logits)  # [1, 1]
        remove_probs = torch.cat(
            [1 - remove_probs, remove_probs], dim=1,
        )  # [1, 2]
        
        # sample a decision
        remove_decision = torch.multinomial(remove_probs, 1, replacement=True)  # [1, 1]
        remove_decision = remove_decision.data.numpy()[0, 0]  # int
        
        return remove_decision, remove_probs
    
    def _known_decision_forward(self, tree, decision):
        _, probs = self.forward(tree)
        return decision, probs
    
    @staticmethod
    def random_decision():
        remove_decision = np.random.randint(0, 2)
        return remove_decision


class InsertActor(nn.Module):
    def __init__(self, compute_candidates, encoder_hidden_size, actor_config):
        super().__init__()
        
        self.compute_candidates = compute_candidates
        self.encoder_hidden_size = encoder_hidden_size
        self.actor_config = actor_config
        
        # define parameters
        self.compute_decider = build_linear_seq(self.encoder_hidden_size, len(self.compute_candidates), self.actor_config)
    
    def init_actor(self, scheme: dict):
        init_linear_seq(self.compute_decider, scheme)
    
    def forward(self, tree: Tree, decision=None):
        if decision:
            return self._known_decision_forward(tree, decision)
        state = tree.get_output()  # [1, hidden_size]
        compute_logits = self.compute_decider(state)  # [1, edge_candidates_num]
        compute_probs = F.softmax(compute_logits, dim=1)  # [1, edge_candidates_num]
        
        # sample a decision
        compute_decision = torch.multinomial(compute_probs, 1, replacement=True)  # [1, 1]
        compute_decision = compute_decision.data.numpy()[0, 0]  # int
        
        return compute_decision, compute_probs
    
    def _known_decision_forward(self, tree, decision):
        _, probs = self.forward(tree)
        return decision, probs
    
    def random_decision(self):
        return np.random.randint(0, len(self.compute_candidates))
