from typing_extensions import Required
import torch.nn as nn
import torch.nn.functional as F
import torch

from ir.tree import Tree
from .utils import lstm_zero_hidden_state


class BottomUpTreeLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_size, max_n=2, type='child-sum&n-ary'):
        super().__init__()
        self.type = type.split('&')
        self.hidden_size = hidden_size
        self.max_n = max_n
        
        # define parameters
        self.iou_x = nn.Linear(input_dim, 3 * hidden_size)  # bias term here
        self.f_x = nn.Linear(input_dim, hidden_size)  # bias term here
        
        if 'child-sum' in self.type:
            # child sum
            self.iou_h_sum = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
            self.f_h_sum = nn.Linear(hidden_size, hidden_size, bias=False)
        if 'n-ary' in self.type:
            # n-ary
            self.iou_h_nArray = nn.Linear(max_n * hidden_size, 3 * hidden_size, bias=False)
            self.f_h_nArray = nn.ModuleList([
                nn.Linear(max_n * hidden_size, hidden_size, bias=False) for _ in range(max_n)
            ])
    
    @property
    def rnn_parameter_list(self):
        # 5 + max_n <weights>, 2 <bias>
        weight_parameters = [self.iou_x.weight, self.f_x.weight]
        bias_parameters = [self.iou_x.bias, self.f_x.bias]
        
        if 'child-sum' in self.type:
            weight_parameters += [self.iou_h_sum.weight, self.f_h_sum.weight]
        if 'n-ary' in self.type:
            weight_parameters += [self.iou_h_nArray.weight] + [self.f_h_nArray[k].weight for k in range(self.max_n)]
        
        return weight_parameters, bias_parameters
    
    def forward(self, op_type, input_x, child_h: list, child_c: list):
        assert op_type in self.type, 'Unsupported operation: %s' % op_type
        child_num = len(child_c)
        
        child_h = torch.cat(child_h, dim=0)  # [child_num, hidden_size]
        child_c = torch.cat(child_c, dim=0)  # [child_num, hidden_size]
        
        if op_type == 'child-sum':
            child_sum_h = torch.sum(child_h, dim=0, keepdim=True)  # [1, hidden_size]
            iou = self.iou_x(input_x) + self.iou_h_sum(child_sum_h)  # [1, 3 * hidden_size]
            i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)  # [1, hidden_size]
            i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)  # [1, hidden_size]
            
            f = F.sigmoid(
                self.f_x(input_x).repeat(child_num, 1) + self.f_h_sum(child_h)
            )  # [child_num, hidden_size]
            fc = torch.mul(f, child_c)  # [child_num, hidden_size]
            
            c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)  # [1, hidden_size]
            h = torch.mul(o, F.tanh(c))  # [1, hidden_size]
        elif op_type == 'n-ary':
            if child_num < self.max_n:
                padding = torch.zeros(self.max_n - child_num, self.hidden_size, requires_grad=True)
                child_cat_h = torch.cat([child_h, padding], dim=0)  # [max_n, hidden_size]
            else:
                child_cat_h = child_h
            
            child_cat_h = child_cat_h.view(1, -1)  # [1, max_n * hidden_size]
            iou = self.iou_x(input_x) + self.iou_h_nArray(child_cat_h)  # [1, 3 * hidden_size]
            i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)  # [1, hidden_size]
            i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)  # [1, hidden_size]
            
            fk = [
                F.sigmoid(self.f_x(input_x) + self.f_h_nArray[k](child_cat_h)) for k in range(child_num)
            ]  # each is [1, hidden_size]
            f = torch.cat(fk, dim=0)  # [child_num, hidden_size]
            fc = torch.mul(f, child_c)  # [child_num, hidden_size]
            
            c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)  # [1, hidden_size]
            h = torch.mul(o, F.tanh(c))  # [1, hidden_size]
        else:
            raise NotImplementedError
        
        return h, c


class TopDownTreeLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_size, max_n=2, type='child-sum&n-ary'):
        super().__init__()
        self.type = type.split('&')
        self.hidden_size = hidden_size
        self.max_n = max_n
        
        # define parameters
        self.iou_x = nn.Linear(input_dim, 3 * hidden_size)  # bias term here
        self.f_x = nn.Linear(input_dim, hidden_size)  # bias term here
        
        if 'child-sum' in self.type:
            self.iou_h_sum_child = nn.Linear(hidden_size, 3 * hidden_size, bias=False)  # h_child_sum
            self.iou_h_sum_parent = nn.Linear(hidden_size, 3 * hidden_size, bias=False)  # h_parent
            
            self.f_h_sum_child = nn.Linear(2 * hidden_size, hidden_size, bias=False)  # [h_parent, h_k]
            self.f_h_sum_parent = nn.Linear(2 * hidden_size, hidden_size, bias=False)  # [h_parent, h_child_sum]
        if 'n-ary' in self.type:
            self.iou_h_nArray_child = nn.Linear(max_n * hidden_size, 3 * hidden_size, bias=False)
            self.iou_h_nArray_parent = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
            
            self.f_h_nArray_child = nn.ModuleList([
                nn.Linear((max_n + 1) * hidden_size, hidden_size, bias=False) for _ in range(max_n)
            ])  # [h_parent, h_1, h_2, ..., h_N]
            self.f_h_nArray_parent = nn.Linear((max_n + 1) * hidden_size, hidden_size, bias=False)
    
    @property
    def rnn_parameter_list(self):
        # 9 + max_n <weight>, 2 <bias>
        weight_parameters = [self.iou_x.weight, self.f_x.weight]
        bias_parameters = [self.iou_x.bias, self.f_x.bias]
        
        if 'child-sum' in self.type:
            weight_parameters += [self.iou_h_sum_child.weight, self.iou_h_sum_parent.weight]
            weight_parameters += [self.f_h_sum_child.weight, self.f_h_sum_parent.weight]
        if 'n-ary' in self.type:
            weight_parameters += [self.iou_h_nArray_child.weight, self.iou_h_nArray_parent.weight]
            weight_parameters += [self.f_h_nArray_child[k].weight for k in range(self.max_n)] + \
                                 [self.f_h_nArray_parent.weight]
        
        return weight_parameters, bias_parameters
    
    def forward(self, op_type, input_x, parent_h, parent_c, child_h: list, child_c: list):
        assert op_type in self.type, 'Unsupported operation: %s' % op_type
        child_num = len(child_c)
        
        child_h = torch.cat(child_h, dim=0)  # [child_num, hidden_size]
        child_c = torch.cat(child_c, dim=0)  # [child_num, hidden_size]
        
        if op_type == 'child-sum':
            child_sum_h = torch.sum(child_h, dim=0, keepdim=True)  # [1, hidden_size], sum of all except <child_idx>
            iou = self.iou_x(input_x) + self.iou_h_sum_child(child_sum_h) + self.iou_h_sum_parent(parent_h)
            i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)  # [1, hidden_size]
            i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)  # [1, hidden_size]
            
            fk_child = [
                F.sigmoid(self.f_x(input_x) + self.f_h_sum_child(torch.cat([parent_h, child_h[k:k + 1]], dim=1)))
                for k in range(child_num)
            ]  # each is [1, hidden_size]
            f_child = torch.cat(fk_child, dim=0)  # [child_num, hidden_size]
            fc_child = torch.mul(f_child, child_c)  # [child_num, hidden_size], <child_idx> row should be zeros
            
            f_parent = F.sigmoid(
                self.f_x(input_x) + self.f_h_sum_parent(torch.cat([parent_h, child_sum_h], dim=1))
            )  # [1, hidden_size]
            fc_parent = torch.mul(f_parent, parent_c)  # [1, hidden_size]
            
            c = torch.mul(i, u) + torch.sum(fc_child, dim=0, keepdim=True) + fc_parent  # [1, hidden_size]
            h = torch.mul(o, F.tanh(c))  # [1, hidden_size]
        elif op_type == 'n-ary':
            if child_num < self.max_n:
                padding = torch.zeros(self.max_n - child_num, self.hidden_size, requires_grad=True)
                child_cat_h = torch.cat([child_h, padding], dim=0)  # [max_n, hidden_size]
            else:
                child_cat_h = child_h
            child_cat_h = child_cat_h.view(1, -1)  # [1, max_n * hidden_size]
            iou = self.iou_x(input_x) + self.iou_h_nArray_child(child_cat_h) + self.iou_h_nArray_parent(parent_h)
            i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)  # [1, hidden_size]
            i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)  # [1, hidden_size]
            
            parent_child_cat_h = torch.cat([parent_h, child_cat_h], dim=1)  # [1, (max_n + 1) * hidden_size]
            fk_child = [
                F.sigmoid(self.f_x(input_x) + self.f_h_nArray_child[k](parent_child_cat_h)) for k in range(child_num)
            ]  # each is [1, hidden_size]
            f_child = torch.cat(fk_child, dim=0)  # [child_num, hidden_size]
            fc_child = torch.mul(f_child, child_c)  # [child_num, hidden_size], <child_idx> row should be zeros
            
            f_parent = F.sigmoid(
                self.f_x(input_x) + self.f_h_nArray_parent(parent_child_cat_h)
            )  # [1, hidden_size]
            fc_parent = torch.mul(f_parent, parent_c)  # [1, hidden_size]
            
            c = torch.mul(i, u) + torch.sum(fc_child, dim=0, keepdim=True) + fc_parent  # [1, hidden_size]
            h = torch.mul(o, F.tanh(c))  # [1, hidden_size]
        else:
            raise NotImplementedError
        
        return h, c


class TreeEncoderNet(nn.Module):
    def __init__(self, merge_vocab, compute_vocab, max_n, embedding_dim, hidden_size, bidirectional=True):
        super().__init__()
        
        self.merge_vocab = merge_vocab
        self.compute_vocab = compute_vocab
        
        self.max_n = max_n  # max children number for N-ary case
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # embedding layer
        self.merge_embedding = nn.Embedding(self.merge_vocab.size, self.embedding_dim, self.merge_vocab.pad_code)
        self.compute_embedding = nn.Embedding(self.compute_vocab.size, self.embedding_dim, self.compute_vocab.pad_code)
        
        # RNN parameters
        self.bottom_up_merge_lstm = \
            BottomUpTreeLSTMCell(self.embedding_dim, self.hidden_size, self.max_n, _type='child-sum&n-ary')
        self.bottom_up_compute_lstm = nn.LSTMCell(self.embedding_dim, self.hidden_size)
        
        if self.bidirectional:
            self.top_down_merge_lstm = \
                TopDownTreeLSTMCell(self.embedding_dim, self.hidden_size, self.max_n, _type='child-sum&n-ary')
            self.top_down_compute_lstm = nn.LSTMCell(self.embedding_dim, self.hidden_size)
    
    def init_embedding(self, scheme: dict):
        _type = scheme.get('type', 'uniform')
        if _type == 'uniform':
            _min = scheme.get('min', -0.1)
            _max = scheme.get('max', 0.1)
            nn.init.uniform(self.merge_embedding.weight, _min, _max)
            nn.init.uniform(self.compute_embedding.weight, _min, _max)
        elif _type == 'normal':
            _mean = scheme.get('mean', 0)
            _std = scheme.get('std', 1)
            nn.init.normal(self.merge_embedding.weight, _mean, _std)
            nn.init.normal(self.compute_embedding.weight, _mean, _std)
        elif _type == 'default':
            pass
        else:
            raise NotImplementedError
        if self.merge_embedding.padding_idx is not None:
            self.merge_embedding.weight.data[self.merge_embedding.padding_idx].fill_(0)
        if self.compute_embedding.padding_idx is not None:
            self.compute_embedding.weight.data[self.compute_embedding.padding_idx].fill_(0)
    
    @property
    def rnn_parameter_list(self):
        weight_parameters, bias_parameters = self.bottom_up_merge_lstm.rnn_parameter_list
        
        weight_parameters += [
            self.bottom_up_compute_lstm.weight_hh, self.bottom_up_compute_lstm.weight_ih
        ]
        if self.bottom_up_merge_lstm.bias:
            bias_parameters += [
                self.bottom_up_compute_lstm.bias_hh, self.bottom_up_compute_lstm.bias_ih
            ]
        
        if self.bidirectional:
            top_down_weight_parameters, top_down_bias_parameters = self.top_down_merge_lstm.rnn_parameter_list
            weight_parameters += top_down_weight_parameters
            bias_parameters += top_down_bias_parameters
            weight_parameters += [
                self.top_down_compute_lstm.weight_hh, self.top_down_compute_lstm.weight_ih
            ]
            if self.top_down_compute_lstm.bias:
                bias_parameters += [
                    self.top_down_compute_lstm.bias_hh, self.top_down_compute_lstm.bias_ih
                ]
        return weight_parameters, bias_parameters
    
    def init_cell(self, scheme: dict):
        _type = scheme.get('type', 'default')
        weight_parameters, bias_parameters = self.rnn_parameter_list
        if _type == 'uniform':
            _min, _max = scheme.get('min', -0.1), scheme.get('max', 0.1)
            for weight in weight_parameters:
                nn.init.uniform(weight, _min, _max)
            for bias in bias_parameters:
                nn.init.uniform(bias, _min, _max)
        elif _type == 'normal':
            _mean, _std = scheme.get('mean', 0), scheme.get('std', 1)
            for weight in weight_parameters:
                nn.init.normal(weight, _mean, _std)
            for bias in bias_parameters:
                nn.init.normal(bias, _mean, _std)
        elif _type == 'orthogonal':
            for weight in weight_parameters:
                nn.init.orthogonal(weight)
            for bias in bias_parameters:
                nn.init.constant(bias, 0.0)
        elif _type == 'default':
            pass
        else:
            raise NotImplementedError
    
    def _zero_hidden_state(self):
        return lstm_zero_hidden_state(self.hidden_size)
    
    def _bottom_up(self, tree: Tree):
        if tree.is_leaf:
            child_h, child_c = self._zero_hidden_state()  # [1, hidden_size]
            child_h, child_c = [child_h], [child_c]
        else:
            child_h, child_c = [], []
            for child in tree.children:
                self._bottom_up(child)
                bottom_up_state = child.state[0]  # (h, c)
                child_h.append(bottom_up_state[0])
                child_c.append(bottom_up_state[1])
        
        if tree.is_merge_type:
            merge_idx = self.merge_vocab.get_code(tree.node_type)
            input_node = torch.LongTensor([merge_idx], requires_grad=True)
            input_node = self.merge_embedding(input_node)  # [1, embedding_dim]
            h, c = self.bottom_up_merge_lstm('n-ary', input_node, child_h, child_c)  # [1, hidden_size]
        else:
            compute_idx = self.compute_vocab.get_code(tree.node_type)  # int
            input_edge = torch.LongTensor([compute_idx], requires_grad=True)
            input_edge = self.compute_embedding(input_edge)  # [1, embedding_dim]
            h, c = self.bottom_up_compute_lstm(input_edge, (child_h[0], child_c[0]))  # [1, hidden_size]
            
        tree.state = [(h, c), None]
    
    def _top_down(self, tree: Tree):
        if tree.is_root:
            h, c = self._zero_hidden_state()  # h: [1, hidden_size], c: [1, hidden_size]
        else:
            parent = tree.parent
            parent_h, parent_c = parent.state[1]  # top_down_state of parent
            
            child_h, child_c = [], []
            for child in parent.children:
                if child.idx == tree.idx:
                    child_bottom_up_state = lstm_zero_hidden_state(self.hidden_size)
                else:
                    child_bottom_up_state = child.state[0]
                child_h.append(child_bottom_up_state[0])
                child_c.append(child_bottom_up_state[1])

            if parent.is_merge_type:
                merge_idx = self.merge_vocab.get_code(parent.node_type)
                input_node = torch.LongTensor([merge_idx], requires_grad=True)
                input_node = self.merge_embedding(input_node)  # [1, embedding_dim]
                h, c = self.top_down_merge_lstm('n-ary', input_node, parent_h, parent_c, child_h, child_c)
            else:
                compute_idx = self.edge_vocab.get_code(tree.edge)  # int
                input_edge = torch.LongTensor([compute_idx], requires_grad=True)
                input_edge = self.compute_embedding(input_edge)  # [1, embedding_dim]
                h, c = self.top_down_compute_lstm(input_edge, (parent_h, parent_c))  # [1, hidden_size]

        tree.state[1] = (h, c)
        for child in tree.children:
            self._top_down(child)
    
    def forward(self, tree: Tree):
        self._bottom_up(tree)
        
        if self.bidirectional:
            self._top_down(tree)
        return tree.get_output()
