'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/resnet.py
BSD 3-Clause License
Copyright (c) Soumith Chintala 2016,
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import sys
import torch.nn as nn
import torch

from models.base_mbmodel import BaseMBModel
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional
from ir.tree import Tree
from .tree_module import TreeModule
from .layers import ResidualBlock, conv3x3


cifar10_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
}

cifar100_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt',
}


class CifarResNet(BaseMBModel):
    state_dict_map = {
        '20': {
            'conv1.weight': 'conv1.weight', 'bn1.weight': 'bn1.weight', 'bn1.bias': 'bn1.bias', 'bn1.running_mean': 'bn1.running_mean', 'bn1.running_var': 'bn1.running_var', 'bn1.num_batches_tracked': 'bn1.num_batches_tracked', 'layer1.0.conv1.weight': 'blocks.0.branches.0.0.conv1.weight', 'layer1.0.bn1.weight': 'blocks.0.branches.0.0.bn1.weight', 'layer1.0.bn1.bias': 'blocks.0.branches.0.0.bn1.bias', 'layer1.0.bn1.running_mean': 'blocks.0.branches.0.0.bn1.running_mean', 'layer1.0.bn1.running_var': 'blocks.0.branches.0.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked': 'blocks.0.branches.0.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight': 'blocks.0.branches.0.0.conv2.weight', 'layer1.0.bn2.weight': 'blocks.0.branches.0.0.bn2.weight', 'layer1.0.bn2.bias': 'blocks.0.branches.0.0.bn2.bias', 'layer1.0.bn2.running_mean': 'blocks.0.branches.0.0.bn2.running_mean', 'layer1.0.bn2.running_var': 'blocks.0.branches.0.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked': 'blocks.0.branches.0.0.bn2.num_batches_tracked', 'layer1.1.conv1.weight': 'blocks.0.branches.0.1.conv1.weight', 'layer1.1.bn1.weight': 'blocks.0.branches.0.1.bn1.weight', 'layer1.1.bn1.bias': 'blocks.0.branches.0.1.bn1.bias', 'layer1.1.bn1.running_mean': 'blocks.0.branches.0.1.bn1.running_mean', 'layer1.1.bn1.running_var': 'blocks.0.branches.0.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked': 'blocks.0.branches.0.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight': 'blocks.0.branches.0.1.conv2.weight', 'layer1.1.bn2.weight': 'blocks.0.branches.0.1.bn2.weight', 'layer1.1.bn2.bias': 'blocks.0.branches.0.1.bn2.bias', 'layer1.1.bn2.running_mean': 'blocks.0.branches.0.1.bn2.running_mean', 'layer1.1.bn2.running_var': 'blocks.0.branches.0.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked': 'blocks.0.branches.0.1.bn2.num_batches_tracked', 'layer1.2.conv1.weight': 'blocks.0.branches.0.2.conv1.weight', 'layer1.2.bn1.weight': 'blocks.0.branches.0.2.bn1.weight', 'layer1.2.bn1.bias': 'blocks.0.branches.0.2.bn1.bias', 'layer1.2.bn1.running_mean': 'blocks.0.branches.0.2.bn1.running_mean', 'layer1.2.bn1.running_var': 'blocks.0.branches.0.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked': 'blocks.0.branches.0.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight': 'blocks.0.branches.0.2.conv2.weight', 'layer1.2.bn2.weight': 'blocks.0.branches.0.2.bn2.weight', 'layer1.2.bn2.bias': 'blocks.0.branches.0.2.bn2.bias', 'layer1.2.bn2.running_mean': 'blocks.0.branches.0.2.bn2.running_mean', 'layer1.2.bn2.running_var': 'blocks.0.branches.0.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked': 'blocks.0.branches.0.2.bn2.num_batches_tracked', 'layer2.0.conv1.weight': 'blocks.1.branches.0.0.conv1.weight', 'layer2.0.bn1.weight': 'blocks.1.branches.0.0.bn1.weight', 'layer2.0.bn1.bias': 'blocks.1.branches.0.0.bn1.bias', 'layer2.0.bn1.running_mean': 'blocks.1.branches.0.0.bn1.running_mean', 'layer2.0.bn1.running_var': 'blocks.1.branches.0.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked': 'blocks.1.branches.0.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight': 'blocks.1.branches.0.0.conv2.weight', 'layer2.0.bn2.weight': 'blocks.1.branches.0.0.bn2.weight', 'layer2.0.bn2.bias': 'blocks.1.branches.0.0.bn2.bias', 'layer2.0.bn2.running_mean': 'blocks.1.branches.0.0.bn2.running_mean', 'layer2.0.bn2.running_var': 'blocks.1.branches.0.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked': 'blocks.1.branches.0.0.bn2.num_batches_tracked', 'layer2.0.downsample.0.weight': 'blocks.1.branches.0.0.downsample.0.weight', 'layer2.0.downsample.1.weight': 'blocks.1.branches.0.0.downsample.1.weight', 'layer2.0.downsample.1.bias': 'blocks.1.branches.0.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean': 'blocks.1.branches.0.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var': 'blocks.1.branches.0.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked': 'blocks.1.branches.0.0.downsample.1.num_batches_tracked', 'layer2.1.conv1.weight': 'blocks.1.branches.0.1.conv1.weight', 'layer2.1.bn1.weight': 'blocks.1.branches.0.1.bn1.weight', 'layer2.1.bn1.bias': 'blocks.1.branches.0.1.bn1.bias', 'layer2.1.bn1.running_mean': 'blocks.1.branches.0.1.bn1.running_mean', 'layer2.1.bn1.running_var': 'blocks.1.branches.0.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked': 'blocks.1.branches.0.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight': 'blocks.1.branches.0.1.conv2.weight', 'layer2.1.bn2.weight': 'blocks.1.branches.0.1.bn2.weight', 'layer2.1.bn2.bias': 'blocks.1.branches.0.1.bn2.bias', 'layer2.1.bn2.running_mean': 'blocks.1.branches.0.1.bn2.running_mean', 'layer2.1.bn2.running_var': 'blocks.1.branches.0.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked': 'blocks.1.branches.0.1.bn2.num_batches_tracked', 'layer2.2.conv1.weight': 'blocks.1.branches.0.2.conv1.weight', 'layer2.2.bn1.weight': 'blocks.1.branches.0.2.bn1.weight', 'layer2.2.bn1.bias': 'blocks.1.branches.0.2.bn1.bias', 'layer2.2.bn1.running_mean': 'blocks.1.branches.0.2.bn1.running_mean', 'layer2.2.bn1.running_var': 'blocks.1.branches.0.2.bn1.running_var', 'layer2.2.bn1.num_batches_tracked': 'blocks.1.branches.0.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight': 'blocks.1.branches.0.2.conv2.weight', 'layer2.2.bn2.weight': 'blocks.1.branches.0.2.bn2.weight', 'layer2.2.bn2.bias': 'blocks.1.branches.0.2.bn2.bias', 'layer2.2.bn2.running_mean': 'blocks.1.branches.0.2.bn2.running_mean', 'layer2.2.bn2.running_var': 'blocks.1.branches.0.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked': 'blocks.1.branches.0.2.bn2.num_batches_tracked', 'layer3.0.conv1.weight': 'blocks.2.branches.0.0.conv1.weight', 'layer3.0.bn1.weight': 'blocks.2.branches.0.0.bn1.weight', 'layer3.0.bn1.bias': 'blocks.2.branches.0.0.bn1.bias', 'layer3.0.bn1.running_mean': 'blocks.2.branches.0.0.bn1.running_mean', 'layer3.0.bn1.running_var': 'blocks.2.branches.0.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked': 'blocks.2.branches.0.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight': 'blocks.2.branches.0.0.conv2.weight', 'layer3.0.bn2.weight': 'blocks.2.branches.0.0.bn2.weight', 'layer3.0.bn2.bias': 'blocks.2.branches.0.0.bn2.bias', 'layer3.0.bn2.running_mean': 'blocks.2.branches.0.0.bn2.running_mean', 'layer3.0.bn2.running_var': 'blocks.2.branches.0.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked': 'blocks.2.branches.0.0.bn2.num_batches_tracked', 'layer3.0.downsample.0.weight': 'blocks.2.branches.0.0.downsample.0.weight', 'layer3.0.downsample.1.weight': 'blocks.2.branches.0.0.downsample.1.weight', 'layer3.0.downsample.1.bias': 'blocks.2.branches.0.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean': 'blocks.2.branches.0.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var': 'blocks.2.branches.0.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked': 'blocks.2.branches.0.0.downsample.1.num_batches_tracked', 'layer3.1.conv1.weight': 'blocks.2.branches.0.1.conv1.weight', 'layer3.1.bn1.weight': 'blocks.2.branches.0.1.bn1.weight', 'layer3.1.bn1.bias': 'blocks.2.branches.0.1.bn1.bias', 'layer3.1.bn1.running_mean': 'blocks.2.branches.0.1.bn1.running_mean', 'layer3.1.bn1.running_var': 'blocks.2.branches.0.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked': 'blocks.2.branches.0.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight': 'blocks.2.branches.0.1.conv2.weight', 'layer3.1.bn2.weight': 'blocks.2.branches.0.1.bn2.weight', 'layer3.1.bn2.bias': 'blocks.2.branches.0.1.bn2.bias', 'layer3.1.bn2.running_mean': 'blocks.2.branches.0.1.bn2.running_mean', 'layer3.1.bn2.running_var': 'blocks.2.branches.0.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked': 'blocks.2.branches.0.1.bn2.num_batches_tracked', 'layer3.2.conv1.weight': 'blocks.2.branches.0.2.conv1.weight', 'layer3.2.bn1.weight': 'blocks.2.branches.0.2.bn1.weight', 'layer3.2.bn1.bias': 'blocks.2.branches.0.2.bn1.bias', 'layer3.2.bn1.running_mean': 'blocks.2.branches.0.2.bn1.running_mean', 'layer3.2.bn1.running_var': 'blocks.2.branches.0.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked': 'blocks.2.branches.0.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight': 'blocks.2.branches.0.2.conv2.weight', 'layer3.2.bn2.weight': 'blocks.2.branches.0.2.bn2.weight', 'layer3.2.bn2.bias': 'blocks.2.branches.0.2.bn2.bias', 'layer3.2.bn2.running_mean': 'blocks.2.branches.0.2.bn2.running_mean', 'layer3.2.bn2.running_var': 'blocks.2.branches.0.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked': 'blocks.2.branches.0.2.bn2.num_batches_tracked', 'fc.weight': 'fc.weight', 'fc.bias': 'fc.bias'
        },
        '32': {
            'conv1.weight': 'conv1.weight', 'bn1.weight': 'bn1.weight', 'bn1.bias': 'bn1.bias', 'bn1.running_mean': 'bn1.running_mean', 'bn1.running_var': 'bn1.running_var', 'bn1.num_batches_tracked': 'bn1.num_batches_tracked', 'layer1.0.conv1.weight': 'blocks.0.branches.0.0.conv1.weight', 'layer1.0.bn1.weight': 'blocks.0.branches.0.0.bn1.weight', 'layer1.0.bn1.bias': 'blocks.0.branches.0.0.bn1.bias', 'layer1.0.bn1.running_mean': 'blocks.0.branches.0.0.bn1.running_mean', 'layer1.0.bn1.running_var': 'blocks.0.branches.0.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked': 'blocks.0.branches.0.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight': 'blocks.0.branches.0.0.conv2.weight', 'layer1.0.bn2.weight': 'blocks.0.branches.0.0.bn2.weight', 'layer1.0.bn2.bias': 'blocks.0.branches.0.0.bn2.bias', 'layer1.0.bn2.running_mean': 'blocks.0.branches.0.0.bn2.running_mean', 'layer1.0.bn2.running_var': 'blocks.0.branches.0.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked': 'blocks.0.branches.0.0.bn2.num_batches_tracked', 'layer1.1.conv1.weight': 'blocks.0.branches.0.1.conv1.weight', 'layer1.1.bn1.weight': 'blocks.0.branches.0.1.bn1.weight', 'layer1.1.bn1.bias': 'blocks.0.branches.0.1.bn1.bias', 'layer1.1.bn1.running_mean': 'blocks.0.branches.0.1.bn1.running_mean', 'layer1.1.bn1.running_var': 'blocks.0.branches.0.1.bn1.running_var', 'layer1.1.bn1.num_batches_tracked': 'blocks.0.branches.0.1.bn1.num_batches_tracked', 'layer1.1.conv2.weight': 'blocks.0.branches.0.1.conv2.weight', 'layer1.1.bn2.weight': 'blocks.0.branches.0.1.bn2.weight', 'layer1.1.bn2.bias': 'blocks.0.branches.0.1.bn2.bias', 'layer1.1.bn2.running_mean': 'blocks.0.branches.0.1.bn2.running_mean', 'layer1.1.bn2.running_var': 'blocks.0.branches.0.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked': 'blocks.0.branches.0.1.bn2.num_batches_tracked', 'layer1.2.conv1.weight': 'blocks.0.branches.0.2.conv1.weight', 'layer1.2.bn1.weight': 'blocks.0.branches.0.2.bn1.weight', 'layer1.2.bn1.bias': 'blocks.0.branches.0.2.bn1.bias', 'layer1.2.bn1.running_mean': 'blocks.0.branches.0.2.bn1.running_mean', 'layer1.2.bn1.running_var': 'blocks.0.branches.0.2.bn1.running_var', 'layer1.2.bn1.num_batches_tracked': 'blocks.0.branches.0.2.bn1.num_batches_tracked', 'layer1.2.conv2.weight': 'blocks.0.branches.0.2.conv2.weight', 'layer1.2.bn2.weight': 'blocks.0.branches.0.2.bn2.weight', 'layer1.2.bn2.bias': 'blocks.0.branches.0.2.bn2.bias', 'layer1.2.bn2.running_mean': 'blocks.0.branches.0.2.bn2.running_mean', 'layer1.2.bn2.running_var': 'blocks.0.branches.0.2.bn2.running_var', 'layer1.2.bn2.num_batches_tracked': 'blocks.0.branches.0.2.bn2.num_batches_tracked', 'layer1.3.conv1.weight': 'blocks.0.branches.0.3.conv1.weight', 'layer1.3.bn1.weight': 'blocks.0.branches.0.3.bn1.weight', 'layer1.3.bn1.bias': 'blocks.0.branches.0.3.bn1.bias', 'layer1.3.bn1.running_mean': 'blocks.0.branches.0.3.bn1.running_mean', 'layer1.3.bn1.running_var': 'blocks.0.branches.0.3.bn1.running_var', 'layer1.3.bn1.num_batches_tracked': 'blocks.0.branches.0.3.bn1.num_batches_tracked', 'layer1.3.conv2.weight': 'blocks.0.branches.0.3.conv2.weight', 'layer1.3.bn2.weight': 'blocks.0.branches.0.3.bn2.weight', 'layer1.3.bn2.bias': 'blocks.0.branches.0.3.bn2.bias', 'layer1.3.bn2.running_mean': 'blocks.0.branches.0.3.bn2.running_mean', 'layer1.3.bn2.running_var': 'blocks.0.branches.0.3.bn2.running_var', 'layer1.3.bn2.num_batches_tracked': 'blocks.0.branches.0.3.bn2.num_batches_tracked', 'layer1.4.conv1.weight': 'blocks.0.branches.0.4.conv1.weight', 'layer1.4.bn1.weight': 'blocks.0.branches.0.4.bn1.weight', 'layer1.4.bn1.bias': 'blocks.0.branches.0.4.bn1.bias', 'layer1.4.bn1.running_mean': 'blocks.0.branches.0.4.bn1.running_mean', 'layer1.4.bn1.running_var': 'blocks.0.branches.0.4.bn1.running_var', 'layer1.4.bn1.num_batches_tracked': 'blocks.0.branches.0.4.bn1.num_batches_tracked', 'layer1.4.conv2.weight': 'blocks.0.branches.0.4.conv2.weight', 'layer1.4.bn2.weight': 'blocks.0.branches.0.4.bn2.weight', 'layer1.4.bn2.bias': 'blocks.0.branches.0.4.bn2.bias', 'layer1.4.bn2.running_mean': 'blocks.0.branches.0.4.bn2.running_mean', 'layer1.4.bn2.running_var': 'blocks.0.branches.0.4.bn2.running_var', 'layer1.4.bn2.num_batches_tracked': 'blocks.0.branches.0.4.bn2.num_batches_tracked', 'layer2.0.conv1.weight': 'blocks.1.branches.0.0.conv1.weight', 'layer2.0.bn1.weight': 'blocks.1.branches.0.0.bn1.weight', 'layer2.0.bn1.bias': 'blocks.1.branches.0.0.bn1.bias', 'layer2.0.bn1.running_mean': 'blocks.1.branches.0.0.bn1.running_mean', 'layer2.0.bn1.running_var': 'blocks.1.branches.0.0.bn1.running_var', 'layer2.0.bn1.num_batches_tracked': 'blocks.1.branches.0.0.bn1.num_batches_tracked', 'layer2.0.conv2.weight': 'blocks.1.branches.0.0.conv2.weight', 'layer2.0.bn2.weight': 'blocks.1.branches.0.0.bn2.weight', 'layer2.0.bn2.bias': 'blocks.1.branches.0.0.bn2.bias', 'layer2.0.bn2.running_mean': 'blocks.1.branches.0.0.bn2.running_mean', 'layer2.0.bn2.running_var': 'blocks.1.branches.0.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked': 'blocks.1.branches.0.0.bn2.num_batches_tracked', 'layer2.0.downsample.0.weight': 'blocks.1.branches.0.0.downsample.0.weight', 'layer2.0.downsample.1.weight': 'blocks.1.branches.0.0.downsample.1.weight', 'layer2.0.downsample.1.bias': 'blocks.1.branches.0.0.downsample.1.bias', 'layer2.0.downsample.1.running_mean': 'blocks.1.branches.0.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var': 'blocks.1.branches.0.0.downsample.1.running_var', 'layer2.0.downsample.1.num_batches_tracked': 'blocks.1.branches.0.0.downsample.1.num_batches_tracked', 'layer2.1.conv1.weight': 'blocks.1.branches.0.1.conv1.weight', 'layer2.1.bn1.weight': 'blocks.1.branches.0.1.bn1.weight', 'layer2.1.bn1.bias': 'blocks.1.branches.0.1.bn1.bias', 'layer2.1.bn1.running_mean': 'blocks.1.branches.0.1.bn1.running_mean', 'layer2.1.bn1.running_var': 'blocks.1.branches.0.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked': 'blocks.1.branches.0.1.bn1.num_batches_tracked', 'layer2.1.conv2.weight': 'blocks.1.branches.0.1.conv2.weight', 'layer2.1.bn2.weight': 'blocks.1.branches.0.1.bn2.weight', 'layer2.1.bn2.bias': 'blocks.1.branches.0.1.bn2.bias', 'layer2.1.bn2.running_mean': 'blocks.1.branches.0.1.bn2.running_mean', 'layer2.1.bn2.running_var': 'blocks.1.branches.0.1.bn2.running_var', 'layer2.1.bn2.num_batches_tracked': 'blocks.1.branches.0.1.bn2.num_batches_tracked', 'layer2.2.conv1.weight': 'blocks.1.branches.0.2.conv1.weight', 'layer2.2.bn1.weight': 'blocks.1.branches.0.2.bn1.weight', 'layer2.2.bn1.bias': 'blocks.1.branches.0.2.bn1.bias', 'layer2.2.bn1.running_mean': 'blocks.1.branches.0.2.bn1.running_mean', 'layer2.2.bn1.running_var': 'blocks.1.branches.0.2.bn1.running_var', 'layer2.2.bn1.num_batches_tracked': 'blocks.1.branches.0.2.bn1.num_batches_tracked', 'layer2.2.conv2.weight': 'blocks.1.branches.0.2.conv2.weight', 'layer2.2.bn2.weight': 'blocks.1.branches.0.2.bn2.weight', 'layer2.2.bn2.bias': 'blocks.1.branches.0.2.bn2.bias', 'layer2.2.bn2.running_mean': 'blocks.1.branches.0.2.bn2.running_mean', 'layer2.2.bn2.running_var': 'blocks.1.branches.0.2.bn2.running_var', 'layer2.2.bn2.num_batches_tracked': 'blocks.1.branches.0.2.bn2.num_batches_tracked', 'layer2.3.conv1.weight': 'blocks.1.branches.0.3.conv1.weight', 'layer2.3.bn1.weight': 'blocks.1.branches.0.3.bn1.weight', 'layer2.3.bn1.bias': 'blocks.1.branches.0.3.bn1.bias', 'layer2.3.bn1.running_mean': 'blocks.1.branches.0.3.bn1.running_mean', 'layer2.3.bn1.running_var': 'blocks.1.branches.0.3.bn1.running_var', 'layer2.3.bn1.num_batches_tracked': 'blocks.1.branches.0.3.bn1.num_batches_tracked', 'layer2.3.conv2.weight': 'blocks.1.branches.0.3.conv2.weight', 'layer2.3.bn2.weight': 'blocks.1.branches.0.3.bn2.weight', 'layer2.3.bn2.bias': 'blocks.1.branches.0.3.bn2.bias', 'layer2.3.bn2.running_mean': 'blocks.1.branches.0.3.bn2.running_mean', 'layer2.3.bn2.running_var': 'blocks.1.branches.0.3.bn2.running_var', 'layer2.3.bn2.num_batches_tracked': 'blocks.1.branches.0.3.bn2.num_batches_tracked', 'layer2.4.conv1.weight': 'blocks.1.branches.0.4.conv1.weight', 'layer2.4.bn1.weight': 'blocks.1.branches.0.4.bn1.weight', 'layer2.4.bn1.bias': 'blocks.1.branches.0.4.bn1.bias', 'layer2.4.bn1.running_mean': 'blocks.1.branches.0.4.bn1.running_mean', 'layer2.4.bn1.running_var': 'blocks.1.branches.0.4.bn1.running_var', 'layer2.4.bn1.num_batches_tracked': 'blocks.1.branches.0.4.bn1.num_batches_tracked', 'layer2.4.conv2.weight': 'blocks.1.branches.0.4.conv2.weight', 'layer2.4.bn2.weight': 'blocks.1.branches.0.4.bn2.weight', 'layer2.4.bn2.bias': 'blocks.1.branches.0.4.bn2.bias', 'layer2.4.bn2.running_mean': 'blocks.1.branches.0.4.bn2.running_mean', 'layer2.4.bn2.running_var': 'blocks.1.branches.0.4.bn2.running_var', 'layer2.4.bn2.num_batches_tracked': 'blocks.1.branches.0.4.bn2.num_batches_tracked', 'layer3.0.conv1.weight': 'blocks.2.branches.0.0.conv1.weight', 'layer3.0.bn1.weight': 'blocks.2.branches.0.0.bn1.weight', 'layer3.0.bn1.bias': 'blocks.2.branches.0.0.bn1.bias', 'layer3.0.bn1.running_mean': 'blocks.2.branches.0.0.bn1.running_mean', 'layer3.0.bn1.running_var': 'blocks.2.branches.0.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked': 'blocks.2.branches.0.0.bn1.num_batches_tracked', 'layer3.0.conv2.weight': 'blocks.2.branches.0.0.conv2.weight', 'layer3.0.bn2.weight': 'blocks.2.branches.0.0.bn2.weight', 'layer3.0.bn2.bias': 'blocks.2.branches.0.0.bn2.bias', 'layer3.0.bn2.running_mean': 'blocks.2.branches.0.0.bn2.running_mean', 'layer3.0.bn2.running_var': 'blocks.2.branches.0.0.bn2.running_var', 'layer3.0.bn2.num_batches_tracked': 'blocks.2.branches.0.0.bn2.num_batches_tracked', 'layer3.0.downsample.0.weight': 'blocks.2.branches.0.0.downsample.0.weight', 'layer3.0.downsample.1.weight': 'blocks.2.branches.0.0.downsample.1.weight', 'layer3.0.downsample.1.bias': 'blocks.2.branches.0.0.downsample.1.bias', 'layer3.0.downsample.1.running_mean': 'blocks.2.branches.0.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var': 'blocks.2.branches.0.0.downsample.1.running_var', 'layer3.0.downsample.1.num_batches_tracked': 'blocks.2.branches.0.0.downsample.1.num_batches_tracked', 'layer3.1.conv1.weight': 'blocks.2.branches.0.1.conv1.weight', 'layer3.1.bn1.weight': 'blocks.2.branches.0.1.bn1.weight', 'layer3.1.bn1.bias': 'blocks.2.branches.0.1.bn1.bias', 'layer3.1.bn1.running_mean': 'blocks.2.branches.0.1.bn1.running_mean', 'layer3.1.bn1.running_var': 'blocks.2.branches.0.1.bn1.running_var', 'layer3.1.bn1.num_batches_tracked': 'blocks.2.branches.0.1.bn1.num_batches_tracked', 'layer3.1.conv2.weight': 'blocks.2.branches.0.1.conv2.weight', 'layer3.1.bn2.weight': 'blocks.2.branches.0.1.bn2.weight', 'layer3.1.bn2.bias': 'blocks.2.branches.0.1.bn2.bias', 'layer3.1.bn2.running_mean': 'blocks.2.branches.0.1.bn2.running_mean', 'layer3.1.bn2.running_var': 'blocks.2.branches.0.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked': 'blocks.2.branches.0.1.bn2.num_batches_tracked', 'layer3.2.conv1.weight': 'blocks.2.branches.0.2.conv1.weight', 'layer3.2.bn1.weight': 'blocks.2.branches.0.2.bn1.weight', 'layer3.2.bn1.bias': 'blocks.2.branches.0.2.bn1.bias', 'layer3.2.bn1.running_mean': 'blocks.2.branches.0.2.bn1.running_mean', 'layer3.2.bn1.running_var': 'blocks.2.branches.0.2.bn1.running_var', 'layer3.2.bn1.num_batches_tracked': 'blocks.2.branches.0.2.bn1.num_batches_tracked', 'layer3.2.conv2.weight': 'blocks.2.branches.0.2.conv2.weight', 'layer3.2.bn2.weight': 'blocks.2.branches.0.2.bn2.weight', 'layer3.2.bn2.bias': 'blocks.2.branches.0.2.bn2.bias', 'layer3.2.bn2.running_mean': 'blocks.2.branches.0.2.bn2.running_mean', 'layer3.2.bn2.running_var': 'blocks.2.branches.0.2.bn2.running_var', 'layer3.2.bn2.num_batches_tracked': 'blocks.2.branches.0.2.bn2.num_batches_tracked', 'layer3.3.conv1.weight': 'blocks.2.branches.0.3.conv1.weight', 'layer3.3.bn1.weight': 'blocks.2.branches.0.3.bn1.weight', 'layer3.3.bn1.bias': 'blocks.2.branches.0.3.bn1.bias', 'layer3.3.bn1.running_mean': 'blocks.2.branches.0.3.bn1.running_mean', 'layer3.3.bn1.running_var': 'blocks.2.branches.0.3.bn1.running_var', 'layer3.3.bn1.num_batches_tracked': 'blocks.2.branches.0.3.bn1.num_batches_tracked', 'layer3.3.conv2.weight': 'blocks.2.branches.0.3.conv2.weight', 'layer3.3.bn2.weight': 'blocks.2.branches.0.3.bn2.weight', 'layer3.3.bn2.bias': 'blocks.2.branches.0.3.bn2.bias', 'layer3.3.bn2.running_mean': 'blocks.2.branches.0.3.bn2.running_mean', 'layer3.3.bn2.running_var': 'blocks.2.branches.0.3.bn2.running_var', 'layer3.3.bn2.num_batches_tracked': 'blocks.2.branches.0.3.bn2.num_batches_tracked', 'layer3.4.conv1.weight': 'blocks.2.branches.0.4.conv1.weight', 'layer3.4.bn1.weight': 'blocks.2.branches.0.4.bn1.weight', 'layer3.4.bn1.bias': 'blocks.2.branches.0.4.bn1.bias', 'layer3.4.bn1.running_mean': 'blocks.2.branches.0.4.bn1.running_mean', 'layer3.4.bn1.running_var': 'blocks.2.branches.0.4.bn1.running_var', 'layer3.4.bn1.num_batches_tracked': 'blocks.2.branches.0.4.bn1.num_batches_tracked', 'layer3.4.conv2.weight': 'blocks.2.branches.0.4.conv2.weight', 'layer3.4.bn2.weight': 'blocks.2.branches.0.4.bn2.weight', 'layer3.4.bn2.bias': 'blocks.2.branches.0.4.bn2.bias', 'layer3.4.bn2.running_mean': 'blocks.2.branches.0.4.bn2.running_mean', 'layer3.4.bn2.running_var': 'blocks.2.branches.0.4.bn2.running_var', 'layer3.4.bn2.num_batches_tracked': 'blocks.2.branches.0.4.bn2.num_batches_tracked', 'fc.weight': 'fc.weight', 'fc.bias': 'fc.bias'
        }
    }

    def __init__(self, block, layers, block_settings=None, num_classes=10, start_trees: Optional[List[Tree]] = None):
        super(CifarResNet, self).__init__()
        inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)

        if block_settings is None:
            block_settings = [
                # c, n, s, f
                [16, 3, 1, 1],
                [32, 3, 2, 1],
                [64, 3, 2, 1]
            ]

        self.idx2layer = {}
        self.blocks = nn.ModuleList([])
        self.downsamples1 = nn.ModuleList([])
        self.downsamples2 = nn.ModuleList([])

        if start_trees is None:
            self.trees: List[Tree] = []
            # building inverted residual blocks
            for c, n, s, f in block_settings:
                if f > 1:
                    root = Tree(None, [], 'copy', {}, inplanes)
                else:
                    root = Tree(None, [], None, {}, inplanes)
                for _ in range(f):
                    branch_inplanes = inplanes
                    parent = root
                    for i in range(n):
                        stride = s if i == 0 else 1
                        tree = Tree(parent, [], 'ResidualBlock', {
                            'inplanes': branch_inplanes,
                            'planes': c,
                            'stride': stride,
                        }, c * block.expansion)
                        branch_inplanes = c * block.expansion
                        parent.children.append(tree)
                        parent = tree
                self.trees.append(root)
                inplanes = branch_inplanes
                self.downsamples1.append(nn.Conv2d(branch_inplanes, branch_inplanes // 2, 1))
                self.downsamples2.append(nn.Conv2d(branch_inplanes, branch_inplanes // 2, 1))
        else:
            self.trees = start_trees

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for tree in self.trees:
            b = TreeModule(tree)
            self.idx2layer.update(b.idx2layer)
            self.blocks.append(b)
        
        self.path = 0

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def copy_weights_from_sequential(self, net: 'CifarResNet'):
        super().copy_weights_from_sequential(net)

        self.conv1.load_state_dict(net.conv1.state_dict())
        self.bn1.load_state_dict(net.bn1.state_dict())
        self.fc.load_state_dict(net.fc.state_dict())

    def copy_weights_from_original(self, net: 'CifarResNet'):
        super().copy_weights_from_original(net)

        self.conv1.load_state_dict(net.conv1.state_dict())
        self.bn1.load_state_dict(net.bn1.state_dict())
        self.fc.load_state_dict(net.fc.state_dict())

    def load_state_dict_from_pretrained(self, state_dict, arch='20'):
        new_state_dict = {
            value: state_dict[key]
            for key, value in self.state_dict_map[arch].items()
        }
        self.load_state_dict(new_state_dict)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for block, downsample1, downsample2 in zip(self.blocks, self.downsamples1, self.downsamples2):
            if self.path == 0:
                x = block(x)
            elif self.path == 1:
                x = block.branches[0](x)
            elif self.path == 2:
                x = block.branches[1](x)
            elif self.path == -1:
                x1 = block.branches[0](x)
                x1 = downsample1(x1)
                x2 = block.branches[1](x)
                x2 = downsample2(x2)
                x = torch.cat([x1, x2], 1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(
    arch: str,
    layers: List[int],
    model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = False,
    **kwargs: Any
) -> CifarResNet:
    model = CifarResNet(ResidualBlock, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress,
                                              model_dir='/data/workspace/wjiany/pretrained')
        model.load_state_dict(state_dict)
    return model


def cifar10_resnet20(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet32(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet44(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet56(*args, **kwargs) -> CifarResNet: pass


def cifar100_resnet20(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet32(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet44(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet56(*args, **kwargs) -> CifarResNet: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for layers, model_name in zip([[3]*3, [5]*3, [7]*3, [9]*3],
                                  ["resnet20", "resnet32", "resnet44", "resnet56"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(_resnet,
                    arch=model_name,
                    layers=layers,
                    model_urls=model_urls,
                    num_classes=num_classes)
        )
