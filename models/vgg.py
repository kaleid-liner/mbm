'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/vgg.py
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
import torch
import torch.nn as nn

from models.tree_module import TreeModule

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functools import partial
from typing import Union, List, Dict, Any, cast

from .base_mbmodel import BaseMBModel
from ir.tree import Tree

cifar10_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg13_bn-c01e4a43.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt',
}

cifar100_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg13_bn-5ebe5778.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt',
}


class VGG(BaseMBModel):
    state_dict_map = {
        'features.0.weight': 'blocks.0.branches.0.0.0.weight', 'features.0.bias': 'blocks.0.branches.0.0.0.bias', 'features.1.weight': 'blocks.0.branches.0.0.1.weight', 'features.1.bias': 'blocks.0.branches.0.0.1.bias', 'features.1.running_mean': 'blocks.0.branches.0.0.1.running_mean', 'features.1.running_var': 'blocks.0.branches.0.0.1.running_var', 'features.1.num_batches_tracked': 'blocks.0.branches.0.0.1.num_batches_tracked', 'features.3.weight': 'blocks.0.branches.0.1.0.weight', 'features.3.bias': 'blocks.0.branches.0.1.0.bias', 'features.4.weight': 'blocks.0.branches.0.1.1.weight', 'features.4.bias': 'blocks.0.branches.0.1.1.bias', 'features.4.running_mean': 'blocks.0.branches.0.1.1.running_mean', 'features.4.running_var': 'blocks.0.branches.0.1.1.running_var', 'features.4.num_batches_tracked': 'blocks.0.branches.0.1.1.num_batches_tracked', 'features.7.weight': 'blocks.2.branches.0.0.0.weight', 'features.7.bias': 'blocks.2.branches.0.0.0.bias', 'features.8.weight': 'blocks.2.branches.0.0.1.weight', 'features.8.bias': 'blocks.2.branches.0.0.1.bias', 'features.8.running_mean': 'blocks.2.branches.0.0.1.running_mean', 'features.8.running_var': 'blocks.2.branches.0.0.1.running_var', 'features.8.num_batches_tracked': 'blocks.2.branches.0.0.1.num_batches_tracked', 'features.10.weight': 'blocks.2.branches.0.1.0.weight', 'features.10.bias': 'blocks.2.branches.0.1.0.bias', 'features.11.weight': 'blocks.2.branches.0.1.1.weight', 'features.11.bias': 'blocks.2.branches.0.1.1.bias', 'features.11.running_mean': 'blocks.2.branches.0.1.1.running_mean', 'features.11.running_var': 'blocks.2.branches.0.1.1.running_var', 'features.11.num_batches_tracked': 'blocks.2.branches.0.1.1.num_batches_tracked', 'features.14.weight': 'blocks.4.branches.0.0.0.weight', 'features.14.bias': 'blocks.4.branches.0.0.0.bias', 'features.15.weight': 'blocks.4.branches.0.0.1.weight', 'features.15.bias': 'blocks.4.branches.0.0.1.bias', 'features.15.running_mean': 'blocks.4.branches.0.0.1.running_mean', 'features.15.running_var': 'blocks.4.branches.0.0.1.running_var', 'features.15.num_batches_tracked': 'blocks.4.branches.0.0.1.num_batches_tracked', 'features.17.weight': 'blocks.4.branches.0.1.0.weight', 'features.17.bias': 'blocks.4.branches.0.1.0.bias', 'features.18.weight': 'blocks.4.branches.0.1.1.weight', 'features.18.bias': 'blocks.4.branches.0.1.1.bias', 'features.18.running_mean': 'blocks.4.branches.0.1.1.running_mean', 'features.18.running_var': 'blocks.4.branches.0.1.1.running_var', 'features.18.num_batches_tracked': 'blocks.4.branches.0.1.1.num_batches_tracked', 'features.20.weight': 'blocks.4.branches.0.2.0.weight', 'features.20.bias': 'blocks.4.branches.0.2.0.bias', 'features.21.weight': 'blocks.4.branches.0.2.1.weight', 'features.21.bias': 'blocks.4.branches.0.2.1.bias', 'features.21.running_mean': 'blocks.4.branches.0.2.1.running_mean', 'features.21.running_var': 'blocks.4.branches.0.2.1.running_var', 'features.21.num_batches_tracked': 'blocks.4.branches.0.2.1.num_batches_tracked', 'features.24.weight': 'blocks.6.branches.0.0.0.weight', 'features.24.bias': 'blocks.6.branches.0.0.0.bias', 'features.25.weight': 'blocks.6.branches.0.0.1.weight', 'features.25.bias': 'blocks.6.branches.0.0.1.bias', 'features.25.running_mean': 'blocks.6.branches.0.0.1.running_mean', 'features.25.running_var': 'blocks.6.branches.0.0.1.running_var', 'features.25.num_batches_tracked': 'blocks.6.branches.0.0.1.num_batches_tracked', 'features.27.weight': 'blocks.6.branches.0.1.0.weight', 'features.27.bias': 'blocks.6.branches.0.1.0.bias', 'features.28.weight': 'blocks.6.branches.0.1.1.weight', 'features.28.bias': 'blocks.6.branches.0.1.1.bias', 'features.28.running_mean': 'blocks.6.branches.0.1.1.running_mean', 'features.28.running_var': 'blocks.6.branches.0.1.1.running_var', 'features.28.num_batches_tracked': 'blocks.6.branches.0.1.1.num_batches_tracked', 'features.30.weight': 'blocks.6.branches.0.2.0.weight', 'features.30.bias': 'blocks.6.branches.0.2.0.bias', 'features.31.weight': 'blocks.6.branches.0.2.1.weight', 'features.31.bias': 'blocks.6.branches.0.2.1.bias', 'features.31.running_mean': 'blocks.6.branches.0.2.1.running_mean', 'features.31.running_var': 'blocks.6.branches.0.2.1.running_var', 'features.31.num_batches_tracked': 'blocks.6.branches.0.2.1.num_batches_tracked', 'features.34.weight': 'blocks.8.branches.0.0.0.weight', 'features.34.bias': 'blocks.8.branches.0.0.0.bias', 'features.35.weight': 'blocks.8.branches.0.0.1.weight', 'features.35.bias': 'blocks.8.branches.0.0.1.bias', 'features.35.running_mean': 'blocks.8.branches.0.0.1.running_mean', 'features.35.running_var': 'blocks.8.branches.0.0.1.running_var', 'features.35.num_batches_tracked': 'blocks.8.branches.0.0.1.num_batches_tracked', 'features.37.weight': 'blocks.8.branches.0.1.0.weight', 'features.37.bias': 'blocks.8.branches.0.1.0.bias', 'features.38.weight': 'blocks.8.branches.0.1.1.weight', 'features.38.bias': 'blocks.8.branches.0.1.1.bias', 'features.38.running_mean': 'blocks.8.branches.0.1.1.running_mean', 'features.38.running_var': 'blocks.8.branches.0.1.1.running_var', 'features.38.num_batches_tracked': 'blocks.8.branches.0.1.1.num_batches_tracked', 'features.40.weight': 'blocks.8.branches.0.2.0.weight', 'features.40.bias': 'blocks.8.branches.0.2.0.bias', 'features.41.weight': 'blocks.8.branches.0.2.1.weight', 'features.41.bias': 'blocks.8.branches.0.2.1.bias', 'features.41.running_mean': 'blocks.8.branches.0.2.1.running_mean', 'features.41.running_var': 'blocks.8.branches.0.2.1.running_var', 'features.41.num_batches_tracked': 'blocks.8.branches.0.2.1.num_batches_tracked', 'classifier.0.weight': 'classifier.0.weight', 'classifier.0.bias': 'classifier.0.bias', 'classifier.3.weight': 'classifier.3.weight', 'classifier.3.bias': 'classifier.3.bias', 'classifier.6.weight': 'classifier.6.weight', 'classifier.6.bias': 'classifier.6.bias'
    }

    def __init__(
        self,
        block_settings=None,
        num_classes: int = 10,
        init_weights: bool = False,
        start_trees=None,
    ) -> None:
        super(VGG, self).__init__()

        if block_settings is None:
            block_settings = [
                # c, n, f
                [64, 2, 1],
                [128, 2, 1],
                [256, 3, 1],
                [512, 3, 1],
                [512, 3, 1],
            ]

        self.idx2layer = {}
        self.blocks = nn.ModuleList([])
        inplanes = 3

        if start_trees is None:
            self.trees: List[Tree] = []
            # building inverted residual blocks
            for c, n, f in block_settings:
                if f > 1:
                    root = Tree(None, [], 'copy', {}, inplanes)
                else:
                    root = Tree(None, [], None, {}, inplanes)
                for _ in range(f):
                    branch_inplanes = inplanes
                    parent = root
                    for i in range(n):
                        tree = Tree(parent, [], 'ConvBNActivation', {
                            'in_planes': branch_inplanes,
                            'out_planes': c,
                            'bias': True,
                        }, c)
                        branch_inplanes = c
                        parent.children.append(tree)
                        parent = tree
                self.trees.append(root)
                inplanes = branch_inplanes
        else:
            self.trees = start_trees

        for tree in self.trees:
            block = TreeModule(tree)
            self.idx2layer.update(block.idx2layer)
            self.blocks.append(block)
            self.blocks.append(nn.MaxPool2d(2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def copy_weights_from_sequential(self, net: 'VGG'):
        super().copy_weights_from_sequential(net)

        self.classifier.load_state_dict(net.classifier.state_dict())

    def copy_weights_from_original(self, net: 'VGG'):
        super().copy_weights_from_original(net)

        self.classifier.load_state_dict(net.classifier.state_dict())


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool,
         model_urls: Dict[str, str],
         pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def cifar10_vgg11_bn(*args, **kwargs) -> VGG: pass
def cifar10_vgg13_bn(*args, **kwargs) -> VGG: pass
def cifar10_vgg16_bn(*args, **kwargs) -> VGG: pass
def cifar10_vgg19_bn(*args, **kwargs) -> VGG: pass


def cifar100_vgg11_bn(*args, **kwargs) -> VGG: pass
def cifar100_vgg13_bn(*args, **kwargs) -> VGG: pass
def cifar100_vgg16_bn(*args, **kwargs) -> VGG: pass
def cifar100_vgg19_bn(*args, **kwargs) -> VGG: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for cfg, model_name in zip(["A", "B", "D", "E"], ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(_vgg,
                    arch=model_name,
                    cfg=cfg,
                    batch_norm=True,
                    model_urls=model_urls,
                    num_classes=num_classes)
        )
