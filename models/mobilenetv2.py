import torch
from torch import nn
from torch import Tensor
from torchvision.models.utils import load_state_dict_from_url
from typing import Callable, Any, Optional, List, OrderedDict

from ir.tree import Tree
from .tree_module import TreeModule
from .layers import ConvBNReLU, InvertedResidual
from .utils import copy_weights_from_identical_module


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        stem_stride: int = 2,
        start_trees: Optional[List[Tree]] = None,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, f
                [1, 16, 1, 1, 1],
                [6, 24, 2, 2, 1],
                [6, 32, 3, 2, 1],
                [6, 64, 4, 2, 1],
                [6, 96, 3, 1, 1],
                [6, 160, 3, 2, 1],
                [6, 320, 1, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 5-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.conv_first = ConvBNReLU(3, input_channel, stride=stem_stride, norm_layer=norm_layer)

        self.blocks = nn.ModuleList([])
        self.idx2layer = {}

        if start_trees is None:
            self.trees: List[Tree] = []
            # building inverted residual blocks
            for t, c, n, s, f in inverted_residual_setting:
                if f > 1:
                    root = Tree(None, [], 'copy', {}, input_channel)
                else:
                    root = Tree(None, [], None, {}, input_channel)
                for _ in range(f):
                    output_channel = _make_divisible(c * width_mult, round_nearest)
                    parent = root
                    inp = input_channel
                    for i in range(n):
                        stride = s if i == 0 else 1
                        tree = Tree(parent, [], 'InvertedResidual', {
                            'inp': inp,
                            'oup': output_channel,
                            'stride': stride,
                            'expand_ratio': t,
                            'norm_layer': norm_layer,
                        }, output_channel)
                        parent.children.append(tree)
                        parent = tree
                        inp = output_channel
                input_channel = output_channel
                self.trees.append(root)
        else:
            self.trees = start_trees

        for tree in self.trees:
            block = TreeModule(tree)
            self.idx2layer.update(block.idx2layer)
            self.blocks.append(block)
            input_channel = tree.get_leaves()[0].output_channel

        # building last several layers
        self.conv_last = ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

    def copy_weights_from_sequential(self, net: 'MobileNetV2'):
        self.conv_first.load_state_dict(net.conv_first.state_dict())
        self.conv_last.load_state_dict(net.conv_last.state_dict())
        self.classifier.load_state_dict(net.classifier.state_dict())

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

    def copy_weights_from_original(self, net: 'MobileNetV2'):
        self.conv_first.load_state_dict(net.conv_first.state_dict())
        self.conv_last.load_state_dict(net.conv_last.state_dict())
        self.classifier.load_state_dict(net.classifier.state_dict())

        for idx, layer in self.idx2layer.items():
            if idx in net.idx2layer:
                layer.load_state_dict(net.idx2layer[idx].state_dict())

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_first(x)

        for block in self.blocks:
            out = block(out)

        out = self.conv_last(out)

        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict, False)
    return model
