import torch
from torch import nn
from torch import Tensor
from torchvision.models.utils import load_state_dict_from_url
from typing import Callable, Any, Optional, List, OrderedDict
from torchvision.models import mobilenetv2

from .tree import TreeModule
from .layers import ConvBNReLU


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


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.in_channels = inp
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


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
        actions = None,
        is_feat=False
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
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if actions is None:
            actions = [
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ]

        assert(len(inverted_residual_setting) == len(actions))

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building layer mapping from torchvision model to multi-branch model
        layer_mapping = {}
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.conv_first = ConvBNReLU(3, input_channel, stride=stem_stride, norm_layer=norm_layer)
        layer_mapping['conv_first'] = 'features.0'

        self.blocks = nn.ModuleList([])
        # building inverted residual blocks
        bid = 0
        lid = 1
        for action, (t, c, n, s) in zip(actions, inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            layers = []
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
            self.blocks.append(TreeModule(layers, action, layer_mapping, 'blocks.{}'.format(bid), 'features', lid))
            bid += 1
            lid += n

        # building last several layers
        self.conv_last = ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)

        layer_mapping['conv_last'] = 'features.{}'.format(lid)
        self.layer_mapping = layer_mapping

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        layer_mapping['classifier'] = 'classifier'

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool, load_from: str = 'tv'):
        if load_from == 'tv':
            new_state_dict = {}
            for prefix, prefix_mapping in self.layer_mapping.items():
                for key, value in state_dict.items():
                    if key.startswith(prefix_mapping + '.'):
                        key = key.replace(prefix_mapping, prefix)
                        new_state_dict[key] = value
        elif load_from == 'self':
            new_state_dict = state_dict
        elif load_from == 'workaround':
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('blocks'):
                    layer_id = int(key[20])
                    layer_id = layer_id + 2
                    key = key[:20] + str(layer_id) + key[21:]
                new_state_dict[key] = value

        super().load_state_dict(new_state_dict, strict)

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

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv_first)
        feat_m.append(self.blocks)
        return feat_m

    def forward(self, x: Tensor, is_feat=False) -> Tensor:
        out = self.conv_first(x)
        f0 = out

        out = self.blocks[0](out)
        f1 = out
        out = self.blocks[1](out)
        f2 = out
        out = self.blocks[2](out)
        f3 = out
        out = self.blocks[3](out)
        f4 = out
        out = self.blocks[4](out)
        f5 = out
        out = self.blocks[5](out)
        f6 = out
        out = self.blocks[6](out)
        f7 = out

        out = self.conv_last(out)

        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        f8 = out
        out = self.classifier(out)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5, f6, f7, f8], out
        else:
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
