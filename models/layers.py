import torch.nn as nn
import torch
from torch import Tensor

from typing import Optional, Callable, List


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
        init_as_identity: bool = False,
        bias: bool = False,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        self.kernel_size = kernel_size
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias) 
        if not init_as_identity:
            super().__init__(
                conv,
                norm_layer(out_planes),
                activation_layer(inplace=False)
            )
        else:
            super().__init__(
                conv,
            )
        self.in_channels = in_planes
        self.out_channels = out_planes

        if init_as_identity:
            self._init_as_identity()

    def _init_as_identity(self):
        mid = self.kernel_size // 2
        self[0].weight.data.zero_()
        weight_init = torch.cat([
            torch.eye(self[0].weight.size(1)) for _ in range(self[0].groups)
        ], dim=0)
        self[0].weight.data[:, :, mid, mid] = weight_init


class DWConvBNActivation(ConvBNActivation):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
        init_as_identity: bool = False,
    ) -> None:
        super().__init__(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            in_planes,
            norm_layer,
            activation_layer,
            dilation,
            init_as_identity,
        )


class MaxPool2d(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation

        self.in_channels = inp
        self.out_channels = oup
        self.pool = nn.MaxPool2d(kernel_size, stride, padding, dilation)

    def forward(self, x):
        return self.pool(x)

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


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in in eq. 3.
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


layer_types = [
    'conv_1x1',
    'dwconv_3x3',
    'dwconv_5x5',
    'dwconv_7x7',
    # 'avgpool',
    # 'maxpool',
]


def get_identity_layer(type, in_channels, out_channels):
    op, kernels = type.split('_')
    kernel = int(kernels[0])
    if op == 'conv':
        return ConvBNActivation(in_channels, out_channels, kernel, init_as_identity=True)
    elif op == 'dwconv':
        return DWConvBNActivation(in_channels, out_channels, kernel, init_as_identity=True)
    elif op == 'maxpool':
        return MaxPool2d(in_channels, out_channels, kernel)
    else:
        raise ValueError('Unsupported type: {}'.format(op))
