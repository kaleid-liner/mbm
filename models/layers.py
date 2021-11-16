import torch.nn as nn
import torch
from torch import Tensor

from typing import Optional, Callable, List


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
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        self.kernel_size = kernel_size
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False) 
        if not init_as_identity:
            super().__init__(
                conv,
                norm_layer(out_planes),
                activation_layer(inplace=True)
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


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


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
    else:
        raise ValueError('Unsupported type: {}'.format(op))
