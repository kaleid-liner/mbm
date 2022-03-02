from models.mobilenetv2 import MobileNetV2
from models.tv_mobilenetv2 import mobilenet_v2 as tv_mobilenet_v2
# from torchvision.models.mobilenetv2 import mobilenet_v2 as tv_mobilenet_v2
from torchinfo import summary

import config
import torch

from utils import get_latency, process_trees_manually


torch.cuda.set_device(0)
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
model = MobileNetV2(num_classes=1000, inverted_residual_setting=inverted_residual_setting, stem_stride=2)
model.eval()

lat = get_latency(model, hw=224, stem_stride=2, main_device='cpu')
print(lat)

lat = get_latency(model, hw=224, stem_stride=2, main_device='gpu')
print(lat)

inverted_residual_setting = [
    # t, c, n, s, f
    [1, 16, 1, 1, 1],
    [6, 24, 2, 2, 2],
    [6, 32, 3, 2, 2],
    [6, 64, 4, 2, 2],
    [6, 96, 3, 1, 2],
    [6, 160, 3, 2, 2],
    [6, 320, 1, 1, 1],
]
model = MobileNetV2(num_classes=1000, inverted_residual_setting=inverted_residual_setting, stem_stride=2)
trees = process_trees_manually(model.trees)
model = MobileNetV2(num_classes=1000, start_trees=trees, stem_stride=2)
model.eval()

lat = get_latency(model, hw=224, stem_stride=2, main_device='gpu')
print(lat)
