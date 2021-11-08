from nn_meter.utils.utils import try_import_torchvision_models
from models.mobilenetv2 import mobilenet_v2
from models.tv_mobilenetv2 import mobilenet_v2 as tv_mobilenet_v2
# from torchvision.models.mobilenetv2 import mobilenet_v2 as tv_mobilenet_v2
from torchinfo import summary

import config
import torch

from utils import get_latency


torch.cuda.set_device(1)
model = mobilenet_v2(False, actions=config.actions, num_classes=100, stem_stride=1)
model.eval()

lat = get_latency(model, hw=224, stem_stride=1, branch=True)
print(lat)
summary(model, (1, 3, 224, 224))
