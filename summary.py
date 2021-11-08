from torchinfo import summary
import torch

from models.mobilenetv2 import mobilenet_v2
import config


torch.cuda.set_device(1)
model_s = mobilenet_v2(True, actions=config.actions)
dummy_input = torch.randn(1, 3, 224, 224)
import pdb; pdb.set_trace()
