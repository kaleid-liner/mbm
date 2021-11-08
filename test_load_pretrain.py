from models.mobilenetv2 import mobilenet_v2
from torchvision.models.mobilenetv2 import mobilenet_v2 as tv_mobilenet_v2

import config

import torch


model_t = tv_mobilenet_v2(num_classes=100)
model_s = mobilenet_v2(num_classes=100, actions=config.actions)
model_t.eval()
model_s.eval()

state_dict = torch.load('./save/models/mobilenetv2_best.pth')
model_t.load_state_dict(state_dict['model'], True)
model_s.load_state_dict(state_dict['model'], False)

x = torch.randn(1, 3, 32, 32)
print(model_t(x) - model_s(x))
