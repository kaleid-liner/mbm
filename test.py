from numpy import squeeze
import torch
import torch.nn as nn

from models.mobilenetv2 import MobileNetV2
from models.resnet import cifar100_resnet32, cifar100_resnet20
from models.vgg import cifar100_vgg16_bn
from vanilla_models.mobilenetv2 import cifar100_mobilenetv2_x1_0
from vanilla_models.vgg import cifar100_vgg16_bn as vanilla_cifar100_vgg16_bn
from datasets.cifar100 import get_cifar100_dataloaders
from trainer.utils import validate
from vanilla_models.resnet import cifar100_resnet32 as vanilla_cifar100_resnet32

from nn_meter import load_latency_predictor

from torchvision.models import vgg16_bn, squeezenet1_0
from torchvision.models import densenet121, mobilenet_v2
import torchvision.models as v_models


device_map = {
    'cpu': 'cortexA76cpu_tflite21',
    'gpu': 'adreno640gpu_tflite21',
    'vpu': 'myriadvpu_openvino2019r2',
}

# block_settings = [
#     # c, n, s, f
#     [16, 5, 1, 1],
#     [32, 5, 2, 1],
#     [64, 5, 2, 1]
# ]
# model = cifar100_resnet32(block_settings=block_settings)
# model = squeezenet1_0()
# model = v_models.efficientnet_b0()
model = cifar100_mobilenetv2_x1_0(pretrained=True)


# model = cifar100_vgg16_bn(pretrained=False)
# state_dict = torch.load('/data/workspace/wjiany/pretrained/cifar100_vgg16_bn-7d8c4031.pt')
# model.load_state_dict_from_pretrained(state_dict)
# v_model = vanilla_cifar100_vgg16_bn(pretrained=False)
# v_model.load_state_dict(state_dict)

train_loader, val_loader = get_cifar100_dataloaders(data_folder='./data')
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    # v_model = v_model.cuda()

options = {
    'print_freq': 100,
}
validate(val_loader, model, criterion, options)
# validate(val_loader, v_model, criterion, options)

# # print({
# #     key: value
# #     for key, value in zip(v_model.state_dict().keys(), model.state_dict().keys())
# # })
