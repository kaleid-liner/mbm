from torchvision.models.mobilenetv2 import mobilenet_v2
from models.mobilenetv2 import MobileNetV2
import torch.nn as nn
import torch

from datasets.imagenet import get_imagenet_dataloaders
from trainer.utils import validate


tv_model = mobilenet_v2(True)

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
model.load_state_dict_from_pretrained(tv_model.state_dict())

train_loader, val_loader = get_imagenet_dataloaders(data_folder='/data/workspace/wjiany/ILSVRC/Data/CLS-LOC')

criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

options = {
    'print_freq': 100,
}
validate(val_loader, model, criterion, options)
