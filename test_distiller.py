from models.resnet import cifar100_resnet32
from models.shufflenetv2 import cifar100_shufflenetv2_x1_0, cifar100_shufflenetv2_x2_0
from models.mobilenetv2 import MobileNetV2
from models.cifar_mobilenetv2 import cifar100_mobilenetv2_x1_0
from vanilla_models.mobilenetv2 import cifar100_mobilenetv2_x1_0 as vanilla_cifar100_mobilenetv2_x1_0
from vanilla_models.shufflenet import cifar100_shufflenetv2_x2_0 as vanilla_cifar100_shufflenetv2_x2_0, cifar100_shufflenetv2_x1_5 as vanilla_cifar100_shufflenetv2_x1_5, cifar100_shufflenetv2_x1_0 as vanilla_cifar100_shufflenetv2_x1_0
from datasets.cifar100 import get_cifar100_dataloaders
from trainer.train_vanilla import train_vanilla
from trainer.train_distill import train_distill
from distiller.wsl_distiller import WSLDistiller
from trainer.utils import validate

import torch
import torch.optim as optim
import torch.nn as nn
import tensorboard_logger as tb_logger
import time
import os


def train(s_net, options, train_loader, epoch, t_net, d_net):
    logger = tb_logger.Logger(logdir=options['tb_path'], flush_secs=2)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        t_net = t_net.cuda()
        s_net = s_net.cuda()
        d_net = d_net.cuda()

    optimizer = optim.SGD(
        s_net.parameters(),
        lr=options['learning_rate'],
        momentum=options['momentum'],
        weight_decay=options['weight_decay'],
        nesterov=options['nesterov'],
    )

    if options['lr_scheduler'] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 60], 0.2)
    elif options['lr_scheduler'] == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.2)
    elif options['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif options['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.1)

    # routine
    best_acc = 0
    for epoch in range(1, epoch + 1):
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train_distill(epoch, train_loader, d_net, optimizer, options, parallel=False)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            break

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, s_net, criterion, options)
        if options['lr_scheduler'] == 'reduce':
            scheduler.step(test_loss)
        else:
            scheduler.step()

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': s_net.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(options['save_folder'], '{}_best.pth'.format(options['model']))
            print('saving the best model!')
            torch.save(state, save_file)


model_type = 'shufflenetv2'
torch.cuda.set_device(0)

if model_type == 'shufflenetv2':
    t_net = vanilla_cifar100_shufflenetv2_x2_0(pretrained=True)
    s_net = cifar100_shufflenetv2_x2_0(
        modified_stages_repeats=[[2, 5, 2], [4, 8, 4]],
        modified_stages_out_channels=[[24, 244, 488, 976, 2048], [24, 116, 232, 464, 1024]],
    )
    d_net = WSLDistiller(t_net, s_net)

    train_loader, train_loader_1, train_loader_2, val_loader = get_cifar100_dataloaders(data_folder='./data', subset=True)

    options = {
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'lr_scheduler': 'multistep',
        'init_epoch': 60,
        'train_epoch': 200,
        'print_freq': 100,
        'nesterov': True,
    }

    options['lr_scheduler'] = 'step'
    options['model'] = 'shufflenetv2'
    options['tb_path'] = './save/cifar100_shufflenet/tensorboards/train_distill_cpu'
    options['save_folder'] = './save/cifar100_shufflenet/models/train_distill_cpu'
    train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

elif model_type == 'mobilenetv2':
    modified_residual_setting = [
        [
            [1, 16, 1, 1],
            [6, 16, 2, 2],
            [6, 24, 3, 2],
            [6, 40, 4, 2],
            [6, 64, 3, 1],
            [6, 112, 3, 1],
            [6, 240, 1, 1],
        ],
        [
            [1, 16, 1, 1],
            [6, 16, 2, 2],
            [6, 24, 3, 2],
            [6, 40, 4, 2],
            [6, 64, 3, 1],
            [6, 112, 3, 1],
            [6, 240, 1, 1],
        ],
    ]
    s_net = cifar100_mobilenetv2_x1_0(modified_residual_setting=modified_residual_setting)
    t_net = vanilla_cifar100_mobilenetv2_x1_0(pretrained=True)
    d_net = WSLDistiller(t_net, s_net)

    train_loader, train_loader_1, train_loader_2, val_loader = get_cifar100_dataloaders(data_folder='./data', subset=True)

    options = {
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'lr_scheduler': 'multistep',
        'init_epoch': 60,
        'train_epoch': 200,
        'print_freq': 100,
        'nesterov': True,
    }

    options['lr_scheduler'] = 'reduce'
    options['model'] = 'mobilenetv2'
    options['tb_path'] = './save/cifar100_mobilenetv2/tensorboards/train_stride'
    options['save_folder'] = './save/cifar100_mobilenetv2/models/train_stride'
    train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)
