from functools import partial
from datasets.cifar100 import get_cifar100_dataloaders
from datasets.imagenet import get_imagenet_dataloaders
from trainer.train_distill import train_distill
from trainer.train_vanilla import train_vanilla
from distiller.wsl_distiller import WSLDistiller
from trainer.utils import validate
from models.efficientnet import EfficientNet, MBConvConfig
from models.imagenet_shufflenetv2 import imagenet_shufflenetv2_x1_0
from models.resnet import cifar100_resnet56, cifar100_resnet32
from models.cifar_mobilenetv2 import cifar100_mobilenetv2_x1_0, cifar100_mobilenetv2_x0_5
from models.shufflenetv2 import cifar100_shufflenetv2_x1_0, cifar100_shufflenetv2_x2_0
from models.b3_shufflenetv2 import cifar100_shufflenetv2_x1_0 as b3_cifar100_shufflenetv2_x1_0
from models.b4_shufflenetv2 import cifar100_shufflenetv2_x1_0 as b4_cifar100_shufflenetv2_x1_0
from models.imagenet_resnet import resnet50
from models.imagenet_mobilenetv2 import imagenet_mobilenetv2_x1_0
from vanilla_models.imagenet_resnet import resnet50 as vanilla_resnet50
from vanilla_models.shufflenet import cifar100_shufflenetv2_x1_0 as vanilla_cifar100_shufflenetv2_x1_0, cifar100_shufflenetv2_x2_0 as vanilla_cifar100_shufflenetv2_x2_0
from vanilla_models.resnet import cifar100_resnet32 as vanilla_cifar100_resnet32, cifar100_resnet56 as vanilla_cifar100_resnet56
from vanilla_models.efficientnet import efficientnet_b3 as vanilla_efficientnet_b3
from vanilla_models.imagenet_shufflenetv2 import shufflenet_v2_x1_0
from vanilla_models.mobilenetv2 import cifar100_mobilenetv2_x1_0 as vanilla_cifar100_mobilenetv2_x1_0, cifar100_mobilenetv2_x0_5 as vanilla_cifar100_mobilenetv2_x0_5
from distributed.utils import init_distributed_mode, is_main_process

from torchvision.models.mobilenetv2 import mobilenet_v2 as vanilla_imagenet_mobilenetv2

import torch
import torch.optim as optim
import torch.nn as nn
import tensorboard_logger as tb_logger
import time
import os

import argparse


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='print frequency')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--model', type=str, default='mobilenetv2')
    parser.add_argument('--save_folder', type=str, default='./save/student_models')
    parser.add_argument('--train_epoch', type=int, default=120, help='number of training epochs')
    parser.add_argument('--lr_scheduler', type=str, default='step')
    parser.add_argument('--better_cpu', dest='better_cpu', action='store_true')
    parser.add_argument('--distill', dest='distill', action='store_true')
    parser.add_argument('--save_freq', type=int, default=30)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dataset", default='imagenet', type=str)
    parser.add_argument("--workers", default=16, type=int, help="number of data loading workers (default: 16)")
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--lr_step_size", default=30, type=int)
    parser.add_argument("--lr_gamma", default=0.1, type=float)

    parser.set_defaults(distill=False)
    parser.set_defaults(better_cpu=False)

    return parser.parse_args()


def train(s_net, options, train_loader, epoch, t_net, d_net):
    logger = tb_logger.Logger(logdir=options['tb_path'], flush_secs=2)

    criterion = nn.CrossEntropyLoss()

    if options['ckpt']:
        state_dict = torch.load(options['ckpt'])
        s_net.load_state_dict(state_dict['model'])

    criterion = criterion.cuda()
    t_net = t_net.cuda()
    s_net = s_net.cuda()
    d_net = d_net.cuda()

    if options['distributed']:
        t_net = nn.parallel.DistributedDataParallel(t_net, device_ids=[args.gpu])
        s_net = nn.parallel.DistributedDataParallel(s_net, device_ids=[args.gpu])
        d_net = nn.parallel.DistributedDataParallel(d_net, device_ids=[args.gpu])

    optimizer = optim.SGD(
        s_net.parameters(),
        lr=options['learning_rate'],
        momentum=options['momentum'],
        weight_decay=options['weight_decay'],
        nesterov=options['nesterov'],
    )

    if options['lr_scheduler'] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 70, 80], 0.1)
    elif options['lr_scheduler'] == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.2)
    elif options['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif options['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, options['lr_step_size'], options['lr_gamma'])

    # routine
    best_acc = 0
    for epoch in range(1, epoch + 1):
        print("==> training...")
        if options['distributed']:
            train_sampler.set_epoch(epoch)

        time1 = time.time()
        if options['distill']:
            train_acc, train_loss = train_distill(epoch, train_loader, d_net, optimizer, options, options['distributed'])
        else:
            train_acc, train_loss = train_vanilla(epoch, train_loader, s_net, criterion, optimizer, options)

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

        if is_main_process():
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

            if epoch % options['save_freq'] == 0:
                state = {
                    'epoch': epoch,
                    'model': s_net.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(options['save_folder'], '{}_{}.pth'.format(options['model'], epoch))
                print('Frequently saving the best model!')
                torch.save(state, save_file)


if __name__ == '__main__':
    args = parse_options()
    options = vars(args)
    model_path = os.path.join(options['save_folder'], 'model')
    tb_path = os.path.join(options['save_folder'], 'tensorboards')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(tb_path, exist_ok=True)
    options.update({
        'nesterov': True,
        'tb_path': tb_path,
        'save_folder': model_path,
    })

    init_distributed_mode(args)
    options = vars(args)

    if options['dataset'] == 'imagenet':
        train_loader, val_loader, train_sampler, test_sampler = get_imagenet_dataloaders(data_folder=options['data_folder'], batch_size=options['batch_size'], num_workers=options['workers'], distributed=options['distributed'])
    elif options['dataset'] == 'cifar100':
        train_loader, val_loader, train_sampler, test_sampler = get_cifar100_dataloaders(data_folder='./data', batch_size=options['batch_size'], num_workers=options['workers'], distributed=options['distributed'])

    if options['model'] == 'efficientnet':
        t_net = vanilla_efficientnet_b3(pretrained=True)

        bneck_conf = partial(MBConvConfig, width_mult=1, depth_mult=1)
        inverted_residual_setting = [
            bneck_conf(expand_ratio=1, kernel=3, stride=1, input_channels=40, out_channels=24, num_layers=2),
            bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=24, out_channels=32, num_layers=3),
            bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=32, out_channels=48, num_layers=3),
            bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=48, out_channels=96, num_layers=5),
            bneck_conf(expand_ratio=6, kernel=5, stride=1, input_channels=96, out_channels=136, num_layers=5),
            bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=136, out_channels=232, num_layers=6),
            bneck_conf(expand_ratio=6, kernel=3, stride=1, input_channels=232, out_channels=384, num_layers=2),
        ]
        modified_settings = [
            [
                bneck_conf(expand_ratio=1, kernel=3, stride=1, input_channels=40, out_channels=24, num_layers=2),
                bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=24, out_channels=32, num_layers=2),
                bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=32, out_channels=48, num_layers=2),
                bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=48, out_channels=96, num_layers=3),
                bneck_conf(expand_ratio=6, kernel=5, stride=1, input_channels=96, out_channels=136, num_layers=3),
                bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=136, out_channels=232, num_layers=3),
                bneck_conf(expand_ratio=6, kernel=3, stride=1, input_channels=232, out_channels=384, num_layers=2),
            ],
            [
                bneck_conf(expand_ratio=1, kernel=3, stride=1, input_channels=40, out_channels=16, num_layers=2),
                bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=24, out_channels=24, num_layers=3),
                bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=32, out_channels=32, num_layers=3),
                bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=48, out_channels=56, num_layers=5),
                bneck_conf(expand_ratio=6, kernel=5, stride=1, input_channels=96, out_channels=80, num_layers=5),
                bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=136, out_channels=140, num_layers=6),
                bneck_conf(expand_ratio=6, kernel=3, stride=1, input_channels=232, out_channels=232, num_layers=2),
            ]
        ]
        if options['better_cpu']:
            modified_settings[1] = [
                bneck_conf(expand_ratio=1, kernel=3, stride=1, input_channels=40, out_channels=16, num_layers=2),
                bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=24, out_channels=24, num_layers=3),
                bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=32, out_channels=40, num_layers=3),
                bneck_conf(expand_ratio=6, kernel=3, stride=2, input_channels=48, out_channels=80, num_layers=5),
                bneck_conf(expand_ratio=6, kernel=5, stride=1, input_channels=96, out_channels=112, num_layers=5),
                bneck_conf(expand_ratio=6, kernel=5, stride=2, input_channels=136, out_channels=192, num_layers=6),
                bneck_conf(expand_ratio=6, kernel=3, stride=1, input_channels=232, out_channels=320, num_layers=2),
            ]
        s_net = EfficientNet(inverted_residual_setting, 0.3, modified_settings=modified_settings)

        d_net = WSLDistiller(t_net, s_net, num_classes=1000)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

    elif options['model'] == 'shufflenetv2':
        t_net = shufflenet_v2_x1_0(pretrained=True)

        s_net = imagenet_shufflenetv2_x1_0(
            modified_stages_repeats=[[2, 5, 2], [2, 5, 2]],
            modified_stages_out_channels=[[24, 116, 232, 464, 1024], [24, 116, 232, 464, 1024]],
        )

        d_net = WSLDistiller(t_net, s_net, num_classes=1000)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

    elif options['model'] == 'cifar100_shufflenetv2':
        t_net = vanilla_cifar100_shufflenetv2_x1_0(pretrained=True)

        s_net = cifar100_shufflenetv2_x1_0(
            modified_stages_repeats=[[2, 5, 2], [2, 5, 2]],
            modified_stages_out_channels=[[24, 116, 232, 464, 1024], [24, 116, 232, 464, 1024]],
        )

        d_net = WSLDistiller(t_net, s_net, num_classes=100)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

    elif options['model'] == 'cifar100_shufflenetv2_x2':
        t_net = vanilla_cifar100_shufflenetv2_x2_0(pretrained=True)

        s_net = cifar100_shufflenetv2_x2_0(
            modified_stages_repeats=[[2, 5, 2], [2, 5, 2]],
            modified_stages_out_channels=[[24, 244, 488, 976, 2048], [24, 244, 488, 976, 2048]],
        )

        d_net = WSLDistiller(t_net, s_net, num_classes=100)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

    elif options['model'] == 'b3_shufflenetv2':
        t_net = vanilla_cifar100_shufflenetv2_x1_0(pretrained=True)

        s_net = b3_cifar100_shufflenetv2_x1_0(
            modified_stages_repeats=[[2, 5, 2], [2, 5, 2], [2, 5, 2]],
            modified_stages_out_channels=[[24, 80, 160, 320, 1024], [24, 80, 160, 320, 1024], [24, 80, 160, 320, 1024]],
            stages_merge_channels=[[40, 40, 36], [80, 80, 72], [160, 160, 144]]
        )

        d_net = WSLDistiller(t_net, s_net, num_classes=100)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

    elif options['model'] == 'b4_shufflenetv2':
        t_net = vanilla_cifar100_shufflenetv2_x1_0(pretrained=True)

        s_net = b4_cifar100_shufflenetv2_x1_0(
            modified_stages_repeats=[[2, 5, 2], [2, 5, 2], [2, 5, 2], [2, 5, 2]],
            modified_stages_out_channels=[[24, 64, 116, 232, 1024], [24, 64, 116, 232, 1024], [24, 64, 116, 232, 1024], [24, 64, 116, 232, 1024]],
            stages_merge_channels=[[29, 29, 29, 29], [58, 58, 58, 58], [116, 116, 116, 116]]
        )

        d_net = WSLDistiller(t_net, s_net, num_classes=100)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

    elif options['model'] == 'resnet32':
        block_settings = [
            # c, n, s, f
            [16, 3, 1, 2],
            [32, 3, 2, 2],
            [64, 3, 2, 2]
        ]
        s_net = cifar100_resnet32(block_settings=block_settings)
        s_net.path = -1

        t_net = vanilla_cifar100_resnet32(pretrained=True)

        d_net = WSLDistiller(t_net, s_net, num_classes=100)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

    elif options['model'] == 'resnet56':
        block_settings = [
            # c, n, s, f
            [16, 5, 1, 2],
            [32, 5, 2, 2],
            [64, 5, 2, 2]
        ]
        s_net = cifar100_resnet56(block_settings=block_settings)
        s_net.path = -1

        t_net = vanilla_cifar100_resnet56(pretrained=True)

        d_net = WSLDistiller(t_net, s_net, num_classes=100)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

    elif options['model'] == 'mobilenetv2':
        modified_residual_setting = [
            [
                [1, 16, 1, 1],
                [6, 16, 2, 1],
                [6, 24, 3, 2],
                [6, 40, 4, 2],
                [6, 64, 3, 1],
                [6, 112, 3, 2],
                [6, 240, 1, 1],
            ],
            [
                [1, 16, 1, 1],
                [6, 16, 2, 1],
                [6, 24, 3, 2],
                [6, 40, 4, 2],
                [6, 64, 3, 1],
                [6, 112, 3, 2],
                [6, 240, 1, 1],
            ],
        ]
        meeting_point = [False, True, True, False, True, False, True]

        s_net = cifar100_mobilenetv2_x1_0(modified_residual_setting=modified_residual_setting, meeting_point=meeting_point)

        t_net = vanilla_cifar100_mobilenetv2_x1_0(pretrained=True)

        d_net = WSLDistiller(t_net, s_net, num_classes=100)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

    elif options['model'] == 'mobilenetv2_x0_5':
        modified_residual_setting = [
            [
                [1, 16, 1, 1],
                [6, 24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10/100
                [6, 32, 2, 2],
                [6, 64, 2, 2],
                [6, 96, 2, 1],
                [6, 160, 2, 2],
                [6, 320, 1, 1],
            ],
            [
                [1, 16, 1, 1],
                [6, 16, 2, 1],
                [6, 24, 3, 2],
                [6, 40, 4, 2],
                [6, 64, 3, 1],
                [6, 112, 3, 2],
                [6, 240, 1, 1],
            ],
        ]
        meeting_point = [False, True, True, False, True, False, True]

        s_net = cifar100_mobilenetv2_x0_5(modified_residual_setting=modified_residual_setting, meeting_point=meeting_point)

        t_net = vanilla_cifar100_mobilenetv2_x0_5(pretrained=True)

        d_net = WSLDistiller(t_net, s_net, num_classes=100)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

    elif options['model'] == 'resnet50':
        s_net = resnet50()

        t_net = vanilla_resnet50(pretrained=True)

        d_net = WSLDistiller(t_net, s_net, num_classes=1000)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)

    elif options['model'] == 'imagenet_mobilenetv2':
        modified_residual_setting = [
            [
                [1, 16, 1, 1],
                [6, 16, 2, 2],
                [6, 24, 3, 2],
                [6, 40, 4, 2],
                [6, 64, 3, 1],
                [6, 112, 3, 2],
                [6, 240, 1, 1],
            ],
            [
                [1, 16, 1, 1],
                [6, 16, 2, 2],
                [6, 24, 3, 2],
                [6, 40, 4, 2],
                [6, 64, 3, 1],
                [6, 112, 3, 2],
                [6, 240, 1, 1],
            ],
        ]
        meeting_point = [True, True, True, False, True, False, True]

        s_net = imagenet_mobilenetv2_x1_0(modified_residual_setting=modified_residual_setting, meeting_point=meeting_point)

        t_net = vanilla_imagenet_mobilenetv2(pretrained=True)

        d_net = WSLDistiller(t_net, s_net, num_classes=1000)

        train(s_net, options, train_loader, options['train_epoch'], t_net, d_net)
