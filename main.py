from numpy import block
from torchinfo import summary
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.mobilenetv2 import mobilenet_v2 as tv_mobilenet_v2
import tensorboard_logger as tb_logger

import argparse
import time
import os

from models.mobilenetv2 import MobileNetV2
from models.resnet import cifar100_resnet20, cifar100_resnet32
from models.vgg import cifar100_vgg16_bn
from trainer.pretrain import init
from distiller.FSP import FSP
from datasets.cifar100 import get_cifar100_dataloaders
from datasets.imagenet import get_imagenet_dataloaders
from trainer.train_vanilla import train_vanilla
from trainer.utils import set_parameter_requires_grad, validate

from utils import process_trees_manually


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--distill', type=str, default='fsp', choices=['kd', 'hint', 'attention', 'similarity',
                                                                       'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                       'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--tb_path', type=str, default='./save/student_tensorboards')
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--model', type=str, default='mobilenetv2')
    parser.add_argument('--save_folder', type=str, default='./save/student_models')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--feature_extract', dest='feature_extract', action='store_true')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--train_student', dest='train_student', action='store_true')
    parser.add_argument('--parallel', dest='parallel', action='store_true')
    parser.add_argument('--lr_scheduler', type=str, default='multistep')
    parser.add_argument('--dataset', type=str, default='cifar100')

    parser.set_defaults(feature_extract=False)
    parser.set_defaults(train_student=False)
    parser.set_defaults(parallel=False)
    return parser.parse_args()


def train(options):
    best_acc = 0

    torch.cuda.set_device(options['device'])
    torch.autograd.set_detect_anomaly(True)

    # dataloader
    if options['dataset'] == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=options['batch_size'],
                                                            num_workers=options['num_workers'],
                                                            is_instance=False,
                                                            data_folder=options['data_folder'])

        if options['model'] == 'mobilenetv2':
            inverted_residual_setting = [
                # t, c, n, s, f
                [1, 16, 1, 1, 1],
                [6, 24, 2, 2, 1],
                [6, 32, 3, 2, 1],
                [6, 64, 4, 2, 1],
                [6, 96, 3, 1, 1],
                [6, 160, 3, 1, 1],
                [6, 320, 1, 1, 1],
            ]
            num_classes = 100
            stem_stride = 1
            model_t = MobileNetV2(num_classes=num_classes, inverted_residual_setting=inverted_residual_setting, stem_stride=stem_stride)

            if options['ckpt']:
                state_dict = torch.load(options['ckpt'])
                model_t.load_state_dict_from_pretrained(state_dict)
        elif options['model'] == 'resnet20':
            block_settings = [
                # c, n, s, f
                [16, 3, 1, 1],
                [32, 3, 2, 1],
                [64, 3, 2, 1]
            ]
            model_t = cifar100_resnet20(block_settings=block_settings)

            if options['ckpt']:
                state_dict = torch.load(options['ckpt'])
                model_t.load_state_dict_from_pretrained(state_dict)
        elif options['model'] == 'resnet32':
            block_settings = [
                # c, n, s, f
                [16, 5, 1, 1],
                [32, 5, 2, 1],
                [64, 5, 2, 1]
            ]
            model_t = cifar100_resnet32(block_settings=block_settings)

            if options['ckpt']:
                state_dict = torch.load(options['ckpt'])
                model_t.load_state_dict_from_pretrained(state_dict, '32')
        elif options['model'] == 'vgg':
            block_settings = [
                # c, n, f
                [64, 2, 1],
                [128, 2, 1],
                [256, 3, 1],
                [512, 3, 1],
                [512, 3, 1],
            ]
            model_t = cifar100_vgg16_bn(block_settings=block_settings)

            if options['ckpt']:
                state_dict = torch.load(options['ckpt'])
                model_t.load_state_dict_from_pretrained(state_dict)

    elif options['dataset'] == 'imagenet':
        train_loader, val_loader = get_imagenet_dataloaders(data_folder='/data/workspace/wjiany/ILSVRC/Data/CLS-LOC')

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
        num_classes = 1000
        stem_stride = 2
        tv_model = tv_mobilenet_v2(True)
        model_t = MobileNetV2(num_classes=num_classes, inverted_residual_setting=inverted_residual_setting, stem_stride=stem_stride)
        model_t.load_state_dict_from_pretrained(tv_model.state_dict())

    if options['train_student']:
        if options['model'] == 'mobilenetv2':
            if options['dataset'] == 'cifar100':
                inverted_residual_setting = [
                    # t, c, n, s, f
                    [1, 16, 1, 1, 1],
                    [6, 24, 2, 2, 2],
                    [6, 32, 3, 2, 2],
                    [6, 64, 4, 2, 2],
                    [6, 96, 3, 1, 2],
                    [6, 160, 3, 1, 2],
                    [6, 320, 1, 1, 1],
                ]
            elif options['dataset'] == 'imagenet':
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
            model = MobileNetV2(num_classes=num_classes, inverted_residual_setting=inverted_residual_setting, stem_stride=stem_stride)
            model.copy_weights_from_sequential(model_t)
            transformed_trees = process_trees_manually(model.trees, options['model'])
            transformed_model = MobileNetV2(num_classes=num_classes, start_trees=transformed_trees, stem_stride=stem_stride)
            transformed_model.copy_weights_from_original(model)
            model = transformed_model

        elif options['model'] == 'resnet20':
            block_settings = [
                # c, n, s, f
                [16, 3, 1, 2],
                [32, 3, 2, 2],
                [64, 3, 2, 2]
            ]
            model = cifar100_resnet20(block_settings=block_settings)
            model.copy_weights_from_original(model_t)
            transformed_trees = process_trees_manually(model.trees, options['model'])
            transformed_model = cifar100_resnet20(start_trees=transformed_trees)
            transformed_model.copy_weights_from_original(model)
            model = transformed_model
        
        elif options['model'] == 'resnet32':
            block_settings = [
                # c, n, s, f
                [16, 5, 1, 2],
                [32, 5, 2, 2],
                [64, 5, 2, 2]
            ]
            model = cifar100_resnet32(block_settings=block_settings)
            model.copy_weights_from_original(model_t)
            transformed_trees = process_trees_manually(model.trees, options['model'])
            transformed_model = cifar100_resnet32(start_trees=transformed_trees)
            transformed_model.copy_weights_from_original(model)
            model = transformed_model

        elif options['model'] == 'vgg':
            block_settings = [
                # c, n, f
                [64, 2, 2],
                [128, 2, 2],
                [256, 3, 2],
                [512, 3, 2],
                [512, 3, 2],
            ]
            model = cifar100_vgg16_bn(block_settings=block_settings)
            model.copy_weights_from_original(model_t)
            transformed_trees = process_trees_manually(model.trees, options['model'])
            transformed_model = cifar100_vgg16_bn(start_trees=transformed_trees)
            transformed_model.copy_weights_from_original(model)
            model = transformed_model

    else:
        model = model_t

    if options['parallel']:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    
    set_parameter_requires_grad(model, feature_extracting=options['feature_extract'])
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t",name)
    if not options['feature_extract']:
        params_to_update = model.parameters()

    # optimizer
    optimizer = optim.SGD(params_to_update,
                          lr=options['learning_rate'],
                          momentum=options['momentum'],
                          weight_decay=options['weight_decay'])
    if options['lr_scheduler'] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160], 0.2)
    elif options['lr_scheduler'] == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.2)
    elif options['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # tensorboard
    logger = tb_logger.Logger(logdir=options['tb_path'], flush_secs=2)

    # routine
    for epoch in range(1, options['epochs'] + 1):
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train_vanilla(epoch, train_loader, model, criterion, optimizer, options)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            break

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, options)
        if options['lr_scheduler'] == 'multistep':
            scheduler.step()
        else:
            scheduler.step(test_loss)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(options['save_folder'], '{}_best.pth'.format(options['model']))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % options['save_freq'] == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(options['save_folder'], 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': options,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(options['save_folder'], '{}_last.pth'.format(options['model']))
    torch.save(state, save_file)


if __name__ == '__main__':
    opt = parse_options()
    train(vars(opt))
