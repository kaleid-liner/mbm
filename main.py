from torchinfo import summary
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.mobilenetv2 import mobilenet_v2 as tv_mobilenet_v2
import tensorboard_logger as tb_logger

import argparse
import time
import os

from models.mobilenetv2 import mobilenet_v2
from trainer.pretrain import init
from distiller.FSP import FSP
from datasets.cifar100 import get_cifar100_dataloaders
import config
from trainer.train_vanilla import train_vanilla
from trainer.utils import validate


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--tb_path', type=str, default='./save/student_tensorboards')
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--model', type=str, default='mobilenetv2')
    parser.add_argument('--save_folder', type=str, default='./save/models')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    return parser.parse_args()


def main():
    opt = parse_options()
        
    logger = tb_logger.Logger(logdir=opt.tb_path, flush_secs=2)

    model_t = mobilenet_v2(pretrained=True)
    model_s = mobilenet_v2(actions=config.actions)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                        num_workers=opt.num_workers,
                                                        is_instance=False,
                                                        data_folder=opt.data_folder)
                                                        
                                                    
    if opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)

        # classification training
        pass


def train():
    best_acc = 0

    opt = parse_options()

    # dataloader
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                        num_workers=opt.num_workers,
                                                        is_instance=False,
                                                        data_folder=opt.data_folder)

    # model
    model = tv_mobilenet_v2()

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_path, flush_secs=2)

    # routine
    for epoch in range(1, opt.epochs + 1):

        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train_vanilla(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

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
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)

if __name__ == '__main__':
    train()
