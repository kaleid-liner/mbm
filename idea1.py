from models.resnet import cifar100_resnet32
from datasets.cifar100 import get_cifar100_dataloaders
from trainer.train_vanilla import train_vanilla
from trainer.utils import validate

import torch
import torch.optim as optim
import torch.nn as nn
import tensorboard_logger as tb_logger
import time
import os


def train(model, options, train_loader, epoch):
    logger = tb_logger.Logger(logdir=options['tb_path'], flush_secs=2)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=options['learning_rate'],
        momentum=options['momentum'],
        weight_decay=options['weight_decay']
    )

    if options['lr_scheduler'] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 60], 0.2)
    elif options['lr_scheduler'] == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.2)
    elif options['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # routine
    for epoch in range(1, epoch + 1):
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


block_settings = [
    # c, n, s, f
    [16, 3, 1, 2],
    [32, 3, 2, 2],
    [64, 3, 2, 2]
]
model = cifar100_resnet32(block_settings=block_settings)
model.init_weight()

train_loader, train_loader_1, train_loader_2, val_loader = get_cifar100_dataloaders(data_folder='./data', subset=True)

options = {
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'lr_scheduler': 'multistep',
    'init_epoch': 60,
    'train_epoch': 200,
    'print_freq': 100,
}

model.path = 1
options['tb_path'] = './save/idea1/tensorboards/init_4'
train(model, options, train_loader_1, options['init_epoch'])

model.path = 2
options['tb_path'] = './save/idea1/tensorboards/init_5'
train(model, options, train_loader_2, options['init_epoch'])

model.path = 0
options['learning_rate'] = 0.02
options['lr_scheduler'] = 'reduce'
options['tb_path'] = './save/idea1/tensorboards/train_2'
train(model, options, train_loader, options['train_epoch'])
