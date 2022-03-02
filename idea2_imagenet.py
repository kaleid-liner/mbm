from models.resnet import cifar100_resnet32
from datasets.cifar100 import get_cifar100_dataloaders
from datasets.imagenet import get_imagenet_dataloaders
from trainer.train_vanilla import train_vanilla
from trainer.utils import validate
from models.imagenet_resnet import resnet34

import torch
import torch.optim as optim
import torch.nn as nn
import tensorboard_logger as tb_logger
import time
import os


def train(model, options, train_loader, epoch):
    best_acc = 0
    logger = tb_logger.Logger(logdir=options['tb_path'], flush_secs=2)

    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

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
    elif options['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.1)

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
        elif options['lr_scheduler'] == 'reduce':
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
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(options['save_folder'], '{}_best.pth'.format(options['model']))
            print('saving the best model!')
            torch.save(state, save_file)


model = resnet34()

train_loader, val_loader = get_imagenet_dataloaders(data_folder='/data/workspace/wjiany/ILSVRC/Data/CLS-LOC')

options = {
    'learning_rate': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_scheduler': 'step',
    'init_epoch': 60,
    'train_epoch': 200,
    'print_freq': 100,
    'model': 'resnet34',
}

options['lr_scheduler'] = 'step'
options['tb_path'] = './save/idea2_imagenet/tensorboards/train'
options['save_folder'] = './save/idea2_imagenet/models'
train(model, options, train_loader, options['train_epoch'])