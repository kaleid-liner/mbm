from torchvision import datasets, transforms
from torch.utils.data import DataLoader, dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import os


def get_imagenet_dataloaders(batch_size=256, num_workers=8, is_instance=False, data_folder='', distributed=False):
    """
    imagenet
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_set = datasets.ImageFolder(
        root=os.path.join(data_folder, 'train'),
        transform=train_transform
    )
    test_set = datasets.ImageFolder(
        root=os.path.join(data_folder, 'val'),
        transform=test_transform
    )

    if distributed:
        train_sampler = DistributedSampler(train_set)
        test_sampler = DistributedSampler(test_set, shuffle=False)
    else:
        train_sampler = RandomSampler(train_set)
        test_sampler = SequentialSampler(test_set)
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    return train_loader, test_loader, train_sampler, test_sampler
