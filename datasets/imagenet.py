from torchvision import datasets, transforms
from torch.utils.data import DataLoader, dataset

import os


def get_imagenet_dataloaders(batch_size=128, num_workers=8, is_instance=False, data_folder=''):
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
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_set = datasets.ImageFolder(
        root=os.path.join(data_folder, 'val'),
        transform=test_transform
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(batch_size/2),
        shuffle=False,
        num_workers=int(num_workers/2)
    )

    return train_loader, test_loader
