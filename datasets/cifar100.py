from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_cifar100_dataloaders(batch_size=128, num_workers=8, is_instance=False, data_folder='', subset=False):
    """
    cifar 100
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(root=data_folder,
                                  download=True,
                                  train=True,
                                  transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if subset:
        train_loader_1 = DataLoader(
            Subset(train_set, range(0, len(train_set), 2)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        train_loader_2 = DataLoader(
            Subset(train_set, range(1, len(train_set), 2)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        return train_loader, train_loader_1, train_loader_2, test_loader

    return train_loader, test_loader
