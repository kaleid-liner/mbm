from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def get_cifar100_dataloaders(batch_size=128, num_workers=8, is_instance=False, data_folder='', distributed=False):
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
    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
                    
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

    return train_loader, test_loader
