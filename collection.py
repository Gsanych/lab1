import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from loguru import logger
from config import settings

def load_data(batch_size=128):
    logger.info("CIFAR100 data loading...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    full_dataset = torchvision.datasets.CIFAR100(root=settings.data_path, train=True, download=True, transform=transform)


    train_size = int(settings.train_ratio * len(full_dataset))
    val_size = int(settings.val_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=2)

    logger.info("CIFAR100 dataset loaded successfully")

    logger.info(f'Training set size: {len(train_dataset)}')
    logger.info(f'Validation set size: {len(val_dataset)}')
    logger.info(f'Test set size: {len(test_dataset)}')

    return train_loader, val_loader, test_loader