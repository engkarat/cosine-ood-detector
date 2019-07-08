from data.load_cifar10 import load_cifar10
from data.load_cifar100 import load_cifar100
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class CifarDataset(Dataset):
    def __init__(self):
        self.labels, self.images, self.transform = [], [], None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.transform:
            image = self.transform(F.to_pil_image(image))
        return image, label


class Cifar10Dataset(CifarDataset):

    """CIFAR-10 dataset."""

    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset (string): One of 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        x_tr, y_tr, x_te, y_te = load_cifar10()
        if dataset == 'train':
            self.images = (x_tr * 255).astype('uint8')
            self.labels = y_tr
        elif dataset == 'test':
            self.images = (x_te * 255).astype('uint8')
            self.labels = y_te
        self.transform = transform


class Cifar100Dataset(CifarDataset):

    """CIFAR-100 dataset."""

    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset (string): One of 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        x_tr, y_tr, x_te, y_te = load_cifar100()
        if dataset == 'train':
            self.images = (x_tr * 255).astype('uint8')
            self.labels = y_tr
        elif dataset == 'test':
            self.images = (x_te * 255).astype('uint8')
            self.labels = y_te
        self.transform = transform