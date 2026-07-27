"""Data transforms and loaders."""

from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataset import DFDCFaceDataset

# ImageNet normalization — required because backbones are ImageNet-pretrained.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def build_transforms(size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def build_loaders(root: str, size: int, batch_size: int,
                  val_split: float = 0.2, num_workers: int = 4,
                  seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """Split one face-crop directory into train and validation loaders.

    Note: for a real cross-dataset benchmark, keep the test set in a
    separate directory entirely — never split it out of train.
    """
    full = DFDCFaceDataset(root, transform=None)

    n_val = int(len(full) * val_split)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=g)

    # Apply the right transform to each split.
    train_set.dataset.transform = build_transforms(size, train=True)
    # val shares the underlying dataset object, so wrap it separately:
    val_view = DFDCFaceDataset(root, transform=build_transforms(size, train=False))
    val_set.dataset = val_view

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
