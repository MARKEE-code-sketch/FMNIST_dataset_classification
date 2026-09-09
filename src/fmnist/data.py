"""Reproducible Fashion-MNIST data loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def build_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    validation_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoaders:
    """Download Fashion-MNIST and create deterministic train/val/test loaders."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ]
    )

    root = str(Path(data_dir))
    full_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    val_size = int(len(full_train) * validation_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    pin_memory = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return DataLoaders(
        train=DataLoader(train_set, shuffle=True, generator=generator, **common),
        validation=DataLoader(val_set, shuffle=False, **common),
        test=DataLoader(test_set, shuffle=False, **common),
    )
