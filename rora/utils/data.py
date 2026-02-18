"""Data loading utilities for experiments."""

import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Optional

_NUM_WORKERS = min(4, os.cpu_count() or 1)


def get_mnist_loaders(
    batch_size: int = 64,
    train_subset: Optional[list] = None,
    test_subset: Optional[list] = None,
    root: str = "./data",
):
    """
    Get MNIST data loaders.

    Args:
        batch_size: Batch size
        train_subset: List of classes to include in training (None = all)
        test_subset: List of classes to include in testing (None = all)
        root: Root directory for data

    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    # Vectorized subset filtering using pre-loaded targets tensor
    if train_subset is not None:
        mask = torch.isin(train_dataset.targets, torch.tensor(train_subset))
        train_dataset = Subset(train_dataset, mask.nonzero(as_tuple=True)[0].tolist())

    if test_subset is not None:
        mask = torch.isin(test_dataset.targets, torch.tensor(test_subset))
        test_dataset = Subset(test_dataset, mask.nonzero(as_tuple=True)[0].tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(_NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(_NUM_WORKERS > 0),
    )

    return train_loader, test_loader


class MultiTaskMNISTDataset(Dataset):
    """Dataset for multi-task learning on MNIST."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, digit = self.base_dataset[idx]

        # Task A: Even/Odd
        task_a_label = 1 if (digit % 2 == 1) else 0

        # Task B: Bit-Parity (popcount mod 2)
        popcount = bin(digit).count("1")
        task_b_label = popcount % 2

        return image, {"digit": digit, "even_odd": task_a_label, "bit_parity": task_b_label}


def multitask_collate_fn(batch):
    """Custom collate function for multi-task dataset."""
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    labels_dict = {
        "digit": torch.tensor([d["digit"] for d in labels], dtype=torch.long),
        "even_odd": torch.tensor([d["even_odd"] for d in labels], dtype=torch.long),
        "bit_parity": torch.tensor([d["bit_parity"] for d in labels], dtype=torch.long),
    }
    return images, labels_dict


def get_multitask_mnist_loaders(batch_size: int = 64, root: str = "./data"):
    """Get multi-task MNIST data loaders."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_multitask = MultiTaskMNISTDataset(train_dataset)
    test_multitask = MultiTaskMNISTDataset(test_dataset)

    train_loader = DataLoader(
        train_multitask,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multitask_collate_fn,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(_NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_multitask,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multitask_collate_fn,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(_NUM_WORKERS > 0),
    )

    return train_loader, test_loader
