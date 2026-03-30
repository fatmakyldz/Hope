"""
CIFAR-100 continual learning task splits.
10 tasks x 10 classes each.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

TASK_CLASSES: list[list[int]] = [list(range(i * 10, i * 10 + 10)) for i in range(10)]


@dataclass
class TaskData:
    task_id: int
    class_ids: list[int]
    train_loader: DataLoader
    test_loader: DataLoader


def get_cifar100_tasks(
    batch_size: int = 64,
    root: str = "./data",
    num_workers: int = 2,
) -> list[TaskData]:
    train_ds = datasets.CIFAR100(root=root, train=True,  download=True, transform=TRANSFORM)
    test_ds  = datasets.CIFAR100(root=root, train=False, download=True, transform=TRANSFORM)

    tasks = []
    for task_id, class_ids in enumerate(TASK_CLASSES):
        class_set = set(class_ids)
        tr_idx = [i for i, y in enumerate(train_ds.targets) if y in class_set]
        te_idx = [i for i, y in enumerate(test_ds.targets)  if y in class_set]

        train_loader = DataLoader(
            Subset(train_ds, tr_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            Subset(test_ds, te_idx),
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )
        tasks.append(TaskData(task_id, class_ids, train_loader, test_loader))
    return tasks
