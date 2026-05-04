"""
CIFAR-10 veri kümesini sürekli öğrenme formatına çeviren modül.
Split CIFAR-10: 10 sınıf, 5 göreve bölünmüş (her görevde 2 yeni sınıf).
"""
from __future__ import annotations

from dataclasses import dataclass
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

VIT_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

# 5 görev × 2 sınıf
TASK_CLASSES: list[list[int]] = [list(range(i * 2, i * 2 + 2)) for i in range(5)]


@dataclass
class TaskData:
    task_id: int
    class_ids: list[int]
    train_loader: DataLoader
    test_loader: DataLoader


def get_cifar10_tasks(
    batch_size: int = 64,
    root: str = "./data",
    num_workers: int = 0,
    transform=None,
    pin_memory: bool = False,
) -> list[TaskData]:
    t = transform if transform is not None else TRANSFORM
    train_ds = datasets.CIFAR10(root=root, train=True,  download=True, transform=t)
    test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=t)

    tasks = []
    for task_id, class_ids in enumerate(TASK_CLASSES):
        class_set = set(class_ids)
        tr_idx = [i for i, y in enumerate(train_ds.targets) if y in class_set]
        te_idx = [i for i, y in enumerate(test_ds.targets)  if y in class_set]

        train_loader = DataLoader(Subset(train_ds, tr_idx), batch_size=batch_size,
                                  shuffle=True,  num_workers=num_workers,
                                  pin_memory=pin_memory)
        test_loader  = DataLoader(Subset(test_ds,  te_idx), batch_size=256,
                                  shuffle=False, num_workers=num_workers,
                                  pin_memory=pin_memory)
        tasks.append(TaskData(task_id, class_ids, train_loader, test_loader))
    return tasks
