"""
CIFAR-100 veri kümesini sürekli öğrenme (continual learning) formatına çeviren modül.
Toplam 100 sınıf, 10 göreve bölünmüştür — her görevde 10 sınıf vardır.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ─── VERİ ÖN İŞLEME ─────────────────────────────────────────────────────────
# Görüntüleri [0,1] aralığına çekip CIFAR-100'e özgü ortalama ve
# standart sapma değerleriyle normalize ediyoruz.
# Bu değerler tüm CIFAR-100 eğitim setinden hesaplanmış sabit istatistiklerdir.
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),   # RGB kanal ortalamaları
                         (0.2675, 0.2565, 0.2761)),   # RGB kanal std'leri
])

# ─── GÖREV TANIMI ────────────────────────────────────────────────────────────
# 100 sınıfı 10 göreve bölen liste.
# Görev 0: sınıf 0-9, Görev 1: sınıf 10-19, ... Görev 9: sınıf 90-99
# Bu Class-Incremental Learning (Class-IL) senaryosudur:
# model her görevde yeni sınıflar görür, eski sınıfları unutmaması beklenir.
TASK_CLASSES: list[list[int]] = [list(range(i * 10, i * 10 + 10)) for i in range(10)]


# ─── GÖREV VERİ YAPISI ───────────────────────────────────────────────────────
# Her görevin bilgilerini bir arada tutan basit veri sınıfı.
@dataclass
class TaskData:
    task_id: int          # Görev numarası (0-9)
    class_ids: list[int]  # Bu görevin sınıf etiketleri
    train_loader: DataLoader
    test_loader: DataLoader


# ─── ANA FONKSİYON: TÜM GÖREVLERİ YÜKLE ────────────────────────────────────
def get_cifar100_task_datasets(root: str = "./data") -> list[dict]:
    """
    Dağıtık eğitim için DataLoader yerine ham dataset döndürür.

    Her node kendi DistributedSampler'ını oluşturabilmek için
    DataLoader değil Subset nesnelerine ihtiyaç duyar.

    Döndürür: [{'task_id', 'class_ids', 'train_subset', 'test_subset'}, ...]
    """
    train_ds = datasets.CIFAR100(root=root, train=True,  download=True, transform=TRANSFORM)
    test_ds  = datasets.CIFAR100(root=root, train=False, download=True, transform=TRANSFORM)

    result = []
    for task_id, class_ids in enumerate(TASK_CLASSES):
        class_set = set(class_ids)
        tr_idx = [i for i, y in enumerate(train_ds.targets) if y in class_set]
        te_idx = [i for i, y in enumerate(test_ds.targets)  if y in class_set]
        result.append({
            "task_id":      task_id,
            "class_ids":    class_ids,
            "train_subset": Subset(train_ds, tr_idx),
            "test_subset":  Subset(test_ds,  te_idx),
        })
    return result


def get_cifar100_tasks(
    batch_size: int = 64,
    root: str = "./data",
    num_workers: int = 2,
) -> list[TaskData]:
    """
    CIFAR-100'ü indirip 10 göreve böler ve her görev için DataLoader döndürür.

    Önemli tasarım kararı: tüm görevler aynı train_ds / test_ds objesinden
    Subset alır — veri iki kez yüklenmez, bellek tasarrufu sağlanır.
    """
    # Tüm veriyi bir kez yükle
    train_ds = datasets.CIFAR100(root=root, train=True,  download=True, transform=TRANSFORM)
    test_ds  = datasets.CIFAR100(root=root, train=False, download=True, transform=TRANSFORM)

    tasks = []
    for task_id, class_ids in enumerate(TASK_CLASSES):
        class_set = set(class_ids)

        # Sadece bu göreve ait örnek indekslerini filtrele
        tr_idx = [i for i, y in enumerate(train_ds.targets) if y in class_set]
        te_idx = [i for i, y in enumerate(test_ds.targets)  if y in class_set]

        # Her görev ~500 eğitim örneği/sınıf × 10 sınıf = ~5000 eğitim örneği içerir
        train_loader = DataLoader(
            Subset(train_ds, tr_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # macOS'ta çok işçi "too many open files" hatasına yol açar
        )
        test_loader = DataLoader(
            Subset(test_ds, te_idx),
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )
        tasks.append(TaskData(task_id, class_ids, train_loader, test_loader))
    return tasks
