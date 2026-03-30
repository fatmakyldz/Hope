"""
Minimal replay buffer — sadece logit kalibrasyonu için.

nested_learning'de bu ihtiyaç yok çünkü dil modeli sürekli eski
token'ları görüyor. Image classification'da classifier head'in
kalibrasyon kaybetmemesi için küçük bir buffer gerekli.

Strateji: class başına en fazla `samples_per_class` örnek sakla.
Buffer sadece classifier'ı kalibre etmek için kullanılır —
backbone ve CMS'i etkilemez.
"""
from __future__ import annotations

import random
import torch
from torch import Tensor


class CalibrationBuffer:
    """Class-balanced reservoir buffer for classifier calibration."""

    def __init__(self, samples_per_class: int = 100) -> None:
        self.samples_per_class = samples_per_class
        self._store: dict[int, list[Tensor]] = {}  # class_id → list of images

    def add(self, images: Tensor, labels: Tensor) -> None:
        """Add batch to buffer using reservoir sampling."""
        for img, lbl in zip(images.cpu(), labels.cpu()):
            cid = int(lbl.item())
            if cid not in self._store:
                self._store[cid] = []
            buf = self._store[cid]
            if len(buf) < self.samples_per_class:
                buf.append(img.clone())
            else:
                # Reservoir sampling
                idx = random.randint(0, self.samples_per_class)
                if idx < self.samples_per_class:
                    buf[idx] = img.clone()

    def sample(self, n: int, device: torch.device) -> tuple[Tensor, Tensor] | tuple[None, None]:
        """Sample n examples uniformly across all stored classes."""
        all_imgs, all_lbls = [], []
        for cid, imgs in self._store.items():
            for img in imgs:
                all_imgs.append(img)
                all_lbls.append(cid)
        if not all_imgs:
            return None, None
        indices = random.sample(range(len(all_imgs)), min(n, len(all_imgs)))
        imgs = torch.stack([all_imgs[i] for i in indices]).to(device)
        lbls = torch.tensor([all_lbls[i] for i in indices], device=device)
        return imgs, lbls

    def __len__(self) -> int:
        return sum(len(v) for v in self._store.values())

    def num_classes(self) -> int:
        return len(self._store)
