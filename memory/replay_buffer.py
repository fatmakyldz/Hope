"""
CalibrationBuffer — Sınıflandırıcı kalibrasyonu için minimal tekrar belleği.

Problem: Softmax sınıflandırıcı, yeni görevler öğrendikçe eski sınıfların
logit değerlerini küçültür (softmax paydasının büyümesi). Bu durum eski
görevlerde doğruluğun sıfıra düşmesine neden olur.

Çözüm: Her sınıftan birkaç örnek sakla ve her eğitim adımında eski sınıfları
da görmeyi sağla. Backbone ve CMS bu replay'den etkilenmez (frozen backbone
veya çok küçük LR ile), yalnızca sınıflandırıcı kalibre edilir.

NCM (Nearest Class Mean) ile birlikte kullanılır:
- Buffer, her sınıfın ortalama özellik vektörünü (class mean) hesaplamak için
  test zamanında da kullanılır.
- Tahmin, softmax yerine cosine benzerliğiyle yapılır → softmax kaymasına karşı
  bağışıklık kazanılır.

Strateji: Reservoir Sampling — her sınıf için eşit sayıda örnek tutulur,
yeni örnekler rastgele eski örneklerin yerine geçer (uniform bias yok).
"""
from __future__ import annotations

import random
import torch
from torch import Tensor


class CalibrationBuffer:
    """
    Sınıf dengeli reservoir buffer.

    Her sınıf için maksimum `samples_per_class` görüntü saklar.
    Bellek dolunca reservoir sampling ile eski örneklerin üzerine yazar.
    """

    def __init__(self, samples_per_class: int = 100) -> None:
        self.samples_per_class = samples_per_class
        # sınıf_id → görüntü listesi eşlemesi
        self._store: dict[int, list[Tensor]] = {}
        # DÜZELTİLDİ: sınıf başına kaç örnek görüldüğünü say (k/n için gerekli)
        self._seen: dict[int, int] = {}

    def add(self, images: Tensor, labels: Tensor) -> None:
        """
        Bir batch görüntüyü buffer'a ekler (doğru reservoir sampling ile).

        Doğru reservoir sampling garantisi:
        - İlk k örnek: direkt eklenir
        - n. örnek (n > k): k/n olasılıkla buffer'daki rastgele bir yeri geçer
        → Her örnek eşit olasılıkla (k/toplam) temsil edilir, yeni/eski bias yok.

        CPU'da saklanır (GPU belleği tasarrufu için).
        """
        for img, lbl in zip(images.cpu(), labels.cpu()):
            cid = int(lbl.item())
            if cid not in self._store:
                self._store[cid] = []
                self._seen[cid] = 0
            buf = self._store[cid]
            self._seen[cid] += 1
            n = self._seen[cid]  # bu sınıf için şimdiye kadar görülen toplam örnek

            if len(buf) < self.samples_per_class:
                # Buffer dolmamış: direkt ekle
                buf.append(img.clone())
            else:
                # DÜZELTİLDİ: k/n olasılığıyla kabul et (n büyüdükçe olasılık azalır)
                # Önceki kod: random.randint(0, k) → sabit k/(k+1) olasılık → yeni örneklere bias
                idx = random.randint(0, n - 1)
                if idx < self.samples_per_class:
                    buf[idx] = img.clone()

    def sample(self, n: int, device: torch.device) -> tuple[Tensor, Tensor] | tuple[None, None]:
        """
        Buffer'daki tüm sınıflardan dengeli şekilde n örnek örnekler.

        Tüm örnekleri düzleştirip rastgele n tane seçer.
        Bu, eski sınıfların replay sırasında eşit temsil edilmesini sağlar.
        """
        all_imgs, all_lbls = [], []
        for cid, imgs in self._store.items():
            for img in imgs:
                all_imgs.append(img)
                all_lbls.append(cid)

        if not all_imgs:
            return None, None  # buffer boş

        # min(n, toplam örnek sayısı) kadar seç (buffer yetersizse hepsini al)
        indices = random.sample(range(len(all_imgs)), min(n, len(all_imgs)))
        imgs = torch.stack([all_imgs[i] for i in indices]).to(device)
        lbls = torch.tensor([all_lbls[i] for i in indices], device=device)
        return imgs, lbls

    def __len__(self) -> int:
        """Toplam saklanan örnek sayısı."""
        return sum(len(v) for v in self._store.values())

    def num_classes(self) -> int:
        """Buffer'da kaç farklı sınıf var."""
        return len(self._store)
