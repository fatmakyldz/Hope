"""
Classifier Alignment — Gaussian örneklerle sınıflandırıcı kalibrasyonu.

TUNA'nın classifer_align() metodundan uyarlanmıştır.

Problem: Yeni görevler öğrenildikçe classifier ağırlıkları yeni sınıflara kayar.
Eski sınıfların logitleri sistematik olarak küçülür (softmax drift).

Çözüm: Gerçek görüntü yerine Gaussian'dan sentetik özellikler üret,
tüm sınıfları aynı anda görerek classifier'ı yeniden kalibre et.

Neden işe yarar?
- Her sınıfın özellik dağılımı Gaussian ile modellenmiş
- Sentetik örnekler gerçek veriyle aynı dağılımdan geliyor
- Tüm sınıflar eşit temsil edildiği için bias oluşmuyor
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─── COSFACE KAYIP FONKSİYONU ────────────────────────────────────────────────
class CosFaceLoss(nn.Module):
    """
    CosFace angular penalty loss (Wang et al. 2018).

    Standart CE'den farkı: logitleri cosine benzerliği üzerinden hesaplar,
    doğru sınıfa ek margin ekler → sınıflar arası açısal mesafe artar.

    s: scale faktörü (cosine değerleri küçük olduğu için CE için büyütülür)
    m: margin (doğru sınıfın cosine değerinden çıkarılır)

    m=0 ile: scaled cosine similarity → NCM ile tutarlı (ikisi de cosine kullanır)
    m>0 ile: daha geniş margin → sınıflar arası ayrım güçlenir
    """

    def __init__(self, s: float = 20.0, m: float = 0.0) -> None:
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, cosine_logits: Tensor, labels: Tensor) -> Tensor:
        """
        cosine_logits: (B, C) — normalize özellik @ normalize ağırlık, [-1, 1]
        labels: (B,) — sınıf indeksleri
        """
        if self.m == 0.0:
            # Sadece scale: standart CE gibi ama cosine üzerinden
            return F.cross_entropy(self.s * cosine_logits, labels)

        # Doğru sınıfın cosine değerinden margin çıkar
        B = cosine_logits.size(0)
        logits = self.s * cosine_logits.clone()
        logits[torch.arange(B), labels] -= self.s * self.m
        return F.cross_entropy(logits, labels)


# ─── CLASSIFIER ALIGNMENT ────────────────────────────────────────────────────
def classifier_align(
    classifier: nn.Linear,
    gaussian_buffer,
    device: torch.device,
    epochs: int = 30,
    lr: float = 0.005,
    n_per_class: int = 256,
    scale: float = 20.0,
    cosface: bool = False,
) -> None:
    """
    Gaussian sentetik özelliklerle sınıflandırıcıyı kalibre eder.

    Sadece classifier ağırlıkları güncellenir — backbone ve CMS dokunulmaz.
    Bu sayede öğrenilmiş özellikler bozulmadan classifier yeniden hizalanır.

    Parametreler:
        epochs     : kalibrasyon epoch sayısı (TUNA: 30)
        n_per_class: her sınıftan üretilecek sentetik örnek sayısı
        scale      : cosine logit ölçek faktörü (CosFace ile tutarlılık için)
        cosface    : True ise CosFace loss, False ise standart CE
    """
    criterion = CosFaceLoss(s=scale, m=0.0) if cosface else None

    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    classifier.train()

    for _ in range(epochs):
        # Tüm sınıflardan Gaussian örnekler al
        feats, labels = gaussian_buffer.sample(n_per_class, device)

        # Karıştır
        perm = torch.randperm(feats.size(0), device=device)
        feats, labels = feats[perm], labels[perm]

        if cosface:
            # CosFace: hem özellik hem ağırlık normalize edilmeli
            norm_feats = F.normalize(feats, dim=1)
            norm_weight = F.normalize(classifier.weight, dim=1)
            cosine_logits = norm_feats @ norm_weight.T  # (B, C)
            if classifier.bias is not None:
                cosine_logits = cosine_logits + classifier.bias
            loss = criterion(cosine_logits, labels)
        else:
            logits = classifier(feats)
            loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    classifier.eval()
