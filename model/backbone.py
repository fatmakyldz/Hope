"""
Görüntü özelliklerini çıkaran ResNet18 omurga (backbone) modülü.

Standart ResNet18, 224x224 boyutlu ImageNet görüntüleri için tasarlanmıştır.
CIFAR-100 görüntüleri 32x32 olduğundan iki uyarlama yapılmıştır:
  1. İlk konvolüsyon katmanı: 7x7 stride-2 → 3x3 stride-1 (bilgi kaybını önler)
  2. MaxPool katmanı kaldırıldı (Identity ile değiştirildi)
Bu sayede küçük görüntüler erken aşamada çok fazla küçültülmeden işlenir.
"""
import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()

        # ─── ÖNCEDEN EĞİTİLMİŞ AĞIRLIKLAR ──────────────────────────────────
        # pretrained=True ise ImageNet'te önceden öğrenilmiş ağırlıklar yüklenir.
        # Bu sayede model sıfırdan öğrenmek zorunda kalmaz; kenar, doku gibi
        # genel görsel özellikler hazır gelir → çok daha yüksek doğruluk.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # ─── CIFAR-100 UYARLAMASI ────────────────────────────────────────────
        # 32x32 görüntü için ilk katmanı küçült: 7x7 stride-2 → 3x3 stride-1
        # Aksi halde görüntü ilk katmandan sonra 4x4'e düşer ve bilgi kaybolur.
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # MaxPool da çıkarılıyor (Identity = geç/işlem yapma)
        resnet.maxpool = nn.Identity()

        # Son sınıflandırma katmanını (fc) çıkarıp sadece özellik çıkarıcıyı al.
        # Bu, modelin çıkışı (B, 512, 1, 1) boyutunda bir özellik haritasıdır.
        self.net = nn.Sequential(*list(resnet.children())[:-1])  # avgpool'a kadar
        self.feature_dim = 512  # ResNet18'in çıkış boyutu

        # ─── DONDURMA SEÇENEĞİ ───────────────────────────────────────────────
        # freeze=True ise backbone'un ağırlıkları eğitim sırasında güncellenmez.
        # Bu, ImageNet özelliklerini korur ama CIFAR'a uyum sağlamaz.
        # freeze=False (varsayılan) ise çok küçük bir LR ile fine-tune yapılır.
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        out = self.net(x)       # (B, 512, 1, 1) — Global Average Pooling sonrası
        return out.flatten(1)   # (B, 512) — Düzleştirip vektör haline getir
