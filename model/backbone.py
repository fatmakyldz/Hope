"""
Görüntü özelliklerini çıkaran omurga (backbone) modülleri.

Bu dosyada iki backbone seçeneği bulunmaktadır:

1. ResNetBackbone — CIFAR-100 için uyarlanmış ResNet18 (512 boyut)
   - Küçük 32×32 görüntüler için ilk konvolüsyon uyarlaması yapılmıştır.

2. ViTBackbone — Vision Transformer ViT-B/16 (768 boyut)
   - 224×224 girdi gerektirir (data/cifar100.py'de Resize(224) eklendi).
   - ImageNet önceden eğitilmiş ağırlıklar kullanılır.
   - CLS token özellik vektörü olarak alınır (tüm patch'lerin özeti).
   - Standart ResNet'e göre daha zengin global bağlam bilgisi taşır.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential
from torchvision import models


# ─── SEÇENEK 1: RESNET18 ──────────────────────────────────────────────────────
class ResNetBackbone(nn.Module):
    """
    CIFAR-100 için uyarlanmış ResNet18 omurgası.

    Standart ResNet18, 224×224 boyutlu ImageNet görüntüleri için tasarlanmıştır.
    CIFAR-100 görüntüleri 32×32 olduğundan iki uyarlama yapılmıştır:
      1. İlk konvolüsyon katmanı: 7×7 stride-2 → 3×3 stride-1 (bilgi kaybını önler)
      2. MaxPool katmanı kaldırıldı (Identity ile değiştirildi)
    Bu sayede küçük görüntüler erken aşamada çok fazla küçültülmeden işlenir.
    """
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # 32×32 görüntü için ilk katmanı küçült: 7×7 stride-2 → 3×3 stride-1
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()

        self.net = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        out = self.net(x)       # (B, 512, 1, 1)
        return out.flatten(1)   # (B, 512)


# ─── SEÇENEK 2: VIT-B/16 ─────────────────────────────────────────────────────
class ViTBackbone(nn.Module):
    """
    Vision Transformer ViT-B/16 omurgası.

    224×224 girdi alır ve 768 boyutlu CLS token özellik vektörü döndürür.

    ViT nasıl çalışır?
    - Görüntü 16×16 piksellik 196 adet "patch"e bölünür.
    - Her patch doğrusal projeksiyon ile 768 boyutlu vektöre dönüştürülür.
    - Başına özel bir [CLS] token eklenir (toplam 197 token).
    - Transformer encoder (12 katman, 12 kafa) tüm tokenlar üzerinde self-attention yapar.
    - Çıkıştaki CLS token, tüm görüntünün global özetini temsil eder.
    - Sınıflandırma kafası (heads) kaldırılır; CLS token özellik vektörü olarak kullanılır.

    ResNet'e göre avantajı:
    - Global bağlam: tüm patch'ler arasında dikkat mekanizması
    - Daha zengin temsil: 768 boyut vs ResNet18'in 512 boyutu
    """

    def __init__(self, pretrained: bool = True, freeze: bool = False,
                 grad_checkpoint: bool = False):
        super().__init__()

        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit = models.vit_b_16(weights=weights)
        vit.heads = nn.Identity()

        self.vit = vit
        self.feature_dim = 768
        self.grad_checkpoint = grad_checkpoint

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        # (B, 3, 224, 224) → (B, 768)
        if self.grad_checkpoint and self.training:
            # Encoder katmanlarını checkpoint_sequential ile sar → ~3x bellek tasarrufu
            vit = self.vit
            x = vit._process_input(x)
            n = x.shape[0]
            batch_class_token = vit.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = checkpoint_sequential(vit.encoder.layers, segments=4, input=x)
            x = vit.encoder.ln(x)
            return x[:, 0]  # CLS token
        return self.vit(x)
