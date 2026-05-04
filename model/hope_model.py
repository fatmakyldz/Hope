"""
HOPEModel: ResNet18 + 4 Kademeli CMS + Lineer Sınıflandırıcı

HOPE (Hierarchical Online Plasticity Engine), iki aşamalı bir eğitim döngüsü kullanır:

  Geçiş-1 (Meta İleri):   logits, features = model(x)
  Öğretme Sinyali:         teach = compute_teach_signal(features, logits, labels, classifier)
  Geçiş-2 (CMS Güncelle): model.cms.update(features, teach)   # sadece hızlı ağırlıklar
  Meta Geri Yayılım:       loss.backward(); optimizer.step()  # backbone + sınıflandırıcı

Bu iki aşamalı yapı, CMS'nin meta optimizer'dan bağımsız öğrenmesini sağlar.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import ResNetBackbone, ViTBackbone
from .cms import CMSModule


# ─── ÖĞRETME SİNYALİ (TEACH SIGNAL) ────────────────────────────────────────
def compute_teach_signal(
    features: Tensor,
    logits: Tensor,
    labels: Tensor,
    classifier: nn.Linear,
) -> Tensor:
    """
    Cross-Entropy kaybının özellik vektörlerine göre kapalı-form gradyanı.

    Matematiksel türetme:
        grad_logits   = (softmax(logits) - one_hot(labels)) / B   # CE'nin logit gradyanı
        grad_features = grad_logits @ W    [W: sınıflandırıcı ağırlığı (C, D)]
        teach         = -grad_features     (iyileştirme = negatif gradyan yönü)

    Bu hesaplama autograd kullanmaz; torch.no_grad() içinde güvenle çağrılabilir.
    nested_learning/training.py'deki compute_teach_signal() ile birebir aynıdır.

    Neden bu yöntemi kullanıyoruz?
    - Geri yayılım yapmadan CMS'nin öğrenmesini sağlar (hızlı ve verimli).
    - Model parametrelerini bozmadan yalnızca CMS hızlı ağırlıklarını günceller.
    """
    with torch.no_grad():
        B = features.size(0)
        p = torch.softmax(logits.detach(), dim=-1)           # (B, C) — sınıf olasılıkları
        p[torch.arange(B, device=p.device), labels] -= 1.0  # one-hot çıkar → gradyan
        p = p / B                                            # batch ortalaması
        W = classifier.weight.detach()                       # (C, D) — sınıflandırıcı ağırlığı
        teach = -(p @ W)                                     # (B, D) — öğretme yönü
    return teach


# ─── ANA MODEL ───────────────────────────────────────────────────────────────
class HOPEModel(nn.Module):
    """
    Üç bileşenli HOPE modeli:
      1. backbone  : ResNet18 — ham görüntüden 512 boyutlu özellik çıkarır
      2. cms       : 4 kademeli CMS — özellikleri hızlı/yavaş bellek ile dönüştürür
      3. classifier: Lineer katman — CMS çıkışından sınıf tahmini yapar
    """

    def __init__(
        self,
        num_classes: int = 100,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        backbone_type: str = "resnet",
        cms_hidden_multiplier: int = 4,
        cms_grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        if backbone_type == "vit":
            self.backbone = ViTBackbone(pretrained=pretrained, freeze=freeze_backbone)
        else:
            self.backbone = ResNetBackbone(pretrained=pretrained, freeze=freeze_backbone)
        self.cms = CMSModule(
            dim=self.backbone.feature_dim,
            hidden_multiplier=cms_hidden_multiplier,
            grad_clip=cms_grad_clip,
        )
        # 512 boyutlu CMS çıkışını 100 sınıfa eşleyen lineer katman
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        İleri geçiş — üç değer döndürür:
          logits       : (B, num_classes) — sınıf skorları
          backbone_feat: (B, dim)         — ham backbone özellikleri (CMS update için)
          cms_out      : (B, dim)         — CMS çıkışı (NCM değerlendirmesi için)

        İki ayrı özellik döndürme nedeni:
        - CMS update backbone özelliğini INPUT olarak alır (kendi girişini öğrenir)
        - NCM cms_out ile yapılır ki CMS'in katkısı değerlendirmeye yansısın
        """
        backbone_feat = self.backbone(x)       # (B, dim) — ham backbone özellikleri
        cms_out = self.cms(backbone_feat)      # (B, dim) — CMS dönüşümü uygulanmış
        logits = self.classifier(cms_out)      # (B, 100) — sınıf skorları
        return logits, backbone_feat, cms_out

    def update_cms(self, features: Tensor, teach: Tensor) -> None:
        """Geçiş-2: öğretme sinyaliyle CMS hızlı ağırlıklarını günceller."""
        self.cms.update(features.detach(), teach.detach())

    def meta_parameters(self) -> list[nn.Parameter]:
        """
        Meta optimizer için parametreler: backbone + sınıflandırıcı.
        CMS hızlı parametreleri HARİÇ tutulur — onlar DeepMomentum ile güncellenir.
        """
        cms_ids = {id(p) for p in self.cms.all_fast_params()}
        return [p for p in self.parameters() if id(p) not in cms_ids]

    def meta_param_groups(
        self,
        backbone_lr: float = 1e-4,
        classifier_lr: float = 1e-3,
    ) -> list[dict]:
        """
        Backbone ve sınıflandırıcı için farklı öğrenme hızı grupları.
        Backbone çok küçük LR alır (önceden öğrenilmiş ağırlıkları bozmamak için).
        Sınıflandırıcı daha büyük LR alır (her task'ta yeni sınıflar eklendiği için).
        """
        cms_ids = {id(p) for p in self.cms.all_fast_params()}
        backbone_params, cls_params = [], []
        for name, p in self.named_parameters():
            if id(p) in cms_ids:
                continue
            if "backbone" in name:
                backbone_params.append(p)
            else:
                cls_params.append(p)
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if cls_params:
            groups.append({"params": cls_params, "lr": classifier_lr})
        return groups

    def on_task_boundary(self) -> None:
        """
        Görev sınırında çağrılır: fast + mid CMS kademe ağırlıklarını sıfırlar.
        slow + ultra korunur — bunlar uzun vadeli belleği temsil eder.
        Bu, nested_learning'in temel tasarım kararıdır.
        """
        self.cms.reset_fast()
