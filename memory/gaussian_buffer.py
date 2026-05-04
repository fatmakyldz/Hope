"""
GaussianBuffer — Sınıf başına Gaussian istatistikleri (mean + covariance) saklar.

TUNA'nın classifier alignment fikrinden uyarlanmıştır.

Geleneksel replay: gerçek görüntüler saklar → bellek pahalı, sınırlı örnek
Bu yaklaşım: her sınıfı Gaussian ile modeller → sonsuz sentetik özellik üretilir

Akış:
  1. Görev sonunda: backbone özelliklerinden mean + cov hesapla
  2. classifier_align() içinde: Gaussian'dan sahte özellikler üret
  3. Sahte özelliklerle classifier'ı kalibre et → softmax drift ortadan kalkar
  4. NCM'de: mean vektörleri class prototype olarak kullanılır
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianBuffer:
    """
    Sınıf başına Gaussian istatistik deposu.

    CalibrationBuffer'ın yerini alır:
    - Görüntü saklamaz → bellek neredeyse sıfır
    - Sonsuz örnek üretebilir → daha güçlü kalibrasyon
    - Hem NCM hem de classifier alignment için kullanılır
    """

    def __init__(self) -> None:
        self.cls_mean: dict[int, Tensor] = {}   # class_id → (dim,) CPU tensörü
        self.cls_cov:  dict[int, Tensor] = {}   # class_id → (dim, dim) CPU tensörü

    @torch.no_grad()
    def update(self, model, loader, device: torch.device, class_ids: list[int]) -> None:
        """
        Verilen sınıflar için backbone özelliklerinden mean + covariance hesaplar.
        Her görev sonunda yalnızca yeni görevin sınıfları için çağrılır.
        """
        model.eval()
        class_feats: dict[int, list[Tensor]] = {cid: [] for cid in class_ids}

        for images, labels in loader:
            images = images.to(device)
            _, backbone_feat, _ = model(images)  # backbone_feat: CMS öncesi, daha kararlı
            for feat, lbl in zip(backbone_feat.cpu(), labels.cpu()):
                cid = int(lbl.item())
                if cid in class_feats:
                    class_feats[cid].append(feat)

        for cid, feats in class_feats.items():
            if not feats:
                continue
            feats_t = torch.stack(feats)  # (N, dim)
            self.cls_mean[cid] = feats_t.mean(dim=0)
            # Covariance + diyagonal düzeltme (pozitif kesin garantisi için)
            cov = torch.cov(feats_t.T) if feats_t.shape[0] > 1 else torch.zeros(feats_t.shape[1], feats_t.shape[1])
            self.cls_cov[cid] = cov + torch.eye(feats_t.shape[1]) * 1e-4

    def sample(self, n_per_class: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Tüm sınıflar için Gaussian'dan sentetik özellik vektörleri üretir.
        Sınıflandırıcı bu sahte özellikler üzerinde kalibre edilir.
        """
        all_feats, all_labels = [], []
        for cid in sorted(self.cls_mean.keys()):
            mean = self.cls_mean[cid].float().to(device)
            cov  = self.cls_cov[cid].float().to(device)
            dist = MultivariateNormal(mean, cov)
            all_feats.append(dist.sample((n_per_class,)))
            all_labels.extend([cid] * n_per_class)
        return torch.cat(all_feats, dim=0), torch.tensor(all_labels, device=device)

    def get_class_means(self) -> dict[int, Tensor]:
        """NCM için L2-normalize edilmiş class mean vektörleri."""
        return {cid: F.normalize(mean, dim=0) for cid, mean in self.cls_mean.items()}

    def num_classes(self) -> int:
        return len(self.cls_mean)

    def __len__(self) -> int:
        return self.num_classes()
