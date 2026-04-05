"""
CMS — Continual Memory System (Sürekli Bellek Sistemi)

HOPE'un kalbi olan bu modül, 4 kademeli hiyerarşik bir bellek yapısı kurar.
Her kademe farklı hızda güncellenir: hızlı olanlar yeni bilgiyi anında alır,
yavaş olanlar ise uzun vadeli bilgiyi korur.

Kademe hiyerarşisi (fast → mid → slow → ultra):
    x0 = backbone özellikleri   (512 boyut)
    x1 = fast(x0)               her adımda güncellenir   (period: 1)
    x2 = mid(x1)                her 4 adımda güncellenir (period: 4)
    x3 = slow(x2)               her 32 adımda güncellenir
    x4 = ultra(x3)              her 128 adımda güncellenir

Her kademe bir residual MLP bloğudur:
    y = x + clip(Linear(GELU(Linear(LayerNorm(x)))))

Neden bu mimari?
- Fast: mevcut görevdeki ani değişimlere hızla uyum sağlar
- Mid: görev içi orta vadeli örüntüleri yakalar
- Slow: görevler arası geçişte bilgiyi korur
- Ultra: tüm görevler boyunca birikmiş uzun vadeli belleği tutar
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from optim.deep_momentum import DeepMomentum


# ─── GÜNCELLEME PERİYOTLARI ─────────────────────────────────────────────────
# Her kademe kaç adımda bir güncelleneceğini belirtir.
# nested_learning pilot.yaml'dan alınmıştır.
PERIODS = {
    "fast":  1,    # her adımda
    "mid":   4,    # her 4 adımda bir
    "slow":  32,   # her 32 adımda bir
    "ultra": 128,  # her 128 adımda bir
}

# ─── ÖĞRENME HIZLARI ─────────────────────────────────────────────────────────
# CMS güncelleme hızları (meta optimizer'dan bağımsız)
LR = {
    "fast":  4e-4,
    "mid":   4e-4,
    "slow":  4e-4,
    "ultra": 4e-4,
}


# ─── TEK KADEME BLOĞU ────────────────────────────────────────────────────────
class CMSBlock(nn.Module):
    """
    Tek CMS kademesinin residual MLP bloğu.

    Yapı: y = x + clip(Linear(GELU(Linear(LayerNorm(x)))))

    LayerNorm: gradyan patlamalarını önler, eğitimi kararlı tutar
    GELU: ReLU'ya göre daha düzgün gradyan akışı sağlar
    Residual bağlantı: girdiyi olduğu gibi çıktıya ekler → bilgi kaybolmaz
    Gradient clip: büyük güncellemeleri sınırlar → kararlı öğrenme
    """

    def __init__(self, dim: int, hidden_multiplier: int = 4, grad_clip: float = 1.0) -> None:
        super().__init__()
        hidden = dim * hidden_multiplier  # gizli katman boyutu: 512 × 4 = 2048
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),   # genişletme: 512 → 2048
            nn.GELU(),
            nn.Linear(hidden, dim),   # daraltma: 2048 → 512
        )
        self.grad_clip = grad_clip

    def forward(self, x: Tensor) -> Tensor:
        delta = self.net(self.norm(x))   # öğrenilmiş dönüşüm
        # Eğitim sırasında büyük delta değerlerini sınırla
        if self.training and self.grad_clip > 0:
            with torch.no_grad():
                nv = delta.norm(dim=-1, keepdim=True).clamp(min=self.grad_clip)
                scale = nv / self.grad_clip
            delta = delta / scale
        return x + delta  # residual bağlantı: girdiyi koru, sadece küçük düzeltme ekle

    def fast_params(self) -> list[nn.Parameter]:
        """Bu kademenin hızlı ağırlıkları (meta optimizer tarafından güncellenmez)."""
        return list(self.net.parameters()) + list(self.norm.parameters())


# ─── 4 KADEMELİ CMS MODÜLÜ ───────────────────────────────────────────────────
class CMSModule(nn.Module):
    """
    4 kademeli CMS: fast → mid → slow → ultra zincirleme çalışır.

    Önemli ayrım:
    - İleri geçiş (forward): tüm kademe çıkışlarını hesaplar, ağırlık güncellenmez
    - Güncelleme (update): öğretme sinyaliyle seçili kademelerin ağırlıklarını değiştirir
    - Meta optimizer bu parametrelere hiç dokunmaz
    """

    def __init__(self, dim: int = 512, hidden_multiplier: int = 4, grad_clip: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        # 4 kademeyi isimli bir ModuleDict içinde sakla
        self.levels = nn.ModuleDict({
            name: CMSBlock(dim, hidden_multiplier, grad_clip)
            for name in ("fast", "mid", "slow", "ultra")
        })
        # Her kademe için bağımsız DeepMomentum optimizer
        # (nn.Module değil — PyTorch state_dict'e dahil edilmez)
        self._opts: dict[str, DeepMomentum] = {
            name: DeepMomentum() for name in ("fast", "mid", "slow", "ultra")
        }
        self._global_step: int = 0  # kaç kez update() çağrıldığını sayar

    # ─── İLERİ GEÇİŞ ─────────────────────────────────────────────────────────
    def forward(self, x: Tensor) -> Tensor:
        """
        4 kademeyi sırayla uygular.
        Her kademenin çıkışı bir sonrakinin girişi olur (zincirleme).
        Ağırlık güncellenmez — sadece ileri hesaplama.
        """
        for name in ("fast", "mid", "slow", "ultra"):
            x = self.levels[name](x)
        return x

    # ─── HIZLI AĞIRLIK GÜNCELLEMESİ ─────────────────────────────────────────
    @torch.no_grad()
    def update(self, x: Tensor, teach_signal: Tensor) -> None:
        """
        Öğretme sinyaliyle CMS hızlı ağırlıklarını günceller (Geçiş-2).

        Her kademe için periyot dolmuşsa:
            kayıp = -mean(teach_signal × delta)   # hizalama kaybı
            grad  = autograd(kayıp, kademe.parametreler)
            param += -lr × deep_momentum(grad)    # Adam benzeri güncelleme

        Bu yöntem nested_learning/memorize.py'deki memorize_tokens() ile aynıdır.

        Neden negatif kayıp?
        Öğretme sinyali "özelliğin hangi yönde iyileşmesi gerektiğini" gösterir.
        delta'yı bu yönle hizalamak (çarpım maksimize etmek) kaybı minimize eder.
        """
        self._global_step += 1
        h = x  # ilk kademeye giren özellik vektörü

        for name in ("fast", "mid", "slow", "ultra"):
            level = self.levels[name]
            period = PERIODS[name]
            lr = LR[name]

            # Bu kademe bu adımda güncellenmeli mi?
            if self._global_step % period == 0:
                with torch.enable_grad():
                    # Gradient hesaplamak için geçici olarak enable_grad aç
                    h_detached = h.detach().requires_grad_(False)
                    delta = level.net(level.norm(h_detached))
                    # Hizalama kaybı: öğretme sinyali ile delta ne kadar uyuşuyor?
                    loss = -(teach_signal.detach() * delta).mean()
                grads = torch.autograd.grad(loss, level.net.parameters(), allow_unused=True)
                opt = self._opts[name]
                for param, grad in zip(level.net.parameters(), grads):
                    if grad is not None:
                        opt.step(param, grad, lr)  # DeepMomentum ile güncelle

            # h'yi bu kademe üzerinden ilerlet (sadece ileri hesaplama)
            h = h + level.net(level.norm(h)).detach()

    # ─── GÖREV SINIRI İŞLEMLERİ ──────────────────────────────────────────────
    def reset_fast(self) -> None:
        """
        Görev sınırında fast + mid kademelerini sıfırlar.
        slow + ultra korunur → uzun vadeli bellek bozulmaz.

        Bu, HOPE'un temel unutmama mekanizmasıdır:
        - Hızlı kademeler yeni göreve sıfırdan başlar (yeni göreve uyum)
        - Yavaş kademeler birikmiş bilgiyi korur (eski görevleri hatırlama)
        """
        for name in ("fast", "mid"):
            block = self.levels[name]
            for m in block.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)  # ağırlıkları sıfırla
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self._opts[name].reset()  # momentum durumunu da sıfırla

    def reset_all(self) -> None:
        """Tüm kademeleri sıfırlar (ablasyon deneyi için)."""
        for name in ("fast", "mid", "slow", "ultra"):
            block = self.levels[name]
            for m in block.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self._opts[name].reset()
        self._global_step = 0

    def all_fast_params(self) -> list[nn.Parameter]:
        """Tüm CMS parametreleri — meta optimizer'a verilmez."""
        params = []
        for level in self.levels.values():
            params.extend(level.fast_params())
        return params
