"""
DeepMomentum — CMS hızlı ağırlık güncellemeleri için Adam benzeri optimizer.

Standart PyTorch optimizer'larından farkı:
- Her parametre tensörü için bağımsız durum (m1, m2) tutar
- Görev sınırında seçici olarak sıfırlanabilir (sadece fast/mid için)
- Meta optimizer ile hiçbir ilişkisi yoktur; sadece CMS içinden çağrılır

nested_learning/src/nested_learning/optim/deep.py ile aynı güncelleme kuralı:
    m1 = β  × m1 + (1-β)  × grad         (1. moment — yön bilgisi)
    m2 = β2 × m2 + (1-β2) × grad²        (2. moment — büyüklük bilgisi)
    m1_hat = m1 / (1 - β^t)              (bias düzeltmesi)
    m2_hat = m2 / (1 - β2^t)
    param  += -lr × m1_hat / (√m2_hat + ε)
"""
from __future__ import annotations
import torch
from torch import Tensor
import torch.nn as nn


class DeepMomentum:
    """
    Parametre başına durum tutan Adam benzeri hızlı ağırlık optimizer'ı.

    Neden standart AdamW kullanmıyoruz?
    - CMS ağırlıkları meta optimizer'a verilmez (nn.Parameter olarak kayıtlı olsalar da)
    - Görev sınırında sadece belirli kademelerin durumunu sıfırlamak gerekir
    - Bu sınıf tam olarak bu esnekliği sağlar
    """

    def __init__(
        self,
        beta: float = 0.9,     # 1. moment katsayısı (momentum)
        beta2: float = 0.999,  # 2. moment katsayısı (adaptif ölçek)
        eps: float = 1e-8,     # sıfıra bölmeyi önleyen küçük sabit
    ) -> None:
        self.beta = beta
        self.beta2 = beta2
        self.eps = eps
        # Her parametre için ayrı durum sözlükleri (param id → tensör)
        self._m1: dict[int, Tensor] = {}   # 1. moment (ortalama gradient)
        self._m2: dict[int, Tensor] = {}   # 2. moment (kare gradient ortalaması)
        self._step: dict[int, int] = {}    # bias düzeltmesi için adım sayacı

    def step(self, param: nn.Parameter, grad: Tensor, lr: float) -> None:
        """
        Tek bir parametre tensörüne bir güncelleme adımı uygular.

        İlk çağrıda m1, m2 sıfırdan başlar (soğuk başlangıç).
        Sonraki çağrılarda birikmeli momentum kullanılır.
        """
        pid = id(param)

        # ─── İLK KEZ GÖRÜLMESİ ───────────────────────────────────────────────
        if pid not in self._m1:
            self._m1[pid] = torch.zeros_like(grad)
            self._m2[pid] = torch.zeros_like(grad)
            self._step[pid] = 0

        self._step[pid] += 1
        t = self._step[pid]

        # ─── MOMENT GÜNCELLEMESİ ─────────────────────────────────────────────
        # m1: gradyanın üstel hareketli ortalaması (yön bilgisi)
        self._m1[pid] = self.beta * self._m1[pid] + (1 - self.beta) * grad
        # m2: gradyan karesinin üstel hareketli ortalaması (büyüklük bilgisi)
        self._m2[pid] = self.beta2 * self._m2[pid] + (1 - self.beta2) * grad * grad

        # ─── BİAS DÜZELTMESİ ─────────────────────────────────────────────────
        # İlk adımlarda m1 ve m2 sıfıra yakındır; bu düzeltme bunu telafi eder
        m1_hat = self._m1[pid] / (1 - self.beta ** t)
        m2_hat = self._m2[pid] / (1 - self.beta2 ** t)

        # ─── PARAMETRE GÜNCELLEMESİ ──────────────────────────────────────────
        # Adaptif adım: büyük gradyanlarda daha küçük güncelleme yapılır
        update = m1_hat / (m2_hat.sqrt() + self.eps)
        param.add_(update, alpha=-lr)  # param -= lr × update

    def reset(self) -> None:
        """
        Tüm durum bilgisini temizler.
        Görev sınırında fast + mid CMS kademeleri için çağrılır.
        Bu sayede yeni görev, eski görevin momentum'undan etkilenmeden başlar.
        """
        self._m1.clear()
        self._m2.clear()
        self._step.clear()
