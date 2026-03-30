"""
Deep Momentum optimizer — per-parameter Adam-like update.

Matches nested_learning/src/nested_learning/optim/deep.py
Used for fast weight updates inside CMS levels (not the meta optimizer).

Update rule:
    m1 = beta  * m1 + (1-beta)  * grad
    m2 = beta2 * m2 + (1-beta2) * grad^2
    update = m1 / (sqrt(m2) + eps)
    param += -lr * update
"""
from __future__ import annotations
import torch
from torch import Tensor
import torch.nn as nn


class DeepMomentum:
    """Stateful per-tensor Adam-like optimizer for fast weight updates."""

    def __init__(
        self,
        beta: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.beta = beta
        self.beta2 = beta2
        self.eps = eps
        self._m1: dict[int, Tensor] = {}
        self._m2: dict[int, Tensor] = {}
        self._step: dict[int, int] = {}

    def step(self, param: nn.Parameter, grad: Tensor, lr: float) -> None:
        """Apply one update step to param using grad."""
        pid = id(param)
        if pid not in self._m1:
            self._m1[pid] = torch.zeros_like(grad)
            self._m2[pid] = torch.zeros_like(grad)
            self._step[pid] = 0

        self._step[pid] += 1
        t = self._step[pid]

        self._m1[pid] = self.beta * self._m1[pid] + (1 - self.beta) * grad
        self._m2[pid] = self.beta2 * self._m2[pid] + (1 - self.beta2) * grad * grad

        # Bias-corrected estimates
        m1_hat = self._m1[pid] / (1 - self.beta ** t)
        m2_hat = self._m2[pid] / (1 - self.beta2 ** t)

        update = m1_hat / (m2_hat.sqrt() + self.eps)
        param.add_(update, alpha=-lr)

    def reset(self) -> None:
        """Clear all state (call at task boundary for fast/mid levels)."""
        self._m1.clear()
        self._m2.clear()
        self._step.clear()
