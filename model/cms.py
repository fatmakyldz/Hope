"""
CMS -- Continual Memory System (4-level hierarchical fast memory).

Faithful port of nested_learning/src/nested_learning/cms.py

Architecture per level:
    y = x + clip(Linear(GELU(Linear(LayerNorm(x)))))   # residual MLP

4 levels (fast->mid->slow->ultra) cascade:
    x0 = backbone_features
    x1 = fast(x0)    updated every step
    x2 = mid(x1)     updated every 4 steps
    x3 = slow(x2)    updated every 32 steps
    x4 = ultra(x3)   updated every 128 steps

Each level has its own DeepMomentum optimizer for fast weight updates.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from optim.deep_momentum import DeepMomentum


# -- Update periods (matches nested_learning pilot.yaml) ----------------------
PERIODS = {
    "fast":  1,
    "mid":   4,
    "slow":  32,
    "ultra": 128,
}

# -- LR per level (all same in nested_learning cms_opt) -----------------------
LR = {
    "fast":  4e-4,
    "mid":   4e-4,
    "slow":  4e-4,
    "ultra": 4e-4,
}


class CMSBlock(nn.Module):
    """Single CMS residual MLP block (one level)."""

    def __init__(self, dim: int, hidden_multiplier: int = 4, grad_clip: float = 1.0) -> None:
        super().__init__()
        hidden = dim * hidden_multiplier
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.grad_clip = grad_clip

    def forward(self, x: Tensor) -> Tensor:
        delta = self.net(self.norm(x))
        if self.training and self.grad_clip > 0:
            with torch.no_grad():
                nv = delta.norm(dim=-1, keepdim=True).clamp(min=self.grad_clip)
                scale = nv / self.grad_clip
            delta = delta / scale
        return x + delta

    def fast_params(self) -> list[nn.Parameter]:
        return list(self.net.parameters()) + list(self.norm.parameters())


class CMSModule(nn.Module):
    """
    4-level CMS: fast -> mid -> slow -> ultra.

    Fast weights updated via teach_signal (deep momentum).
    Meta optimizer never touches these parameters.
    """

    def __init__(self, dim: int = 512, hidden_multiplier: int = 4, grad_clip: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.levels = nn.ModuleDict({
            name: CMSBlock(dim, hidden_multiplier, grad_clip)
            for name in ("fast", "mid", "slow", "ultra")
        })
        # Per-level deep momentum optimizers (NOT nn.Module, not saved in state_dict)
        self._opts: dict[str, DeepMomentum] = {
            name: DeepMomentum() for name in ("fast", "mid", "slow", "ultra")
        }
        self._global_step: int = 0

    # -- Forward --------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Cascade all 4 levels. No weight update here."""
        for name in ("fast", "mid", "slow", "ultra"):
            x = self.levels[name](x)
        return x

    # -- Fast weight update ---------------------------------------------------

    @torch.no_grad()
    def update(self, x: Tensor, teach_signal: Tensor) -> None:
        """
        Update CMS fast weights using teach_signal.

        For each level that 'should update' at this step:
            loss = -mean(teach_signal * delta)   # alignment loss
            grad = autograd(loss, level.params)
            param += -lr * deep_momentum(grad)

        Matches nested_learning/memorize.py memorize_tokens() inner loop.
        """
        self._global_step += 1
        h = x  # input to first level

        for name in ("fast", "mid", "slow", "ultra"):
            level = self.levels[name]
            period = PERIODS[name]
            lr = LR[name]

            if self._global_step % period == 0:
                with torch.enable_grad():
                    h_detached = h.detach().requires_grad_(False)
                    delta = level.net(level.norm(h_detached))
                    loss = -(teach_signal.detach() * delta).mean()
                grads = torch.autograd.grad(loss, level.net.parameters(), allow_unused=True)
                opt = self._opts[name]
                for param, grad in zip(level.net.parameters(), grads):
                    if grad is not None:
                        opt.step(param, grad, lr)

            # Advance h through this level (no_grad, just forward)
            h = h + level.net(level.norm(h)).detach()

    # -- Task boundary --------------------------------------------------------

    def reset_fast(self) -> None:
        """Reset fast+mid weights and optimizers (call at task boundary)."""
        for name in ("fast", "mid"):
            block = self.levels[name]
            for m in block.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self._opts[name].reset()

    def reset_all(self) -> None:
        """Reset all levels (for ablation)."""
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
        """All CMS parameters -- excluded from meta optimizer."""
        params = []
        for level in self.levels.values():
            params.extend(level.fast_params())
        return params
