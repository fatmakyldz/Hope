"""
HOPEModel: ResNet18 + 4-level CMS + Linear classifier.

Two-pass interface (faithful to nested_learning):
  Pass-1: logits, features = model(x)
  Teach:  teach = compute_teach_signal(features, logits, labels, classifier)
  Pass-2: model.cms.update(features, teach)   # fast weight update only
  Meta:   loss.backward(); optimizer.step()   # backbone + classifier
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import ResNetBackbone
from .cms import CMSModule


def compute_teach_signal(
    features: Tensor,
    logits: Tensor,
    labels: Tensor,
    classifier: nn.Linear,
) -> Tensor:
    """
    Closed-form CE gradient w.r.t. features.

    Derivation:
        grad_logits  = (softmax(logits) - one_hot(labels)) / B
        grad_features = grad_logits @ W    [W: classifier.weight (C, D)]
        teach = -grad_features             (improvement = negative gradient)

    No autograd call -- safe inside torch.no_grad().
    Faithful to nested_learning/training.py compute_teach_signal().
    """
    with torch.no_grad():
        B = features.size(0)
        p = torch.softmax(logits.detach(), dim=-1)           # (B, C)
        p[torch.arange(B, device=p.device), labels] -= 1.0  # subtract one-hot
        p = p / B
        W = classifier.weight.detach()                       # (C, D)
        teach = -(p @ W)                                     # (B, D)
    return teach


class HOPEModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        cms_hidden_multiplier: int = 4,
        cms_grad_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained, freeze=freeze_backbone)
        self.cms = CMSModule(
            dim=self.backbone.feature_dim,
            hidden_multiplier=cms_hidden_multiplier,
            grad_clip=cms_grad_clip,
        )
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (logits, backbone_features)."""
        features = self.backbone(x)          # (B, 512)
        cms_out = self.cms(features)          # (B, 512)
        logits = self.classifier(cms_out)     # (B, num_classes)
        return logits, features

    def update_cms(self, features: Tensor, teach: Tensor) -> None:
        """Pass-2: update CMS fast weights."""
        self.cms.update(features.detach(), teach.detach())

    def meta_parameters(self) -> list[nn.Parameter]:
        """Parameters for the meta optimizer (backbone + classifier, NOT CMS)."""
        cms_ids = {id(p) for p in self.cms.all_fast_params()}
        return [p for p in self.parameters() if id(p) not in cms_ids]

    def meta_param_groups(
        self,
        backbone_lr: float = 1e-4,
        classifier_lr: float = 1e-3,
    ) -> list[dict]:
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
        """Reset fast+mid CMS levels at task boundary."""
        self.cms.reset_fast()
