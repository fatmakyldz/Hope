"""
Training engine -- 2-pass HOPE training loop.

Pass-1: meta forward (backbone + CMS forward + classifier)
Teach:  compute_teach_signal (closed-form, no autograd)
Pass-2: cms.update(features, teach)  -- fast weights only
Meta:   loss.backward(); optimizer.step()

Minimal replay: sadece classifier kalibrasyonu için (class başına 100 örnek).
CMS ve backbone replay'den etkilenmez.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from model.hope_model import HOPEModel, compute_teach_signal


def train_one_epoch(
    model: HOPEModel,
    loader,
    optimizer: Optimizer,
    device: torch.device,
    current_class_ids: list[int],
    run_teach: bool = True,
    buffer=None,          # CalibrationBuffer | None
    replay_batch: int = 32,
    replay_weight: float = 1.0,
    dynamic_replay: bool = True,   # scale replay_batch with num old classes
) -> float:
    model.train()
    total_loss = 0.0

    # Dynamic replay: scale batch size so old classes get comparable signal
    if dynamic_replay and buffer is not None:
        n_old = max(buffer.num_classes(), 1)
        effective_replay = min(replay_batch * max(n_old // 10, 1), 256)
    else:
        effective_replay = replay_batch

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # -- Buffer'a ekle (eğitimden önce, bias olmadan) --------------------
        if buffer is not None:
            buffer.add(images, labels)

        # -- Replay ile batch birleştir --------------------------------------
        if buffer is not None and len(buffer) > 0:
            rep_imgs, rep_lbls = buffer.sample(effective_replay, device)
            if rep_imgs is not None:
                all_imgs = torch.cat([images, rep_imgs], dim=0)
                all_lbls = torch.cat([labels, rep_lbls], dim=0)
                n_cur = images.size(0)
            else:
                all_imgs, all_lbls, n_cur = images, labels, images.size(0)
        else:
            all_imgs, all_lbls, n_cur = images, labels, images.size(0)

        # -- Pass-1: meta forward --------------------------------------------
        logits, features = model(all_imgs)

        # Ayrı ağırlıklı CE: current task normal, replay hafif
        cur_loss = F.cross_entropy(logits[:n_cur], all_lbls[:n_cur])
        if all_imgs.size(0) > n_cur:
            rep_loss = F.cross_entropy(logits[n_cur:], all_lbls[n_cur:])
            loss = cur_loss + replay_weight * rep_loss
        else:
            loss = cur_loss

        # -- Meta backward ---------------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient masking: tüm görülen sınıflar için güncelle
        _mask_classifier_grads(model.classifier, current_class_ids, device)

        torch.nn.utils.clip_grad_norm_(model.meta_parameters(), max_norm=1.0)
        optimizer.step()

        # -- Pass-2: teach signal -> CMS fast update (sadece current) --------
        if run_teach:
            with torch.no_grad():
                teach = compute_teach_signal(
                    features=features[:n_cur],
                    logits=logits[:n_cur].detach(),
                    labels=all_lbls[:n_cur],
                    classifier=model.classifier,
                )
            model.update_cms(features[:n_cur], teach)

        total_loss += cur_loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model: HOPEModel, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


@torch.no_grad()
def compute_class_means(
    model: HOPEModel,
    buffer,
    device: torch.device,
) -> dict[int, Tensor]:
    """Compute L2-normalized mean feature vector per class from buffer."""
    model.eval()
    class_means: dict[int, Tensor] = {}
    for cid, imgs in buffer._store.items():
        feats = []
        for i in range(0, len(imgs), 64):
            batch = torch.stack(imgs[i : i + 64]).to(device)
            _, feat = model(batch)
            feats.append(feat.cpu())
        mean = torch.cat(feats, dim=0).mean(dim=0)
        class_means[cid] = F.normalize(mean, dim=0)
    return class_means


@torch.no_grad()
def evaluate_ncm(
    model: HOPEModel,
    loader,
    device: torch.device,
    class_means: dict[int, Tensor],
) -> float:
    """Nearest Class Mean evaluation — immune to softmax calibration drift."""
    model.eval()
    class_ids = sorted(class_means.keys())
    means = torch.stack([class_means[c] for c in class_ids]).to(device)  # [C, D]

    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        _, features = model(images)                          # [B, D]
        features = F.normalize(features, dim=1)
        sims = features @ means.T                            # [B, C]
        pred_indices = sims.argmax(dim=1).cpu()
        pred_labels = torch.tensor([class_ids[i] for i in pred_indices.tolist()])
        correct += (pred_labels == labels.cpu()).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def _mask_classifier_grads(
    classifier: nn.Linear,
    allowed_class_ids: list[int],
    device: torch.device,
) -> None:
    """Zero out classifier gradients for classes not in allowed_class_ids."""
    if classifier.weight.grad is None:
        return
    num_classes = classifier.weight.size(0)
    mask = torch.zeros(num_classes, device=device)
    for cid in allowed_class_ids:
        if 0 <= cid < num_classes:
            mask[cid] = 1.0
    classifier.weight.grad *= mask.unsqueeze(1)
    if classifier.bias is not None and classifier.bias.grad is not None:
        classifier.bias.grad *= mask
