#!/usr/bin/env python3
"""
HOPE-CIFAR — Smoke Test (2 GPU, birkaç saniye)

Her kod path'ini hızlıca doğrular:
  DDP init → forward → loss → CMS sync → evaluate → checkpoint

Başlatma:
  torchrun --nproc_per_node=2 train_smoke.py
  torchrun --nproc_per_node=2 train_smoke.py --backbone vit   # ViT testi

Geçti/Kaldı kriterleri (otomatik kontrol):
  [OK] Loss her epoch sonunda azaldı
  [OK] Son task accuracy > %10 (random baseline, 10 sinif)
  [OK] DDP all_reduce senkronu tamamlandi
  [OK] Checkpoint yazildi ve geri okundu
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(__file__))

from data.cifar100 import VIT_TRANSFORM, get_cifar100_task_datasets
from model.hope_model import HOPEModel
from training.engine import evaluate_ncm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone",    type=str,   default="resnet", choices=["resnet", "vit"])
    p.add_argument("--num_tasks",   type=int,   default=2,   help="Kac gorev test edilsin (2 yeterli)")
    p.add_argument("--epochs",      type=int,   default=2,   help="Gorev basina epoch")
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--max_batches", type=int,   default=8,   help="Epoch basina max batch (hiz siniri)")
    p.add_argument("--backbone_lr", type=float, default=1e-4)
    p.add_argument("--ckpt_dir",    type=str,   default="./checkpoints_smoke")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ─── Renk kodları (terminal çıktısı) ─────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   return f"{GREEN}[OK]{RESET}  {msg}"
def fail(msg): return f"{RED}[FAIL]{RESET} {msg}"
def info(msg): return f"{CYAN}[--]{RESET}  {msg}"
def head(msg): return f"{BOLD}{CYAN}{msg}{RESET}"


# ─── CMS manuel all_reduce ────────────────────────────────────────────────────
def cms_sync(model_unwrapped, world_size):
    for p in model_unwrapped.cms.all_fast_params():
        if p.data is not None:
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
            p.data /= world_size


# ─── Tek epoch (max_batches ile kısıtlı) ─────────────────────────────────────
def train_limited(model, loader, optimizer, device, seen_classes,
                  max_batches, run_teach=True):
    model.train()
    total_loss = 0.0
    n = 0
    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)

        logits, backbone_feat, _ = model(images)
        loss = F.cross_entropy(logits, labels)

        # Teach signal
        if run_teach:
            with torch.no_grad():
                W_snap = model.module.classifier.weight.detach().clone()
                B = backbone_feat.size(0)
                p = torch.softmax(logits.detach(), dim=-1)
                p[torch.arange(B, device=device), labels] -= 1.0
                p = p / B
                teach = -(p @ W_snap)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient maskeleme
        if model.module.classifier.weight.grad is not None:
            mask = torch.zeros(model.module.classifier.weight.size(0), device=device)
            for cid in seen_classes:
                mask[cid] = 1.0
            model.module.classifier.weight.grad *= mask.unsqueeze(1)

        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if run_teach:
            model.module.update_cms(backbone_feat.detach(), teach)

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ─── Hızlı softmax evaluate (NCM kurmadan) ───────────────────────────────────
@torch.no_grad()
def quick_eval(model, loader, device, max_batches=20):
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        logits, _, _ = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # DDP init
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        torch.backends.cudnn.benchmark = True

    is_main = (rank == 0)
    t_global = time.time()

    if is_main:
        print()
        print(head("=" * 58))
        print(head("  HOPE-CIFAR  SMOKE TEST"))
        print(head("=" * 58))
        print(info(f"World size   : {world_size} GPU"))
        print(info(f"Backbone     : {args.backbone.upper()}"))
        print(info(f"Gorevler     : {args.num_tasks}"))
        print(info(f"Epoch/gorev  : {args.epochs}  (max {args.max_batches} batch/epoch)"))
        device_name = torch.cuda.get_device_name(rank) if torch.cuda.is_available() else "CPU"
        print(info(f"Device       : {device_name}"))
        print(head("=" * 58))

    # ── Veri ──────────────────────────────────────────────────────────────────
    transform = VIT_TRANSFORM if args.backbone == "vit" else None
    task_datasets = get_cifar100_task_datasets(root="./data", transform=transform)

    # ── Model ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    model = HOPEModel(num_classes=100, pretrained=True,
                      backbone_type=args.backbone).to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None,
                find_unused_parameters=False)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        t_load = time.time() - t0
        print(info(f"Model yuklendi: {n_params:,} param  ({t_load:.1f}s)"))

    optimizer = optim.AdamW(
        model.module.meta_param_groups(backbone_lr=args.backbone_lr, classifier_lr=1e-3),
        weight_decay=1e-4,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Sonuç takibi ──────────────────────────────────────────────────────────
    results   = {}   # task_id -> {losses, acc, time}
    passed    = []
    failed    = []

    seen_classes = []

    # ── Görev döngüsü ─────────────────────────────────────────────────────────
    for td in task_datasets[:args.num_tasks]:
        task_id   = td["task_id"]
        class_ids = td["class_ids"]
        seen_classes.extend(class_ids)
        t_task = time.time()

        sampler = DistributedSampler(td["train_subset"], num_replicas=world_size,
                                     rank=rank, shuffle=True, drop_last=True)
        loader  = DataLoader(td["train_subset"], batch_size=args.batch_size,
                             sampler=sampler, num_workers=0,
                             pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(td["test_subset"], batch_size=128,
                                 shuffle=False, num_workers=0)

        if is_main:
            print()
            print(head(f"  Gorev {task_id}  |  Siniflar {class_ids[0]}-{class_ids[-1]}"))

        losses = []
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            t_ep = time.time()

            loss = train_limited(model, loader, optimizer, device,
                                 seen_classes, args.max_batches)

            # CMS sync
            cms_sync(model.module, world_size)
            dist.barrier()

            t_ep = time.time() - t_ep
            losses.append(loss)
            if is_main:
                mem = f"  GPU:{torch.cuda.memory_allocated()/1e9:.2f}GB" \
                      if torch.cuda.is_available() else ""
                print(f"    Epoch {epoch+1}/{args.epochs} | "
                      f"Loss: {loss:.4f} | {t_ep:.1f}s{mem}")

        # CMS reset
        model.module.on_task_boundary()

        # Evaluate (rank 0 only)
        acc = 0.0
        if is_main:
            acc = quick_eval(model, test_loader, device)
            print(f"    Accuracy (softmax, {args.max_batches*2} batch): {acc:.2f}%")

        # Checkpoint yaz + geri oku (sadece task 0)
        ckpt_ok = True
        if task_id == 0 and is_main:
            ckpt_path = os.path.join(args.ckpt_dir, "smoke_ckpt.pt")
            torch.save({"model": model.module.state_dict(),
                        "task": task_id}, ckpt_path)
            loaded = torch.load(ckpt_path, map_location=device, weights_only=True)
            ckpt_ok = "model" in loaded and loaded["task"] == 0

        dist.barrier()

        results[task_id] = {"losses": losses, "acc": acc,
                             "time": time.time() - t_task}

    # ── PASS / FAIL değerlendirmesi (rank 0) ───────────────────────────────
    if is_main:
        print()
        print(head("=" * 58))
        print(head("  SMOKE TEST SONUÇLARI"))
        print(head("=" * 58))

        # 1. Loss azaldı mı?
        for tid, r in results.items():
            ls = r["losses"]
            if len(ls) >= 2 and ls[-1] < ls[0]:
                passed.append(ok(f"Gorev {tid}: loss azaldi  {ls[0]:.4f} -> {ls[-1]:.4f}"))
            elif len(ls) == 1:
                passed.append(ok(f"Gorev {tid}: loss={ls[0]:.4f} (tek epoch, karsilastirma yok)"))
            else:
                failed.append(fail(f"Gorev {tid}: loss azalmadi  {ls[0]:.4f} -> {ls[-1]:.4f}"))

        # 2. Loss nan/inf degil mi?
        for tid, r in results.items():
            bad = any((l != l or l == float("inf")) for l in r["losses"])
            if bad:
                failed.append(fail(f"Gorev {tid}: NaN/Inf loss tespit edildi!"))
            else:
                passed.append(ok(f"Gorev {tid}: Loss sayi araliginda (NaN/Inf yok)"))

        # 3. Son task accuracy > rastgele (%10)
        last_tid  = max(results.keys())
        last_acc  = results[last_tid]["acc"]
        threshold = 12.0   # %10 random + biraz marj
        if last_acc >= threshold:
            passed.append(ok(f"Son gorev accuracy: {last_acc:.2f}% > {threshold}% (random baseline)"))
        else:
            failed.append(fail(f"Son gorev accuracy: {last_acc:.2f}% < {threshold}% — model ogrenmedi!"))

        # 4. DDP barrier'lar tamamlandi (buraya geldiyse tamamlandi)
        passed.append(ok(f"DDP all_reduce / barrier: {world_size} rank senkronize"))

        # 5. Checkpoint
        if ckpt_ok:
            passed.append(ok("Checkpoint yazildi ve geri okundu"))
        else:
            failed.append(fail("Checkpoint okuma hatasi!"))

        # Sonuc
        print()
        for msg in passed:
            print(f"  {msg}")
        for msg in failed:
            print(f"  {msg}")

        total_time = time.time() - t_global
        print()
        print(head(f"  Toplam sure : {total_time:.1f}s"))
        if failed:
            print(f"  {RED}{BOLD}SONUC: {len(failed)} HATA — detaylari yukarda incele{RESET}")
        else:
            print(f"  {GREEN}{BOLD}SONUC: TUM TESTLER GECTI ({len(passed)}/{len(passed)}){RESET}")
        print(head("=" * 58))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
