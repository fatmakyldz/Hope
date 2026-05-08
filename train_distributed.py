#!/usr/bin/env python3
"""
HOPE-CIFAR — Dağıtık Eğitim (Distributed Training) Scripti

Başlatma:
  torchrun --nproc_per_node=2 train_distributed.py [args]              # tek makine, 2 GPU
  torchrun --nproc_per_node=4 train_distributed.py [args]              # tek makine, 4 GPU
  torchrun --nnodes=2 --nproc_per_node=4 train_distributed.py [args]  # 2 makine × 4 GPU

─── DAĞITIK MİMARİ ──────────────────────────────────────────────────────────
  - Backbone + Classifier  : DDP (otomatik all_reduce gradient senkronu)
  - CMS parametreleri      : Manuel all_reduce (DeepMomentum autograd dışı)
  - Gaussian Alignment     : Sadece rank 0 hesaplar, classifier broadcast edilir
  - Replay buffer          : Her node'da yerel (divergent — tasarım kararı)
  - NCM class mean'leri    : all_reduce ile tüm node'larda eşitleniyor
  - Değerlendirme          : Sadece rank 0 yapar ve yazdırır
  - Checkpoint             : Sadece rank 0 kaydeder, tüm rank'lar yükler

─── CHECKPOINT & RESUME ──────────────────────────────────────────────────────
  --resume ./checkpoints/ckpt_task3.pt   # kaldığı yerden devam et
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(__file__))

from data.cifar100 import VIT_TRANSFORM, get_cifar100_task_datasets
from memory.gaussian_buffer import GaussianBuffer
from memory.replay_buffer import CalibrationBuffer
from model.hope_model import HOPEModel
from training.classifier_align import classifier_align
from training.engine import compute_class_means, evaluate, evaluate_ncm, train_one_epoch
from utils.metrics import ContinualMetrics


# ─── ARGÜMANLAR ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HOPE-CIFAR Distributed")
    p.add_argument("--epochs",            type=int,   default=10)
    p.add_argument("--batch_size",        type=int,   default=64,  help="Her GPU'daki batch boyutu")
    p.add_argument("--num_tasks",         type=int,   default=10)
    p.add_argument("--backbone_lr",       type=float, default=1e-5)
    p.add_argument("--classifier_lr",     type=float, default=1e-3)
    p.add_argument("--weight_decay",      type=float, default=1e-4)
    p.add_argument("--backbone",          type=str,   default="resnet", choices=["resnet", "vit"])
    p.add_argument("--freeze_backbone",   action="store_true")
    p.add_argument("--no_pretrained",     action="store_true")
    p.add_argument("--no_teach",          action="store_true")
    p.add_argument("--reset_all_cms",     action="store_true")
    p.add_argument("--grad_checkpoint",   action="store_true", help="ViT gradient checkpointing")
    # Gaussian Alignment
    p.add_argument("--gaussian_align",    action="store_true", help="Gaussian kalibrasyon etkinleştir")
    p.add_argument("--align_epochs",      type=int,   default=30)
    p.add_argument("--cosface",           action="store_true")
    p.add_argument("--cosface_scale",     type=float, default=20.0)
    # Replay
    p.add_argument("--replay",            action="store_true")
    p.add_argument("--samples_per_class", type=int,   default=500)
    p.add_argument("--replay_batch",      type=int,   default=64)
    p.add_argument("--replay_weight",     type=float, default=1.0)
    # Dizinler
    p.add_argument("--data_dir",          type=str,   default="./data")
    p.add_argument("--results_dir",       type=str,   default="./results")
    p.add_argument("--checkpoint_dir",    type=str,   default="./checkpoints")
    p.add_argument("--resume",            type=str,   default=None)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--amp",               action="store_true", help="Mixed precision (FP16) eğitim")
    return p.parse_args()


# ─── CHECKPOINT KAYDET ────────────────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, buffer, task_id, epoch):
    state = {
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "cms":          model.cms.cms_state_dict(),
        "buffer_store": {k: v for k, v in buffer._store.items()} if buffer else None,
        "buffer_seen":  dict(buffer._seen) if buffer else None,
        "task_id":      task_id,
        "epoch":        epoch,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ─── CHECKPOINT YÜKLE ─────────────────────────────────────────────────────────
def load_checkpoint(path, model, optimizer, buffer):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    model.cms.cms_load_state_dict(state["cms"])
    if buffer is not None and state.get("buffer_store") is not None:
        buffer._store = state["buffer_store"]
        buffer._seen  = state.get("buffer_seen", {})
    return state["task_id"], state["epoch"]


# ─── NCM CLASS MEAN'LERİ SENKRONIZE ET ───────────────────────────────────────
def sync_class_means(class_means, world_size, device):
    synced = {}
    for cid, mean in class_means.items():
        m = mean.to(device).clone()
        dist.all_reduce(m, op=dist.ReduceOp.SUM)
        m /= world_size
        synced[cid] = F.normalize(m, dim=0)
    return synced


# ─── ANA FONKSİYON ───────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed + rank)
    is_master = (rank == 0)

    def _barrier():
        if world_size > 1:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[local_rank])
            else:
                dist.barrier()

    if is_master:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"dist_{args.backbone}_t{args.num_tasks}_e{args.epochs}"
        if args.gaussian_align:
            tag += "_galign"
        result_dir = os.path.join(args.results_dir, f"{tag}_{ts}")
        os.makedirs(result_dir, exist_ok=True)
        print("=" * 60)
        print("  HOPE-CIFAR -- Distributed Continual Learning")
        print("=" * 60)
        print(f"  World size     : {world_size} GPU")
        print(f"  Backend        : {backend}")
        print(f"  Backbone       : {args.backbone.upper()}")
        print(f"  Gorevler       : {args.num_tasks} x 10 sinif")
        print(f"  Epoch/gorev    : {args.epochs}")
        print(f"  Batch/GPU      : {args.batch_size}  (toplam: {args.batch_size * world_size})")
        align_str = f"ACIK (epochs={args.align_epochs}, cosface={args.cosface})" if args.gaussian_align else "KAPALI"
        print(f"  Gaussian align : {align_str}")
        replay_str = f"ACIK (spc={args.samples_per_class})" if args.replay else "KAPALI"
        print(f"  Replay buffer  : {replay_str}")
        print("=" * 60)

    # ─── VERİ YÜKLEME ─────────────────────────────────────────────────────────
    transform = VIT_TRANSFORM if args.backbone == "vit" else None
    task_datasets = get_cifar100_task_datasets(root=args.data_dir, transform=transform)

    # ─── MODEL OLUŞTURMA ──────────────────────────────────────────────────────
    model = HOPEModel(
        num_classes=100,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        backbone_type=args.backbone,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)

    if world_size > 1:
        # grad_checkpoint açıkken recompute sırasında DDP param takibini kaybedebilir.
        # Normal çalışmada False → daha hızlı.
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=False)

    raw_model: HOPEModel = model.module if world_size > 1 else model

    # ─── OPTİMİZER ────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        raw_model.meta_param_groups(
            backbone_lr=args.backbone_lr,
            classifier_lr=args.classifier_lr,
        ),
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler("cuda", enabled=args.amp and torch.cuda.is_available())

    buffer         = CalibrationBuffer(samples_per_class=args.samples_per_class) if args.replay else None
    gaussian_buffer = GaussianBuffer() if args.gaussian_align else None

    start_task, start_epoch = 0, 0
    if args.resume and os.path.isfile(args.resume):
        start_task, start_epoch = load_checkpoint(args.resume, raw_model, optimizer, buffer)
        if is_master:
            print(f"[Checkpoint] Gorev {start_task}, Epoch {start_epoch}'den devam ediliyor")
        _barrier()

    def cms_sync():
        if world_size > 1:
            raw_model.cms.sync_params(world_size)

    metrics      = ContinualMetrics(num_tasks=args.num_tasks) if is_master else None
    seen_classes: list[int] = []

    for task_info in task_datasets[:args.num_tasks]:
        task_id   = task_info["task_id"]
        class_ids = task_info["class_ids"]

        if task_id < start_task:
            seen_classes.extend(class_ids)
            continue

        seen_classes.extend(class_ids)

        if is_master:
            print(f"\n{'='*60}")
            print(f"  Gorev {task_id}  |  Siniflar {class_ids[0]}-{class_ids[-1]}")
            print(f"{'='*60}")

        train_sampler = DistributedSampler(
            task_info["train_subset"],
            num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed + task_id,
        )
        train_loader = DataLoader(
            task_info["train_subset"],
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=0,
        )
        test_loader = DataLoader(
            task_info["test_subset"], batch_size=256, shuffle=False, num_workers=0,
        ) if is_master else None

        # ─── EĞİTİM ───────────────────────────────────────────────────────────
        first_epoch = start_epoch if task_id == start_task else 0
        for epoch in range(first_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            # cms_sync_fn=None: her batch'te all_reduce DDP backward ile çakışıyor.
            # CMS senkronu epoch sonunda manuel yapılıyor.
            loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                current_class_ids=seen_classes,
                run_teach=not args.no_teach,
                buffer=buffer,
                replay_batch=args.replay_batch,
                replay_weight=args.replay_weight,
                cms_sync_fn=None,
                scaler=scaler,
            )
            # Epoch bittikten sonra CMS senkronu — DDP backward'dan bağımsız
            if world_size > 1:
                _barrier()
                cms_sync()
            if is_master:
                print(f"  Epoch {epoch+1}/{args.epochs} | Kayip: {loss:.4f}")

        # ─── GÖREV SINIRI ─────────────────────────────────────────────────────
        if world_size > 1:
            _barrier()

        if args.reset_all_cms:
            raw_model.cms.reset_all()
        else:
            raw_model.on_task_boundary()

        if world_size > 1:
            raw_model.cms.sync_params(world_size)

        # ─── GAUSSIAN KALİBRASYON (sadece rank 0) ────────────────────────────
        # Rank 0 tüm görülen sınıflar için Gaussian hesaplar ve classifier'ı
        # kalibre eder. Sonra güncel classifier ağırlıkları tüm rank'lara
        # broadcast edilir.
        if gaussian_buffer is not None:
            if is_master:
                print(f"  Gaussian istatistikleri guncelleniyor ({len(seen_classes)} sinif)...")
                for prev_td in task_datasets[:task_id + 1]:
                    tmp_loader = DataLoader(
                        prev_td["train_subset"], batch_size=128, shuffle=False, num_workers=0
                    )
                    gaussian_buffer.update(raw_model, tmp_loader, device, prev_td["class_ids"])
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print(f"  Siniflandirici kalibre ediliyor (epoch={args.align_epochs})...")
                classifier_align(
                    classifier=raw_model.classifier,
                    gaussian_buffer=gaussian_buffer,
                    device=device,
                    epochs=args.align_epochs,
                    lr=0.005,
                    n_per_class=256,
                    scale=args.cosface_scale,
                    cosface=args.cosface,
                )
                print(f"  Kalibrasyon tamamlandi.")

            # Güncel classifier ağırlıklarını tüm rank'lara yay
            if world_size > 1:
                for param in raw_model.classifier.parameters():
                    dist.broadcast(param.data, src=0)
                _barrier()

        # ─── NCM CLASS MEAN'LERİ ──────────────────────────────────────────────
        if gaussian_buffer is not None and is_master:
            class_means = gaussian_buffer.get_class_means()
        elif buffer is not None:
            class_means = compute_class_means(raw_model, buffer, device)
            if world_size > 1:
                class_means = sync_class_means(class_means, world_size, device)
        else:
            class_means = None

        # ─── CHECKPOINT (rank 0) ──────────────────────────────────────────────
        if is_master:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_task{task_id}.pt")
            save_checkpoint(ckpt_path, raw_model, optimizer, buffer, task_id, args.epochs)
            print(f"  [Checkpoint] -> {ckpt_path}")

        # ─── DEĞERLENDİRME (rank 0) ───────────────────────────────────────────
        if is_master:
            print(f"\n  Gorev {task_id} sonrasi degerlendirme:")
            accs = []
            for prev_td in task_datasets[:task_id + 1]:
                prev_loader = DataLoader(
                    prev_td["test_subset"], batch_size=256, shuffle=False, num_workers=0
                )
                if class_means is not None:
                    acc = evaluate_ncm(
                        raw_model, prev_loader, device, class_means,
                        use_backbone=(gaussian_buffer is not None),
                    )
                else:
                    acc = evaluate(raw_model, prev_loader, device)
                accs.append(acc)
                marker = " <- mevcut gorev" if prev_td["task_id"] == task_id else ""
                print(f"    Gorev {prev_td['task_id']}: {acc:.2f}%{marker}")
            metrics.record(after_task=task_id, accs=accs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start_epoch = 0

    # ─── SONUÇLAR ─────────────────────────────────────────────────────────────
    if is_master:
        print("\n" + "=" * 60)
        print("  SONUCLAR -- HOPE-CIFAR Distributed")
        print("=" * 60)
        metrics.print_matrix()
        print()
        print(metrics.summary())
        metrics.save(os.path.join(result_dir, "metrics.json"))
        with open(os.path.join(result_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"\nSonuclar kaydedildi -> {result_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
