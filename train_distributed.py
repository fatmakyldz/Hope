#!/usr/bin/env python3
"""
HOPE-CIFAR — Dağıtık Eğitim (Distributed Training) Scripti

Başlatma:
  torchrun --nproc_per_node=4 train_distributed.py [args]              # tek makine, 4 GPU
  torchrun --nnodes=2 --nproc_per_node=4 train_distributed.py [args]  # 2 makine × 4 GPU

─── DAĞITIK MİMARİ ──────────────────────────────────────────────────────────
  - Backbone + Classifier  : DDP (otomatik all_reduce gradient senkronu)
  - CMS parametreleri      : Manuel all_reduce (DeepMomentum autograd dışı)
  - Replay buffer          : Her node'da yerel (divergent — tasarım kararı)
  - NCM class mean'leri    : all_reduce ile tüm node'larda eşitleniyor
  - Değerlendirme          : Sadece rank 0 yapar ve yazdırır
  - Checkpoint             : Sadece rank 0 kaydeder, tüm rank'lar yükler

─── CHECKPOINT & RESUME ──────────────────────────────────────────────────────
  --resume ./checkpoints/ckpt_task3.pt   # kaldığı yerden devam et

─── NEDEN DDP? ───────────────────────────────────────────────────────────────
  Data parallelism: her GPU farklı mini-batch görür, gradyanlar all_reduce ile
  ortalalanır. Model parallelism (katmanları GPU'lara dağıtmak) Ethernet
  gecikmesi yüzünden bu model ölçeğinde daha kötü olur.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(__file__))

from data.cifar100 import get_cifar100_task_datasets
from memory.replay_buffer import CalibrationBuffer
from model.hope_model import HOPEModel
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
    p.add_argument("--freeze_backbone",   action="store_true")
    p.add_argument("--no_pretrained",     action="store_true")
    p.add_argument("--no_teach",          action="store_true")
    p.add_argument("--reset_all_cms",     action="store_true")
    p.add_argument("--replay",            action="store_true")
    p.add_argument("--samples_per_class", type=int,   default=500)
    p.add_argument("--replay_batch",      type=int,   default=64)
    p.add_argument("--replay_weight",     type=float, default=1.0)
    p.add_argument("--data_dir",          type=str,   default="./data")
    p.add_argument("--results_dir",       type=str,   default="./results")
    p.add_argument("--checkpoint_dir",    type=str,   default="./checkpoints")
    p.add_argument("--resume",            type=str,   default=None,  help="Checkpoint dosya yolu")
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


# ─── CHECKPOINT KAYDET ────────────────────────────────────────────────────────
def save_checkpoint(
    path: str,
    model: HOPEModel,
    optimizer,
    buffer: CalibrationBuffer | None,
    task_id: int,
    epoch: int,
) -> None:
    """
    Tam checkpoint: model ağırlıkları + meta optimizer + CMS optimizer + buffer.
    Sadece rank 0 çağırır.
    """
    state = {
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "cms":        model.cms.cms_state_dict(),     # DeepMomentum durumları
        "buffer_store": {k: v for k, v in buffer._store.items()} if buffer else None,
        "buffer_seen":  dict(buffer._seen) if buffer else None,
        "task_id":    task_id,
        "epoch":      epoch,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ─── CHECKPOINT YÜKLE ─────────────────────────────────────────────────────────
def load_checkpoint(
    path: str,
    model: HOPEModel,
    optimizer,
    buffer: CalibrationBuffer | None,
) -> tuple[int, int]:
    """
    Checkpoint'ten tüm durumu geri yükler.
    Tüm rank'lar çağırır (her biri kendi kopyasına yükler).
    Döndürür: (task_id, epoch) — kaldığı yer
    """
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    model.cms.cms_load_state_dict(state["cms"])
    if buffer is not None and state.get("buffer_store") is not None:
        buffer._store = state["buffer_store"]
        buffer._seen  = state.get("buffer_seen", {})
    return state["task_id"], state["epoch"]


# ─── NCM CLASS MEAN'LERİ SENKRONIZE ET ───────────────────────────────────────
def sync_class_means(
    class_means: dict,
    world_size: int,
    device: torch.device,
) -> dict:
    """
    Her node farklı veri gördüğü için class mean'ler farklı hesaplanır.
    all_reduce ile tüm node'ların ortalamalarını toplayıp dünya genelinde
    tek tutarlı class mean seti elde ediyoruz.
    """
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

    # ─── DAĞITIK ORTAMI BAŞLAT ────────────────────────────────────────────────
    # torchrun bu değişkenleri otomatik set eder
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    rank        = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))

    # CUDA varsa NCCL (GPU arası en hızlı), yoksa GLOO (CPU/test)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    # Her process kendi GPU'suna bağlanır
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    # Tekrarlanabilirlik: her rank farklı seed → farklı veri karıştırması
    torch.manual_seed(args.seed + rank)

    is_master = (rank == 0)  # sadece rank 0 dosya yazar ve ekrana basar

    if is_master:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(args.results_dir, f"dist_t{args.num_tasks}_e{args.epochs}_{ts}")
        os.makedirs(result_dir, exist_ok=True)
        print("=" * 60)
        print("  HOPE-CIFAR -- Distributed Continual Learning")
        print("=" * 60)
        print(f"  World size     : {world_size} process")
        print(f"  Backend        : {backend}")
        print(f"  Device/rank    : {device} (rank {rank})")
        print(f"  Gorevler       : {args.num_tasks} x 10 sinif")
        print(f"  Epoch/gorev    : {args.epochs}")
        print(f"  Batch/GPU      : {args.batch_size}  (toplam: {args.batch_size * world_size})")
        print("=" * 60)

    # ─── VERİ YÜKLEME ─────────────────────────────────────────────────────────
    # Ham dataset: her node kendi DistributedSampler'ını oluşturur
    # CIFAR-100 download=True → her node bağımsız indirebilir
    task_datasets = get_cifar100_task_datasets(root=args.data_dir)

    # ─── MODEL OLUŞTURMA ──────────────────────────────────────────────────────
    model = HOPEModel(
        num_classes=100,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # DDP: backbone + classifier gradyanlarını otomatik senkronlar
    # CMS parametreleri de DDP içinde ama DeepMomentum ile ayrıca güncellenir
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # DDP sarmalayıcısını geçerek asıl modele eriş
    raw_model: HOPEModel = model.module if world_size > 1 else model

    # ─── OPTİMİZER ────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        raw_model.meta_param_groups(
            backbone_lr=args.backbone_lr,
            classifier_lr=args.classifier_lr,
        ),
        weight_decay=args.weight_decay,
    )

    # ─── REPLAY BUFFER ────────────────────────────────────────────────────────
    # Her node yerel buffer tutar — replay divergent ama pratik
    buffer = CalibrationBuffer(samples_per_class=args.samples_per_class) if args.replay else None

    # ─── CHECKPOINT'TEN DEVAM ─────────────────────────────────────────────────
    start_task, start_epoch = 0, 0
    if args.resume and os.path.isfile(args.resume):
        start_task, start_epoch = load_checkpoint(args.resume, raw_model, optimizer, buffer)
        if is_master:
            print(f"[Checkpoint] Gorev {start_task}, Epoch {start_epoch}'den devam ediliyor")
        # Tüm rank'ların aynı noktadan başlaması için bariyer
        dist.barrier()

    # ─── CMS SENKRON CALLBACK ─────────────────────────────────────────────────
    def cms_sync():
        """Her CMS update sonrası tüm node'larda CMS parametrelerini eşitle."""
        if world_size > 1:
            raw_model.cms.sync_params(world_size)

    # ─── SÜREKLİ ÖĞRENME DÖNGÜSÜ ─────────────────────────────────────────────
    metrics = ContinualMetrics(num_tasks=args.num_tasks) if is_master else None
    seen_classes: list[int] = []

    for task_info in task_datasets[:args.num_tasks]:
        task_id    = task_info["task_id"]
        class_ids  = task_info["class_ids"]

        # Checkpoint'ten devam ediyorsak tamamlanmış görevleri atla
        if task_id < start_task:
            seen_classes.extend(class_ids)
            continue

        seen_classes.extend(class_ids)

        if is_master:
            print(f"\n{'='*60}")
            print(f"  Gorev {task_id}  |  Siniflar {class_ids[0]}-{class_ids[-1]}")
            print(f"{'='*60}")

        # ─── DİSTRİBUTED SAMPLER ──────────────────────────────────────────────
        # Her node farklı veri parçasını görür; shuffle=True ile epoch başı karıştırır
        train_sampler = DistributedSampler(
            task_info["train_subset"],
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed + task_id,
        )
        train_loader = DataLoader(
            task_info["train_subset"],
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=0,
        )
        # Test loader: değerlendirme sadece rank 0 yapıyor, diğerleri boş bırakabilir
        test_loader = DataLoader(
            task_info["test_subset"],
            batch_size=256,
            shuffle=False,
            num_workers=0,
        ) if is_master else None

        # ─── EĞİTİM ───────────────────────────────────────────────────────────
        first_epoch = start_epoch if task_id == start_task else 0
        for epoch in range(first_epoch, args.epochs):
            # DistributedSampler her epoch'ta farklı karıştırma için epoch set eder
            train_sampler.set_epoch(epoch)

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
                cms_sync_fn=cms_sync,  # her adımda CMS senkronu
            )

            if is_master:
                print(f"  Epoch {epoch+1}/{args.epochs} | Kayip: {loss:.4f}")

        # ─── GÖREV SINIRI ─────────────────────────────────────────────────────
        # Tüm node'lar görev sınırına aynı anda gelsin
        if world_size > 1:
            dist.barrier()

        if args.reset_all_cms:
            raw_model.cms.reset_all()
        else:
            raw_model.on_task_boundary()

        # CMS sıfırlandıktan sonra tekrar senkronize et
        if world_size > 1:
            raw_model.cms.sync_params(world_size)

        # ─── NCM CLASS MEAN'LERİ ──────────────────────────────────────────────
        # Her node kendi buffer'ından hesaplar, sonra all_reduce ile birleştirir
        if buffer is not None:
            class_means = compute_class_means(raw_model, buffer, device)
            if world_size > 1:
                class_means = sync_class_means(class_means, world_size, device)
        else:
            class_means = None

        # ─── CHECKPOINT KAYDET (rank 0) ───────────────────────────────────────
        if is_master:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"ckpt_task{task_id}.pt"
            )
            save_checkpoint(ckpt_path, raw_model, optimizer, buffer, task_id, args.epochs)
            print(f"  [Checkpoint] -> {ckpt_path}")

        # ─── DEĞERLENDİRME (sadece rank 0) ───────────────────────────────────
        if is_master:
            print(f"\n  Gorev {task_id} sonrasi degerlendirme:")
            accs = []
            for prev_info in task_datasets[:task_id + 1]:
                prev_loader = DataLoader(
                    prev_info["test_subset"], batch_size=256, shuffle=False, num_workers=0
                )
                if class_means is not None:
                    acc = evaluate_ncm(raw_model, prev_loader, device, class_means)
                else:
                    acc = evaluate(raw_model, prev_loader, device)
                accs.append(acc)
                marker = " <- mevcut gorev" if prev_info["task_id"] == task_id else ""
                print(f"    Gorev {prev_info['task_id']}: {acc:.2f}%{marker}")
            metrics.record(after_task=task_id, accs=accs)

        # start_epoch sadece ilk görev için geçerli
        start_epoch = 0

    # ─── SONUÇLARI YAZDIR VE KAYDET ───────────────────────────────────────────
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
