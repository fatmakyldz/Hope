#!/usr/bin/env python3
"""
HOPE-CIFAR — GPU Odaklı Eğitim Scripti

train.py ile aynı mantık; GPU için ek optimizasyonlar:
  - Mixed Precision (AMP): fp16 hesaplama, ~2x hız artışı
  - pin_memory + num_workers: CPU→GPU veri transfer hızlanması
  - cudnn.benchmark: sabit input için cuDNN otomatik kernel seçimi
  - Gradient accumulation: küçük GPU'da büyük batch etkisi

─── KULLANIM ─────────────────────────────────────────────────────────────────
  # Temel (ViT + Gaussian + CosFace — lab için önerilen):
  python train_gpu.py --backbone vit --gaussian_align --cosface

  # Hafif test (ResNet, daha hızlı):
  python train_gpu.py --backbone resnet --gaussian_align

  # Büyük batch + gradient accumulation:
  python train_gpu.py --backbone vit --gaussian_align --cosface --batch_size 64 --accum_steps 4

  # Backbone dondur (sadece CMS + classifier öğrenir):
  python train_gpu.py --backbone vit --freeze_backbone --gaussian_align --cosface
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))

from data.cifar100 import VIT_TRANSFORM, get_cifar100_tasks
from memory.gaussian_buffer import GaussianBuffer
from memory.replay_buffer import CalibrationBuffer
from model.hope_model import HOPEModel
from training.classifier_align import classifier_align
from training.engine import compute_class_means, evaluate, evaluate_ncm
from utils.metrics import ContinualMetrics


# ─── ARGÜMANLAR ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HOPE-CIFAR GPU Eğitimi")
    # Eğitim hiperparametreleri
    p.add_argument("--epochs",           type=int,   default=15,   help="Her görev için epoch sayısı")
    p.add_argument("--batch_size",       type=int,   default=128,  help="GPU'ya sığan batch boyutu")
    p.add_argument("--accum_steps",      type=int,   default=1,    help="Gradient accumulation adımı (etkili batch = batch_size × accum_steps)")
    p.add_argument("--num_tasks",        type=int,   default=10)
    # Optimizer
    p.add_argument("--backbone_lr",      type=float, default=1e-4)
    p.add_argument("--classifier_lr",    type=float, default=1e-3)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    # Backbone
    p.add_argument("--backbone",         type=str,   default="vit", choices=["resnet", "vit"],
                                         help="vit: ViT-B/16 (güçlü, lab), resnet: ResNet18 (hızlı)")
    p.add_argument("--freeze_backbone",  action="store_true")
    p.add_argument("--no_pretrained",    action="store_true")
    # Gaussian Classifier Alignment
    p.add_argument("--gaussian_align",   action="store_true", help="Gaussian kalibrasyon etkinleştir")
    p.add_argument("--align_epochs",     type=int,   default=30)
    p.add_argument("--cosface",          action="store_true", help="CosFace loss kullan")
    p.add_argument("--cosface_scale",    type=float, default=20.0)
    # Ablasyon
    p.add_argument("--no_teach",         action="store_true")
    p.add_argument("--reset_all_cms",    action="store_true")
    p.add_argument("--no_amp",           action="store_true", help="Mixed precision kapat (debug için)")
    # Replay
    p.add_argument("--replay",           action="store_true")
    p.add_argument("--samples_per_class",type=int,   default=100)
    p.add_argument("--replay_batch",     type=int,   default=32)
    p.add_argument("--replay_weight",    type=float, default=0.5)
    # Dizinler
    p.add_argument("--data_dir",         type=str,   default="./data")
    p.add_argument("--results_dir",      type=str,   default="./results")
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


# ─── GPU EĞİTİM DÖNGÜSÜ (AMP DESTEKLİ) ──────────────────────────────────────
def train_one_epoch_gpu(
    model: HOPEModel,
    loader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    current_class_ids: list[int],
    run_teach: bool = True,
    buffer=None,
    replay_batch: int = 32,
    replay_weight: float = 1.0,
    accum_steps: int = 1,
    use_amp: bool = True,
) -> float:
    """
    AMP (fp16) + gradient accumulation destekli GPU eğitim döngüsü.

    autocast bloğu: forward + loss fp16'da hesaplanır → hız + bellek kazancı
    GradScaler    : fp16 underflow'a karşı gradient ölçekleme
    accum_steps   : her accum_steps adımda bir optimizer.step() → büyük batch etkisi
    """
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    model.train()
    total_loss = 0.0

    # Dinamik replay boyutu
    if buffer is not None and buffer.num_classes() > 0:
        n_old = max(buffer.num_classes(), 1)
        effective_replay = min(replay_batch * max(n_old // 10, 1), 256)
    else:
        effective_replay = replay_batch

    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Replay birleştir
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

        if buffer is not None:
            buffer.add(images, labels)

        # ─── FORWARD (fp16) ───────────────────────────────────────────────────
        with autocast(enabled=use_amp):
            logits, backbone_feat, cms_out = model(all_imgs)
            cur_loss = F.cross_entropy(logits[:n_cur], all_lbls[:n_cur])
            if all_imgs.size(0) > n_cur:
                rep_loss = F.cross_entropy(logits[n_cur:], all_lbls[n_cur:])
                loss = cur_loss + replay_weight * rep_loss
            else:
                loss = cur_loss
            # Gradient accumulation: loss'u adım sayısına böl
            loss = loss / accum_steps

        # ─── TEACH SİNYALİ (fp32'de, daha kararlı) ───────────────────────────
        if run_teach:
            with torch.no_grad():
                W_snap = model.classifier.weight.detach().float().clone()
                B_cur = backbone_feat[:n_cur].size(0)
                p = torch.softmax(logits[:n_cur].detach().float(), dim=-1)
                p[torch.arange(B_cur, device=p.device), all_lbls[:n_cur]] -= 1.0
                p = p / B_cur
                teach = -(p @ W_snap)

        # ─── BACKWARD (scaler ile fp16 gradient) ─────────────────────────────
        scaler.scale(loss).backward()

        # Her accum_steps adımda bir optimizer güncelle
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            # Gradient maskeleme (fp32'ye unscale sonrası)
            scaler.unscale_(optimizer)
            _mask_classifier_grads_gpu(model.classifier, current_class_ids, device)
            clip_grad_norm_(model.meta_parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ─── CMS GÜNCELLE (backward sonrası, fp32) ────────────────────────────
        if run_teach:
            model.update_cms(backbone_feat[:n_cur].detach().float(), teach)

        total_loss += cur_loss.item()

    return total_loss / max(len(loader), 1)


def _mask_classifier_grads_gpu(classifier, allowed_class_ids, device):
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


# ─── ANA FONKSİYON ───────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # ─── DONANIM KONTROLÜ ─────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("[UYARI] CUDA bulunamadı! train.py (CPU/MPS) kullanın.")
        print("        Bu script NVIDIA GPU gerektiriyor.")
        sys.exit(1)

    device = torch.device("cuda")
    # cuDNN sabit-boyutlu input için en hızlı konvolüsyon algoritmasını seç
    torch.backends.cudnn.benchmark = True

    use_amp = not args.no_amp
    effective_batch = args.batch_size * args.accum_steps

    # ─── SONUÇ DİZİNİ ─────────────────────────────────────────────────────────
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"gpu_{args.backbone}_t{args.num_tasks}_e{args.epochs}_b{effective_batch}"
    if args.gaussian_align:
        tag += "_galign"
        if args.cosface:
            tag += "_cosface"
    result_dir = os.path.join(args.results_dir, f"{tag}_{ts}")
    os.makedirs(result_dir, exist_ok=True)

    # ─── KONFİGÜRASYON YAZDIR ────────────────────────────────────────────────
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 60)
    print("  HOPE-CIFAR GPU -- Continual Learning")
    print("=" * 60)
    print(f"  GPU            : {gpu_name}")
    print(f"  Backbone       : {args.backbone.upper()}")
    print(f"  Gorevler       : {args.num_tasks} x 10 sinif")
    print(f"  Epoch/gorev    : {args.epochs}")
    print(f"  Batch boyutu   : {args.batch_size} x {args.accum_steps} adim = {effective_batch} efektif")
    print(f"  Mixed Precision: {'ACIK (fp16)' if use_amp else 'KAPALI (fp32)'}")
    print(f"  Backbone donuk : {args.freeze_backbone}")
    print(f"  Ogretme sig.   : {not args.no_teach}")
    align_str = f"ACIK (epochs={args.align_epochs}, cosface={args.cosface})" if args.gaussian_align else "KAPALI"
    print(f"  Gaussian align : {align_str}")
    print("=" * 60)

    # ─── VERİ YÜKLEME ─────────────────────────────────────────────────────────
    data_transform = VIT_TRANSFORM if args.backbone == "vit" else None
    # GPU'da pin_memory=True: sayfalı bellek → DMA transferi → hız artışı
    tasks = get_cifar100_tasks(
        batch_size=args.batch_size,
        root=args.data_dir,
        num_workers=4,          # GPU sunucularında 4-8 worker mantıklı
        transform=data_transform,
        pin_memory=True,        # CPU→GPU transfer için
    )

    # ─── MODEL OLUŞTURMA ──────────────────────────────────────────────────────
    model = HOPEModel(
        num_classes=100,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        backbone_type=args.backbone,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] Toplam param    : {total_params:,}")
    print(f"[Model] GPU bellek      : {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ─── OPTİMİZER + AMP SCALER ──────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.meta_param_groups(
            backbone_lr=args.backbone_lr,
            classifier_lr=args.classifier_lr,
        ),
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=use_amp)

    # ─── SÜREKLİ ÖĞRENME DÖNGÜSÜ ─────────────────────────────────────────────
    metrics       = ContinualMetrics(num_tasks=args.num_tasks)
    buffer        = CalibrationBuffer(samples_per_class=args.samples_per_class) if args.replay else None
    gaussian_buffer = GaussianBuffer() if args.gaussian_align else None
    seen_classes: list[int] = []

    for task in tasks[:args.num_tasks]:
        seen_classes.extend(task.class_ids)

        print(f"\n{'='*60}")
        print(f"  Gorev {task.task_id}  |  Siniflar {task.class_ids[0]}-{task.class_ids[-1]}")
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU bellek kullanimi: {gpu_mem:.2f} GB")
        print(f"{'='*60}")

        # ─── EĞİTİM ───────────────────────────────────────────────────────────
        for epoch in range(args.epochs):
            loss = train_one_epoch_gpu(
                model=model,
                loader=task.train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                current_class_ids=seen_classes,
                run_teach=not args.no_teach,
                buffer=buffer,
                replay_batch=args.replay_batch,
                replay_weight=args.replay_weight,
                accum_steps=args.accum_steps,
                use_amp=use_amp,
            )
            print(f"  Epoch {epoch+1}/{args.epochs} | Kayip: {loss:.4f}")

        # ─── GÖREV SINIRI ─────────────────────────────────────────────────────
        if args.reset_all_cms:
            model.cms.reset_all()
        else:
            model.on_task_boundary()

        # ─── GAUSSIAN KALİBRASYON ─────────────────────────────────────────────
        if gaussian_buffer is not None:
            print(f"  Gaussian istatistikleri guncelleniyor ({len(seen_classes)} sinif)...")
            # Backbone fine-tune sonrası feature space değişti → TÜM görülen sınıflar yeniden hesaplanmalı
            for prev_task in tasks[:task.task_id + 1]:
                gaussian_buffer.update(model, prev_task.train_loader, device, prev_task.class_ids)
            print(f"  Siniflandirici kalibre ediliyor (epoch={args.align_epochs})...")
            classifier_align(
                classifier=model.classifier,
                gaussian_buffer=gaussian_buffer,
                device=device,
                epochs=args.align_epochs,
                lr=0.005,
                n_per_class=256,
                scale=args.cosface_scale,
                cosface=args.cosface,
            )
            print(f"  Kalibrasyon tamamlandi.")

        # ─── NCM SINIFLARI HESAPLA ────────────────────────────────────────────
        if gaussian_buffer is not None:
            class_means = gaussian_buffer.get_class_means()
        elif buffer is not None:
            class_means = compute_class_means(model, buffer, device)
        else:
            class_means = None

        # ─── DEĞERLENDİR ──────────────────────────────────────────────────────
        print(f"\n  Gorev {task.task_id} sonrasi degerlendirme:")
        accs = []
        for prev_task in tasks[:task.task_id + 1]:
            if class_means is not None:
                acc = evaluate_ncm(
                    model, prev_task.test_loader, device, class_means,
                    use_backbone=(gaussian_buffer is not None),
                )
            else:
                acc = evaluate(model, prev_task.test_loader, device)
            accs.append(acc)
            marker = " <- mevcut gorev" if prev_task.task_id == task.task_id else ""
            print(f"    Gorev {prev_task.task_id}: {acc:.2f}%{marker}")

        metrics.record(after_task=task.task_id, accs=accs)
        torch.cuda.empty_cache()

    # ─── SONUÇLAR ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SONUCLAR -- HOPE-CIFAR GPU")
    print("=" * 60)
    metrics.print_matrix()
    print()
    print(metrics.summary())

    metrics.save(os.path.join(result_dir, "metrics.json"))
    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nSonuclar kaydedildi -> {result_dir}")


if __name__ == "__main__":
    main()
