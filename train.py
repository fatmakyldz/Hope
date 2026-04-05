#!/usr/bin/env python3
"""
HOPE-CIFAR — CIFAR-100 Sürekli Öğrenme (Continual Learning) Ana Scripti

Bu script, HOPE mimarisini CIFAR-100 üzerinde çalıştırır.
100 sınıf, 10 göreve bölünmüştür (her görevde 10 yeni sınıf).

─── MİMARİ ───────────────────────────────────────────────────────────────────
  ResNet18 (ImageNet önceden eğitilmiş)
    ↓ 512 boyutlu özellik vektörü
  4 Kademeli CMS (fast → mid → slow → ultra)
    ↓ 512 boyutlu dönüştürülmüş özellik
  Lineer Sınıflandırıcı (512 → 100 sınıf)

─── STANDART YÖNTEMLERDEN FARKI ─────────────────────────────────────────────
  1. CMS: Hızlı/yavaş bellek hiyerarşisi ile unutmayı engeller
  2. Öğretme Sinyali: Kapalı-form CE gradyanı (backprop yerine)
  3. NCM Değerlendirme: Softmax kaymasına karşı bağışıklıklı tahmin
  4. Kalibrasyon Replay: Minimal buffer (sınıflandırıcı kalibrasyonu için)

─── KULLANIM ─────────────────────────────────────────────────────────────────
  python train.py                        # varsayılan ayarlar
  python train.py --freeze_backbone      # backbone dondurulmuş
  python train.py --replay               # replay buffer etkin
  python train.py --no_teach             # teach signal ablasyonu
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))

from data.cifar100 import get_cifar100_tasks
from memory.replay_buffer import CalibrationBuffer
from model.hope_model import HOPEModel
from training.engine import compute_class_means, evaluate, evaluate_ncm, train_one_epoch
from utils.metrics import ContinualMetrics


# ─── ARGÜMANLAR ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HOPE-CIFAR Continual Learning")
    # Eğitim hiperparametreleri
    p.add_argument("--epochs",           type=int,   default=10,  help="Her görev için epoch sayısı")
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--num_tasks",        type=int,   default=10,  help="Toplam görev sayısı (max 10)")
    # Optimizer ayarları
    p.add_argument("--backbone_lr",      type=float, default=1e-4, help="Backbone öğrenme hızı (küçük tutulmalı)")
    p.add_argument("--classifier_lr",    type=float, default=1e-3, help="Sınıflandırıcı öğrenme hızı")
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    # Backbone seçenekleri
    p.add_argument("--freeze_backbone",  action="store_true", help="Backbone ağırlıklarını dondur")
    p.add_argument("--no_pretrained",    action="store_true", help="ImageNet ağırlıkları kullanma")
    # Ablasyon bayrakları (HOPE bileşenlerini tek tek kapatmak için)
    p.add_argument("--no_teach",         action="store_true", help="Öğretme sinyalini devre dışı bırak")
    p.add_argument("--reset_all_cms",    action="store_true", help="Görev sınırında TÜM CMS kademelerini sıfırla")
    # Replay buffer ayarları
    p.add_argument("--replay",           action="store_true", help="Kalibrasyon replay buffer'ı etkinleştir")
    p.add_argument("--samples_per_class",type=int,   default=100, help="Buffer'da sınıf başına örnek sayısı")
    p.add_argument("--replay_batch",     type=int,   default=32,  help="Her adımda replay'den alınan örnek sayısı")
    p.add_argument("--replay_weight",    type=float, default=0.5, help="Replay kaybının ağırlığı")
    # Dizin ayarları
    p.add_argument("--data_dir",         type=str,   default="./data")
    p.add_argument("--results_dir",      type=str,   default="./results")
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


# ─── ANA FONKSİYON ───────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Tekrarlanabilirlik için rastgele tohum sabitle
    torch.manual_seed(args.seed)

    # ─── DONANIM SEÇIMI ───────────────────────────────────────────────────────
    # Öncelik sırası: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # M1/M2 Mac GPU
    else:
        device = torch.device("cpu")

    # Sonuç dizini: zaman damgalı, her çalıştırma için ayrı klasör
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"cifar100_t{args.num_tasks}_e{args.epochs}"
    if args.freeze_backbone:
        tag += "_frozen"
    result_dir = os.path.join(args.results_dir, f"{tag}_{ts}")
    os.makedirs(result_dir, exist_ok=True)

    # Çalıştırma konfigürasyonunu yazdır
    print("=" * 60)
    print("  HOPE-CIFAR -- Continual Learning")
    print("=" * 60)
    print(f"  Donanim        : {device}")
    print(f"  Gorevler       : {args.num_tasks} x 10 sinif")
    print(f"  Epoch/gorev    : {args.epochs}")
    print(f"  Backbone donuk : {args.freeze_backbone}")
    print(f"  Onceden egit.  : {not args.no_pretrained}")
    print(f"  Ogretme sig.   : {not args.no_teach}")
    print(f"  CMS kademeleri : fast(1) mid(4) slow(32) ultra(128)")
    replay_str = f"ACIK (spc={args.samples_per_class}, w={args.replay_weight})" if args.replay else "KAPALI"
    print(f"  Replay buffer  : {replay_str}")
    print("=" * 60)

    # ─── VERİ YÜKLEME ─────────────────────────────────────────────────────────
    # 10 görev × 10 sınıf = 100 sınıf, sıralı olarak sunulur
    tasks = get_cifar100_tasks(
        batch_size=args.batch_size,
        root=args.data_dir,
        num_workers=2,
    )

    # ─── MODEL OLUŞTURMA ──────────────────────────────────────────────────────
    model = HOPEModel(
        num_classes=100,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    meta_params  = sum(p.numel() for p in model.meta_parameters())
    cms_params   = sum(p.numel() for p in model.cms.all_fast_params())
    print(f"\n[Model] Toplam param    : {total_params:,}")
    print(f"[Model] Meta param      : {meta_params:,}")
    print(f"[Model] CMS hizli param : {cms_params:,}")

    # ─── OPTİMİZER ────────────────────────────────────────────────────────────
    # Meta optimizer: sadece backbone + sınıflandırıcı (CMS HARİÇ)
    # CMS kendi DeepMomentum optimizer'ını kullanır
    # Backbone ve sınıflandırıcıya farklı öğrenme hızı verilir
    optimizer = optim.AdamW(
        model.meta_param_groups(
            backbone_lr=args.backbone_lr,
            classifier_lr=args.classifier_lr,
        ),
        weight_decay=args.weight_decay,
    )

    # ─── SÜREKLİ ÖĞRENME DÖNGÜSÜ ─────────────────────────────────────────────
    metrics = ContinualMetrics(num_tasks=args.num_tasks)
    # Replay buffer: sınıflandırıcı kalibrasyonu için eski görev örneklerini saklar
    buffer = CalibrationBuffer(samples_per_class=args.samples_per_class) if args.replay else None
    seen_classes: list[int] = []  # şimdiye kadar görülen tüm sınıf IDleri

    for task in tasks[:args.num_tasks]:
        # Bu görevin sınıflarını görülen sınıflar listesine ekle
        seen_classes.extend(task.class_ids)

        print(f"\n{'='*60}")
        print(f"  Gorev {task.task_id}  |  Siniflar {task.class_ids[0]}-{task.class_ids[-1]}")
        print(f"{'='*60}")

        # ─── EĞİTİM ───────────────────────────────────────────────────────────
        # Her epoch'ta 2-geçişli HOPE eğitim döngüsü çalışır
        for epoch in range(args.epochs):
            loss = train_one_epoch(
                model=model,
                loader=task.train_loader,
                optimizer=optimizer,
                device=device,
                current_class_ids=seen_classes,
                run_teach=not args.no_teach,
                buffer=buffer,
                replay_batch=args.replay_batch,
                replay_weight=args.replay_weight,
            )
            print(f"  Epoch {epoch+1}/{args.epochs} | Kayip: {loss:.4f}")

        # ─── GÖREV SINIRI: CMS SIFIRLA ────────────────────────────────────────
        # Fast + mid kademeleri sıfırlanır → yeni göreve hazır
        # Slow + ultra korunur → eski görevlerden birikmiş bilgi kaybolmaz
        if args.reset_all_cms:
            model.cms.reset_all()  # ablasyon: hepsini sıfırla
        else:
            model.on_task_boundary()  # sadece fast + mid sıfırla

        # ─── NCM İÇİN SINIF ORTALAMALARINI HESAPLA ───────────────────────────
        # Buffer'daki görüntüler üzerinde model çalıştırılarak her sınıfın
        # ortalama özellik vektörü hesaplanır. Bu vektörler test zamanında
        # softmax yerine mesafe hesabı için kullanılır.
        if buffer is not None:
            class_means = compute_class_means(model, buffer, device)
        else:
            class_means = None

        # ─── TÜM GÖRÜLEN GÖREVLERİ DEĞERLENDİR ──────────────────────────────
        # Sürekli öğrenmede başarı: hem yeni görevi öğrenmek hem eski görevleri
        # hatırlamak. Her görev sonunda tüm önceki görevler de test edilir.
        print(f"\n  Gorev {task.task_id} sonrasi degerlendirme:")
        accs = []
        for prev_task in tasks[:task.task_id + 1]:
            if class_means is not None:
                # NCM: softmax kaymasına karşı bağışıklıklı değerlendirme
                acc = evaluate_ncm(model, prev_task.test_loader, device, class_means)
            else:
                # Standart softmax değerlendirme
                acc = evaluate(model, prev_task.test_loader, device)
            accs.append(acc)
            marker = " <- mevcut gorev" if prev_task.task_id == task.task_id else ""
            print(f"    Gorev {prev_task.task_id}: {acc:.2f}%{marker}")

        metrics.record(after_task=task.task_id, accs=accs)

        # Apple Silicon bellek temizliği
        if device.type == "mps":
            torch.mps.empty_cache()

    # ─── SONUÇLARI YAZDIR VE KAYDET ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SONUCLAR -- HOPE-CIFAR")
    print("=" * 60)
    metrics.print_matrix()
    print()
    print(metrics.summary())

    # JSON olarak kaydet: sonraki analizler için
    metrics.save(os.path.join(result_dir, "metrics.json"))
    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nSonuclar kaydedildi -> {result_dir}")


if __name__ == "__main__":
    main()
