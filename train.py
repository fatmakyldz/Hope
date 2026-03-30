#!/usr/bin/env python3
"""
HOPE-CIFAR -- Faithful HOPE implementation for CIFAR-100 continual learning.

Mimari:
  ResNet18 (pretrained) -> 4-level CMS (fast/mid/slow/ultra) -> Linear classifier

Nested_learning'e sadik:
  - Replay buffer YOK
  - Fast/mid sifirla, slow/ultra koru (task sinirinda)
  - Deep Momentum per CMS level
  - Teach signal: closed-form CE gradient

Kullanim:
  python train.py
  python train.py --freeze_backbone
  python train.py --epochs 10 --batch_size 128
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HOPE-CIFAR Continual Learning")
    p.add_argument("--epochs",           type=int,   default=10)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--num_tasks",        type=int,   default=10)
    p.add_argument("--backbone_lr",      type=float, default=1e-4)
    p.add_argument("--classifier_lr",    type=float, default=1e-3)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--freeze_backbone",  action="store_true")
    p.add_argument("--no_pretrained",    action="store_true")
    p.add_argument("--no_teach",         action="store_true", help="Disable teach signal (ablation)")
    p.add_argument("--reset_all_cms",    action="store_true", help="Reset ALL CMS at task boundary (ablation)")
    p.add_argument("--replay",           action="store_true", help="Use calibration replay buffer")
    p.add_argument("--samples_per_class",type=int,   default=100, help="Buffer size per class")
    p.add_argument("--replay_batch",     type=int,   default=32)
    p.add_argument("--replay_weight",    type=float, default=0.5)
    p.add_argument("--data_dir",         type=str,   default="./data")
    p.add_argument("--results_dir",      type=str,   default="./results")
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Results dir
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"cifar100_t{args.num_tasks}_e{args.epochs}"
    if args.freeze_backbone:
        tag += "_frozen"
    result_dir = os.path.join(args.results_dir, f"{tag}_{ts}")
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print("  HOPE-CIFAR -- Continual Learning")
    print("=" * 60)
    print(f"  Device         : {device}")
    print(f"  Tasks          : {args.num_tasks} x 10 classes")
    print(f"  Epochs/task    : {args.epochs}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Pretrained     : {not args.no_pretrained}")
    print(f"  Teach signal   : {not args.no_teach}")
    print(f"  CMS levels     : fast(1) mid(4) slow(32) ultra(128)")
    replay_str = f"YES (spc={args.samples_per_class}, w={args.replay_weight})" if args.replay else "NONE"
    print(f"  Replay buffer  : {replay_str}")
    print("=" * 60)

    # Data
    tasks = get_cifar100_tasks(
        batch_size=args.batch_size,
        root=args.data_dir,
        num_workers=2,
    )

    # Model
    model = HOPEModel(
        num_classes=100,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    meta_params  = sum(p.numel() for p in model.meta_parameters())
    cms_params   = sum(p.numel() for p in model.cms.all_fast_params())
    print(f"\n[Model] Total params   : {total_params:,}")
    print(f"[Model] Meta params    : {meta_params:,}")
    print(f"[Model] CMS fast params: {cms_params:,}")

    # Optimizer (meta only -- CMS uses deep momentum internally)
    optimizer = optim.AdamW(
        model.meta_param_groups(
            backbone_lr=args.backbone_lr,
            classifier_lr=args.classifier_lr,
        ),
        weight_decay=args.weight_decay,
    )

    metrics = ContinualMetrics(num_tasks=args.num_tasks)
    buffer = CalibrationBuffer(samples_per_class=args.samples_per_class) if args.replay else None
    seen_classes: list[int] = []

    for task in tasks[:args.num_tasks]:
        seen_classes.extend(task.class_ids)

        print(f"\n{'='*60}")
        print(f"  Task {task.task_id}  |  Classes {task.class_ids[0]}-{task.class_ids[-1]}")
        print(f"{'='*60}")

        # -- Train -----------------------------------------------------------
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
            print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f}")

        # -- Task boundary: reset fast+mid CMS --------------------------------
        if args.reset_all_cms:
            model.cms.reset_all()
        else:
            model.on_task_boundary()  # reset fast+mid, keep slow+ultra

        # -- Compute class means for NCM (if replay enabled) ------------------
        if buffer is not None:
            class_means = compute_class_means(model, buffer, device)
        else:
            class_means = None

        # -- Evaluate all seen tasks ------------------------------------------
        print(f"\n  Evaluation after Task {task.task_id}:")
        accs = []
        for prev_task in tasks[:task.task_id + 1]:
            if class_means is not None:
                acc = evaluate_ncm(model, prev_task.test_loader, device, class_means)
            else:
                acc = evaluate(model, prev_task.test_loader, device)
            accs.append(acc)
            marker = " <- current" if prev_task.task_id == task.task_id else ""
            print(f"    Task {prev_task.task_id}: {acc:.2f}%{marker}")

        metrics.record(after_task=task.task_id, accs=accs)

        if device.type == "mps":
            torch.mps.empty_cache()

    # -- Final results --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SONUCLAR -- HOPE-CIFAR")
    print("=" * 60)
    metrics.print_matrix()
    print()
    print(metrics.summary())

    # Save
    metrics.save(os.path.join(result_dir, "metrics.json"))
    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nSonuclar kaydedildi -> {result_dir}")


if __name__ == "__main__":
    main()
