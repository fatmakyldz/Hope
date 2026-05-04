# HOPE-CIFAR: Continual Learning on CIFAR-10/100

HOPE (Hierarchical Online Plasticity Engine) implementasyonu — CIFAR-10 ve CIFAR-100 üzerinde sürekli öğrenme (continual learning).

## Mimari

```
Görüntü → Backbone (ResNet18 / ViT-B/16) → CMS (4 kademe) → Sınıflandırıcı
```

- **Backbone**: ResNet18 (hızlı) veya ViT-B/16 (güçlü, ImageNet pretrained)
- **CMS**: fast(1) · mid(4) · slow(32) · ultra(128) — DeepMomentum optimizer
- **Gaussian Alignment**: TUNA'dan uyarlanmış sınıflandırıcı kalibrasyonu
- **Replay Buffer**: eski sınıflardan örnek saklama

---

## Kurulum

```bash
git clone https://github.com/fatmakyldz/hope_cifar100.git
cd hope_cifar100
pip install torch torchvision timm
```

---

## Çalıştırma Komutları

### 1. Temel Eğitim (`train.py`) — CPU / MPS / GPU

```bash
# Varsayılan: ResNet18, CIFAR-100, 10 görev
python train.py

# ViT backbone ile
python train.py --backbone vit

# Gaussian alignment ekle
python train.py --gaussian_align

# Gaussian + CosFace
python train.py --gaussian_align --cosface

# Gaussian + Replay buffer
python train.py --gaussian_align --replay --samples_per_class 50

# Backbone dondur (sadece CMS + classifier öğrenir)
python train.py --freeze_backbone --gaussian_align

# CIFAR-10 (5 görev × 2 sınıf)
python train.py --dataset cifar10 --gaussian_align --replay

# Teach signal olmadan (ablasyon)
python train.py --no_teach
```

### 2. GPU Eğitimi (`train_gpu.py`) — NVIDIA GPU gerekli

```bash
# ViT + Gaussian + Replay (önerilen)
python train_gpu.py --backbone vit --gaussian_align --replay \
  --samples_per_class 50 --epochs 5 --batch_size 48 --grad_checkpoint

# ResNet + Gaussian (daha hızlı)
python train_gpu.py --backbone resnet --gaussian_align --replay \
  --samples_per_class 50 --epochs 10 --batch_size 128

# CIFAR-10 üzerinde
python train_gpu.py --dataset cifar10 --backbone vit --gaussian_align --replay \
  --samples_per_class 50 --epochs 5 --batch_size 48 --grad_checkpoint

# Gradient accumulation (bellek tasarrufu)
python train_gpu.py --backbone vit --gaussian_align \
  --batch_size 32 --accum_steps 4

# Mixed precision kapat (hata ayıklama)
python train_gpu.py --backbone vit --no_amp
```

### 3. Google Colab (T4 GPU)

```python
# Hücre 1
!git clone https://github.com/fatmakyldz/hope_cifar100.git

# Hücre 2
!pip install timm -q

# Hücre 3
import os
os.chdir("hope_cifar100")

# Hücre 4 — CIFAR-100, ViT, Gaussian + Replay
!python train_gpu.py --backbone vit --gaussian_align --replay \
  --samples_per_class 50 --epochs 5 --num_tasks 5 \
  --batch_size 48 --grad_checkpoint

# Hücre 4 (alternatif) — CIFAR-10, daha hızlı
!python train_gpu.py --dataset cifar10 --backbone vit --gaussian_align --replay \
  --samples_per_class 50 --epochs 5 --batch_size 48 --grad_checkpoint
```

### 4. Kaggle (2× T4 GPU — DDP Smoke Test)

> Settings → Internet: ON · Accelerator: GPU T4 × 2

```python
# Hücre 1
!git clone https://github.com/fatmakyldz/hope_cifar100.git

# Hücre 2
!pip install timm -q

# Hücre 3
import os
os.chdir("hope_cifar100")

# Hücre 4 — 2 GPU DDP smoke test
!torchrun --nproc_per_node=2 train_smoke.py --backbone vit
```

### 5. Lab / Sunucu (Çok GPU)

```bash
# 2 GPU — smoke test (hızlı doğrulama)
torchrun --nproc_per_node=2 train_smoke.py --backbone vit

# 2 GPU — tam eğitim
torchrun --nproc_per_node=2 train_distributed.py \
  --backbone vit --gaussian_align --epochs 15

# 4 GPU
torchrun --nproc_per_node=4 train_distributed.py \
  --backbone vit --gaussian_align --epochs 15 --batch_size 64
```

---

## Argümanlar

| Argüman | Varsayılan | Açıklama |
|---|---|---|
| `--dataset` | `cifar100` | `cifar100` (10 görev × 10 sınıf) veya `cifar10` (5 görev × 2 sınıf) |
| `--backbone` | `resnet` | `resnet` (ResNet18) veya `vit` (ViT-B/16) |
| `--epochs` | `10` | Görev başına epoch sayısı |
| `--batch_size` | `64` | Batch boyutu |
| `--num_tasks` | otomatik | Kaç görev çalıştırılacak |
| `--gaussian_align` | kapalı | Gaussian sınıflandırıcı kalibrasyonu |
| `--align_epochs` | `30` | Kalibrasyon epoch sayısı |
| `--cosface` | kapalı | CosFace loss kullan |
| `--replay` | kapalı | Replay buffer etkinleştir |
| `--samples_per_class` | `100` | Buffer'da sınıf başına örnek |
| `--freeze_backbone` | kapalı | Backbone ağırlıklarını dondur |
| `--no_teach` | kapalı | Teach sinyalini kapat (ablasyon) |
| `--grad_checkpoint` | kapalı | ViT gradient checkpointing (OOM önleme) |
| `--no_amp` | kapalı | Mixed precision kapat |
| `--seed` | `42` | Rastgele tohum |

---

## Deney Sonuçları (ResNet18, CIFAR-100)

| Yöntem | Ortalama Doğruluk |
|---|---|
| Baseline (CE only) | 44.50% |
| + Gaussian Align | 56.69% |
| + Gaussian + CosFace | 27.90% |
| + Freeze Backbone + Gaussian | 22.20% |

ViT + Gaussian + Replay → beklenen: **70-85%** (lab GPU sonuçları bekleniyor)

---

## Dosya Yapısı

```
hope_cifar100/
├── train.py              # Ana eğitim scripti (CPU/MPS/GPU)
├── train_gpu.py          # GPU optimize (AMP, pin_memory, grad checkpoint)
├── train_smoke.py        # 2-GPU DDP smoke test
├── train_distributed.py  # Çok GPU DDP eğitimi
├── data/
│   ├── cifar100.py       # CIFAR-100 görev yükleyici (10 × 10 sınıf)
│   └── cifar10.py        # CIFAR-10 görev yükleyici (5 × 2 sınıf)
├── model/
│   ├── backbone.py       # ResNet18 + ViT-B/16
│   ├── cms.py            # CMS modülü (4 kademe)
│   └── hope_model.py     # Ana model
├── memory/
│   ├── gaussian_buffer.py  # Gaussian dağılım saklama
│   └── replay_buffer.py    # Örnek replay buffer
├── training/
│   ├── engine.py           # Eğitim + değerlendirme fonksiyonları
│   └── classifier_align.py # Gaussian kalibrasyon
└── utils/
    └── metrics.py          # Sürekli öğrenme metrikleri
```
