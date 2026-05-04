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

## Platform Desteği

| Platform | train.py | train_gpu.py | train_smoke.py | train_distributed.py |
|---|---|---|---|---|
| Windows (CPU) | ✓ | — | — | — |
| Windows (NVIDIA GPU) | ✓ | ✓ | — | — |
| macOS (CPU / Apple Silicon MPS) | ✓ | — | — | — |
| Linux (CPU) | ✓ | — | — | — |
| Linux (NVIDIA GPU) | ✓ | ✓ | ✓ | ✓ |
| Google Colab T4 | ✓ | ✓ | ✓ | ✓ |
| Kaggle 2× T4 | ✓ | ✓ | ✓ | ✓ |

---

## Kurulum

```bash
git clone https://github.com/fatmakyldz/hope_cifar100.git
cd hope_cifar100
pip install torch torchvision timm
```

---

## 1. `train.py` — Tüm Platformlar (CPU / MPS / GPU)

Mac, Windows, Linux ve GPU'suz ortamlar için. Donanımı otomatik seçer: CUDA → MPS → CPU.

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

# CIFAR-10 (5 görev × 2 sınıf — daha hızlı test)
python train.py --dataset cifar10 --gaussian_align --replay

# Hızlı deneme: 3 görev, 5 epoch
python train.py --num_tasks 3 --epochs 5 --gaussian_align

# Teach signal kapalı (ablasyon)
python train.py --no_teach
```

---

## 2. `train_gpu.py` — NVIDIA GPU (Linux / Windows)

AMP (fp16), pin_memory, gradient checkpointing destekli. CUDA yoksa çalışmaz.  
Windows'ta `num_workers` otomatik 0'a düşer.

```bash
# ViT + Gaussian + Replay — önerilen
python train_gpu.py --backbone vit --gaussian_align --replay \
  --samples_per_class 50 --epochs 5 --batch_size 48 --grad_checkpoint

# ResNet + Gaussian (daha hızlı)
python train_gpu.py --backbone resnet --gaussian_align --replay \
  --samples_per_class 50 --epochs 10 --batch_size 128

# CIFAR-10
python train_gpu.py --dataset cifar10 --backbone vit --gaussian_align --replay \
  --samples_per_class 50 --epochs 5 --batch_size 48 --grad_checkpoint

# Sadece 5 görev (hızlı test)
python train_gpu.py --backbone vit --gaussian_align --num_tasks 5 \
  --epochs 5 --batch_size 48 --grad_checkpoint

# Gradient accumulation (bellek tasarrufu, büyük batch etkisi)
python train_gpu.py --backbone vit --gaussian_align \
  --batch_size 32 --accum_steps 4

# Mixed precision kapat (hata ayıklama)
python train_gpu.py --backbone vit --no_amp
```

---

## 3. Google Colab (T4 GPU)

```python
# Hücre 1 — Repo kur
!git clone https://github.com/fatmakyldz/hope_cifar100.git
!pip install timm -q

# Hücre 2 — Dizine gir
import os
os.chdir("hope_cifar100")

# Hücre 3 — CIFAR-100, ViT + Gaussian + Replay (5 görev, ~15 dk)
!python train_gpu.py --backbone vit --gaussian_align --replay \
  --samples_per_class 50 --epochs 5 --num_tasks 5 \
  --batch_size 48 --grad_checkpoint

# Hücre 3 (alternatif) — CIFAR-10, daha hızlı (~8 dk)
!python train_gpu.py --dataset cifar10 --backbone vit --gaussian_align --replay \
  --samples_per_class 50 --epochs 5 --batch_size 48 --grad_checkpoint

# Hücre 3 (alternatif) — ResNet, tam 10 görev (~20 dk)
!python train_gpu.py --backbone resnet --gaussian_align --replay \
  --samples_per_class 50 --epochs 10 --batch_size 128
```

> **Not**: Runtime yeniden başlatıldıktan sonra `os.chdir` tekrar çalıştırılmalıdır.

---

## 4. Kaggle (2× T4 GPU)

> **Gereksinimler**: Settings → Internet: **ON** · Accelerator: **GPU T4 × 2**  
> Telefon doğrulaması gerekebilir (GPU seçimi için).

```python
# Hücre 1
!git clone https://github.com/fatmakyldz/hope_cifar100.git
!pip install timm -q

# Hücre 2
import os
os.chdir("/kaggle/working/hope_cifar100")

# Hücre 3 — 2 GPU smoke test (hızlı doğrulama, ~1-2 dk)
!torchrun --nproc_per_node=2 train_smoke.py --backbone vit

# Hücre 3 (alternatif) — tek GPU, tam eğitim
!python train_gpu.py --backbone vit --gaussian_align --replay \
  --samples_per_class 50 --epochs 5 --num_tasks 5 \
  --batch_size 48 --grad_checkpoint
```

---

## 5. Lab / Sunucu (Çok GPU, Linux)

```bash
# 2 GPU — smoke test (saniyeler içinde tamamlanır)
torchrun --nproc_per_node=2 train_smoke.py --backbone vit

# 2 GPU — tam eğitim
torchrun --nproc_per_node=2 train_distributed.py \
  --backbone vit --epochs 15

# 4 GPU
torchrun --nproc_per_node=4 train_distributed.py \
  --backbone vit --epochs 15 --batch_size 64

# Arka planda çalıştır, log kaydet
nohup torchrun --nproc_per_node=2 train_distributed.py \
  --backbone vit --epochs 15 > run.log 2>&1 &
```

---

## Argümanlar

| Argüman | Varsayılan | Açıklama |
|---|---|---|
| `--dataset` | `cifar100` | `cifar100` (10×10 sınıf) veya `cifar10` (5×2 sınıf) |
| `--backbone` | `resnet` | `resnet` (ResNet18, hızlı) veya `vit` (ViT-B/16, güçlü) |
| `--epochs` | `10` | Görev başına epoch sayısı |
| `--batch_size` | `64` | Batch boyutu |
| `--num_tasks` | otomatik | Kaç görev çalıştırılacak |
| `--gaussian_align` | kapalı | Gaussian sınıflandırıcı kalibrasyonu |
| `--align_epochs` | `30` | Kalibrasyon epoch sayısı |
| `--cosface` | kapalı | CosFace loss kullan |
| `--replay` | kapalı | Replay buffer etkinleştir |
| `--samples_per_class` | `100` | Buffer'da sınıf başına örnek |
| `--replay_weight` | `0.5` | Replay kaybının ağırlığı |
| `--freeze_backbone` | kapalı | Backbone ağırlıklarını dondur |
| `--no_teach` | kapalı | Teach sinyalini kapat (ablasyon) |
| `--grad_checkpoint` | kapalı | ViT gradient checkpointing — T4 OOM önleme |
| `--no_amp` | kapalı | Mixed precision kapat (debug) |
| `--backbone_lr` | `1e-4` | Backbone öğrenme hızı |
| `--classifier_lr` | `1e-3` | Sınıflandırıcı öğrenme hızı |
| `--seed` | `42` | Rastgele tohum |

---

## Deney Sonuçları (ResNet18, CIFAR-100, 10 görev)

| Yöntem | Ortalama Doğruluk |
|---|---|
| Baseline (CE only) | 44.50% |
| + Gaussian Align | 56.69% |
| + Gaussian + Replay | 56.69% |
| + Gaussian + CosFace | 27.90% |
| + Freeze Backbone + Gaussian | 22.20% |

> ViT-B/16 + Gaussian + Replay beklenen: **70–85%** (lab GPU sonuçları bekleniyor)

---

## Dosya Yapısı

```
hope_cifar100/
├── train.py              # Ana script — CPU / MPS / GPU, tüm platformlar
├── train_gpu.py          # GPU optimize — AMP, pin_memory, grad checkpoint
├── train_smoke.py        # 2-GPU DDP smoke test (Linux/Colab/Kaggle)
├── train_distributed.py  # Çok GPU DDP eğitimi (Linux/Colab/Kaggle)
├── data/
│   ├── cifar100.py       # CIFAR-100 görev yükleyici (10 görev × 10 sınıf)
│   └── cifar10.py        # CIFAR-10 görev yükleyici (5 görev × 2 sınıf)
├── model/
│   ├── backbone.py       # ResNet18 + ViT-B/16 (grad checkpoint destekli)
│   ├── cms.py            # CMS modülü — 4 kademe DeepMomentum
│   └── hope_model.py     # Ana model
├── memory/
│   ├── gaussian_buffer.py  # Gaussian dağılım saklama (mean + cov)
│   └── replay_buffer.py    # Örnek replay buffer
├── training/
│   ├── engine.py           # Eğitim + NCM değerlendirme
│   └── classifier_align.py # Gaussian kalibrasyon + CosFace
└── utils/
    └── metrics.py          # Sürekli öğrenme metrikleri (ACC, BWT, FWT)
```
