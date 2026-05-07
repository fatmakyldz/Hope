"""
Eğitim motoru — HOPE 2 geçişli eğitim döngüsü ve değerlendirme fonksiyonları.

Eğitim her batch için şu adımları izler:
  1. Geçiş-1 (Meta İleri): backbone → CMS → sınıflandırıcı → logit, özellik
  2. Öğretme Sinyali: kapalı-form CE gradyanı hesapla (autograd yok)
  3. Geçiş-2 (CMS Güncelle): öğretme sinyaliyle CMS hızlı ağırlıklarını güncelle
  4. Meta Geri Yayılım: CE kaybını backbone + sınıflandırıcıya geri yay

Değerlendirme iki yöntemle yapılır:
  - evaluate():     Standart softmax argmax (tek görev veya sadece mevcut görev için)
  - evaluate_ncm(): Nearest Class Mean (NCM) — tüm görevlerde softmax kaymasına karşı bağışıklık

Replay entegrasyonu:
  - Buffer'dan eski örnekler alınarak mevcut batch ile birleştirilir
  - CMS güncellemesi (Geçiş-2) sadece mevcut göreve ait örnekleri kullanır
  - Replay örnekleri yalnızca sınıflandırıcıyı kalibre eder
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from model.hope_model import HOPEModel


# ─── ANA EĞİTİM FONKSİYONU ───────────────────────────────────────────────────
def train_one_epoch(
    model: HOPEModel,
    loader,
    optimizer: Optimizer,
    device: torch.device,
    current_class_ids: list[int],  # bu görevde görülen tüm sınıf IDleri
    run_teach: bool = True,
    buffer=None,               # CalibrationBuffer | None
    replay_batch: int = 32,    # her adımda replay'den kaç örnek alınacak
    replay_weight: float = 1.0, # replay kaybının ağırlığı
    dynamic_replay: bool = True, # eski sınıf sayısıyla orantılı replay büyüklüğü
    cms_sync_fn=None,          # dağıtık eğitimde CMS senkronizasyon callback'i
) -> float:
    model.train()
    total_loss = 0.0

    # ─── DİNAMİK REPLAY BOYUTU ───────────────────────────────────────────────
    # Görev sayısı arttıkça eski sınıflar için daha fazla replay örneği gerekir.
    # Örnek: Görev 5'te 50 eski sınıf var → replay_batch × 5 = 160 örnek/adım
    # Bu olmadan: 10 yeni sınıf 64 örnek alırken, 90 eski sınıf 32 örnek alır → dengesizlik
    if dynamic_replay and buffer is not None:
        n_old = max(buffer.num_classes(), 1)
        effective_replay = min(replay_batch * max(n_old // 10, 1), 256)
    else:
        effective_replay = replay_batch

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # ─── REPLAY İLE BATCH BİRLEŞTİR ──────────────────────────────────────
        # DÜZELTİLDİ: Önce eski örnekleri çek, sonra mevcut batch'i ekle.
        # Önceki sıra (önce add, sonra sample) mevcut batch'in kendi replay'ine
        # karışmasına yol açıyordu.
        if buffer is not None and len(buffer) > 0:
            rep_imgs, rep_lbls = buffer.sample(effective_replay, device)
            if rep_imgs is not None:
                all_imgs = torch.cat([images, rep_imgs], dim=0)
                all_lbls = torch.cat([labels, rep_lbls], dim=0)
                n_cur = images.size(0)  # mevcut görev örneklerinin sayısı
            else:
                all_imgs, all_lbls, n_cur = images, labels, images.size(0)
        else:
            all_imgs, all_lbls, n_cur = images, labels, images.size(0)

        # Mevcut batch'i sample'dan sonra buffer'a ekle (aynı batch replay'e karışmasın)
        if buffer is not None:
            buffer.add(images, labels)

        # ─── GEÇİŞ-1: META İLERİ HESAPLAMA ──────────────────────────────────
        logits, backbone_feat, cms_out = model(all_imgs)

        # Mevcut görev kaybı (normal ağırlık)
        cur_loss = F.cross_entropy(logits[:n_cur], all_lbls[:n_cur])

        # Replay kaybı (eski görevler için sınıflandırıcıyı kalibre eder)
        if all_imgs.size(0) > n_cur:
            rep_loss = F.cross_entropy(logits[n_cur:], all_lbls[n_cur:])
            loss = cur_loss + replay_weight * rep_loss
        else:
            loss = cur_loss

        # ─── ÖĞRETME SİNYALİ HAZIRLA (backward'dan önce W snapshot'ı al) ─────
        # W backward'dan önce klonlanır: logits bu W ile hesaplandı → teach tutarlı.
        # CMS update backward'dan SONRA yapılır → in-place değişiklik graph'ı bozmaz.
        # DDP sarmalayıcısını geç: model.module varsa onu, yoksa model'i kullan
        raw = model.module if hasattr(model, "module") else model

        if run_teach:
            with torch.no_grad():
                W_snap = raw.classifier.weight.detach().clone()
                B_cur = backbone_feat[:n_cur].size(0)
                p = torch.softmax(logits[:n_cur].detach(), dim=-1)
                p[torch.arange(B_cur, device=p.device), all_lbls[:n_cur]] -= 1.0
                p = p / B_cur
                teach = -(p @ W_snap)  # (B, dim) — öğretme yönü

        # ─── META GERİ YAYILIM ────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient maskeleme: sınıflandırıcı sadece görülen sınıflar için güncellenir.
        _mask_classifier_grads(raw.classifier, current_class_ids, device)

        # Gradient patlamalarını önlemek için norm kırpma
        torch.nn.utils.clip_grad_norm_(raw.meta_parameters(), max_norm=1.0)
        optimizer.step()

        # ─── GEÇİŞ-2: CMS GÜNCELLE (backward'dan sonra — graph artık serbest) ─
        if run_teach:
            raw.update_cms(backbone_feat[:n_cur], teach)
            # Dağıtık modda CMS parametrelerini tüm node'larda eşitle
            if cms_sync_fn is not None:
                cms_sync_fn()

        total_loss += cur_loss.item()

    return total_loss / max(len(loader), 1)


# ─── STANDART DEĞERLENDİRME (SOFTMAX) ────────────────────────────────────────
@torch.no_grad()
def evaluate(model: HOPEModel, loader, device: torch.device) -> float:
    """
    Standart softmax argmax değerlendirmesi.
    Tek görevde iyi çalışır, ancak çok görevde softmax kaymasından etkilenir.
    """
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _, _ = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# ─── SINIF ORTALAMASI HESAPLAMA (NCM İÇİN) ───────────────────────────────────
@torch.no_grad()
def compute_class_means(
    model: HOPEModel,
    buffer,
    device: torch.device,
) -> dict[int, Tensor]:
    """
    Buffer'daki her sınıf için L2 normalize edilmiş ortalama özellik vektörü hesaplar.

    Bu vektörler NCM değerlendirmesinde mesafe hesabı için kullanılır.
    Buffer görüntüleri üzerinde model çalıştırılarak gerçek özellikler alınır.
    Backbone ağırlıkları değiştikçe class mean'leri de otomatik güncellenir
    (her görev sonunda yeniden hesaplanır).
    """
    model.eval()
    class_means: dict[int, Tensor] = {}
    for cid, imgs in buffer._store.items():
        feats = []
        # Büyük sınıflarda bellek aşımını önlemek için batch'ler halinde işle
        for i in range(0, len(imgs), 64):
            batch = torch.stack(imgs[i : i + 64]).to(device)
            _, _, feat = model(batch)  # cms_out kullan → NCM CMS katkısını yansıtır
            feats.append(feat.cpu())
        mean = torch.cat(feats, dim=0).mean(dim=0)  # tüm örneklerin ortalaması
        class_means[cid] = F.normalize(mean, dim=0)  # birim vektör → cosine mesafe için
    return class_means


# ─── NCM DEĞERLENDİRME ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_ncm(
    model: HOPEModel,
    loader,
    device: torch.device,
    class_means: dict[int, Tensor],
    use_backbone: bool = False,
) -> float:
    """
    Nearest Class Mean (En Yakın Sınıf Ortalaması) değerlendirmesi.

    Softmax kullanmaz — bunun yerine test örneğinin özelliğini,
    tüm sınıfların ortalama özellik vektörlerine cosine benzerliğiyle karşılaştırır.
    En yüksek benzerlik hangi sınıfa aitse o sınıf tahmin edilir.

    Neden daha iyi?
    Softmax ile değerlendirmede yeni sınıfların yüksek logitleri,
    eski sınıfların olasılıklarını sistematik olarak küçültür.
    NCM bu problemi tamamen ortadan kaldırır çünkü sınıflar arası
    "logit rekabeti" yoktur — her sınıf bağımsız olarak mesafe ile değerlendirilir.
    """
    model.eval()
    class_ids = sorted(class_means.keys())
    # Tüm sınıf ortalamalarını tek bir matrise topla: (C, 512)
    means = torch.stack([class_means[c] for c in class_ids]).to(device)

    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        _, backbone_feat, cms_out = model(images)
        features = backbone_feat if use_backbone else cms_out
        features = F.normalize(features, dim=1)        # L2 normalize → cosine benzerliği için
        sims = features @ means.T                      # (B, C) — her sınıfa cosine benzerlik skoru
        pred_indices = sims.argmax(dim=1).cpu()        # en yüksek benzerlik indeksi
        pred_labels = torch.tensor([class_ids[i] for i in pred_indices.tolist()])
        correct += (pred_labels == labels.cpu()).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# ─── GRADİENT MASKELEME ───────────────────────────────────────────────────────
def _mask_classifier_grads(
    classifier: nn.Linear,
    allowed_class_ids: list[int],
    device: torch.device,
) -> None:
    """
    Sınıflandırıcının yalnızca görülen sınıflar için güncellenmesini sağlar.

    Neden gerekli?
    100 sınıflı bir sınıflandırıcıda, görev 0 sadece 0-9 arası sınıfları görür.
    Ama loss.backward() tüm 100 sınıfın ağırlığı için gradient hesaplar.
    Bu maske, görülmemiş sınıfların gradyanlarını sıfırlayarak korunmalarını sağlar.
    Görev ilerledikçe izin verilen sınıflar (seen_classes) genişler.
    """
    if classifier.weight.grad is None:
        return
    num_classes = classifier.weight.size(0)
    mask = torch.zeros(num_classes, device=device)
    for cid in allowed_class_ids:
        if 0 <= cid < num_classes:
            mask[cid] = 1.0
    # Her sınıfın ağırlık satırını maskele: (num_classes, feature_dim)
    classifier.weight.grad *= mask.unsqueeze(1)
    if classifier.bias is not None and classifier.bias.grad is not None:
        classifier.bias.grad *= mask
