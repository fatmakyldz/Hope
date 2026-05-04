#!/usr/bin/env python3
"""HOPE-CIFAR Progress Report - PDF Generator"""
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "HOPE_CIFAR_Report.pdf")

ARIAL     = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_B   = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ARIAL_I   = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
COURIER_N = "/System/Library/Fonts/Supplemental/Courier New.ttf"
COURIER_B = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"

DARK_BLUE   = (23,  54,  93)
MID_BLUE    = (31,  78, 121)
ACCENT_BLUE = (68, 114, 196)
LIGHT_BLUE  = (189, 215, 238)
GREEN       = (56, 142,  60)
ORANGE      = (230, 126,  34)
RED_SOFT    = (192,  57,  43)
WHITE       = (255, 255, 255)
LIGHT_GRAY  = (245, 245, 245)
MID_GRAY    = (189, 189, 189)
DARK_GRAY   = (66,  66,  66)
BLACK       = (30,  30,  30)


class Report(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)
        self.add_font("Arial",    "", ARIAL)
        self.add_font("Arial",    "B", ARIAL_B)
        self.add_font("Arial",    "I", ARIAL_I)
        self.add_font("CourierN", "", COURIER_N)
        self.add_font("CourierN", "B", COURIER_B)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*DARK_BLUE)
        self.rect(0, 0, 210, 10, "F")
        self.set_y(2)
        self.set_font("Arial", "B", 8)
        self.set_text_color(*LIGHT_BLUE)
        self.cell(0, 6, "HOPE-CIFAR: Hierarchical Online Plasticity Engine — İlerleme Raporu",
                  align="C")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "", 8)
        self.set_text_color(*MID_GRAY)
        self.cell(0, 5, f"Sayfa {self.page_no()}", align="C")

    def section_title(self, text):
        self.ln(4)
        self.set_fill_color(*MID_BLUE)
        self.set_text_color(*WHITE)
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, f"  {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.set_text_color(*BLACK)
        self.ln(2)

    def sub_title(self, text):
        self.set_font("Arial", "B", 10)
        self.set_text_color(*ACCENT_BLUE)
        self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*BLACK)

    def body(self, text, indent=0):
        self.set_font("Arial", "", 9.5)
        self.set_text_color(*DARK_GRAY)
        self.set_x(self.l_margin + indent)
        self.multi_cell(0, 5.5, text)
        self.set_x(self.l_margin)

    def bullet(self, text, indent=4):
        self.set_font("Arial", "", 9.5)
        self.set_text_color(*DARK_GRAY)
        x = self.l_margin + indent
        self.set_x(x)
        self.cell(5, 5.5, "•")
        self.set_x(x + 5)
        self.multi_cell(0, 5.5, text)
        self.set_x(self.l_margin)

    def code(self, text):
        self.set_fill_color(*LIGHT_GRAY)
        self.set_font("CourierN", "", 8.5)
        self.set_text_color(60, 60, 60)
        self.set_x(self.l_margin + 4)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 8,
                        5, text, fill=True, border=0)
        self.set_font("Arial", "", 9.5)
        self.set_text_color(*DARK_GRAY)
        self.set_x(self.l_margin)
        self.ln(1)

    def draw_accuracy_row(self, label, accs, highlight_idx, col_w=17, row_h=6):
        self.set_font("Arial", "B" if highlight_idx < 0 else "", 8.5)
        self.set_fill_color(*LIGHT_BLUE if highlight_idx < 0 else WHITE)
        self.set_text_color(*DARK_BLUE if highlight_idx < 0 else DARK_GRAY)
        self.cell(22, row_h, label, border=1, fill=(highlight_idx < 0))
        for i, v in enumerate(accs):
            if v is None:
                self.set_fill_color(*LIGHT_GRAY)
                self.cell(col_w, row_h, "", border=1, fill=True)
            elif i == highlight_idx:
                self.set_fill_color(*ACCENT_BLUE)
                self.set_text_color(*WHITE)
                self.set_font("Arial", "B", 8.5)
                self.cell(col_w, row_h, str(v), border=1, fill=True, align="C")
                self.set_text_color(*DARK_GRAY)
                self.set_font("Arial", "", 8.5)
                self.set_fill_color(*WHITE)
            else:
                fgt = (highlight_idx >= 0 and i < highlight_idx and v < 50)
                self.set_fill_color(*(255, 235, 235) if (fgt and v < 35) else WHITE)
                self.cell(col_w, row_h, str(v), border=1,
                          fill=(fgt and v < 35), align="C")
        self.ln()

    def metric_box(self, x, y, w, h, title, value, sub, color):
        self.set_fill_color(*color)
        self.rect(x, y, w, h, "F")
        self.set_text_color(*WHITE)
        self.set_font("Arial", "", 8)
        self.set_xy(x + 2, y + 2)
        self.cell(w - 4, 5, title)
        self.set_font("Arial", "B", 16)
        self.set_xy(x + 2, y + 7)
        self.cell(w - 4, 9, value, align="C")
        self.set_font("Arial", "", 7.5)
        self.set_xy(x + 2, y + 17)
        self.cell(w - 4, 4, sub, align="C")
        self.set_text_color(*BLACK)


def build_report():
    pdf = Report()

    # ── SAYFA 1: Kapak ──────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*DARK_BLUE)
    pdf.rect(0, 0, 210, 85, "F")

    pdf.set_text_color(*WHITE)
    pdf.set_font("Arial", "B", 26)
    pdf.set_xy(0, 22)
    pdf.cell(210, 14, "HOPE-CIFAR", align="C")

    pdf.set_font("Arial", "B", 14)
    pdf.set_xy(0, 38)
    pdf.cell(210, 9, "Hierarchical Online Plasticity Engine", align="C")

    pdf.set_font("Arial", "", 11)
    pdf.set_xy(0, 50)
    pdf.cell(210, 7, "CIFAR-100  |  Class-Incremental Continual Learning", align="C")

    pdf.set_font("Arial", "", 9)
    pdf.set_xy(0, 65)
    pdf.cell(210, 6, "Geliştirme & Deney İlerleme Raporu  —  Mayıs 2026", align="C")

    pdf.set_text_color(*BLACK)

    pdf.metric_box(18,  95, 80, 28, "EN İYİ SONUÇ", "56.69%",
                   "Avg Accuracy (Replay + Gaussian)", ACCENT_BLUE)
    pdf.metric_box(112, 95, 80, 28, "EN AZ UNUTMA", "10.30%",
                   "Forgetting (lr=1e-5, 5 epoch)", GREEN)

    pdf.set_y(132)
    pdf.section_title("Proje Özeti")
    pdf.body(
        "HOPE, 100 sınıflı CIFAR-100 üzerinde Class-Incremental Learning (CIL) problemi için "
        "geliştirilmiş bir mimaridir. Model, 10 göreve bölünmüş 100 sınıfı sıralı olarak öğrenir "
        "ve her yeni görev öğrenilirken eski görevlerin unutulmaması hedeflenir.\n\n"
        "Bu rapor; mevcut geliştirmeleri, çalıştırılan deneyleri ve laboratuvar GPU ortamında "
        "beklenen sonuçları özetlemektedir."
    )

    pdf.section_title("Mimari Genel Bakış")
    pdf.body("Görüntüden özelliğe, özellikten sınıfa akış:")
    pdf.ln(1)
    pdf.code(
        "  CIFAR-100 Görüntüsü (32x32 veya 224x224)\n"
        "         |\n"
        "  [Backbone]  ResNet18 (512-dim)  veya  ViT-B/16 (768-dim)\n"
        "         |\n"
        "  [CMS]  4 Kademe: fast(1) -> mid(4) -> slow(32) -> ultra(128)\n"
        "         |\n"
        "  [Classifier]  Linear(512/768 -> 100 sinif)"
    )
    pdf.ln(2)
    pdf.sub_title("CMS (Continual Memory System)")
    pdf.bullet("Hızlı kademe: tek epoch içinde güncellenir, yeni göreve hızla uyum sağlar")
    pdf.bullet("Yavaş kademeler: uzun vadeli bilgiyi korur, görev sınırında sıfırlanmaz")
    pdf.bullet("DeepMomentum optimizer: her kademe için bağımsız öğrenme dinamiği")
    pdf.ln(2)
    pdf.sub_title("Öğretme Sinyali (Teach Signal)")
    pdf.bullet("CE kayıp gradyanının kapalı-form hesabı: backprop olmadan CMS güncellenir")
    pdf.bullet("W snapshot: backward öncesi alınır, matematiksel tutarlılık sağlar")

    # ── SAYFA 2: Geliştirmeler ──────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Yapılan Geliştirmeler")

    pdf.sub_title("1. Hata Düzeltmeleri (6 Kritik Bug Fix)")
    fixes = [
        ("Bug 1", "CalibrationBuffer.add() eksikti — buffer hiç dolmuyordu"),
        ("Bug 2", "CMS update backward'dan önce yapılıyordu — computation graph bozuluyordu"),
        ("Bug 3", "NCM, cms_out yerine backbone_feat kullanmıyordu — CMS katkısı ölçülmüyordu"),
        ("Bug 4", "LayerNorm parametreleri CMS gradyanından hariçti — norm katmanları öğrenemiyordu"),
        ("Bug 5", "Teach signal W snapshot almıyordu — doğru gradient hesaplanamıyordu"),
        ("Bug 6", "num_workers macOS uyumlu hale getirildi (0 olarak sabitlendi)"),
    ]
    for code_lbl, desc in fixes:
        pdf.set_font("Arial", "B", 9.5)
        pdf.set_text_color(*RED_SOFT)
        pdf.cell(16, 6, code_lbl + ":")
        pdf.set_font("Arial", "", 9.5)
        pdf.set_text_color(*DARK_GRAY)
        pdf.multi_cell(0, 6, desc)
        pdf.set_x(pdf.l_margin)
    pdf.set_text_color(*BLACK)

    pdf.ln(2)
    pdf.sub_title("2. Gaussian Classifier Alignment (TUNA'dan Uyarlama — ICCV 2025)")
    pdf.body(
        "TUNA makalesindeki classifier_align() yöntemi ResNet/ViT mimarisine uyarlandı. "
        "Gerçek görüntü saklamak yerine her sınıfı Gaussian dağılımla modeller."
    )
    pdf.bullet("memory/gaussian_buffer.py  —  GaussianBuffer: mean + covariance her sınıf için")
    pdf.bullet("training/classifier_align.py  —  CosFaceLoss + classifier_align() fonksiyonu")
    pdf.bullet("Her görev sonunda TÜM görülen sınıfların istatistikleri yenilenir (backbone drift düzeltmesi)")
    pdf.ln(1)
    pdf.code(
        "  [Görev T bitti]\n"
        "  -> Tüm görülen görevlerin train loader'larından backbone_feat topla\n"
        "  -> Her sınıf için mean + covariance hesapla\n"
        "  -> MultivariateNormal.sample() ile 256 sentetik örnek/sınıf üret\n"
        "  -> Classifier'ı SGD + CosineAnnealingLR ile 30 epoch kalibre et"
    )

    pdf.ln(2)
    pdf.sub_title("3. CosFace Loss")
    pdf.body("Angular penalty loss — cosine benzerliğine margin ekler, sınıflar arası açıklık artar.")
    pdf.code(
        "  m=0.0 (varsayılan): scaled cosine CE  ->  s * cos(W, f)\n"
        "  m>0.0             : doğru sınıfın cosine değerinden m çıkarılır\n"
        "  Varsayılan: s=20.0, m=0.0"
    )

    pdf.ln(2)
    pdf.sub_title("4. Çoklu Backbone Desteği  (--backbone resnet/vit)")
    for name, desc in [
        ("ResNet18",  "32×32 CIFAR girişi, 512-dim, Mac/CPU için optimize, hızlı test"),
        ("ViT-B/16",  "224×224 girişi, 768-dim, ImageNet-21k ön-eğitim, lab GPU için"),
    ]:
        pdf.set_font("Arial", "B", 9.5)
        pdf.set_text_color(*MID_BLUE)
        pdf.cell(22, 6, name + ":")
        pdf.set_font("Arial", "", 9.5)
        pdf.set_text_color(*DARK_GRAY)
        pdf.multi_cell(0, 6, desc)
        pdf.set_x(pdf.l_margin)
    pdf.set_text_color(*BLACK)

    pdf.ln(2)
    pdf.sub_title("5. GPU Odaklı Eğitim  (train_gpu.py)")
    for f in [
        "Mixed Precision AMP (fp16) — yaklaşık 2x hız artışı, daha az VRAM kullanımı",
        "pin_memory=True + num_workers=4 — CPU→GPU veri transfer hızlandırması",
        "torch.backends.cudnn.benchmark — sabit input için cuDNN otomatik kernel seçimi",
        "Gradient accumulation (--accum_steps) — küçük GPU'da büyük batch etkisi",
        "CUDA kontrolü: GPU yoksa çalışmayı reddeder, net uyarı verir",
    ]:
        pdf.bullet(f)

    pdf.ln(2)
    pdf.sub_title("6. Dağıtık Eğitim  (train_distributed.py)")
    pdf.bullet("PyTorch DDP (DistributedDataParallel) ile çok-GPU desteği")
    pdf.bullet("CMS parametreleri: manuel all_reduce ile senkronize (autograd dışında)")
    pdf.bullet("Checkpoint kayıt/yükleme: --resume ile kesilen eğitim devam eder")
    pdf.code("  torchrun --nproc_per_node=4 train_distributed.py --backbone vit ...")

    pdf.ln(2)
    pdf.sub_title("7. Yeni Komut Satırı Bayrakları")
    flags = [
        ("--backbone",       "resnet/vit  — backbone seçimi"),
        ("--gaussian_align", "Gaussian kalibrasyon etkinleştir"),
        ("--cosface",        "CosFace loss kullan"),
        ("--align_epochs",   "Kalibrasyon epoch sayısı (varsayılan: 30)"),
        ("--cosface_scale",  "CosFace ölçek faktörü (varsayılan: 20.0)"),
        ("--accum_steps",    "Gradient accumulation adım sayısı (GPU için)"),
        ("--no_amp",         "Mixed precision kapat (debug için)"),
    ]
    for flag, desc in flags:
        pdf.set_font("CourierN", "B", 9)
        pdf.set_text_color(*ACCENT_BLUE)
        pdf.cell(40, 6, flag)
        pdf.set_font("Arial", "", 9.5)
        pdf.set_text_color(*DARK_GRAY)
        pdf.multi_cell(0, 6, desc)
        pdf.set_x(pdf.l_margin)
    pdf.set_text_color(*BLACK)

    # ── SAYFA 3: Deney Sonuçları ────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Deney Sonuçları — ResNet18, CIFAR-100 (10 Görev × 10 Sınıf)")

    # Özet tablo başlık
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(*MID_BLUE)
    pdf.set_text_color(*WHITE)
    for cw, h in [(72, "Konfigürasyon"), (26, "Avg Acc"), (26, "Forgetting"), (48, "Temel Bulgu")]:
        pdf.cell(cw, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    results = [
        ("Baseline (önceki en iyi)",              "44.50%", "11.56%", "Referans",                    LIGHT_GRAY,      BLACK),
        ("gaussian_align (tek başına)",            "36.24%", "36.36%", "Backbone drift problemi",     (255, 235, 235), RED_SOFT),
        ("gaussian_align + cosface",               "27.90%", "46.48%", "CE/CosFace uyumsuzluğu",      (255, 220, 220), RED_SOFT),
        ("gaussian_align + freeze_backbone",       "22.20%", "10.68%", "ResNet özellikleri yetersiz", (255, 235, 235), ORANGE),
        ("gaussian_align + lr=1e-5 + 5 epoch",     "44.88%", "10.30%", "Baseline eşiti, az unutma",   LIGHT_GRAY,      DARK_GRAY),
        ("gaussian_align + replay  ★",             "56.69%", "16.64%", "EN İYİ SONUÇ (+12.2%)",      (220, 245, 220), GREEN),
    ]
    pdf.set_font("Arial", "", 9)
    for cfg, acc, fgt, note, bg, tc in results:
        is_best = "EN İYİ" in note
        pdf.set_fill_color(*bg)
        pdf.set_text_color(*tc)
        if is_best:
            pdf.set_font("Arial", "B", 9)
        pdf.cell(72, 6.5, cfg,  border=1, fill=True)
        pdf.cell(26, 6.5, acc,  border=1, fill=True, align="C")
        pdf.cell(26, 6.5, fgt,  border=1, fill=True, align="C")
        pdf.cell(48, 6.5, note, border=1, fill=True, align="C")
        if is_best:
            pdf.set_font("Arial", "", 9)
        pdf.ln()
    pdf.set_text_color(*BLACK)
    pdf.ln(3)

    # Accuracy matrisi
    pdf.sub_title("En İyi Konfigürasyon: gaussian_align + replay  (Avg Acc: 56.69%)")
    pdf.body("Görev 9 sonrası tam accuracy matrisi (satır = o görevin ardından, sütun = test görevi):")
    pdf.ln(2)

    CW = 17
    pdf.set_font("Arial", "B", 7.5)
    pdf.set_fill_color(*LIGHT_BLUE)
    pdf.set_text_color(*DARK_BLUE)
    pdf.cell(22, 6, "After / Test", border=1, fill=True, align="C")
    for t in range(10):
        pdf.cell(CW, 6, f"T{t:02d}", border=1, fill=True, align="C")
    pdf.ln()

    matrix = [
        ("T00", [85.3, None, None, None, None, None, None, None, None, None], 0),
        ("T01", [74.6, 75.5, None, None, None, None, None, None, None, None], 1),
        ("T02", [69.1, 69.4, 77.8, None, None, None, None, None, None, None], 2),
        ("T03", [64.7, 62.9, 72.8, 70.2, None, None, None, None, None, None], 3),
        ("T04", [61.4, 59.8, 69.7, 66.3, 72.2, None, None, None, None, None], 4),
        ("T05", [59.3, 56.7, 66.8, 62.7, 62.8, 70.9, None, None, None, None], 5),
        ("T06", [55.3, 53.7, 63.0, 58.5, 59.7, 65.3, 70.8, None, None, None], 6),
        ("T07", [53.9, 51.3, 60.8, 57.3, 57.9, 62.5, 63.6, 63.4, None, None], 7),
        ("T08", [54.5, 50.1, 61.0, 56.2, 56.0, 60.3, 58.9, 60.1, 66.3, None], 8),
        ("T09", [52.9, 49.9, 59.0, 53.0, 55.0, 56.8, 56.9, 55.3, 63.4, 61.6], 9),
    ]
    for label, accs, hi in matrix:
        pdf.draw_accuracy_row(label, accs, hi, col_w=CW)
    pdf.ln(3)

    pdf.sub_title("Temel Bulgular")
    pdf.bullet(
        "Replay + Gaussian alignment kombinasyonu en güçlü sonucu verdi (56.69%). "
        "Replay backbone'u eski sınıflar için kararlı tutuyor; Gaussian ise classifier kaymasını düzeltiyor."
    )
    pdf.bullet(
        "CosFace tek başına zarar verdi: backbone CE ile eğitilip kalibrasyon CosFace ile "
        "yapılınca tutarsızlık oluşuyor. ViT + frozen backbone durumunda daha uyumlu olması bekleniyor."
    )
    pdf.bullet(
        "Frozen backbone ResNet18 için yetersiz: ImageNet özellikleri CIFAR-100 uyarlamasız "
        "zayıf kalıyor (%22.20). ViT-B/16'da tam tersi bekleniyor — ön-eğitim özellikleri çok daha güçlü."
    )
    pdf.bullet(
        "Backbone drift asıl sorun: Gaussian ortalamalar her görev sonunda TÜM görülen "
        "görevler için yenilenmezse eski sınıfların prototipleri geçersiz kalıyor."
    )

    # ── SAYFA 4: Lab GPU Beklentileri ───────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Lab GPU Beklentileri — ViT-B/16 + Dağıtık Eğitim")

    pdf.sub_title("Referans: TUNA (ICCV 2025)")
    pdf.body(
        "TUNA (Frozen ViT-B/16 + task-specific adapters + universal adapter) CIFAR-100'de "
        "92.15% final accuracy rapor etmiştir. Bizim yaklaşımımız adapter mekanizması "
        "içermemekle birlikte Gaussian alignment ve CMS ile rekabetçi sonuçlar hedeflenmektedir."
    )
    pdf.ln(2)

    pdf.sub_title("Neden ViT Daha İyi?")
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(*LIGHT_BLUE)
    pdf.set_text_color(*DARK_BLUE)
    for cw, h in [(52, "Özellik"), (62, "ResNet18"), (62, "ViT-B/16")]:
        pdf.cell(cw, 7, h, border=1, fill=True, align="C")
    pdf.ln()
    comparisons = [
        ("Özellik boyutu",     "512-dim",                  "768-dim"),
        ("Ön-eğitim verisi",   "ImageNet-1K",              "ImageNet-21K (14x büyük)"),
        ("Bağlam",             "Konvolüsyon (yerel)",       "Self-Attention (global)"),
        ("Frozen performans",  "Zayıf (~%22)",              "Güçlü (~%70-80 beklenti)"),
    ]
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(*DARK_GRAY)
    for feat, rn, vit in comparisons:
        pdf.set_fill_color(*LIGHT_GRAY)
        pdf.cell(52, 6, feat, border=1, fill=True)
        pdf.set_fill_color(*WHITE)
        pdf.cell(62, 6, rn, border=1, align="C")
        pdf.set_fill_color(220, 245, 220)
        pdf.cell(62, 6, vit, border=1, fill=True, align="C")
        pdf.ln()
    pdf.set_text_color(*BLACK)
    pdf.ln(3)

    pdf.sub_title("Beklenen Doğruluk Aralıkları")
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(*MID_BLUE)
    pdf.set_text_color(*WHITE)
    for cw, h in [(88, "Konfigürasyon"), (22, "Avg Acc"), (22, "Forgetting"), (40, "Not")]:
        pdf.cell(cw, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    scenarios = [
        ("ViT + freeze + gaussian_align",                "%70–78", "%12–18", "TUNA mantığıyla tutarlı"),
        ("ViT + freeze + gaussian_align + cosface",      "%72–80", "%10–16", "CosFace frozen ViT'e uygun"),
        ("ViT + replay + gaussian_align",                "%78–85", "%10–15", "En güçlü beklenti"),
        ("ViT + DDP (4 GPU) + replay + gaussian  ★",    "%80–87", "%8–13",  "Tam lab konfigürasyonu"),
    ]
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(*DARK_GRAY)
    for cfg, acc, fgt, note in scenarios:
        is_best = "★" in cfg
        pdf.set_fill_color(*(220, 245, 220) if is_best else WHITE)
        pdf.set_font("Arial", "B" if is_best else "", 9)
        pdf.cell(88, 6, cfg,  border=1, fill=is_best)
        pdf.cell(22, 6, acc,  border=1, fill=is_best, align="C")
        pdf.cell(22, 6, fgt,  border=1, fill=is_best, align="C")
        pdf.cell(40, 6, note, border=1, fill=is_best, align="C")
        pdf.ln()
    pdf.set_text_color(*BLACK)
    pdf.ln(3)

    pdf.sub_title("Lab Çalıştırma Komutları")
    pdf.body("Tek GPU (test ve doğrulama):")
    pdf.code(
        "  python3 train_gpu.py --backbone vit --gaussian_align --freeze_backbone\n"
        "  python3 train_gpu.py --backbone vit --gaussian_align --cosface --freeze_backbone\n"
        "  python3 train_gpu.py --backbone vit --gaussian_align --replay --samples_per_class 50"
    )
    pdf.body("4 GPU dağıtık eğitim (tam konfigürasyon):")
    pdf.code(
        "  torchrun --nproc_per_node=4 train_distributed.py \\\n"
        "    --backbone vit --gaussian_align --cosface \\\n"
        "    --replay --samples_per_class 50 \\\n"
        "    --epochs 15 --batch_size 64"
    )
    pdf.body("Küçük GPU (8GB VRAM) için — gradient accumulation:")
    pdf.code(
        "  python3 train_gpu.py --backbone vit --gaussian_align \\\n"
        "    --batch_size 32 --accum_steps 4   # efektif batch = 128"
    )
    pdf.ln(2)

    pdf.sub_title("Hız Tahmini (4× NVIDIA A100 80GB)")
    times = [
        ("ResNet18, 10 epoch/görev, 10 görev — 1 GPU",   "~8 dakika"),
        ("ViT-B/16, 15 epoch/görev, 10 görev — 1 GPU",   "~45 dakika"),
        ("ViT-B/16, 15 epoch/görev, 10 görev — 4 GPU",   "~12 dakika"),
        ("+ Gaussian alignment (30 ep × 10 görev)",       "+5 dakika"),
    ]
    for scen, t in times:
        pdf.set_font("Arial", "", 9.5)
        pdf.set_text_color(*DARK_GRAY)
        pdf.cell(115, 6, scen)
        pdf.set_font("Arial", "B", 9.5)
        pdf.set_text_color(*ACCENT_BLUE)
        pdf.cell(50, 6, t)
        pdf.ln()
    pdf.set_text_color(*BLACK)
    pdf.ln(3)

    pdf.sub_title("Sonraki Adımlar")
    for i, s in enumerate([
        "Lab'da ViT + freeze_backbone + gaussian_align çalıştır — ablasyon için temel al",
        "CosFace margin (m=0.1, 0.2) dene — frozen ViT ile margin anlamlı olabilir",
        "Gaussian n_per_class hiperparametresini karşılaştır: 128 vs 256 vs 512",
        "DDP ile tekrarlanabilirlik testi (birden fazla seed)",
        "TUNA'nın adapter mekanizmasını eklemeyi değerlendir (uzun vadeli hedef)",
    ], 1):
        pdf.set_font("Arial", "B", 9.5)
        pdf.set_text_color(*ACCENT_BLUE)
        pdf.cell(8, 6, f"{i}.")
        pdf.set_font("Arial", "", 9.5)
        pdf.set_text_color(*DARK_GRAY)
        pdf.multi_cell(0, 6, s)
        pdf.set_x(pdf.l_margin)
    pdf.set_text_color(*BLACK)

    # ── SAYFA 5: Dosya Yapısı & Hızlı Başvuru ──────────────────────────────
    pdf.add_page()
    pdf.section_title("Proje Dosya Yapısı")
    pdf.code(
        "hope_cifar/\n"
        "  train.py              Ana eğitim scripti (Mac / CPU / MPS)\n"
        "  train_gpu.py          GPU odaklı eğitim (AMP + pin_memory + cudnn)\n"
        "  train_distributed.py  Dağıtık eğitim (DDP, torchrun)\n"
        "\n"
        "  model/\n"
        "    backbone.py         ResNetBackbone + ViTBackbone\n"
        "    cms.py              4-kademe CMS (fast/mid/slow/ultra)\n"
        "    hope_model.py       Ana model: backbone + CMS + classifier\n"
        "\n"
        "  training/\n"
        "    engine.py           train_one_epoch, evaluate, evaluate_ncm\n"
        "    classifier_align.py CosFaceLoss + classifier_align()\n"
        "\n"
        "  memory/\n"
        "    gaussian_buffer.py  GaussianBuffer (mean + covariance per class)\n"
        "    replay_buffer.py    CalibrationBuffer (görüntü tabanlı replay)\n"
        "\n"
        "  data/\n"
        "    cifar100.py         CIFAR-100 görev yükleme, TRANSFORM, VIT_TRANSFORM\n"
        "\n"
        "  optim/\n"
        "    deep_momentum.py    DeepMomentum: CMS için özel optimizer\n"
        "\n"
        "  utils/\n"
        "    metrics.py          ContinualMetrics: accuracy matrisi + forgetting"
    )

    pdf.ln(3)
    pdf.section_title("Hızlı Başvuru: Komut Örnekleri")
    commands = [
        ("Mac test (varsayılan):",             "python3 train.py"),
        ("Mac + Gaussian:",                    "python3 train.py --gaussian_align"),
        ("Mac + Gaussian + Replay (en iyi):",  "python3 train.py --gaussian_align --replay"),
        ("Mac + Frozen + Gaussian:",           "python3 train.py --gaussian_align --freeze_backbone"),
        ("Lab GPU (önerilen):",                "python3 train_gpu.py --backbone vit --gaussian_align --freeze_backbone"),
        ("Lab GPU + CosFace:",                 "python3 train_gpu.py --backbone vit --gaussian_align --cosface"),
        ("Lab 4-GPU dağıtık:",                "torchrun --nproc_per_node=4 train_distributed.py --backbone vit ..."),
    ]
    for label, cmd in commands:
        pdf.set_font("Arial", "B", 9)
        pdf.set_text_color(*DARK_GRAY)
        pdf.cell(66, 6, label)
        pdf.set_font("CourierN", "", 8.5)
        pdf.set_text_color(*ACCENT_BLUE)
        pdf.multi_cell(0, 6, cmd)
        pdf.set_x(pdf.l_margin)
        pdf.set_text_color(*BLACK)

    pdf.ln(3)
    pdf.section_title("Sonuç")
    pdf.body(
        "HOPE-CIFAR, ResNet18 üzerinde 6 kritik hata düzeltmesi, Gaussian Classifier Alignment "
        "(TUNA'dan uyarlama), CosFace loss, çoklu backbone desteği ve GPU/dağıtık eğitim "
        "altyapısıyla güçlü bir sürekli öğrenme sistemi haline getirilmiştir.\n\n"
        "Mevcut en iyi ResNet18 sonucu: %56.69 avg accuracy / %16.64 forgetting "
        "(gaussian_align + replay ile baseline'ı 12.2 puan geçtik).\n\n"
        "Lab'da ViT-B/16 + frozen backbone + Gaussian alignment kombinasyonuyla "
        "%70–85 aralığında bir sonuç beklenmektedir. Bu sonuç, TUNA'nın %92.15 değerine "
        "yakın olmasa da adapter mekanizması olmaksızın rekabetçi bir performans olacaktır."
    )

    pdf.output(OUT_PATH)
    print(f"PDF oluşturuldu: {OUT_PATH}")


if __name__ == "__main__":
    build_report()
