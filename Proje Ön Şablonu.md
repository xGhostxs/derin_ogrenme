# PROJE ÖN ŞABLONU — v2.0
## Ekip Görev Dağılımı ve Uygulama Planı

| | |
| :--- | :--- |
| **Proje adı** | Bulut Altı Biyo-İndikatör Analizi: EVA-02 Mimarisi Üzerinde IA³ ve AdaLoRA Adaptasyon Metotları ile Fitoplankton Tür Sınıflandırması |
| **Ekip** | Cankat (teknik lider), Salih, Buse, Yusuf |
| **Ders / Danışman** | ................................ / ................................ |
| **Teslim tarihi** | ................................ |
| **Doküman sürümü** | v2.0 — görev yeniden dağıtımı |
| **Son güncelleme** | 18.11.2026 |

---

## 0. BİR BAKIŞTA GÖREV DAĞILIMI

| Üye | Yeni rol | Zorluk | Tahmini emek | Ana çıktı | Sunum |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Cankat** | Model & PEFT Lideri + Teknik Entegratör | ★★★★★ 5/5 | 36–40 saat | `src/models.py`, `configs/*.yaml`, parametre/rank analizi, final kod incelemesi | 3:00 |
| **Salih** | Veri & Eğitim Mühendisi | ★★★★☆ 4/5 | 30–34 saat | `src/data.py`, `src/train.py`, `data/splits.csv`, checkpoint ve loglar | 2:30 |
| **Buse** | Değerlendirme, XAI & İstatistik Uzmanı | ★★★☆☆ 3/5 | 24–28 saat | `src/evaluate.py`, `src/xai.py`, figür seti, McNemar/bootstrap | 2:30 |
| **Yusuf** | Dokümantasyon, Sunum & Deney Takibi | ★★☆☆☆ 2/5 | 18–22 saat | PDF rapor, PPT, README, sonuç tablosu, teslim kontrolü | 1:30 |

> **Zorluk tanımı:** Teknik karmaşıklık + hata ayıklama yükü + bağımlılık sayısı.  
> **Tasarım kararı:** Zorluk sırası bilinçli olarak **Cankat > Salih > Buse > Yusuf** şeklindedir.  
> **Önemli not:** "Kolay" görev, "önemsiz" görev demek değildir. Yusuf'un işi teknik riski düşük ama teslim tarihine bağlı olduğu için **kritik yol** üzerindedir.

### Neden bu dağılım?

- **Cankat — en zor:** PEFT ile timm/EVA-02 uyumunu çözmek, `head` ve hedef modül yönetimi, AdaLoRA'nın rank bütçesi, iki yöntemin ortak arayüzü, herkesin tıkandığı yerde hata ayıklama ve final kod incelemesi. Projenin teknik omurgası.
- **Salih — zor:** Veri kalitesi, sızıntısız split, sınıf dengesizliği ve eğitim döngüsü sonuçların güvenilirliğini belirler. İş hacmi yüksek ve hata toleransı düşük.
- **Buse — orta:** Metrikler standarttır ancak yorumlama, istatistiksel testler ve Grad-CAM hook'ları dikkat ister. Salih'in checkpoint'lerine bağımlıdır.
- **Yusuf — en kolay:** Şablon, kontrol listesi ve küçük bir sonuç birleştirme script'i ağırlıklıdır. Model/eğitim hatası ayıklamaz; asıl işi düzen, tutarlılık ve teslim takibidir.

### Eski şablon → yeni ekip eşleşmesi

| Eski rol | Yeni üye | Yapılan değişiklik |
| :--- | :--- | :--- |
| Görevli 1 — Model & PEFT | **Cankat** | + teknik entegrasyon, rank analizi, ablasyon, kod inceleme |
| Görevli 2 — Veri & Eğitim | **Salih** | + veri sızıntısı kontrolü, deney logları, checkpoint yönetimi |
| Görevli 3 — Analiz & CAM | **Buse** | + McNemar/bootstrap, hata analizi, figür standardı |
| Görevli 4 — Rapor & Sunum | **Yusuf** | + notebook koşumu, sonuç tablosu, teslim kontrolü |

---

## 1. PROJE ÖZETİ

| Parametre | İçerik |
| :--- | :--- |
| **Foundational model** | **EVA-02-Base** (ViT-B/14, ~86M parametre). ImageNet-22K üzerinde MIM ile ön eğitim, ardından CLIP tabanlı öğretmen modelinden bilgi damıtma (knowledge distillation) ile hibrit eğitim. |
| **Problem** | WHOI-Plankton (Imaging FlowCytobot) mikroskopi görüntülerinden **12 fitoplankton türünün** çok sınıflı sınıflandırılması. |
| **Neden özgün?** | Türler arası morfolojik farklar ince tanelidir (fine-grained); EVA-02'nin MIM ile kazandığı detay odaklı temsillerin PEFT ile uyarlanması bilimsel olarak anlamlıdır. Fitoplanktonlar su kalitesi ve ötrofikasyonun biyo-indikatörüdür. |
| **Yöntem 1** | **IA³:** Ağırlıkları dondurup attention ve FFN aktivasyonlarını öğrenilebilir çarpımsal vektörlerle ölçekler. |
| **Yöntem 2** | **AdaLoRA:** LoRA rank'larını katman önemine göre SVD tabanlı budama ile eğitim sırasında dinamik dağıtır. |
| **Birincil metrik** | **Macro-F1** (sınıf dengesizliği nedeniyle), yanında balanced accuracy ve Cohen's kappa. |
| **Teslimler** | PDF rapor, 10 dakikalık sunum, temiz çalışan Jupyter Notebook, kaynak kod, figür/metrik seti. |
| **Başarı kriteri** | İki yöntem de uçtan uca eğitilmiş, aynı split/seed ile adil karşılaştırılmış, istatistiksel testle desteklenmiş ve yeniden üretilebilir olmalı. Sayısal eşik, veri EDA'sı sonrası danışmanla netleştirilecek. |

---

## 2. ORTAK KURALLAR (HERKES İÇİN BAĞLAYICI)

1. **Test seti kutsaldır.** Hafta 3'te metrikler dondurulana kadar test setinde ayar yapılmaz. Test yalnızca final değerlendirmede kullanılır.
2. **Split dosyası dondurulur.** `data/splits.csv` commit'lenir; sonradan değiştirilirse tüm deneyler geçersiz sayılır.
3. **Her koşu loglanır.** Config + seed + git commit + epoch metrikleri + süre + VRAM. Logsuz sonuç rapora girmez.
4. **Tek doğruluk kaynağı:** Sayılar `results/metrics/*.json` dosyalarından gelir. Rapora/sunuma elle sayı yazılmaz.
5. **Notebook çakışması yasak.** Herkes kendi `.py` modülünde çalışır; final notebook'u Yusuf birleştirir. Aynı anda aynı notebook düzenlenmez.
6. **Branch disiplini:** `cankat/model`, `salih/data-train`, `buse/eval-xai`, `yusuf/docs`. `main`'e merge yalnızca Cankat'ın incelemesiyle yapılır.
7. **Herkes kendi kodunu savunur.** Sunumda her üye kendi bölümünün kodunu ve teorisini açıklayabilmelidir.
8. **Blocker kuralı:** 2 saatten uzun takılma → Cankat'a haber ver; 1 günden uzun → ekip toplantısı. Sessiz beklemek yok.
9. **Yeniden üretilebilirlik:** Kurulum adımları `README.md`'de; sürümler `requirements.lock` ile sabitlenir.
10. **Etik:** WHOI-Plankton lisansı ve atıf kuralları rapora eklenir; veri dışarıya sızdırılmaz.

---

## 3. GÖREV PAKETLERİ (DETAYLI)

### 3.1 Cankat — Model & PEFT Lideri + Teknik Entegratör (Zorluk 5/5)

**Misyon:** EVA-02-Base'i IA³ ve AdaLoRA ile sorunsuz çalıştırmak, parametre verimliliğini ölçmek ve tüm kodun teknik entegrasyonunu yönetmek. **Projenin en zor görevi.**

**Sorumluluklar**
1. **Ortam ve model doğrulama:** `timm`, `peft`, `torch` sürümlerini sabitle; `scripts/inspect_modules.py` ile EVA-02 modül ağacını çıkar; `attn.q_proj`, `attn.k_proj`, `attn.v_proj`, `attn.proj`, `mlp.fc1_g`, `mlp.fc1_x`, `mlp.fc2`, `head` isimlerini doğrula.
2. **Model fabrikası:** `build_model(method, num_classes, total_step)` yaz. IA³ ve AdaLoRA config'lerini kur; `modules_to_save=["head"]` ile sınıflandırıcı başın eğitilebilir kalmasını sağla.
3. **Parametre verimliliği:** Toplam/eğitilebilir parametre ve oran tablosu; IA³ vektör normları; AdaLoRA'da **efektif rank** (sıfırlanmayan `lora_E` girdileri) tarihçesi ve rank dağılım grafiği. `ranknum` PEFT'te `init_r` değerinde kalır; gerçek bütçe `lora_E` maskesindedir. Maskeler arasında sıfırlanan tripletler yeniden aktifleşebilir; bu yüzden eğri monoton olmayabilir, nihai bütçe `tfinal` sonrasında sabitlenir.
4. **AdaLoRA entegrasyonu:** `total_step = epochs × steps_per_epoch` hesabı; `tinit/tfinal/deltaT` ayarı; `update_and_allocate(step)` çağrısını Salih'in döngüsüne yerleştirme; 1 epoch smoke test.
5. **Kaydetme/yükleme:** Adapter + `head` ağırlıklarını kaydet; `PeftModel.from_pretrained` ile yükle; logit round-trip testi (tolerans `1e-5`).
6. **Yeniden üretilebilirlik:** Seed, config, git commit ve run adı standardı; `configs/ia3.yaml`, `configs/adalora.yaml`.
7. **Ablasyon (stretch):** IA³ hedef seti (yalnız attention vs attention+MLP) veya AdaLoRA `target_r=4/8`. Süre yetmezse raporlanacak "gelecek çalışma" olur.
8. **Kod inceleme ve merge:** Salih/Buse/Yusuf branch'lerini review et; final notebook'un teknik bölümlerini onayla; takılan üyeye aynı gün destek ol.
9. **Sunum:** EVA-02, IA³ ve AdaLoRA slaytları; Q&A'de mimari/PEFT soruları.

**Çıktılar**
- `src/models.py`
- `configs/ia3.yaml`, `configs/adalora.yaml`
- `scripts/inspect_modules.py`
- `results/tables/param_efficiency.csv`
- `results/tables/ia3_vector_norms.csv`
- `results/figures/adalora_rank_evolution.png`
- Adapter checkpoint drive linki + yükleme talimatı
- Kod inceleme notları

**Bağımlılıklar / el değiştirme**
- Salih → Cankat: gerçek `steps_per_epoch`, OOM geri bildirimi, eğitim hataları.
- Cankat → Salih: stabil `build_model`, config, optimizer parametre grupları.
- Cankat → Buse: checkpoint yükleme talimatı, Grad-CAM hedef katmanı (`blocks[-1].norm1`).
- Cankat → Yusuf: teknik metinler, parametre tablosu, figür altyazıları.
- Cankat → tüm ekip: kod inceleme SLA'sı (aynı gün).

**Kabul kriteri (Definition of Done)**
- [ ] İki yöntem de CPU ve GPU'da kuruluyor; forward/backward sonlu loss üretiyor.
- [ ] `print_trainable_parameters()` çıktıları kayıtlı; ölçülen değerler: IA³ ~%0.10, AdaLoRA ~%2.40 (tüm attention + SwiGLU hedefleri, `init_r=12`/`target_r=8`). Kesin değerler ve hedef seti raporlanacak.
- [ ] `head` eğitilebilir; adapter save/load sonrası logitler `allclose`.
- [ ] AdaLoRA rank dağılımı adımlar ilerledikçe değişiyor ve grafikleniyor.
- [ ] `total_step` Salih'in döngüsüyle birebir uyumlu; 1 epoch dry run başarılı.
- [ ] Hafta 2'den sonra arayüz bozulmuyor; değişiklikler ekipçe duyuruluyor.
- [ ] Final notebook'un teknik bölümleri temiz koşuda hatasız.

**Sözlü savunma için hazırlan**
- SubLN nedir, EVA-02'de neden kullanılır?
- IA³'te `l_k`, `l_v`, `l_ffn` neyi ölçekler? Neden çarpımsal?
- AdaLoRA'da SVD tabanlı rank bütçesi nasıl çalışır? `tinit`, `tfinal`, `deltaT` ne yapar?
- timm EVA-02'de `q_proj`, `k_proj`, `v_proj` ayrı; IA³ bu üçüne ayrı ayrı mı yoksa aynı vektörle mi uygulanıyor; sınırlılığı ne?
- `modules_to_save=["head"]` olmasa ne olur?
- Hedef modüller neden `attn.q_proj`, `attn.k_proj`, `attn.v_proj`, `attn.proj`, `mlp.fc1_g`, `mlp.fc1_x`, `mlp.fc2`?
- IA³ ve AdaLoRA parametre bütçeleri eşit değilse karşılaştırma nasıl adil yapıldı?

---

### 3.2 Salih — Veri & Eğitim Mühendisi (Zorluk 4/5)

**Misyon:** Veri setini temiz, dengeli ve sızıntısız hale getirmek; IA³ ve AdaLoRA için aynı koşullarda çalışan güvenilir bir eğitim döngüsü kurmak.

**Sorumluluklar**
1. **Veri indirme ve doğrulama:** WHOI-Plankton kaynağını, lisansı ve dosya formatını doğrula; `data/manifest.csv` üret; 12 sınıfı seç; bozuk ve tekrar eden görüntüleri ayıkla.
2. **EDA:** Sınıf dağılımı, görüntü boyutu/kanal bilgisi, parlaklık/kontrast, kısa kuyruk sınıflar ve olası etiket gürültüsü. Sonuçları `results/figures/eda_*.png` olarak kaydet.
3. **Split:** Stratified %70/%15/%15, `random_state=42`. Metadata varsa aynı örnek/cruise'a ait ROI'lerin farklı split'e düşmemesi için group split uygula. `data/splits.csv` dosyasını dondur.
4. **Dataset ve transform:** Gri görüntüyü 3 kanala çoğalt; eğitimde morfolojiyi bozmayan augmentation (flip, 180° rotation, hafif crop/color jitter); validasyonda tam kare resize; ImageNet normalizasyonu.
5. **Sınıf dengesizliği:** `WeightedRandomSampler` **veya** sınıf ağırlıklı CE seç; ikisini birden kullanma (çift dengeleme). Seçimi ve dağılımı raporla.
6. **Eğitim döngüsü:** AdamW, cosine + %5 warmup, AMP, gradient clipping `1.0`, `CrossEntropyLoss(label_smoothing=0.1)`, val macro-F1'e göre early stopping, best/last checkpoint, CSV+JSON log, resume desteği.
7. **Koşular:** IA³ ve AdaLoRA'yı Cankat'ın fabrikasıyla eğit; aynı split/seed/augmentation; config + commit logla; test tahminlerini kaydet.
8. **Teslim:** Buse'ye checkpoint + split + test tahminleri; Yusuf'a loglar ve run tablosu; README'ye çalıştırma komutları.
9. **Yeniden üretilebilirlik:** İkinci bir kısa koşuyla aynı seed'in aynı sonucu verdiğini doğrula.

**Çıktılar**
- `src/data.py`, `src/train.py`
- `data/manifest.csv`, `data/splits.csv`
- `results/logs/{run}.csv`, `results/logs/{run}.json`
- `results/checkpoints/` (büyükse Drive linki)
- `results/predictions/test_{method}.csv`
- `results/figures/eda_class_distribution.png`
- `notebooks/01_data_eda.ipynb` (opsiyonel)

**Bağımlılıklar / el değiştirme**
- Cankat → Salih: `build_model`, config, `total_step` hesabı.
- Salih → Cankat: gerçek adım sayısı, OOM/hata geri bildirimi.
- Salih → Buse: checkpoint, `splits.csv`, test tahminleri (Hafta 3 başı).
- Salih → Yusuf: eğitim logları, süre/VRAM değerleri, veri bölümü metni.

**Kabul kriteri (Definition of Done)**
- [ ] Tüm görüntüler yükleniyor; bozuk dosya ve duplicate kontrolü tamam.
- [ ] Split'ler ayrık; sınıf sayıları tablosu ve test seti dokunulmamış.
- [ ] İki yöntem de 1 epoch uçtan uca eğitiliyor; AMP çalışıyor.
- [ ] En iyi checkpoint val macro-F1'e göre seçiliyor; resume testi geçiyor.
- [ ] Loglarda config, seed, commit, epoch metrikleri, süre ve VRAM var.
- [ ] Buse, checkpoint + split ile metrikleri birebir yeniden üretebiliyor.
- [ ] README komutları temiz ortamda çalışıyor.

**Sözlü savunma için hazırlan**
- Mikroskopi verisinde hangi augmentation'lar neden seçildi? Güçlü renk bozulması neden yok?
- AdamW vs SGD; learning rate, warmup ve cosine stratejisi.
- Sınıf dengesizliğinde sampler mı class weight mi; neden ikisi birden değil?
- Overfitting/underfitting nasıl teşhis edildi; early stopping nasıl çalışıyor?
- AMP, gradient accumulation ve OOM yönetimi.
- Veri sızıntısı nasıl önlendi (duplicate hash, group split, test freeze)?

---

### 3.3 Buse — Değerlendirme, XAI & İstatistik Uzmanı (Zorluk 3/5)

**Misyon:** İki yöntemi aynı test setinde adil biçimde karşılaştırmak; metrikleri, istatistiksel testleri ve Grad-CAM analizini üretmek.

**Sorumluluklar**
1. **Değerlendirme sözleşmesi:** Checkpoint + `splits.csv` ile aynı test setini yükle; testte hiçbir ayar yapma; tahminleri dondur.
2. **Metrikler:** Accuracy, macro-F1, balanced accuracy, Cohen's kappa, sınıf bazlı precision/recall/F1 ve karmaşıklık matrisi. JSON + CSV olarak kaydet.
3. **Eğriler ve karşılaştırma:** Train/val loss, val macro-F1, eğitim süresi, peak VRAM ve parametre verimliliği grafikleri. Renk körü dostu palet, 300 dpi.
4. **İstatistiksel testler:** Aynı test örnekleri üzerinde McNemar testi (paired); macro-F1 ve fark için 1000 tekrarlı bootstrap %95 güven aralığı; p-değeri ve etki büyüklüğü.
5. **Grad-CAM:** Son blok `blocks[-1].norm1` hedefiyle doğru ve yanlış örnekler için ısı haritaları; overlay; morfolojik yorum; artifact/arka plan bağımlılığı kontrolü.
6. **Hata analizi:** En sık karışan sınıf çiftleri, sınıf bazlı başarısızlıklar, yanlış sınıflandırılan görüntü galerisi, nadir sınıf/ görüntü kalitesi ilişkisi.
7. **Teslim:** Figürleri ve metrik JSON'larını Yusuf'a ver; sonuç/tartışma bölümünü yaz; sayıların raporla uyumunu kontrol et.
8. **Dürüstlük:** Kötü sonuçları gizleme; tüm sınıfları ve başarısız örnekleri göster; cherry-picking yapma.

**Çıktılar**
- `src/evaluate.py`, `src/xai.py`
- `results/metrics/{method}_test.json`
- `results/tables/classification_report_{method}.csv`
- `results/tables/statistical_tests.csv`
- `results/figures/curves_{method}.png`, `confusion_{method}.png`, `gradcam_{method}.png`, `comparison_*.png`
- Rapor "Sonuçlar ve Tartışma" taslağı

**Bağımlılıklar / el değiştirme**
- Salih → Buse: checkpoint, split, test tahminleri.
- Cankat → Buse: checkpoint yükleme kodu, Grad-CAM hedef katmanı.
- Buse → Yusuf: final figürler, metrik JSON'ları, istatistik tablosu.
- Buse → Cankat: sonuçların teknik yorumu ve Q&A desteği.

**Kabul kriteri (Definition of Done)**
- [ ] Metrikler checkpoint + split'ten birebir yeniden üretiliyor.
- [ ] Rapordaki/sunumdaki tüm sayılar JSON ile eşleşiyor.
- [ ] Karmaşıklık matrisleri satır-normalize + örnek sayıları görünür.
- [ ] McNemar p-değeri ve bootstrap güven aralığı raporlanmış.
- [ ] Her yöntem için en az 3 doğru + 3 yanlış Grad-CAM örneği var.
- [ ] Figürler 300 dpi, okunabilir font, renk körü dostu.
- [ ] Hata analizi morfolojik yorum ve sınırlılık içeriyor.
- [ ] Test setinde tuning yapılmamış; tahminler dondurulmuş.

**Sözlü savunma için hazırlan**
- Neden birincil metrik macro-F1? Kappa ve balanced accuracy ne ekliyor?
- McNemar testinin sıfır hipotezi ne; neden paired tasarım?
- Bootstrap güven aralığı nasıl yorumlanır; çakışan aralıklar ne anlama gelir?
- Grad-CAM hook mekanizması; neden son blok; nedensellik iddiası neden yok?
- En sık karışan sınıf çiftlerinin morfolojik açıklaması.
- Veri seti kaynaklı bias'lar sonuçları nasıl etkileyebilir?

---

### 3.4 Yusuf — Dokümantasyon, Sunum & Deney Takibi (Zorluk 2/5)

**Misyon:** Projenin tüm çıktılarını tek, tutarlı ve teslim edilebilir bir pakete dönüştürmek. **Teknik riski en düşük görev.**

**Sorumluluklar**
1. **Repo ve dokümantasyon:** `README.md`, klasör düzeni, `requirements.txt` ve `.gitignore`; WHOI-Plankton atıf/lisans bilgisi.
2. **Notebook koşumu:** Final notebook'u temiz ortamda baştan sona çalıştır; çıktıları kaydet; elle sayı yazma; kurulum adımlarını belgele.
3. **Sonuç tablosu:** `scripts/collect_results.py` ile `results/metrics/*.json` dosyalarından CSV/Markdown tablo üret; Buse'nin figürleriyle çapraz kontrol et; uyuşmazlığı bildir.
4. **Sunum:** 10 slaytlık şablon; figürleri yerleştir; alt text ekle; 10:00 süre planı; provaları koordine et.
5. **PDF rapor:** Bölümleri birleştir; figür/tablo numaraları, altyazılar, kaynakça, sayfa numaraları; rubrik ve sayfa limiti kontrolü; yazım denetimi.
6. **Teslim:** `Ad_Soyad_proje.pdf` adlandırması, PPTX ve notebook; yükleme; yedek link; kontrol listesi.
7. **Sunum açılış/kapanış:** Motivasyon ve sonuç/sınırlılıklar slaytları; Q&A lojistiği.
8. **Hafif kod:** `collect_results.py` ve opsiyonel `check_submission.py`. Model/eğitim kodu yazmaz.

**Çıktılar**
- `README.md`, `requirements.txt`, `.gitignore`
- `notebooks/final_project.ipynb` (temiz koşu)
- `results/tables/final_results.md`
- `report/*.pdf`, `slides/proje_sunum.pptx`
- `submission_checklist.md`

**Bağımlılıklar / el değiştirme**
- Cankat → Yusuf: teknik metinler, parametre tablosu, figür altyazıları.
- Salih → Yusuf: eğitim logları, veri bölümü metni, çalıştırma komutları.
- Buse → Yusuf: final figürler, metrik JSON'ları, sonuç metni.
- Yusuf → tüm ekip: şablon, taslak, deadline hatırlatmaları, teslim kontrolü.
- Yusuf → Cankat: final PDF/PPT teknik onayı.

**Kabul kriteri (Definition of Done)**
- [ ] Notebook temiz ortamda baştan sona hatasız çalışıyor; çıktılar kayıtlı.
- [ ] Rapordaki/sunumdaki tüm sayılar JSON ile eşleşiyor; yetim figür yok.
- [ ] PDF format/sayfa limiti/isim kuralına uygun; kaynakça tam.
- [ ] PPT 10 slayt, 10:00 akışına uygun; tüm konuşmacılar prova yapmış.
- [ ] Teslim kontrol listesi tamam; yedek link hazır.
- [ ] Her üyenin bölüm metni alınmış ve düzenlenmiş.
- [ ] Q&A lojistiği hazır (soru dağılımı, sıra).

**Sözlü savunma için hazırlan**
- Projenin çevresel/akademik motivasyonu nedir?
- Sonucu tek cümleyle özetle: hangi yöntem önde, ne kadar?
- Sınırlılıklar ve gelecek çalışmalar neler?
- Daha fazla veri/compute olsa ne yapardın?
- Ekip içi görev dağılımı ve kendi katkın neydi?
- Neden EVA-02 ve PEFT? (temel düzey; Cankat destek verir)

---

## 4. ZAMAN PLANI (4 HAFTA)

| Hafta | Cankat | Salih | Buse | Yusuf | Kapı (Gate) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hafta 1** | Ortam, model smoke test, modül inceleme, config'ler | Veri indirme, EDA, split dondurma | Metrik/eval iskeleti, test sözleşmesi | Repo/docs iskeleti, run log şablonu | **G1:** Dummy batch ile iki yöntem forward/backward; split donduruldu |
| **Hafta 2** | IA³ config final, AdaLoRA scheduler testi, parametre tablosu | Eğitim döngüsü, IA³ tam koşu, AdaLoRA dry run | 1-epoch checkpoint üzerinde eval, figür şablonları | README/requirements, slayt taslağı, `collect_results.py` | **G2:** İki yöntem 1 epoch, loglar tam |
| **Hafta 3** | AdaLoRA tam koşu desteği, rank grafiği, ablasyon (ops.), kod inceleme | Final koşular, checkpoint, tahminler, loglar | Metrikler, eğriler, karmaşıklık, McNemar/bootstrap, Grad-CAM | Sonuç tablosu, rapor taslağı, slayt taslağı | **G3:** Test metrikleri donduruldu; figürler v1 |
| **Hafta 4** | Teknik inceleme, notebook entegrasyonu, Q&A hazırlığı | Yeniden üretilebilirlik kontrolü, README komutları, ekler | Final figürler, sonuç metni, istatistik incelemesi | PDF/PPT final, prova, teslim | **G4:** 10:00 prova + PDF/PPT/notebook kontrolü |

**Gate tanımları**
- **G1 — Teknik fizibilite:** Model kuruluyor, iki PEFT yöntemi de çalışıyor, split donduruldu.
- **G2 — Eğitim hattı:** İki yöntem 1 epoch eğitiliyor; log/checkpoint düzeni oturdu.
- **G3 — Sonuç dondurma:** Test metrikleri ve figürler kilitlendi; artık testte değişiklik yok.
- **G4 — Teslim hazırlığı:** Rapor, sunum, notebook ve yedekler tamam.

---

## 5. BAĞIMLILIK VE TESLİM MATRİSİ

| Kimden → Kime | Ne | Ne zaman |
| :--- | :--- | :--- |
| Cankat → Salih | `build_model()`, config'ler, hedef modül listesi, `total_step` formülü | Hafta 1 sonu |
| Salih → Cankat | Gerçek `steps_per_epoch`, OOM/hata geri bildirimi | Hafta 2 |
| Salih → Buse | Checkpoint, `splits.csv`, test tahminleri | Hafta 3 başı |
| Cankat → Buse | Checkpoint yükleme kodu, Grad-CAM hedef katmanı | Hafta 3 başı |
| Buse → Yusuf | Metrik JSON'ları, final figürler, istatistik tablosu | Hafta 3 sonu |
| Yusuf → Tüm ekip | Rapor/sunum taslağı, deadline hatırlatması | Hafta 4 başı |
| Tüm ekip → Yusuf | Kendi bölüm metni + altyazı + kaynak | Hafta 4 başı |
| Cankat → Tüm ekip | Kod incelemesi ve final teknik onay | Hafta 4 |

### Akış şeması

```text
  Cankat (model/PEFT) ──► Salih (veri + eğitim) ──► Buse (metrik + XAI) ──► Yusuf (rapor + sunum)
        ▲                        │                          │                        │
        └────── geri bildirim ───┴────── metrik/figür ───────┴───── final kontrol ───┘
```

> **Kritik yol:** Cankat'ın model fabrikası → Salih'in eğitimi → Buse'nin metrikleri → Yusuf'un teslimi. Bu zincirdeki her gecikme doğrudan teslim tarihini etkiler.

---

## 6. KOD İSKELETİ VE ARAYÜZ SÖZLEŞMESİ

### 6.1 Klasör yapısı

```text
proje/
├─ configs/
│  ├─ ia3.yaml
│  └─ adalora.yaml
├─ data/
│  ├─ raw/                  # git'e eklenmez
│  ├─ manifest.csv
│  └─ splits.csv
├─ notebooks/
│  ├─ 01_data_eda.ipynb
│  └─ final_project.ipynb
├─ src/
│  ├─ models.py             # Cankat
│  ├─ data.py               # Salih
│  ├─ train.py              # Salih
│  ├─ evaluate.py           # Buse
│  └─ xai.py                # Buse
├─ scripts/
│  ├─ inspect_modules.py    # Cankat
│  └─ collect_results.py    # Yusuf
├─ results/
│  ├─ logs/
│  ├─ checkpoints/          # büyükse Drive linki
│  ├─ predictions/
│  ├─ metrics/
│  ├─ tables/
│  └─ figures/
├─ report/
├─ slides/
├─ README.md
├─ requirements.txt
└─ .gitignore
```

### 6.2 Ortak arayüz (değişmez sözleşme)

```python
# src/models.py  — Cankat
def build_model(method: str, num_classes: int = 12, total_step: int | None = None): ...

# src/data.py   — Salih
def build_dataloaders(cfg: dict): ...

# src/train.py  — Salih
def train(cfg: dict) -> dict: ...

# src/evaluate.py — Buse
def evaluate(model, loader, device) -> tuple[dict, list[int], list[int]]: ...
```

> Bu dört imza Hafta 1'de dondurulur. Değişiklik gerekiyorsa Cankat onayı ve ekip duyurusu şarttır.

### 6.3 `models.py` — Cankat

```python
# src/models.py
import timm
from peft import AdaLoraConfig, IA3Config, get_peft_model

# timm 1.x: 224 px EVA-02-Base etiketi; "_ft_in1k" yalnızca 448 px varyantında var.
MODEL_NAME = "eva02_base_patch14_224.mim_in22k"
# timm EVA-02 ayrı q/k/v projeksiyonları ve SwiGLU MLP kullanır.
TARGET_MODULES = [
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj",
    "mlp.fc1_g", "mlp.fc1_x", "mlp.fc2",
]
FEEDFORWARD_MODULES = ["mlp.fc1_g", "mlp.fc1_x", "mlp.fc2"]


def build_model(method="ia3", num_classes=12, total_step=None):
    # timm EVA-02'de sınıflandırıcı başlığın adı: head
    base = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)

    if method == "ia3":
        cfg = IA3Config(
            # timm için task_type=None: generic PeftModel kullanılır.
            target_modules=TARGET_MODULES,
            feedforward_modules=FEEDFORWARD_MODULES,
            modules_to_save=["head"],
        )
    elif method == "adalora":
        if total_step is None:
            raise ValueError("AdaLoRA için total_step zorunlu.")
        # PEFT'te tinit = başlangıç warm-up, tfinal = SONDAKİ fine-tune adımı.
        # Bütçe dağıtımı [tinit, total_step - tfinal] aralığında yapılır.
        tinit = max(2, int(0.10 * total_step))
        tfinal = max(2, int(0.20 * total_step))
        cfg = AdaLoraConfig(
            # timm için task_type=None: generic PeftModel kullanılır.
            target_modules=TARGET_MODULES,
            init_r=12,
            target_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            total_step=total_step,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=10,
            modules_to_save=["head"],
        )
    else:
        raise ValueError(f"Bilinmeyen yöntem: {method}")

    model = get_peft_model(base, cfg)
    model.print_trainable_parameters()
    return model
```

### 6.4 Hedef modül doğrulama — Cankat

```python
# scripts/inspect_modules.py
import timm
from src.models import MODEL_NAME

model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=12)
for name, module in model.named_modules():
    if name.endswith(("attn.q_proj", "attn.k_proj", "attn.v_proj",
                      "attn.proj", "mlp.fc1_g", "mlp.fc1_x", "mlp.fc2", "head")):
        print(f"{name:55s} {type(module).__name__:12s} "
              f"{str(getattr(module, 'in_features', '-')):>5} -> "
              f"{getattr(module, 'out_features', '-')}")
```

> **Kritik teknik not:** `head` modülü `modules_to_save` listesine eklenmezse PEFT onu dondurur ve yeni başlatılan 12 sınıflı sınıflandırıcı öğrenemez. timm 1.x EVA-02'de attention **ayrı `q_proj`, `k_proj`, `v_proj`** modülleriyle, MLP ise **SwiGLU (`fc1_g`, `fc1_x`, `fc2`)** ile kurulur; `attn.qkv` veya `mlp.fc1` yoktur. IA³ bu ayrık projeksiyonlara ayrı vektörler uygular; bu, K/V ayrımını korur.

### 6.5 Eğitim döngüsü iskeleti — Salih

```python
# src/train.py (iskelet)
from torch.cuda.amp import GradScaler, autocast

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=0.01)
# %5 warmup için: LinearLR + CosineAnnealingLR -> SequentialLR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler(enabled=(device.type == "cuda"))
global_step = 0

for epoch in range(cfg["epochs"]):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # AdaLoRA rank bütçesi bu çağrı olmadan güncellenmez.
        if cfg["method"] == "adalora":
            model.base_model.update_and_allocate(global_step)
        global_step += 1

    val_metrics = evaluate(model, val_loader, device)
    # early stopping + best checkpoint burada
```

> `total_steps = epochs × len(train_loader)`. `total_step` config'i bu değerle birebir aynı olmalı; yanlışsa AdaLoRA'nın rank bütçesi bozulur. `scheduler.step()` her batch'te çağrılıyorsa `T_max` da toplam adım olmalı.

### 6.6 Veri hattı — Salih

```python
# src/data.py (iskelet)
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_tf = T.Compose([
    T.Grayscale(num_output_channels=3),          # WHOI görüntüleri gri; ImageNet modeli 3 kanal ister
    T.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(180),                        # mikroskop yönü keyfi
    T.ColorJitter(brightness=0.15, contrast=0.15),  # morfolojiyi bozmayacak kadar hafif
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((224, 224)),                         # merkez kırpma yerine tam kare
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Sınıf dengesizliği: sampler VEYA class weight, ikisi birden değil.
class_counts = np.bincount(train_labels, minlength=cfg["num_classes"])
sample_weights = 1.0 / class_counts[train_labels]
sampler = torch.utils.data.WeightedRandomSampler(
    sample_weights, num_samples=len(sample_weights), replacement=True
)
```

### 6.7 Değerlendirme ve istatistik — Buse

```python
# src/evaluate.py (iskelet)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score)
from statsmodels.stats.contingency_tables import mcnemar

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

# Aynı test örnekleri: b = IA3 doğru & AdaLoRA yanlış, c = tersi
table = [[both_correct, b], [c, both_wrong]]
result = mcnemar(table, exact=False, correction=True)
print("McNemar p =", result.pvalue)

# Bootstrap %95 GA (macro-F1 farkı için)
# 1000 kez test setinden yerine koyarak örnekle, delta_f1 dağılımının
# 2.5 ve 97.5 yüzdeliklerini raporla.
```

### 6.8 Grad-CAM — Buse

```python
# src/xai.py (iskelet)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

model.eval()
target_layers = [model.base_model.model.blocks[-1].norm1]
cam = GradCAM(model=model, target_layers=target_layers)

grayscale_cam = cam(
    input_tensor=x,                                  # [1, 3, 224, 224]
    targets=[ClassifierOutputTarget(int(y))],
)[0]
# show_cam_on_image ile orijinal görüntünün üzerine bindir.
```

> Hook hedefini `print(model)` ile doğrula; PEFT sarmalayıcısı nedeniyle yol `model.base_model.model...` şeklindedir. Grad-CAM nedensellik kanıtı değildir; "modelin baktığı bölge" yorumuyla sınırlı tutulmalıdır.

### 6.9 Sonuç birleştirme — Yusuf

```python
# scripts/collect_results.py
import json
from pathlib import Path
import pandas as pd

rows = []
for path in sorted(Path("results/metrics").glob("*_test.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"method": path.stem.replace("_test", ""), **data})

df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
df.to_csv("results/tables/final_results.csv", index=False)
Path("results/tables/final_results.md").write_text(
    df.to_markdown(index=False), encoding="utf-8"
)
print(df.to_string(index=False))
```

---

## 7. DENEYSEL PROTOKOL (ADİL KARŞILAŞTIRMA)

| Kural | Uygulama |
| :--- | :--- |
| **Aynı split** | `data/splits.csv`; hash'i commit mesajına yazılır. |
| **Aynı seed** | 42 (bütçe varsa 3 seed: 42/43/44; ortalama ± std). |
| **Aynı epoch** | 20 üst sınır; early stopping patience 4. |
| **Aynı batch** | 32; OOM'da 16 + gradient accumulation 2. |
| **Aynı augmentation** | Tüm yöntemlerde birebir aynı train/val transform. |
| **Optimizer** | AdamW; IA³ lr `1e-3`, AdaLoRA lr `5e-4`; wd `0.01`. |
| **Scheduler** | Cosine + %5 warmup. |
| **Loss** | `CrossEntropyLoss(label_smoothing=0.1)`. |
| **Model seçimi** | Val macro-F1; test yalnızca finalde. |
| **Raporlama** | Trainable param, %, süre, peak VRAM, val/test macro-F1, balanced acc, kappa, McNemar p. |
| **Yasak** | Test setinde hiperparametre ayarı, sınıf birleştirme (sonradan), cherry-picking. |

### Sonuç tablosu şablonu

| Yöntem | Trainable param | % | Eğitim süresi | Peak VRAM | Val macro-F1 | Test macro-F1 | Balanced acc | Kappa | McNemar p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IA³ | — | — | — | — | — | — | — | — | — |
| AdaLoRA | — | — | — | — | — | — | — | — | — |
| Linear probe (ops.) | — | — | — | — | — | — | — | — | — |

> **Adil karşılaştırma notu:** IA³ ve AdaLoRA'nın eğitilebilir parametre bütçeleri birebir eşit değildir. Bu, yöntemlerin doğası gereği kabul edilir; ancak fark tabloda açıkça raporlanır ve "sınırlılık" bölümünde tartışılır. Gerekirse AdaLoRA `target_r=4` ile daha düşük bütçeli bir ek koşu yapılır.

---

## 8. RAPOR VE SUNUM PLANI

### 8.1 Rapor bölümleri ve sahipleri

| Bölüm | İçerik | Sahip | İnceleme |
| :--- | :--- | :--- | :--- |
| 1. Giriş ve motivasyon | Fitoplankton, su kalitesi, problem tanımı | Yusuf | Cankat |
| 2. İlgili çalışmalar ve EVA-02 | Foundation model, MIM, KD, PEFT literatürü | Cankat | Yusuf |
| 3. Veri seti ve ön işleme | WHOI-Plankton, EDA, split, augmentation | Salih | Cankat |
| 4. Yöntem: IA³ ve AdaLoRA | Matematiksel temel, config, hedef modüller | Cankat | Salih |
| 5. Deneysel kurulum | Donanım, hiperparametre, metrikler | Salih | Buse |
| 6. Sonuçlar ve tartışma | Tablolar, eğriler, karmaşıklık, Grad-CAM, testler | Buse | Cankat |
| 7. Sınırlılıklar ve gelecek çalışmalar | Veri, compute, yöntem sınırları | Yusuf + Buse | Cankat |
| 8. Sonuç | 3–5 paragraf, net mesaj | Yusuf | Cankat |
| Kaynaklar | APA/BibTeX, WHOI atıfı | Yusuf | Tüm ekip |
| Ekler | Kod, ek figürler, hiperparametre tablosu | Yusuf | Tüm ekip |

### 8.2 10 dakikalık sunum akışı

| # | Slayt | Konuşmacı | Süre |
| :---: | :--- | :--- | :---: |
| 1 | Başlık, ekip, tek cümlelik problem | Yusuf | 0:20 |
| 2 | Motivasyon: fitoplankton ve su kalitesi | Yusuf | 0:40 |
| 3 | Veri seti ve zorluklar | Salih | 1:15 |
| 4 | Veri hattı ve augmentation | Salih | 1:15 |
| 5 | EVA-02: neden foundation model? | Cankat | 1:00 |
| 6 | IA³: çalışma prensibi ve config | Cankat | 1:00 |
| 7 | AdaLoRA: SVD tabanlı rank bütçesi | Cankat | 1:00 |
| 8 | Sonuçlar: metrikler ve karşılaştırma | Buse | 1:15 |
| 9 | Grad-CAM ve hata analizi | Buse | 1:15 |
| 10 | Sonuç, sınırlılıklar, gelecek çalışmalar | Yusuf | 0:30 |
| — | Geçişler + buffer | — | 0:30 |
| **Toplam** | | | **10:00** |

**Konuşmacı toplamları:** Cankat 3:00 · Salih 2:30 · Buse 2:30 · Yusuf 1:30.

### 8.3 Sunum provası ve Q&A

- **Prova 1 (teknik):** Cankat + Buse geri bildirimi; grafiklerin okunurluğu ve sayı tutarlılığı.
- **Prova 2 (süreli):** Yusuf yönetir; 10:00 kronometre; geçişler ve buffer test edilir.
- **Q&A drill:** Aşağıdaki 10 soru rastgele dağıtılır; herkes kendi alanını savunur.

| Soru | Birincil cevaplayıcı |
| :--- | :--- |
| Neden EVA-02-Base, neden ViT-B/14? | Cankat |
| IA³ neden parametre verimli? | Cankat |
| AdaLoRA ile klasik LoRA farkı ne? | Cankat |
| `head` neden ayrıca eğitildi? | Cankat |
| Veri dengesizliği nasıl çözüldü? | Salih |
| Neden macro-F1, accuracy değil? | Buse |
| Sonuçlar istatistiksel olarak anlamlı mı? | Buse |
| Grad-CAM neden son blok? | Buse |
| Projenin çevresel katkısı ne? | Yusuf |
| Daha fazla veri/compute ile ne yapardınız? | Yusuf + Cankat |

---

## 9. RİSK PLANI

| Risk | Olasılık | Etki | Önlem | Sorumlu |
| :--- | :---: | :---: | :--- | :--- |
| WHOI-Plankton erişim/format sorunu | Orta | Yüksek | Hafta 1'de indir; mirror ve manifest; gerekirse 8 sınıfa düşür ve raporla | Salih |
| PEFT–timm uyumsuzluğu / hedef modül isimleri | Orta | Yüksek | `inspect_modules.py`, sürüm sabitleme, Hafta 1 smoke test | Cankat |
| AdaLoRA `total_step` / rank scheduler hatası | Orta | Orta | 1 epoch dry run, `update_and_allocate` logu, rank grafiği | Cankat |
| GPU OOM | Yüksek | Orta | Batch 16 + grad accum 2, AMP, 224 çözünürlük, Colab/Kaggle | Salih |
| Sınıf dengesizliği / nadir sınıf | Yüksek | Yüksek | Sampler veya class weight, macro-F1, sınıf birleştirme yalnızca gerekçeli | Salih + Buse |
| Overfitting | Yüksek | Orta | Early stopping, label smoothing, augmentation, weight decay | Salih |
| Test sızıntısı | Düşük | Çok yüksek | Split freeze, hash kontrolü, testte tuning yasağı, kod incelemesi | Cankat |
| Notebook merge çakışması | Orta | Orta | Modüler `src/`, final notebook'u yalnızca Yusuf birleştirir | Yusuf |
| Deadline gecikmesi | Orta | Yüksek | Haftalık gate'ler, erken taslak, Hafta 4 buffer | Yusuf |
| Rapor/sunum sayı uyuşmazlığı | Orta | Orta | Tek kaynak JSON, `collect_results.py`, çapraz kontrol | Yusuf + Buse |

---

## 10. TESLİM KONTROL LİSTESİ

- [ ] `data/splits.csv` donduruldu ve commit'lendi.
- [ ] İki yöntem aynı split/seed/augmentation ile eğitildi; loglar tam.
- [ ] Test metrikleri JSON olarak kaydedildi; rapor/sunum sayıları eşleşiyor.
- [ ] McNemar p-değeri ve bootstrap %95 GA raporlandı.
- [ ] Her yöntem için Grad-CAM grid'i (en az 3 doğru + 3 yanlış) hazır.
- [ ] Final notebook temiz ortamda baştan sona hatasız çalıştı.
- [ ] PDF format/sayfa limiti/isim kuralına uygun (`Ad_Soyad_proje.pdf`).
- [ ] PPT 10 slayt, 10:00 prova tamamlandı.
- [ ] Kaynakça ve WHOI-Plankton atıfı tam.
- [ ] `README.md` kurulum ve çalıştırma komutları içeriyor.
- [ ] Her üye kendi kodunu ve teorisini açıklayabiliyor.
- [ ] Yedekler (PDF + PPT + notebook) Drive'da; teslim yüklemesi yapıldı.

---

## EK A — KURULUM

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / Colab
# source .venv/bin/activate

# CUDA sürümünüze uygun index-url ile:
pip install torch torchvision
pip install timm peft scikit-learn pandas numpy matplotlib seaborn \
            pytorch-grad-cam statsmodels tqdm pyyaml pillow tabulate \
            jupyterlab
pip freeze > requirements.lock
```

> Colab/Kaggle kullanılıyorsa runtime'ı GPU'ya alın; `torch.cuda.is_available()` ile doğrulayın. EVA-02-Base 224 px, batch 32 ve AMP ile T4 16 GB'da genellikle rahat çalışır; OOM'da batch 16 + grad accumulation 2 kullanın.

---

## EK B — ORİJİNAL ŞABLONDAN DÜZELTİLEN TEKNİK NOKTALAR

1. **Patch boyutu ve model adı:** EVA-02-Base **ViT-B/14**'tür; orijinaldeki "ViT-B/16" ifadesi düzeltildi. timm'de 224 px için doğru etiket **`eva02_base_patch14_224.mim_in22k`**; `_ft_in1k` yalnızca 448 px varyantında bulunur.
2. **IA³ parametre oranı:** 86M'lik EVA-02-Base için ölçülen değer **~%0.10** (89.100 / 85.86M). Literatürdeki "%0.01" iddiası çok daha büyük modeller (ör. T0) için geçerlidir; kesin değer `print_trainable_parameters()` ile raporlanmalıdır.
3. **timm modül isimleri:** EVA-02'de `attn.qkv` veya `mlp.fc1` yoktur. Doğru hedefler: **`attn.q_proj`, `attn.k_proj`, `attn.v_proj`, `attn.proj`, `mlp.fc1_g`, `mlp.fc1_x`, `mlp.fc2`**. AdaLoRA bu hedeflerle **~%2.40** eğitilebilir parametre üretir (2.11M / 87.88M); IA³ ile bütçe farkı raporlanmalıdır.
4. **Sınıflandırıcı baş:** PEFT uygulandıktan sonra `head`'in öğrenmesi için **`modules_to_save=["head"]`** şarttır.
5. **AdaLoRA scheduler:** `total_step`, `tinit`, `tfinal`, `deltaT` doğru ayarlanmalı ve eğitim döngüsünde **`update_and_allocate(step)`** çağrılmalıdır. PEFT'te `tfinal` "sona kalan fine-tune adımı"dır, mutlak bitiş adımı değildir; bütçe `[tinit, total_step - tfinal]` aralığında dağıtılır. Öneri: `tinit ≈ %10`, `tfinal ≈ %20`.
6. **AdaLoRA rank takibi:** `SVDLinear.ranknum` PEFT'te `init_r` değerinde kalır; gerçek bütçe `lora_E` içindeki sıfırlanmış girdilerle temsil edilir. Grafik için **etkin rank = sıfır olmayan `lora_E` sayısı** kullanılmalıdır.
7. **Task type:** timm görüntü modelinde transformer task type'ları (`SEQ_CLS`, `FEATURE_EXTRACTION`) yerine **generic `PeftModel` (`task_type=None`)** + özel baş kullanılır; aksi halde PEFT sarmalayıcısı `input_ids` bekleyip timm forward'ını bozabilir. Kullanılan PEFT sürümünde mutlaka smoke test yapılır.
8. **Görüntü formatı:** WHOI-Plankton görüntüleri gri tonlamalıdır (tipik 148×148); ImageNet ön eğitimli model için **3 kanala çoğaltma** ve 224'e resize gerekir.
9. **Test disiplini:** Test seti Hafta 3'e kadar kullanılmaz; split dosyası dondurulur.

---

## EK C — HIZLI SÖZLÜK

| Terim | Kısa açıklama |
| :--- | :--- |
| **PEFT** | Parameter-Efficient Fine-Tuning; tüm modeli eğitmeden az sayıda parametreyle uyarlama. |
| **IA³** | Aktivasyonları öğrenilebilir çarpımsal vektörlerle ölçekleyen PEFT yöntemi. |
| **AdaLoRA** | Rank bütçesini katman önemine göre SVD ile dinamik dağıtan LoRA varyantı. |
| **MIM** | Masked Image Modeling; görüntü yamalarını maskeleyip temsil öğrenme. |
| **KD** | Knowledge Distillation; öğretmen modelin çıktılarına öğrenci modelini hizalama. |
| **SubLN** | Sub-Layer Normalization; EVA-02 bloklarında stabilizasyon için kullanılan normalizasyon. |
| **Macro-F1** | Sınıf bazlı F1'lerin ortalaması; dengesiz veride azınlık sınıfları eşit ağırlıklar. |
| **Cohen's kappa** | Şans düzeltmeli uyum ölçüsü; sınıf dağılımından bağımsız karşılaştırma sağlar. |
| **Grad-CAM** | Son katman aktivasyonlarıyla modelin hangi bölgeye baktığını gösteren ısı haritası. |
| **McNemar** | Aynı test örneklerinde iki modelin doğru/yanlışlarını karşılaştıran paired test. |

---

## EK D — EKİP ONAYI

| Üye | Görev | Sorumluluk kapsamı | İmza |
| :--- | :--- | :--- | :--- |
| **Cankat** | Model & PEFT Lideri + Teknik Entegratör | `models.py`, config'ler, parametre/rank analizi, kod inceleme | ................ |
| **Salih** | Veri & Eğitim Mühendisi | `data.py`, `train.py`, split, checkpoint, loglar | ................ |
| **Buse** | Değerlendirme, XAI & İstatistik Uzmanı | `evaluate.py`, `xai.py`, figürler, istatistiksel testler | ................ |
| **Yusuf** | Dokümantasyon, Sunum & Deney Takibi | README, notebook koşumu, PDF, PPT, teslim | ................ |

> Bu şablon, proje başlamadan önce tüm ekip tarafından okunur ve imzalanır. Görev değişikliği yalnızca ekip onayı ve Cankat'ın teknik değerlendirmesiyle yapılır.

---

*Doküman sonudur. Sürüm: v2.0 — 18.11.2026.*












