# Derin Öğrenme Projesi — EVA-02 + IA³ / AdaLoRA

Fitoplankton tür sınıflandırması için EVA-02-Base üzerinde iki PEFT yönteminin
(IA³ ve AdaLoRA) karşılaştırılması.

> Bu README G1 aşaması için hazırlandı. Rapor/sunum bölümleri Yusuf tarafından
> genişletilecek.

## Ekip

| Üye | Görev |
| :--- | :--- |
| Cankat | Model & PEFT + teknik entegrasyon |
| Salih | Veri & eğitim döngüsü |
| Buse | Değerlendirme, XAI & istatistik |
| Yusuf | Dokümantasyon, sunum & teslim |

## Klasör yapısı

```text
src/models.py              # EVA-02 + IA³ / AdaLoRA fabrikası (Cankat)
scripts/inspect_modules.py # hedef modül isimlerini doğrular
scripts/smoke_test.py      # G1: dataset olmadan forward/backward testi
notebooks/                 # Colab notebook'ları
configs/                   # IA³ ve AdaLoRA hiperparametreleri
results/                   # JSON, figür, checkpoint (checkpoint'ler git dışı)
```

## Hızlı başlangıç (Colab)

1. Bu klasörü GitHub'a push edin (veya Drive'a yükleyin).
2. Colab'da yeni notebook açın; runtime türünü **GPU** yapın.
3. Repoyu klonlayın:

```bash
!git clone https://github.com/KULLANICI/REPO.git /content/derin-ogrenme-proje
%cd /content/derin-ogrenme-proje
```

4. Bağımlılıkları kurun (torch/torchvision Colab'da hazır):

```bash
!pip -q install "timm==1.0.31" "peft==0.21.3"
```

5. Hedef modülleri doğrulayın:

```bash
!python scripts/inspect_modules.py
```

6. G1 smoke testini çalıştırın:

```bash
!python scripts/smoke_test.py --method both \
    --output-dir /content/drive/MyDrive/derin_ogrenme_proje/results
```

Colab free oturumları kopabilir; JSON ve checkpoint'leri mutlaka Drive'a yazın.

## Yerel kurulum (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision
pip install -r requirements.txt
python scripts/inspect_modules.py
python scripts/smoke_test.py --method both --no-pretrained --steps 2
```

`--no-pretrained --steps 2` hızlı yapı testidir; gerçek G1 testi için
`--pretrained` ve AdaLoRA için `--steps 20` kullanın.

## G1 kabul kriteri

- [ ] IA³ ve AdaLoRA modelleri kuruluyor.
- [ ] Dummy `[2, 3, 224, 224]` girişte çıkış `[2, 12]`, loss sonlu.
- [ ] `head` eğitilebilir; parametre oranı kayıtlı.
- [ ] AdaLoRA etkin rankı (sıfır olmayan `lora_E`) mask adımlarında değişiyor; `tfinal` sonrası hedef bütçede sabitleniyor.
- [ ] Adapter save/load sonrası logitler `allclose`.
- [ ] `results/g1_smoke_test.json` üretildi.

## G1 ölçümleri (yerel CPU, 22 Kasım 2026)

| Yöntem | Eğitilebilir / Toplam | Oran | Round-trip | Not |
| :--- | ---: | ---: | :---: | :--- |
| IA³ | 89.100 / 85.857.048 | %0.1038 | ✅ | 2 adım |
| AdaLoRA | 2.111.484 / 87.879.516 | %2.4027 | ✅ | 20 adım, etkin rank 1008 → 672 |

- `results/g1_smoke_test.json` pretrained ağırlıklarla üretildi; iki yöntemde de
  loss sonlu ve çıkış `[1, 12]`.
- AdaLoRA 20 adımlık smoke testinde `tinit=2`, `tfinal=4`, `deltaT=10`.
  Etkin rank 16. adımda hedef bütçeye (84 katmanda toplam 672) iniyor ve
  orada sabitleniyor. Maskeler arasında tripletler yeniden aktifleşebildiği
  için ara adımlarda (10 → 11) geçici artış görülebilir.
- Bu tablo eğitim başarısı değil, G1 altyapı doğrulamasıdır; Colab GPU koşusu
  tekrarlanacak.

## Notlar

- `task_type=None` kullanılıyor; timm modelleri Hugging Face transformer değildir.
- `modules_to_save=["head"]` olmadan 12 sınıflı baş öğrenmez.
- AdaLoRA'da `total_step` gerçek eğitim adım sayısıyla eşleşmeli ve eğitim
  döngüsünde `model.base_model.update_and_allocate(step)` çağrılmalı.
- AdaLoRA'da `SVDLinear.ranknum` `init_r` değerinde kalır; etkin rank
  `lora_E` içindeki sıfır olmayan girdilerle ölçülür. Maskeler arasında
  sıfırlanan tripletler yeniden aktifleşebilir; `tfinal` sonrasında mask her
  adımda uygulanır.
