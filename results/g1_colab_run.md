# G1 Colab GPU run log

- **Tarih:** 24 Kasım 2026, 11:18 (Colab sunucu saati)
- **GPU:** Tesla T4 (Colab free)
- **Python:** 3.13.16
- **torch:** 2.14.1+cu130
- **timm:** 1.0.31
- **peft:** 0.21.3 (`pip show peft` ile doğrulandı)
- **Model:** `eva02_base_patch14_224.mim_in22k`
- **Hedef modüller:** 85 eşleşme; tümü bulundu.

## Smoke test özeti

| Yöntem | Adım | Cihaz | Çıkış | Eğitilebilir / Toplam | Oran | Round-trip |
| :--- | ---: | :---: | :---: | ---: | ---: | :---: |
| IA³ | 2 | cuda | `[2, 12]` | 89.100 / 85.857.048 | %0.1038 | ✅ |
| AdaLoRA | 20 | cuda | `[2, 12]` | 2.111.484 / 87.879.516 | %2.4027 | ✅ |

- IA³ loss: `1.5033`
- AdaLoRA loss: `0.0590`
- AdaLoRA: `tinit=2`, `tfinal=4`, `deltaT=10`
- Etkin rank: `1008 → 672` (84 katman, ortalama 8)
- `rank_budget_ok=True`
- Rank değişim adımları: `[10, 11, 16]`; 16. adımdan sonra 672 sabit.
- Her iki yöntemde de Grad-CAM hedefi: `LayerNorm`
- Pretrained ağırlıklar HF Hub'dan indirildi (~343 MB).

Tam JSON: `results/g1_smoke_test.json`

Checkpoint'ler Colab Drive'da:
`/content/drive/MyDrive/derin_ogrenme_proje/results/checkpoints/`

> Bu çıktı eğitim başarısı değil, G1 altyapı doğrulamasıdır.
