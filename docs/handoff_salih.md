# Salih → Entegrasyon Sözleşmesi (G1 çıkışı)

**Durum:** Cankat'ın G1 model/PEFT işi Colab T4'te tamamlandı (24 Kasım 2026).
Bu belge, Salih'in veri/eğitim döngüsünün Cankat'ın fabrikasına nasıl bağlanacağını
tanımlar. Referans: `results/g1_colab_run.md`, `results/g1_smoke_test.json`,
`scripts/smoke_test.py`.

## 1. Model fabrikası

```python
from src.models import build_model, count_parameters, get_gradcam_target_layer

model = build_model(
    method="ia3",          # "ia3" | "adalora"
    num_classes=12,
    total_step=None,       # yalnızca AdaLoRA için zorunlu
    pretrained=True,
)
```

- **IA³:** `total_step` gerekmez; adapter doğrudan kurulur.
- **AdaLoRA:** `total_step = epochs * len(train_loader)` **model kurulmadan önce**
  hesaplanmalı ve fabrikaya verilmelidir. En az 10 olmalıdır.
- AdaLoRA takvimi fabrikada otomatik seçilir:
  `tinit = max(2, %10)`, `tfinal = max(2, %20)`, `deltaT = 10`.
- Sınıflandırıcı baş `modules_to_save=["head"]` ile eğitilebilir kalır.
- Dönen nesne `PeftModel`'dır; `.to(device)` sonradan çağrılır.

## 2. Eğitim döngüsü sözleşmesi

AdaLoRA rank bütçesi otomatik güncellenmez. **Her optimizer adımından sonra**,
koşu boyunca artan **0 tabanlı global adım** ile bir kez çağrılmalıdır:

```python
model.base_model.update_and_allocate(global_step)
```

- `global_step` epoch başına sıfırlanmaz; resume sonrasında kaldığı yerden devam eder.
- Çağrı sırası: `optimizer.step()` (AMP'de `scaler.step` + `scaler.update`) →
  `scheduler.step()` → `update_and_allocate(step)`.
- Çağrı atlanırsa rank bütçesi güncellenmez; fazla çağrılırsa takvim kayar.
- `tfinal` sonrasında (yaklaşık son %20) bütçe hedefe kilitlenir. G1'de 20 adımlık
  koşuda etkin rank `1008 → 672` oldu ve 16. adımdan sonra sabitlendi.

### Minimal iskelet

```python
steps_per_epoch = len(train_loader)
total_step = epochs * steps_per_epoch

model = build_model("adalora", num_classes=12, total_step=total_step).to(device)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-4,
    weight_decay=0.01,
)
scheduler = ...  # cosine + %5 warmup
scaler = torch.amp.GradScaler("cuda")

global_step = 0
for epoch in range(start_epoch, epochs):
    model.train()
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if method == "adalora":
            model.base_model.update_and_allocate(global_step)

        global_step += 1
```

## 3. Kaydetme / yükleme

```python
model.save_pretrained(checkpoint_dir)   # adapter + head birlikte
```

Yükleme için base model **pretrained olmadan** kurulur, sonra adapter eklenir:

```python
base = timm.create_model("eva02_base_patch14_224.mim_in22k",
                         pretrained=False, num_classes=12)
model = PeftModel.from_pretrained(base, checkpoint_dir).to(device)
```

G1'de adapter + head round-trip testi `atol=1e-5` ile geçti. Checkpoint'ler
Colab'da Drive'a yazılmalı:
`/content/drive/MyDrive/derin_ogrenme_proje/results/checkpoints/`

## 4. Ölçülen referans değerler (Colab T4)

| Yöntem | Eğitilebilir / Toplam | Oran |
| :--- | ---: | ---: |
| IA³ | 89.100 / 85.857.048 | %0.1038 |
| AdaLoRA | 2.111.484 / 87.879.516 | %2.4027 |

- Sabitlenen sürümler: `timm==1.0.31`, `peft==0.21.3`.
- Grad-CAM hedefi: `get_gradcam_target_layer(model)` → `blocks[-1].norm1`
  (`LayerNorm`); Buse'ye bu arayüz devredildi.

## 5. Salih'ten beklenen geri bildirim

1. Gerçek `steps_per_epoch`, `total_step`, batch size ve VRAM kullanımı.
2. 1 epoch dry run logu (loss, val macro-F1, süre).
3. OOM veya eğitim hatası olursa aynı gün Cankat'a bildirim.
