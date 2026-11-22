"""EVA-02-Base + IA^3 / AdaLoRA model factory.

Cankat — G1 deliverable.

Notes
-----
* timm models are not Hugging Face ``PreTrainedModel`` instances, so we keep
  ``task_type=None`` and let PEFT build a generic ``PeftModel``.  Transformer
  task types (``SEQ_CLS``, ``FEATURE_EXTRACTION``) can make PEFT expect
  ``input_ids`` and break the timm forward pass.
* The freshly initialized 12-class ``head`` must be listed in
  ``modules_to_save``; otherwise PEFT freezes it.
"""

from __future__ import annotations

from typing import Any

import timm
import torch.nn as nn
from peft import AdaLoraConfig, IA3Config, PeftModel, get_peft_model

# timm'de 224 px EVA-02-Base için doğru etiket budur; "_ft_in1k" yalnızca 448 px
# varyantında bulunur. 448 Colab free için ağırdır.
MODEL_NAME = "eva02_base_patch14_224.mim_in22k"
# timm 1.x EVA-02 uses separate q/k/v projections and a SwiGLU MLP
# (fc1_g, fc1_x, fc2).  It does NOT expose a fused "attn.qkv" or "mlp.fc1".
TARGET_MODULES = [
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.proj",
    "mlp.fc1_g",
    "mlp.fc1_x",
    "mlp.fc2",
]
FEEDFORWARD_MODULES = ["mlp.fc1_g", "mlp.fc1_x", "mlp.fc2"]
HEAD_MODULE = "head"


def _validate_total_step(total_step: int | None) -> int:
    if total_step is None:
        raise ValueError(
            "AdaLoRA için total_step zorunlu. "
            "total_step = epochs * len(train_loader) olarak ver."
        )
    total_step = int(total_step)
    if total_step < 10:
        raise ValueError(
            f"AdaLoRA için total_step en az 10 olmalı, gelen: {total_step}"
        )
    return total_step


def _adalora_schedule(total_step: int) -> tuple[int, int, int]:
    """Return (tinit, tfinal, deltaT) for PEFT's AdaLoRA schedule.

    PEFT semantics: ``tinit`` is the warm-up at the START; ``tfinal`` is the
    number of steps at the END reserved for final fine-tuning.  Budget
    allocation runs between ``tinit`` and ``total_step - tfinal``.
    """
    tinit = max(2, int(0.10 * total_step))
    tfinal = max(2, int(0.20 * total_step))
    if tinit + tfinal >= total_step:
        tinit = max(1, total_step // 10)
        tfinal = max(1, total_step // 5)
    return tinit, tfinal, 10


def build_model(
    method: str,
    num_classes: int = 12,
    total_step: int | None = None,
    pretrained: bool = True,
    base_model: nn.Module | None = None,
    verbose: bool = True,
) -> PeftModel:
    """Build an EVA-02-Base model with an IA^3 or AdaLoRA adapter.

    Parameters
    ----------
    method:
        ``"ia3"`` or ``"adalora"``.
    num_classes:
        Number of phytoplankton classes (default 12).
    total_step:
        Required for AdaLoRA; total optimizer steps of the run.
    pretrained:
        Load ImageNet-22K pretrained timm weights. Ignored when ``base_model``
        is supplied.
    base_model:
        Optional pre-built timm model. Useful for adapter save/load tests.
    verbose:
        Print the PEFT trainable-parameter summary.
    """
    method = method.lower().strip()
    if method not in {"ia3", "adalora"}:
        raise ValueError(f"Bilinmeyen yöntem: {method!r}. Seçenekler: 'ia3', 'adalora'.")

    base = base_model
    if base is None:
        base = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    if method == "ia3":
        config: Any = IA3Config(
            target_modules=TARGET_MODULES,
            feedforward_modules=FEEDFORWARD_MODULES,
            modules_to_save=[HEAD_MODULE],
        )
    else:
        resolved_total_step = _validate_total_step(total_step)
        tinit, tfinal, delta_t = _adalora_schedule(resolved_total_step)
        config = AdaLoraConfig(
            target_modules=TARGET_MODULES,
            init_r=12,
            target_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            total_step=resolved_total_step,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=delta_t,
            modules_to_save=[HEAD_MODULE],
        )

    model = get_peft_model(base, config)
    if verbose:
        model.print_trainable_parameters()
    return model


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    """Return total, trainable and trainable-ratio statistics."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = (100.0 * trainable / total) if total else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "trainable_ratio_pct": ratio,
    }


def get_gradcam_target_layer(model: PeftModel):
    """Last transformer block's pre-attention LayerNorm.

    PEFT wraps the timm model, hence the ``base_model.model`` path.
    """
    return model.base_model.model.blocks[-1].norm1
