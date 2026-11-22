"""G1 smoke test: IA^3 and AdaLoRA on EVA-02-Base.

Runs without the phytoplankton dataset, using random tensors.

Usage
-----
    python scripts/smoke_test.py --method both
    python scripts/smoke_test.py --method adalora --steps 20
    python scripts/smoke_test.py --method both --no-pretrained --steps 2

On Colab, write the JSON summary to Drive:
    python scripts/smoke_test.py --method both \
        --output-dir /content/drive/MyDrive/derin_ogrenme_proje/results
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import PeftModel  # noqa: E402

from src.models import (  # noqa: E402
    MODEL_NAME,
    build_model,
    count_parameters,
    get_gradcam_target_layer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["ia3", "adalora", "both"],
        default="both",
    )
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total-step",
        type=int,
        default=20,
        help="AdaLoRA schedule length; use the real run length in training.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of dummy optimizer steps (AdaLoRA rank updates need >= 10).",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download/use timm pretrained weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
    )
    return parser.parse_args()


def make_base(num_classes: int, pretrained: bool):
    return timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def rank_snapshot(model) -> dict[str, int]:
    """Effective AdaLoRA rank per layer.

    PEFT's ``SVDLinear.ranknum`` stays at ``init_r``; the actual budget is
    encoded by zeroed entries in ``lora_E``.  Count non-zeros instead.
    """
    ranks: dict[str, int] = {}
    for name, module in model.named_modules():
        lora_e = getattr(module, "lora_E", None)
        if lora_e is not None and hasattr(lora_e, "items"):
            for adapter_name, param in lora_e.items():
                ranks[f"{name}.{adapter_name}"] = int(
                    (param.detach() != 0).sum().item()
                )
    return ranks


def run_method(method: str, args: argparse.Namespace, device: torch.device) -> dict:
    torch.manual_seed(args.seed)

    base = make_base(args.num_classes, args.pretrained)
    base_state = copy.deepcopy(base.state_dict())

    model = build_model(
        method=method,
        num_classes=args.num_classes,
        total_step=args.total_step if method == "adalora" else None,
        base_model=base,
        verbose=True,
    ).to(device)

    stats = count_parameters(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=0.01,
    )

    x = torch.randn(args.batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    ranks_before = rank_snapshot(model) if method == "adalora" else {}
    steps = args.steps if method == "adalora" else min(2, args.steps)
    if method == "adalora" and steps < 10:
        print("UYARI: AdaLoRA rank güncellemesi için --steps >= 10 önerilir.")

    model.train()
    started = time.time()
    loss_value = float("nan")
    rank_history: list[dict] = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        if method == "adalora":
            # AdaLoRA rank budget is NOT updated automatically.
            model.base_model.update_and_allocate(step)
            ranks = rank_snapshot(model)
            rank_history.append(
                {
                    "step": step,
                    "rank_sum": sum(ranks.values()),
                    "rank_min": min(ranks.values()),
                    "rank_max": max(ranks.values()),
                }
            )

        loss_value = float(loss.detach().cpu())

    elapsed = time.time() - started
    ranks_after = rank_snapshot(model) if method == "adalora" else {}

    model.eval()
    with torch.no_grad():
        eval_logits = model(x)

    if not torch.isfinite(eval_logits).all():
        raise RuntimeError(f"{method}: logits contain NaN/Inf")
    if tuple(eval_logits.shape) != (args.batch_size, args.num_classes):
        raise RuntimeError(
            f"{method}: beklenen {(args.batch_size, args.num_classes)}, "
            f"gelen {tuple(eval_logits.shape)}"
        )

    # Verify the Grad-CAM path before Buse needs it.
    target = get_gradcam_target_layer(model)
    target_type = type(target).__name__

    # Adapter + head save/load round-trip.
    save_dir = args.output_dir / "checkpoints" / f"smoke_{method}"
    model.save_pretrained(save_dir)

    new_base = make_base(args.num_classes, pretrained=False)
    new_base.load_state_dict(base_state)
    loaded = PeftModel.from_pretrained(new_base, save_dir).to(device)
    loaded.eval()
    with torch.no_grad():
        loaded_logits = loaded(x)

    roundtrip_ok = torch.allclose(eval_logits, loaded_logits, atol=1e-5, rtol=1e-4)

    result: dict = {
        "method": method,
        "model": MODEL_NAME,
        "device": str(device),
        "steps": steps,
        "elapsed_sec": round(elapsed, 3),
        "loss": loss_value,
        "logits_shape": list(eval_logits.shape),
        "total_params": stats["total"],
        "trainable_params": stats["trainable"],
        "trainable_ratio_pct": round(float(stats["trainable_ratio_pct"]), 4),
        "roundtrip_ok": roundtrip_ok,
        "gradcam_target_type": target_type,
        "checkpoint_dir": str(save_dir),
    }

    if method == "adalora":
        changed = sum(
            1 for name, rank in ranks_after.items()
            if ranks_before.get(name) != rank
        )
        changed_steps = [
            item["step"]
            for previous, item in zip(rank_history, rank_history[1:])
            if item["rank_sum"] != previous["rank_sum"]
        ]
        rank_distribution = {
            int(rank): count
            for rank, count in sorted(
                Counter(ranks_after.values()).items()
            )
        }
        adalora_cfg = model.peft_config["default"]
        rank_total = sum(ranks_after.values())
        rank_budget_ok = None
        if steps > adalora_cfg.total_step - adalora_cfg.tfinal:
            expected_rank_total = adalora_cfg.target_r * len(ranks_after)
            rank_budget_ok = rank_total == expected_rank_total
            if not rank_budget_ok:
                raise RuntimeError(
                    "AdaLoRA final rank budget mismatch: "
                    f"expected {expected_rank_total}, got {rank_total}"
                )
        result.update(
            {
                "adalora_total_step": adalora_cfg.total_step,
                "adalora_tinit": adalora_cfg.tinit,
                "adalora_tfinal": adalora_cfg.tfinal,
                "adalora_delta_t": adalora_cfg.deltaT,
                "adalora_init_r": adalora_cfg.init_r,
                "adalora_target_r": adalora_cfg.target_r,
                "rank_layers": len(ranks_after),
                "rank_total": rank_total,
                "rank_budget_ok": rank_budget_ok,
                "rank_layers_changed": changed,
                "rank_steps_changed": changed_steps,
                "rank_distribution": rank_distribution,
                "rank_history": rank_history,
                "ranks_before_sample": dict(list(ranks_before.items())[:5]),
                "ranks_after_sample": dict(list(ranks_after.items())[:5]),
            }
        )

    return result


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("UYARI: GPU yok; smoke test yavaş çalışır. --no-pretrained --steps 2 kullan.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["ia3", "adalora"] if args.method == "both" else [args.method]
    results = [run_method(method, args, device) for method in methods]

    json_path = args.output_dir / "g1_smoke_test.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== G1 ÖZET ===")
    for result in results:
        print(
            f"{result['method']:8s} | "
            f"trainable={result['trainable_params']:,} "
            f"({result['trainable_ratio_pct']:.3f}%) | "
            f"roundtrip={result['roundtrip_ok']} | "
            f"loss={result['loss']:.4f}"
        )
        if result["method"] == "adalora":
            print(
                f"         | rank layers={result['rank_layers']}, "
                f"total={result['rank_total']}, "
                f"budget_ok={result['rank_budget_ok']}, "
                f"changed={result['rank_layers_changed']}, "
                f"steps={result['rank_steps_changed']}"
            )
    print(f"\nJSON: {json_path}")


if __name__ == "__main__":
    main()
