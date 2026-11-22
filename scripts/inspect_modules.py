"""Print the timm EVA-02 modules that PEFT will target.

Usage
-----
    python scripts/inspect_modules.py
    python scripts/inspect_modules.py --pretrained
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import timm  # noqa: E402

from src.models import (  # noqa: E402
    HEAD_MODULE,
    MODEL_NAME,
    TARGET_MODULES,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained weights (downloads ~350 MB).",
    )
    parser.add_argument("--num-classes", type=int, default=12)
    args = parser.parse_args()

    model = timm.create_model(
        MODEL_NAME,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
    )

    wanted = tuple(TARGET_MODULES + [HEAD_MODULE])
    found: list[str] = []

    print(f"Model: {MODEL_NAME}")
    print(f"{'module':55s} {'type':14s} {'in':>6s} -> {'out':>6s}")
    print("-" * 90)

    for name, module in model.named_modules():
        if name.endswith(wanted):
            found.append(name)
            in_features = getattr(module, "in_features", "-")
            out_features = getattr(module, "out_features", "-")
            print(f"{name:55s} {type(module).__name__:14s} {str(in_features):>6s} -> {str(out_features):>6s}")

    missing = [
        target
        for target in wanted
        if not any(name.endswith(target) for name in found)
    ]
    print("-" * 90)
    print(f"Eşleşen modül: {len(found)}")
    if missing:
        raise SystemExit(f"HATA: eksik hedef modüller: {missing}")
    print("Tüm hedef modüller bulundu.")


if __name__ == "__main__":
    main()
