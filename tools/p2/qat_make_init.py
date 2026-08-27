#!/usr/bin/env python3
"""P3 QAT helper: materialize the untouched official DeiT-T checkpoint as a
QAT eval checkpoint (baseline anchor for quick-run comparisons).

The file layout mirrors p2_qat.train's ``save`` (a ``floats`` key in
HeatViT layout), so ``p2_qat.py eval --checkpoint p2_out/qat/init.pt``
measures the exact pre-training bit-exact Top-1 through the very same
``exact_forward`` path used after training (expected == the P2 PTQ
baseline of the deployment contract, 76.06% @5k).

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/qat_make_init.py \\
      --out p2_out/qat/init.pt
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_quantize import load_state_dict, to_heatvit_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="p2_out/qat/init.pt")
    args = parser.parse_args()
    floats = to_heatvit_tensors(load_state_dict())
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"floats": floats, "epoch": 0, "step": 0, "best_exact": 0.0},
               out)
    print(f"saved {len(floats)} tensors -> {out}")


if __name__ == "__main__":
    main()
