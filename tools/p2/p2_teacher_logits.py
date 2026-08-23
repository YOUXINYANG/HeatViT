#!/usr/bin/env python3
"""Precompute unpruned-backbone teacher logits for selector distillation.

Runs the exact integer simulator WITHOUT pruning over the same training
subset that tools/p2/p2_train_selector.py uses (seed 20260815), and saves
logits keyed by subset index so the training loop can look them up.

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p2_teacher_logits.py \
      --max-images 8192 --out p2_out/teacher_logits.pt
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torchvision import datasets, transforms

from tools.p2.p2_quantize import (
    build_model,
    load_state_dict,
    make_val_loader,
    to_heatvit_tensors,
)
from tools.p2.p2_sim_ivit import NonlinConfig, forward_batch_cfg
from tools.p2.scale_table import ScaleTable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--table", default="p2_out/ivit/scale_table_legacy.json")
    parser.add_argument("--out", default="p2_out/teacher_logits.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    state = load_state_dict()
    floats = to_heatvit_tensors(state)
    table = ScaleTable.load(REPO_ROOT / args.table).validate()
    model = build_model(floats, table, device)
    cfg = NonlinConfig()

    # Deterministic val subset (CenterCrop, first N images): identical to
    # the selector training set, so the teacher logits stay valid across
    # epochs (no random augmentation mismatch).
    loader = make_val_loader(args.max_images, batch_size=args.batch_size)

    logits_all = []
    t0 = time.time()
    with torch.no_grad():
        for img, label in loader:
            logits, _ = forward_batch_cfg(model, img.to(device), cfg)
            logits_all.append(logits.cpu())
    logits_all = torch.cat(logits_all, dim=0)
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"logits": logits_all, "count": logits_all.shape[0]}, out_path)
    print(f"teacher logits [{logits_all.shape[0]}, 1000] -> {out_path} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
