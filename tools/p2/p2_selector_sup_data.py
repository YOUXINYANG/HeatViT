#!/usr/bin/env python3
"""Collect supervised selector training data: (features, importance labels).

For each training image:
  * features: the EXACT integer simulator's dequantized block outputs at
    the three selector positions (b3_out / b6_out / b9_out, CLS excluded)
    — the true RTL selector inputs;
  * labels: the float DeiT-T teacher's per-token importance = the CLS
    attention row (mean over heads), aggregated over the remaining
    blocks after each selector and normalized to [0, 1] per image.

The pairing bypasses the integer simulator's gradient blockage: the
selectors are then trained directly on these pairs (BCE), so the task
signal reaches the selector parameters without flowing through the
backbone.

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p2_selector_sup_data.py \\
      --max-images 8192 --out p2_out/selector_sup_data.pt
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_quantize import (
    build_model,
    load_state_dict,
    make_val_loader,
    to_heatvit_tensors,
)
from tools.p2.p2_sim_ivit import NonlinConfig, forward_batch_cfg
from tools.p2.scale_table import ScaleTable

SELECTOR_BLOCKS = (4, 7, 10)


def collect_features(model, table, loader, device):
    """Dequantized b3/b6/b9_out candidate features from the int sim."""
    cfg = NonlinConfig()
    feats = {1: [], 2: [], 3: []}
    with torch.no_grad():
        for img, _ in loader:
            rec = {}
            forward_batch_cfg(model, img.to(device), cfg, rec)
            for stage, blk in enumerate(SELECTOR_BLOCKS, start=1):
                t = rec[f"b{blk}_out"]                      # [B,197,192] int8
                exp = table.activation_exp(f"b{blk}_out")
                deq = t.float() * (2.0 ** exp)              # [B,197,192]
                feats[stage].append(deq[:, 1:].half().cpu())  # candidates
    return {s: torch.cat(v, dim=0) for s, v in feats.items()}


def collect_labels(model, loader, device):
    """CLS attention importance aggregated over the remaining blocks."""
    qkv = {}
    hooks = []
    for i, blk in enumerate(model.blocks):
        def h(m, a, o, idx=i):
            qkv[idx] = o.detach()
        hooks.append(blk.attn.qkv.register_forward_hook(h))

    labels = {1: [], 2: [], 3: []}
    heads = 3
    head_dim = 64
    try:
        with torch.no_grad():
            for img, _ in loader:
                qkv.clear()
                model(img.to(device))
                b = img.shape[0]
                for stage, start in enumerate(SELECTOR_BLOCKS, start=1):
                    agg = None
                    for i in range(start - 1, 12):
                        o = qkv[i]                          # [B,197,576]
                        q, k, _ = o.reshape(b, 197, 3, 3, head_dim) \
                            .permute(2, 3, 0, 1, 4)
                        attn = torch.softmax(
                            (q @ k.transpose(-2, -1)) / (head_dim ** 0.5),
                            dim=-1)                         # [B,3,197,197]
                        cls_row = attn[:, :, 0, :].mean(dim=0)  # [B,197]
                        agg = cls_row if agg is None else agg + cls_row
                    imp = agg[:, 1:]                        # candidates
                    imp = imp / imp.max(dim=1, keepdim=True).values \
                        .clamp(min=1e-6)
                    labels[stage].append(imp.half().cpu())
    finally:
        for h in hooks:
            h.remove()
    return {s: torch.cat(v, dim=0) for s, v in labels.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--table", default="p2_out/ivit/scale_table_legacy.json")
    parser.add_argument("--out", default="p2_out/selector_sup_data.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    state = load_state_dict()
    floats = to_heatvit_tensors(state)
    table = ScaleTable.load(REPO_ROOT / args.table).validate()
    model = build_model(floats, table, device)

    loader = make_val_loader(args.max_images, batch_size=args.batch_size)

    print("collecting int8 features ...", flush=True)
    t0 = time.time()
    feats = collect_features(model, table, loader, device)
    print(f"  features done in {time.time() - t0:.0f}s")

    import timm
    fm = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    fm.load_state_dict(state, strict=True)
    fm = fm.to(device).eval()

    print("collecting teacher attention labels ...", flush=True)
    t0 = time.time()
    labels = collect_labels(fm, loader, device)
    print(f"  labels done in {time.time() - t0:.0f}s")

    out_path = REPO_ROOT / args.out
    payload = {
        "features": {str(s): feats[s] for s in (1, 2, 3)},
        "labels": {str(s): labels[s] for s in (1, 2, 3)},
        "count": args.max_images,
    }
    torch.save(payload, out_path)
    for s in (1, 2, 3):
        print(f"stage {s}: features {tuple(feats[s].shape)} "
              f"labels mean {labels[s].mean().item():.4f} "
              f"frac>0.5 { (labels[s] > 0.5).float().mean().item():.4f}")
    print(f"data -> {out_path}")


if __name__ == "__main__":
    main()
