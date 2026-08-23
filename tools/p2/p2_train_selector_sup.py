#!/usr/bin/env python3
"""Supervised Token Selector training on (int8-feature, importance) pairs.

Trains the three RTL-shaped selectors directly on the cached pairs from
p2_selector_sup_data.py: the features are the EXACT integer simulator's
dequantized block outputs (true RTL inputs), the labels are the float
teacher's CLS attention importance. The task signal therefore reaches the
selector parameters without flowing through the integer backbone.

Loss = BCE(fused, label) + sparsity-weight * (keep_fraction - target)^2
       + range-weight * contract-range penalty.

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p2_train_selector_sup.py \\
      --data p2_out/selector_sup_data.pt --epochs 30 --lr 1e-3 \\
      --out p2_out/selectors_sup.pt
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from tools.p2.p2_train_selector import (
    STAGE_TARGETS,
    TrainSelector,
    export_selector,
    selector_scale_table,
)
from tools.p2.scale_table import ScaleTable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", default="p2_out/selector_sup_data2.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sparsity-weight", type=float, default=15.0)
    parser.add_argument("--range-weight", type=float, default=1.0)
    parser.add_argument("--threshold-offset", type=float, default=-0.008)
    parser.add_argument("--threshold-offsets", default=None,
                        help="per-stage colon-separated list "
                             "(stage1:stage2:stage3) overriding "
                             "--threshold-offset")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--table", default="p2_out/ivit/scale_table_legacy.json")
    parser.add_argument("--out", default="p2_out/selectors_sup.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    data = torch.load(REPO_ROOT / args.data, map_location="cpu",
                      weights_only=False)
    feats = {s: data["features"][str(s)].float().to(device)
             for s in (1, 2, 3)}
    labels = {s: data["labels"][str(s)].float().to(device)
              for s in (1, 2, 3)}
    masks = {s: data["masks"][str(s)].to(device) for s in (1, 2, 3)}
    n = feats[1].shape[0]

    table = ScaleTable.load(REPO_ROOT / args.table).validate()
    in_exps = {1: table.activation_exp("b3_out"),
               2: table.activation_exp("b6_out"),
               3: table.activation_exp("b9_out")}
    sel_acts = {i: dict(local_out=-6, concat_out=-6, h1_out=-6,
                        h2_out=-6, logits_out=-7,
                        stats_out=in_exps[i], hw_hidden_out=-3)
                for i in (1, 2, 3)}

    selectors = nn.ModuleList([TrainSelector().to(device) for _ in range(3)])
    opt = torch.optim.AdamW(selectors.parameters(), lr=args.lr,
                            weight_decay=0.05)
    if args.threshold_offsets:
        thr = [float(x) for x in args.threshold_offsets.split(":")]
        assert len(thr) == 3, "need 3 per-stage offsets"
        thresholds = [0.5 + t for t in thr]
    else:
        thresholds = [0.5 + args.threshold_offset] * 3

    steps = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(n, device=device)
        run_bce = 0.0
        run_sp = 0.0
        run_rg = 0.0
        for b0 in range(0, n, args.batch_size):
            idx = perm[b0:b0 + args.batch_size]
            loss = torch.tensor(0.0, device=device)
            for s in (1, 2, 3):
                sel = selectors[s - 1]
                fused, keep_soft, rl = sel(feats[s][idx])
                lab = labels[s][idx]
                mk = masks[s][idx].float()
                bce_el = nn.functional.binary_cross_entropy(
                    fused, lab, reduction="none")
                bce = (bce_el * mk).sum() / mk.sum().clamp(min=1e-6)
                hard = fused >= thresholds[s - 1]
                mask = hard.float() + fused - fused.detach()
                rate = (mask * mk).sum() / mk.sum().clamp(min=1e-6)
                sp = (rate - STAGE_TARGETS[s - 1]) ** 2
                loss = loss + bce + args.sparsity_weight * sp \
                    + args.range_weight * rl
                run_bce += bce.item()
                run_sp += sp.item()
                run_rg += rl.item()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selectors.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for sel in selectors:
                    for head in sel.score:
                        head[-1].bias.clamp_(-0.5, 0.5)
                    for m in sel.modules():
                        if isinstance(m, nn.Linear):
                            m.weight.clamp_(-0.9, 0.9)
            steps += 1
        if epoch % 3 == 0:
            print(f"epoch {epoch}: bce {run_bce / (n / args.batch_size):.4f} "
                  f"sparsity {run_sp / (n / args.batch_size):.4f} "
                  f"range {run_rg / (n / args.batch_size):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    sel_wexps = {}
    payload_selectors = []
    for s, sel in enumerate(selectors, start=1):
        q, wexp = export_selector(sel, device, in_exps[s], sel_acts[s])
        payload_selectors.append(q)
        sel_wexps[str(s)] = wexp
    out_path = REPO_ROOT / args.out
    payload = {
        "selectors": payload_selectors,
        "weight_exps": sel_wexps,
        "sel_acts": {str(i): sel_acts[i] for i in (1, 2, 3)},
        "in_exps": {str(i): in_exps[i] for i in (1, 2, 3)},
        "selectors_float": [s.state_dict() for s in selectors],
    }
    torch.save(payload, out_path)
    print(f"selectors -> {out_path}")


if __name__ == "__main__":
    main()
