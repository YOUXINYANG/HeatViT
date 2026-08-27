#!/usr/bin/env python3
"""P3 D3 adjudication probe: ablate the post-QAT activation-recalib
table changes and measure their bit-exact Top-1 impact.

The Q2 recalib (p2_out/qat/short128k/scale_table_after.json) changed 32
activation exponents vs the frozen PTQ table and cost -4.70pp on 5k
(77.86% -> 73.16%). This probe partitions the changes and evaluates each
subset through the exact eval path (same machinery as p2_qat.eval), so
the toxic changes are identified mechanistically:

  none    frozen table only (the 77.86% reference)
  all     every recalib change (the 73.16% reference)
  ln      only LN-input tensors (b<N>_y / b<N>_out)        [19]
  nonln   everything else (ffn/ln2/msa/qkv/hidden/final)   [13]
  plus1   only the +1 exponent changes                     [18]
  big2    only the |delta| = 2 changes                     [11]

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/qat_recalib_probe.py \\
      --checkpoint p2_out/qat/short128k/best.pt --images 5000
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_quantize import LN_INPUT_NAMES
from tools.p2.scale_table import ScaleTable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint",
                        default="p2_out/qat/short128k/best.pt")
    parser.add_argument("--frozen-table", default="p2_out/scale_table.json")
    parser.add_argument("--recalib-table",
                        default="p2_out/qat/short128k/scale_table_after.json")
    parser.add_argument("--images", type=int, default=5000)
    parser.add_argument("--subsets",
                        default="none,ln,nonln,plus1,big2,all")
    parser.add_argument("--dump", default=None,
                        help="subset name: save that table JSON instead "
                             "of evaluating (use with --out)")
    parser.add_argument("--out", default=None,
                        help="output path for --dump")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    frozen = ScaleTable.load(REPO_ROOT / args.frozen_table)
    recalib = ScaleTable.load(REPO_ROOT / args.recalib_table)
    diff = []
    for name, exp in recalib.activations.items():
        old = frozen.activations.get(name)
        if old != exp:
            diff.append((name, old, exp))
    print(f"diff size: {len(diff)} changes")

    def build(subset):
        acts = dict(frozen.activations)
        for name, old, new in diff:
            if subset == "all":
                acts[name] = new
            elif subset == "ln" and name in LN_INPUT_NAMES:
                acts[name] = new
            elif subset == "nonln" and name not in LN_INPUT_NAMES:
                acts[name] = new
            elif subset == "plus1" and new - old == 1:
                acts[name] = new
            elif subset == "big2" and abs(new - old) == 2:
                acts[name] = new
        table = ScaleTable(weights=dict(frozen.weights), activations=acts)
        return table.validate()

    ckpt = torch.load(REPO_ROOT / args.checkpoint, map_location="cpu",
                      weights_only=False)
    floats = ckpt["floats"]

    if args.dump:
        table = build(args.dump)
        out = REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        table.save(out)
        n_chg = sum(1 for name, old, new in diff
                    if table.activations[name] != frozen.activations[name])
        print(f"dumped [{args.dump}] with {n_chg} changes -> {out}")
        return

    from tools.p2 import p2_qat

    loader = p2_qat.make_val_loader(args.images, batch_size=32)
    for subset in args.subsets.split(","):
        subset = subset.strip()
        table = build(subset)
        qat = p2_qat.build_qat(floats, table, device)
        acc_e, c_e, n_e = p2_qat.eval_val(qat, loader, device, exact=True)
        n_chg = sum(1 for name, old, new in diff
                    if table.activations[name] != frozen.activations[name])
        print(f"[{subset:6s}] exact top1={acc_e:.2f}% ({c_e}/{n_e}) "
              f"changes applied={n_chg}")


if __name__ == "__main__":
    main()
