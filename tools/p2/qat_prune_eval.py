#!/usr/bin/env python3
"""Evaluate QAT-trained weights WITH token pruning in the bit-exact
I-ViT contract simulator (the deployment path).

Plugs the P2-C trained Token Selectors into the SAME contract path that
produces the unpruned QAT numbers (p2_sim_ivit.forward_image_cfg with
prune=True; NonlinConfig() = deployment contract), so the pruning cost
is measured against the unpruned anchors 76.06% (init) / 77.86%
(short128k best) on the identical 5k val subset.

Selectors stay frozen (P4 starting point): the selector per-tensor scale
entries (s{i}_*) come from the selector checkpoint, the backbone entries
from --table. Token counts report the actual per-stage keep rates
(targets 88 / 45 / 32).

Unlike tools/p2/p2_selector_eval.py (which uses the pre-I-ViT p2_sim
path), this tool uses p2_sim_ivit so results are directly comparable
with the QAT eval pipeline.

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/qat_prune_eval.py \\
      --checkpoint p2_out/qat/init.pt --images 1000
  .venv-torch\\Scripts\\python tools/p2/qat_prune_eval.py \\
      --checkpoint p2_out/qat/short128k/best.pt --images 5000
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_quantize import build_model, make_val_loader
from tools.p2.p2_sim import SelectorP
from tools.p2.p2_sim_ivit import NonlinConfig, forward_image_cfg
from tools.p2.scale_table import ScaleTable


def load_selectors(path, model, device, table):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Merge the calibrated selector scale entries into the table (the
    # checkpoint carries them; the backbone entries come from --table).
    if "weight_exps" in payload:
        for i in (1, 2, 3):
            for name, exp in payload["weight_exps"][str(i)].items():
                table.weights[f"s{i}_{name}"] = exp
            for name, exp in payload["sel_acts"][str(i)].items():
                table.activations[f"s{i}_{name}"] = exp
        table.validate()
    for i, sel in enumerate(payload["selectors"]):
        model.selectors[i] = SelectorP(
            local_w=sel["local_w"].to(device),
            local_b=sel["local_b"].to(device),
            score_w1=sel["score_w1"].to(device),
            score_b1=sel["score_b1"].to(device),
            score_w2=sel["score_w2"].to(device),
            score_b2=sel["score_b2"].to(device),
            score_w3=sel["score_w3"].to(device),
            score_b3=sel["score_b3"].to(device),
            hw_w1=sel["hw_w1"].to(device),
            hw_b1=sel["hw_b1"].to(device),
            hw_w2=sel["hw_w2"].to(device),
            hw_b2=sel["hw_b2"].to(device),
        )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", required=True,
                        help="QAT checkpoint with a 'floats' key "
                             "(p2_out/qat/init.pt or a train output)")
    parser.add_argument("--selectors", default="p2_out/selectors_sup4.pt")
    parser.add_argument("--table", default="p2_out/scale_table.json")
    parser.add_argument("--images", type=int, default=5000)
    parser.add_argument("--sampling", choices=("head", "random", "stratified"),
                        default="head",
                        help="validation subset selection (head preserves "
                             "historical first-N results)")
    parser.add_argument("--seed", type=int, default=20260815,
                        help="subset seed for random/stratified sampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    ckpt = torch.load(REPO_ROOT / args.checkpoint, map_location="cpu",
                      weights_only=False)
    floats = ckpt["floats"]
    table = ScaleTable.load(REPO_ROOT / args.table).validate()
    model = build_model(floats, table, device)
    load_selectors(REPO_ROOT / args.selectors, model, device, table)

    loader = make_val_loader(args.images, sampling=args.sampling,
                             seed=args.seed)
    correct = total = 0
    counts = [0, 0, 0]
    t0 = time.time()
    with torch.no_grad():
        for img, label in loader:
            logits, token_counts, _ = forward_image_cfg(
                model, img[0].to(device), NonlinConfig(), prune=True)
            correct += (logits.argmax().item() == label.item())
            total += 1
            for k in range(3):
                counts[k] += token_counts[k + 1]
            if total % 500 == 0:
                print(f"  {total}/{args.images} "
                      f"acc={100.0 * correct / total:.2f}% "
                      f"counts={[round(c / total, 1) for c in counts]}",
                      flush=True)
    elapsed = time.time() - t0
    acc = 100.0 * correct / total
    print(f"pruned Top-1 on {total} val images: {acc:.2f}% "
          f"({elapsed:.0f}s)")
    print(f"mean token counts per stage: {counts[0] / total:.1f} / "
          f"{counts[1] / total:.1f} / {counts[2] / total:.1f} "
          f"(target 88 / 45 / 32)")
    print(f"checkpoint={args.checkpoint} selectors={args.selectors}")


if __name__ == "__main__":
    main()
