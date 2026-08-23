#!/usr/bin/env python3
"""Evaluate trained Token Selectors in the exact integer simulator.

Loads the selectors exported by tools/p2/p2_train_selector.py, plugs them
into the contract-faithful QuantDeiT simulator and measures pruned Top-1
with per-stage token counts (RTL semantics: threshold 0.5 inclusive,
package token weighted by keep scores).

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p2_selector_eval.py \
      --selectors p2_out/selectors_newgelu.pt \
      --table p2_out/ivit/scale_table_legacy.json --eval 3000
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
from tools.p2.p2_sim import SelectorP, forward_image
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selectors", required=True)
    parser.add_argument("--table", default="p2_out/scale_table.json")
    parser.add_argument("--eval", type=int, default=3000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    state = load_state_dict()
    floats = to_heatvit_tensors(state)
    table = ScaleTable.load(REPO_ROOT / args.table).validate()
    model = build_model(floats, table, device)
    load_selectors(REPO_ROOT / args.selectors, model, device, table)

    loader = make_val_loader(args.eval)
    correct = total = 0
    counts = [0, 0, 0]
    t0 = time.time()
    with torch.no_grad():
        for img, label in loader:
            logits, token_counts, _ = forward_image(
                model, img[0].to(device), prune=True)
            correct += (logits.argmax().item() == label.item())
            total += 1
            for k in range(3):
                counts[k] += token_counts[k + 1]
            if total % 500 == 0:
                print(f"  {total}/{args.eval} acc={100.0 * correct / total:.2f}% "
                      f"counts={[c / total for c in counts]}", flush=True)
    elapsed = time.time() - t0
    acc = 100.0 * correct / total
    print(f"pruned Top-1 on {total} val images: {acc:.2f}% "
          f"({elapsed:.0f}s)")
    print(f"mean token counts per stage: "
          f"{counts[0] / total:.1f} / {counts[1] / total:.1f} / "
          f"{counts[2] / total:.1f} (target 88 / 45 / 32)")


if __name__ == "__main__":
    main()
