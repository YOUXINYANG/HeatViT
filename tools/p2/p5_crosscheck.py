#!/usr/bin/env python3
"""P5 cross-check: verification golden model vs p2_sim_ivit, bit-exact.

Runs the pure-integer verification golden model (HeatViTModel.infer with
the per-tensor selector scales) and the torch deployment simulator
(p2_sim_ivit.forward_image_cfg — the accuracy authority behind the 5k
eval numbers) on the same val images and asserts every shared checkpoint,
the selector local activations and the 1000 logits agree byte-for-byte.

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p5_crosscheck.py \\
      --checkpoint p2_out/qat/p4a_rate5_16k/best.pt \\
      --selectors p2_out/selectors_sup4.pt --images 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tools.p2.p2_export_weights import (
    golden_params_from_real,
    merge_selector_scales,
    real_image_list,
)
from tools.p2.p2_quantize import build_model, make_val_loader
from tools.p2.p2_sim import SelectorP
from tools.p2.p2_sim_ivit import NonlinConfig, forward_image_cfg
from tools.p2.scale_table import ScaleTable
from verification.heatvit_ref.model import HeatViTModel


def load_selectors(path, model, table):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    merge_selector_scales(table, payload)
    for i, sel in enumerate(payload["selectors"]):
        model.selectors[i] = SelectorP(
            local_w=sel["local_w"], local_b=sel["local_b"],
            score_w1=sel["score_w1"], score_b1=sel["score_b1"],
            score_w2=sel["score_w2"], score_b2=sel["score_b2"],
            score_w3=sel["score_w3"], score_b3=sel["score_b3"],
            hw_w1=sel["hw_w1"], hw_b1=sel["hw_b1"],
            hw_w2=sel["hw_w2"], hw_b2=sel["hw_b2"])
    return model


def flat(t):
    return t.reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",
                        default="p2_out/qat/p4a_rate5_16k/best.pt")
    parser.add_argument("--selectors", default="p2_out/selectors_sup4.pt")
    parser.add_argument("--table", default="p2_out/scale_table.json")
    parser.add_argument("--images", type=int, default=3)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu",
                      weights_only=False)
    floats = {k: v for k, v in ckpt["floats"].items()}
    table = ScaleTable.load(args.table).validate()
    payload = torch.load(args.selectors, map_location="cpu",
                         weights_only=False)
    merge_selector_scales(table, payload)

    model = build_model(floats, table, torch.device("cpu"))
    load_selectors(args.selectors, model, table)

    params = golden_params_from_real(floats, table, payload)
    gm = HeatViTModel()

    shared = {"patch": "act_tokens"}
    shared.update({f"block_{n:02d}": f"b{n}_out" for n in range(1, 13)})
    shared["final_ln"] = "final_ln_out"

    loader = make_val_loader(args.images)
    all_ok = True
    for idx, (img, _label) in enumerate(loader):
        image = real_image_list(img[0], table)
        res = gm.infer(image, params)
        rec = {}
        logits, counts, _ = forward_image_cfg(model, img[0], NonlinConfig(),
                                              rec=rec, prune=True)
        bad = []
        if flat(logits) != list(res.logits):
            bad.append("logits")
        expected_counts = [197] + [e["output_tokens"]
                                   for e in res.selector_summary]
        if list(counts) != expected_counts:
            bad.append(f"token_counts ({list(counts)} vs {expected_counts})")
        for name, key in shared.items():
            if flat(rec[key]) != [v for row in res.checkpoints[name]
                                  for v in row]:
                bad.append(name)
        if bad:
            all_ok = False
            print(f"img{idx}: MISMATCH in {bad}")
        else:
            print(f"img{idx}: bit-exact OK "
                  f"(counts={counts[1:]}, top1_argmax="
                  f"{int(logits.argmax())})")
    print("CROSSCHECK " + ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
