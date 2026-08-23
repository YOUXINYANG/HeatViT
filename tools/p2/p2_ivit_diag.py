#!/usr/bin/env python3
"""Per-block error attribution for an I-ViT fusion config.

Loads a scale table produced by tools/p2/p2_ivit.py (legacy or relax
variant), builds the quantized model under the given NonlinConfig, runs
one image with activation recording, and compares per-block activations
against the float timm model (max abs error + correlation), plus the
top-1 agreement. Mirrors tools/p2/p2_diag.py for the I-ViT variants.

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p2_ivit_diag.py --table p2_out/ivit/scale_table_relax.json --cfg i-vit
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_ivit import CONFIGS, build_model, eval_top1
from tools.p2.p2_quantize import (
    load_state_dict,
    make_val_loader,
    to_heatvit_tensors,
)
from tools.p2.p2_sim_ivit import NonlinConfig, forward_image_cfg
from tools.p2.scale_table import ScaleTable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--table", required=True)
    parser.add_argument("--cfg", default="i-vit", choices=sorted(CONFIGS))
    parser.add_argument("--images", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    cfg_kwargs, _kind, use_chan = CONFIGS[args.cfg]
    cfg = NonlinConfig(**cfg_kwargs)

    state = load_state_dict()
    floats = to_heatvit_tensors(state)
    table = ScaleTable.load(REPO_ROOT / args.table).validate()
    model = build_model(floats, table, device, use_chan)

    import timm
    fm = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    fm.load_state_dict(state, strict=True)
    fm = fm.to(device).eval()

    loader = make_val_loader(args.images, batch_size=8)
    images = [img for img, _ in loader]

    img = images[0][0].to(device)
    rec = {}
    with torch.no_grad():
        logits, counts, logit_exp = forward_image_cfg(
            model, img, cfg, rec, prune=False)
        fl = fm(img.unsqueeze(0))
    ql = logits.float() * (2.0 ** logit_exp)
    print(f"float top1 = {fl[0].argmax().item()}  quant top1 = "
          f"{ql.argmax().item()}")
    print(f"logits corr = {torch.corrcoef(torch.stack([fl[0].cpu(), ql.cpu()]))[0, 1].item():.4f}")

    cap = {}
    for i, blk in enumerate(fm.blocks):
        def h(_, a, o, idx=i):
            cap[f"b{idx + 1}"] = o.detach().clone()
        blk.register_forward_hook(h)
    with torch.no_grad():
        fm(img.unsqueeze(0))

    print(f"{'tensor':18s} {'maxerr':>9s} {'corr':>7s} {'exp':>4s}")
    for n in range(1, 13):
        f = cap[f"b{n}"][0].float().cpu()
        q = rec[f"b{n}_out"].float() * (2.0 ** table.activations[f"b{n}_out"])
        err = (f - q).abs().max().item()
        corr = torch.corrcoef(torch.stack(
            [f.reshape(-1), q.reshape(-1)]))[0, 1].item()
        print(f"b{n:02d}_out           {err:9.3f} {corr:7.4f} "
              f"{table.activations[f'b{n}_out']:4d}")
    for name in ("b4_ln2_out", "b4_hidden", "b12_ln1_out", "b12_qkv_out",
                 "b12_context_out", "final_ln_out"):
        if name not in rec:
            continue
        q = rec[name].float() * (2.0 ** table.activations[name])
        print(f"{name:18s} quant maxabs={q.abs().max().item():9.3f} "
              f"exp={table.activations[name]:4d}")

    acc, elapsed, correct, total = eval_top1(
        model, cfg, make_val_loader(256, batch_size=32), device)
    print(f"cfg={args.cfg} Top-1 on first 256 val images: {acc:.2f}% "
          f"({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
