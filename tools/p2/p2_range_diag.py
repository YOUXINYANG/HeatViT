#!/usr/bin/env python3
"""Diagnostic: float activation ranges for the PTQ scale-table names.

Hooks the float timm DeiT-T and reports, per named activation tensor:
max_abs, p99.9 abs, std, and the int8 power-of-2 exp that would be needed
to cover max_abs (ceil(log2(max_abs/127))). Used to quantify how much the
LayerNorm input contract clamp (exp <= 0, range +-127) clips the real
residual streams, and to pick a sane upper bound for the relaxed I-ViT
variant (I-LayerNorm with wider input scales).

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p2_range_diag.py --images 128
"""

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_quantize import (
    load_state_dict,
    make_val_loader,
)
from tools.p2.scale_table import ACTIVATION_NAMES


def collect_stats(state, images, device):
    import timm
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    acc = {name: None for name in ACTIVATION_NAMES}
    block_inputs = {}

    def upd(name, tensor):
        v = tensor.detach().float().cpu()
        mx = v.abs().max().item()
        q999 = v.abs().flatten().kthvalue(
            max(1, int(v.numel() * 0.999))).values.item()
        std = v.std().item()
        cur = acc[name]
        if cur is None:
            acc[name] = [mx, q999, std]
        else:
            acc[name][0] = max(cur[0], mx)
            acc[name][1] = max(cur[1], q999)
            acc[name][2] = max(cur[2], std)

    hooks = []
    hooks.append(model.patch_embed.register_forward_hook(
        lambda m, a, o: upd("act_patch_embed", o)))

    def patch_hook(m, a, o):
        upd("act_patch_embed", o)
        pos = model.pos_embed
        cls_tok = model.cls_token
        batch = o.shape[0]
        tokens0 = cls_tok.expand(batch, -1, -1) + pos[:, :1]
        tokens1 = o + pos[:, 1:]
        upd("act_tokens", torch.cat([tokens0, tokens1], dim=1))
    hooks.append(model.patch_embed.register_forward_hook(patch_hook))

    for n, blk in enumerate(model.blocks, start=1):
        def blk_pre(m, a, idx=n - 1):
            block_inputs[idx] = a[0].detach()

        def y_hook(m, a, o, idx=n - 1, nn=n):
            upd(f"b{nn}_msa_out", o)
            x = block_inputs.get(idx)
            if x is not None:
                upd(f"b{nn}_y", x + o)

        hooks.append(blk.register_forward_pre_hook(blk_pre))
        hooks.append(blk.attn.proj.register_forward_hook(y_hook))
        hooks.append(blk.norm1.register_forward_hook(
            lambda m, a, o, nn=n: upd(f"b{nn}_ln1_out", o)))
        hooks.append(blk.attn.qkv.register_forward_hook(
            lambda m, a, o, nn=n: upd(f"b{nn}_qkv_out", o)))
        hooks.append(blk.attn.proj.register_forward_pre_hook(
            lambda m, a, nn=n: upd(f"b{nn}_context_out", a[0])))
        hooks.append(blk.norm2.register_forward_hook(
            lambda m, a, o, nn=n: upd(f"b{nn}_ln2_out", o)))
        hooks.append(blk.mlp.act.register_forward_hook(
            lambda m, a, o, nn=n: upd(f"b{nn}_hidden", o)))
        hooks.append(blk.mlp.fc2.register_forward_hook(
            lambda m, a, o, nn=n: upd(f"b{nn}_ffn_out", o)))
        hooks.append(blk.register_forward_hook(
            lambda m, a, o, nn=n: upd(f"b{nn}_out", o)))
    hooks.append(model.norm.register_forward_hook(
        lambda m, a, o: upd("final_ln_out", o)))

    try:
        with torch.no_grad():
            for img in images:
                upd("input", img)
                upd("act_patch_matrix", img)
                model(img.to(device))
    finally:
        for h in hooks:
            h.remove()
    return acc


def required_exp(max_abs):
    if max_abs <= 0:
        return None
    return max(-32, min(31, math.ceil(math.log2(max_abs / 127.0))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--images", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")

    state = load_state_dict()
    loader = make_val_loader(args.images, batch_size=32)
    images = [img for img, _ in loader]
    print(f"collecting float stats on {len(images)} images ...")
    stats = collect_stats(state, images, device)

    print(f"{'name':20s} {'max_abs':>10s} {'p99.9':>10s} "
          f"{'std':>8s} {'req_exp':>8s}  clip@exp<=0")
    rows = []
    for name in ACTIVATION_NAMES:
        cur = stats.get(name)
        if cur is None or cur[0] <= 0:
            continue
        mx, q999, std = cur
        rexp = required_exp(mx)
        clip = "YES" if (rexp is not None and rexp > 0) else ""
        rows.append((name, mx, q999, std, rexp, clip))
    for name, mx, q999, std, rexp, clip in rows:
        print(f"{name:20s} {mx:10.3f} {q999:10.3f} {std:8.3f} "
              f"{str(rexp):>8s}  {clip}")


if __name__ == "__main__":
    main()
