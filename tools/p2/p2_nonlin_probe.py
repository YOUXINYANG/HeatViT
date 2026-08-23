#!/usr/bin/env python3
"""Nonlinearity quality probe: contract vs I-ViT vs float reference.

Hooks the float timm DeiT-T over a few images and collects
  * attention logits (QK^T / sqrt(64), per head, per block),
  * FFN pre-GELU hidden values (mlp.fc1 output),
then compares the integer softmax/GELU candidates against the float
reference:
  contract  = HeatViT golden contract (quadratic exp approx / erf poly)
  shiftmax  = I-ViT Shiftmax (linear 2^x approx, slope 1/2)
  shiftmax-ln2 = same with slope 11/16 (ln2)
  shiftgelu  = I-ViT ShiftGELU (x * sigmoid(1.702x) via shift exp)
  shiftgelu-ln2 = same with slope 11/16

Metrics: mean |err| and KL divergence (softmax), mean |err| / mean
relative err (GELU). Runs fully in torch; no ImageNet labels needed.

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p2_nonlin_probe.py --images 32
"""

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_quantize import load_state_dict, make_val_loader

LN2_Q16 = 45426
QUAD_Q16 = 23495
OFFSET_Q16 = 88670
CONST_Q16 = 22544
INV_SQRT2_Q16 = 46341
GELU_A_Q16 = -18927
GELU_B_Q16 = -115933
GELU_DELTA_Q16 = 32768


def rsa(x, shift):
    if shift == 0:
        return x
    if shift < 0:
        return x << (-shift)
    mag = x.abs()
    r = (mag + (1 << (shift - 1))) >> shift
    return torch.where(x < 0, -r, r)


def contract_softmax_row(q16):
    """Golden-contract attention softmax on one row of Q16 logits."""
    m = q16.max()
    x = q16 - m
    z = (-x) // LN2_Q16
    p = x + z * LN2_Q16
    sq = rsa((p + OFFSET_Q16) ** 2, 16)
    e = rsa(QUAD_Q16 * sq, 16) + CONST_Q16
    e = e >> z
    s = e.sum()
    recip = torch.div(1 << 32, s, rounding_mode="trunc")
    recip = torch.where((2 * ((1 << 32) % s)) >= s, recip + 1, recip)
    ratio = rsa(e * recip, 16)
    return rsa(ratio, 8).clamp(0, 255).to(torch.uint8)


def shiftmax_row(q16, slope="half"):
    """I-ViT Shiftmax on one row of Q16 logits; returns UQ0.8 probs."""
    m = q16.max()
    i_d = q16 - m                      # <= 0
    i_p = i_d + (i_d >> 1) - (i_d >> 4)  # * log2(e) ~ 1.4375
    i0 = 65536
    q = (-i_p) // i0
    r = (-i_p) - q * i0               # in [0, i0)
    if slope == "half":
        frac = (r + 1) >> 1           # ceil(r/2); 2^x ~ 1 + x/2
    else:
        frac = (r * 11 + 15) >> 4     # 11/16 ~ ln2
    i_b = (i0 - frac).clamp(min=1)
    e = torch.where(q <= 16, i_b >> q, torch.zeros_like(i_b))
    s = e.sum()
    recip = torch.div(1 << 32, s, rounding_mode="trunc")
    recip = torch.where((2 * ((1 << 32) % s)) >= s, recip + 1, recip)
    ratio = rsa(e * recip, 16)
    return rsa(ratio, 8).clamp(0, 255).to(torch.uint8)


def contract_gelu_row(x_q16):
    u = rsa(x_q16 * INV_SQRT2_Q16, 16)
    clip = u.abs().clamp(max=-GELU_B_Q16)
    t = clip + GELU_B_Q16
    t2 = rsa(t * t, 16)
    poly = rsa(GELU_A_Q16 * t2, 16) + 65536
    erf_mag = rsa(GELU_DELTA_Q16 * poly, 16)
    sign = (u > 0).to(torch.int64) - (u < 0).to(torch.int64)
    l_erf = sign * erf_mag
    y = rsa(x_q16 * (65536 + l_erf), 17)
    return y.clamp(-(1 << 23), (1 << 23) - 1)


def shift_exp(n, slope):
    """2^(n*2^-16) in Q16 for arbitrary signed int64 n (sat 24 bits)."""
    i0 = 65536
    neg = n < 0
    a = n.abs()
    q = a // i0
    r = a - q * i0
    if slope == "half":
        frac = (r + 1) >> 1          # slope 1/2
    else:
        frac = (r * 11 + 15) >> 4    # slope 11/16 ~ ln2
    i_b_neg = (i0 - frac).clamp(min=1)
    e_neg = torch.where(q <= 16, i_b_neg >> q, torch.zeros_like(i_b_neg))
    i_b_pos = i0 + frac
    e_pos = torch.where(q <= 7, i_b_pos << q,
                        torch.full_like(i_b_pos, 1 << 23))
    e = torch.where(neg, e_neg, e_pos)
    return e.clamp(0, (1 << 23) - 1)


def shiftgelu_row(x_q16, slope="half"):
    """I-ViT ShiftGELU (robust per-element form): x*sigmoid(1.702x)."""
    i_p = x_q16 + (x_q16 >> 1) + (x_q16 >> 3) + (x_q16 >> 4)  # *1.6875
    i_p2 = i_p + (i_p >> 1) - (i_p >> 4)                       # * log2 e
    e = shift_exp(i_p2, slope)
    one = 65536
    den = (one + e).clamp(min=1)
    sig = (((e.to(torch.int64) << 16) + (den >> 1)) // den).clamp(0, 65536)
    y = rsa(x_q16.to(torch.int64) * sig, 16)
    return y.clamp(-(1 << 23), (1 << 23) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--images", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")

    state = load_state_dict()
    import timm
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    loader = make_val_loader(args.images, batch_size=8)
    images = [img for img, _ in loader]

    logits_by_block = {n: [] for n in range(1, 13)}
    hidden_by_block = {n: [] for n in range(1, 13)}
    hooks = []
    for n, blk in enumerate(model.blocks, start=1):
        def qkv_hook(m, a, o, nn=n):
            b = o.shape[0]
            q, k, v = o.reshape(b, 197, 3, 3, 64).permute(2, 3, 0, 1, 4)
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(64.0)
            logits_by_block[nn].append(logits.detach().cpu().reshape(-1, 197))
        hooks.append(blk.attn.qkv.register_forward_hook(qkv_hook))

        def fc1_hook(m, a, o, nn=n):
            hidden_by_block[nn].append(o.detach().cpu().reshape(-1))
        hooks.append(blk.mlp.fc1.register_forward_hook(fc1_hook))

    try:
        with torch.no_grad():
            for img in images:
                model(img.to(device))
    finally:
        for h in hooks:
            h.remove()

    print("=== attention softmax (Q16 logits -> UQ0.8 probs) ===")
    print(f"{'method':14s} {'mean|err|':>10s} {'max|err|':>10s} {'meanKL':>10s}")
    sm_err = {"contract": [], "shiftmax": [], "shiftmax-ln2": []}
    sm_kl = {"contract": [], "shiftmax": [], "shiftmax-ln2": []}
    for n in range(1, 13):
        rows = torch.cat(logits_by_block[n], dim=0)   # [B*3, 197]
        q16 = (rows * 65536.0).round().clamp(-(1 << 23), (1 << 23) - 1).to(
            torch.int64)
        ref = torch.softmax(rows, dim=-1)             # float prob
        for name in ("contract", "shiftmax", "shiftmax-ln2"):
            if name == "contract":
                out = torch.stack([contract_softmax_row(r) for r in q16])
            elif name == "shiftmax":
                out = torch.stack([shiftmax_row(r, "half") for r in q16])
            else:
                out = torch.stack([shiftmax_row(r, "ln2") for r in q16])
            p = out.float() / 256.0
            err = (p - ref).abs().mean().item()
            kl = (ref * ((ref + 1e-9) / (p + 1e-9)).log()).sum(-1).mean().item()
            sm_err[name].append(err)
            sm_kl[name].append(kl)
    for name in ("contract", "shiftmax", "shiftmax-ln2"):
        print(f"{name:14s} {sum(sm_err[name])/12:10.6f} "
              f"{max(sm_err[name]):10.6f} {sum(sm_kl[name])/12:10.6f}")

    print("\n=== GELU (Q16 in -> Q16 out) ===")
    print(f"{'method':14s} {'mean|err|Q16':>12s} {'max|err|Q16':>12s} "
          f"{'meanrel%':>10s}")
    ge_err = {"contract": [], "shiftgelu": [], "shiftgelu-ln2": []}
    ge_rel = {"contract": [], "shiftgelu": [], "shiftgelu-ln2": []}
    for n in range(1, 13):
        vals = torch.cat(hidden_by_block[n], dim=0)
        q16 = (vals * 65536.0).round().clamp(-(1 << 23), (1 << 23) - 1).to(
            torch.int64)
        ref = torch.nn.functional.gelu(vals, approximate="none") * 65536.0
        for name in ("contract", "shiftgelu", "shiftgelu-ln2"):
            if name == "contract":
                out = contract_gelu_row(q16).float()
            elif name == "shiftgelu":
                out = shiftgelu_row(q16, "half").float()
            else:
                out = shiftgelu_row(q16, "ln2").float()
            err = (out - ref).abs().mean().item()
            rel = ((out - ref).abs() / (ref.abs() + 1e-3)).mean().item() * 100
            ge_err[name].append(err)
            ge_rel[name].append(rel)
    for name in ("contract", "shiftgelu", "shiftgelu-ln2"):
        print(f"{name:14s} {sum(ge_err[name])/12:12.1f} "
              f"{max(ge_err[name]):12.1f} {sum(ge_rel[name])/12:10.4f}")


if __name__ == "__main__":
    main()
