#!/usr/bin/env python3
"""Throwaway probe: measure train-path vs integer-contract deviations.

Measures:
  1. shiftgelu_float vs gelu_q16 (RTL ShiftGELU-ln2) over the Q8.16 grid
  2. contract softmax (quadratic exp approx) vs exact softmax
  3. QatDeiT train forward vs exact_forward, per named boundary
     (bin distance on the contract grid, on random small-model weights)
Run: .venv-torch\\Scripts\\python tools/p2/qat_probe.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_sim import gelu_q16, softmax_attention
from tools.p2.qat_fakeq import shiftgelu_float
from tools.p2.qat_model import QatDeiT, exact_forward
from tools.p2.scale_table import ACTIVATION_NAMES, WEIGHT_NAMES, ScaleTable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={DEVICE}")


def probe_gelu():
    xs = (torch.linspace(-8.0, 8.0, 8193) * 65536).round().clamp(
        -(1 << 23), (1 << 23) - 1).to(torch.int64)
    ref = gelu_q16(xs).to(torch.float64) / 65536.0
    got = shiftgelu_float(xs.to(torch.float64) / 65536.0).double()
    err = (got - ref).abs()
    print(f"[gelu]   mean|err|={err.mean():.5f}  max|err|={err.max():.5f}")
    return err.mean().item(), err.max().item()


def probe_softmax():
    rows = (torch.randn(64, 197) * 8 * 65536).round().clamp(
        -(1 << 23), (1 << 23) - 1).to(torch.int64)
    ref = softmax_attention(rows).to(torch.float64) / 256.0
    exact = torch.softmax(rows.to(torch.float64) / 65536.0, dim=-1)
    err = (exact - ref).abs()
    print(f"[softmax] mean|err|={err.mean():.6f}  max|err|={err.max():.6f}")


def probe_wiring():
    torch.manual_seed(20260815)
    table = ScaleTable(weights={n: -7 for n in WEIGHT_NAMES},
                       activations={n: -7 for n in ACTIVATION_NAMES})

    def r(*shape, std=0.05):
        return torch.randn(*shape) * std

    floats = {
        "patch_w": r(768, 192, std=0.04), "patch_b": r(192, std=0.1),
        "cls": r(192, std=0.05), "pos": r(197, 192, std=0.05),
    }
    for n in range(1, 13):
        floats[f"b{n}_gamma1"] = 0.8 + r(192, std=0.05)
        floats[f"b{n}_beta1"] = r(192, std=0.05)
        floats[f"b{n}_wqkv"] = r(192, 576, std=0.04)
        floats[f"b{n}_bqkv"] = r(576, std=0.1)
        floats[f"b{n}_wproj"] = r(192, 192, std=0.04)
        floats[f"b{n}_bproj"] = r(192, std=0.1)
        floats[f"b{n}_gamma2"] = 0.8 + r(192, std=0.05)
        floats[f"b{n}_beta2"] = r(192, std=0.05)
        floats[f"b{n}_w1"] = r(192, 768, std=0.04)
        floats[f"b{n}_b1"] = r(768, std=0.1)
        floats[f"b{n}_w2"] = r(768, 192, std=0.04)
        floats[f"b{n}_b2"] = r(192, std=0.1)
    floats["final_gamma"] = 0.8 + r(192, std=0.05)
    floats["final_beta"] = r(192, std=0.05)
    floats["head_w"] = r(192, 1000, std=0.04)
    floats["head_b"] = r(1000, std=0.1)

    qat = QatDeiT(floats, table).to(DEVICE)
    images = (torch.rand(2, 3, 224, 224) * 0.9 - 0.45).to(DEVICE)

    with torch.no_grad():
        rec_t = {}
        logits_t = qat(images, rec_t)
        rec_e = {}
        logits_e, scale_e = exact_forward(qat, images, rec_e)

    print(f"[wiring] logits exact {logits_e.shape} @ {scale_e}, "
          f"float {logits_t.shape}")
    rel = ((logits_t.cpu() - logits_e.cpu().to(torch.float32)
            * (2.0 ** scale_e)).abs()
           / (logits_e.cpu().abs().to(torch.float32) * (2.0 ** scale_e)
              + 1.0))
    print(f"[wiring] logits mean|rel|={rel.mean():.4f} max={rel.max():.4f}")

    for name, t in rec_t.items():
        if name not in rec_e:
            print(f"[wiring] {name:16s} train-only")
            continue
        e = rec_e[name]
        exp = table.activations.get(name)
        if exp is None:
            continue
        e_deq = e.to(torch.float32) * (2.0 ** exp)
        step = 2.0 ** exp
        bins = ((t - e_deq).abs() / step)
        print(f"[wiring] {name:16s} mean_bins={bins.mean():.4f} "
              f"max_bins={bins.max():.4f}")


if __name__ == "__main__":
    probe_gelu()
    probe_softmax()
    probe_wiring()
