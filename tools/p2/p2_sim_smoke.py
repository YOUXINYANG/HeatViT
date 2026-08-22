#!/usr/bin/env python3
"""Smoke test: P2 simulator vs the pure-integer golden model.

Builds the QuantDeiT simulator from the deterministic synthetic weights
(verification.heatvit_ref.weights.build_params) with the uniform scale table
from config/heatvit_t.json, runs one full image on GPU and asserts bitwise
equality of logits, token counts and key activation checkpoints against the
golden model (which is itself bit-exact versus XSim).

Usage (torch venv): .venv-torch\\Scripts\\python tools/p2/p2_sim_smoke.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.generate_e2e_vectors import det_image
from tools.p2.p2_sim import BlockP, QuantDeiT, SelectorP, forward_image
from tools.p2.scale_table import (
    ACTIVATION_NAMES,
    WEIGHT_NAMES,
    ScaleTable,
)
from verification.heatvit_ref.model import HeatViTModel
from verification.heatvit_ref.weights import SEED, build_params


def build_uniform_table():
    """Mirror config/heatvit_t.json synthetic_scale_exp values."""
    weights = {}
    for name in WEIGHT_NAMES:
        if name.endswith(("_gamma1", "_gamma2")) or name == "final_gamma":
            weights[name] = -6
        else:
            weights[name] = -7
    activations = {name: -7 for name in ACTIVATION_NAMES}
    return ScaleTable(weights=weights, activations=activations)


def _m(rows):
    return torch.tensor([list(r) for r in rows], dtype=torch.int8)


def _v(values):
    return torch.tensor(list(values), dtype=torch.int32)


def build_sim_model(gp):
    blocks = []
    for b in gp.blocks:
        blocks.append(BlockP(
            gamma1=_v(b.mhsa.ln_gamma), beta1=_v(b.mhsa.ln_beta),
            wqkv=_m(b.mhsa.wqkv), bqkv=_v(b.mhsa.bqkv),
            wproj=_m(b.mhsa.wproj), bproj=_v(b.mhsa.bproj),
            gamma2=_v(b.ffn.ln_gamma), beta2=_v(b.ffn.ln_beta),
            w1=_m(b.ffn.w1), b1=_v(b.ffn.b1),
            w2=_m(b.ffn.w2), b2=_v(b.ffn.b2),
        ))
    selectors = []
    for sel in gp.selectors:
        selectors.append(SelectorP(
            local_w=torch.stack([_m(sel.local.w[h]) for h in range(3)]),
            local_b=torch.stack([_v(sel.local.b[h]) for h in range(3)]),
            score_w1=torch.stack([_m(sel.score.w1[h]) for h in range(3)]),
            score_b1=torch.stack([_v(sel.score.b1[h]) for h in range(3)]),
            score_w2=torch.stack([_m(sel.score.w2[h]) for h in range(3)]),
            score_b2=torch.stack([_v(sel.score.b2[h]) for h in range(3)]),
            score_w3=torch.stack([_m(sel.score.w3[h]) for h in range(3)]),
            score_b3=torch.stack([_v(sel.score.b3[h]) for h in range(3)]),
            hw_w1=_m(sel.head_weight.w1), hw_b1=_v(sel.head_weight.b1),
            hw_w2=_m(sel.head_weight.w2), hw_b2=_v(sel.head_weight.b2),
        ))
    return QuantDeiT(
        scales=build_uniform_table(),
        patch_w=_m(gp.patch.patch_weight), patch_b=_v(gp.patch.patch_bias),
        cls=_v(gp.patch.cls), pos=_m(gp.patch.pos),
        blocks=blocks, selectors=selectors,
        final_gamma=_v(gp.final_gamma), final_beta=_v(gp.final_beta),
        head_w=_m(gp.head_w), head_b=_v(gp.head_b),
    )


def main():
    image = det_image(SEED)                      # flat NHWC int8 list
    gp, summary, token_counts = build_params(image)
    result = HeatViTModel().infer(image, gp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sim_model(gp).to(device)

    # image float CHW at input scale 2^-7; sim requantizes it back exactly
    img_float = (torch.tensor(image, dtype=torch.float32)
                 .reshape(224, 224, 3).permute(2, 0, 1) * (2.0 ** -7))
    rec = {}
    logits, sim_counts, logit_exp = forward_image(
        model, img_float.to(device), rec)

    assert logit_exp == -14, logit_exp
    assert sim_counts == token_counts, (sim_counts, token_counts)
    golden_logits = torch.tensor(list(result.logits), dtype=torch.int32)
    assert torch.equal(logits.cpu(), golden_logits), \
        f"logits mismatch: {logits[:5]} vs {golden_logits[:5]}"

    checkpoint_map = {
        "patch": "act_tokens",
        "block_01": "b1_out",
        "block_04": "b4_out",
        "block_12": "b12_out",
        "final_ln": "final_ln_out",
    }
    for golden_name, rec_name in checkpoint_map.items():
        g = torch.tensor([list(r) for r in result.checkpoints[golden_name]],
                         dtype=torch.int8)
        assert torch.equal(rec[rec_name].to(torch.int8), g), \
            f"checkpoint {golden_name} ({rec_name}) mismatch"

    print(f"SMOKE PASS on {device}: logits bit-exact, token counts "
          f"{token_counts}, {len(checkpoint_map)} checkpoints bit-exact")


if __name__ == "__main__":
    main()
