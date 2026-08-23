#!/usr/bin/env python3
"""I-ViT fusion PTQ driver: calibration + ablation matrix evaluation.

Fuses the I-ViT integer-only methods (Shiftmax, ShiftGELU, I-LayerNorm
with relaxed input scales; plus per-channel weight exponents as a
contract-extension axis) into the HeatViT P2-B PTQ pipeline and measures
Top-1 accuracy per configuration against the current contract baseline
(~0.82% per README P2, 5k val).

Scale tables: the legacy histogram grid (+-64, 4096 bins) calibrates the
contract LayerNorm input clamp (exp <= 0); the wide grid (+-512, 16384
bins) calibrates the relaxed I-ViT variant (LN input exp <= 6). Both come
from the same float model hooks, so one ``--calib`` run of each suffices
for the whole matrix.

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p2_ivit.py --calib 2048 --eval 5000
  .venv-torch\\Scripts\\python tools/p2/p2_ivit.py --unit

Artifacts: p2_out/ivit/results.json and p2_out/ivit/scale_table_*.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_quantize import (
    collect_float_histograms,
    load_state_dict,
    make_val_loader,
    pick_activation_exps,
    quantize_bias,
    quantize_int8,
    to_heatvit_tensors,
    build_weight_exps,
)
from tools.p2.p2_sim import (
    BlockP,
    QuantDeiT,
    SelectorP,
    forward_image,
    layernorm,
)
from tools.p2.p2_sim_ivit import (
    NonlinConfig,
    forward_batch_cfg,
    forward_image_cfg,
    gemm_int8_chan,
    shiftgelu_q16,
    shiftmax_attention,
)
from tools.p2.scale_table import ScaleTable

# config name -> (NonlinConfig, table variant, uses channel weights)
CONFIGS = {
    "contract":     (dict(softmax="contract", gelu="contract", ln="contract",
                          wq="tensor"), "legacy", False),
    "shiftmax":     (dict(softmax="shiftmax", gelu="contract", ln="contract",
                          wq="tensor", shift_slope="half"), "legacy", False),
    "shiftmax-ln2": (dict(softmax="shiftmax", gelu="contract", ln="contract",
                          wq="tensor", shift_slope="ln2"), "legacy", False),
    "shiftgelu":    (dict(softmax="contract", gelu="shiftgelu",
                          ln="contract", wq="tensor",
                          shift_slope="half"), "legacy", False),
    "shiftgelu-ln2": (dict(softmax="contract", gelu="shiftgelu",
                           ln="contract", wq="tensor",
                           shift_slope="ln2"), "legacy", False),
    "plangelu":      (dict(softmax="contract", gelu="plan",
                           ln="contract", wq="tensor"), "legacy", False),
    "poly":          (dict(softmax="contract", gelu="poly",
                           ln="contract", wq="tensor"), "legacy", False),
    "i-vit-plan":    (dict(softmax="shiftmax", gelu="plan",
                           ln="newton", wq="tensor",
                           shift_slope="ln2"), "relax", False),
    "shift-both":   (dict(softmax="shiftmax", gelu="shiftgelu",
                          ln="contract", wq="tensor",
                          shift_slope="half"), "legacy", False),
    "relaxln":      (dict(softmax="contract", gelu="contract", ln="relaxed",
                          wq="tensor"), "relax", False),
    "newtonln":     (dict(softmax="contract", gelu="contract", ln="newton",
                          wq="tensor"), "relax", False),
    "i-vit":        (dict(softmax="shiftmax", gelu="shiftgelu",
                          ln="newton", wq="tensor",
                          shift_slope="half"), "relax", False),
    "i-vit-ln2":    (dict(softmax="shiftmax", gelu="shiftgelu",
                          ln="newton", wq="tensor",
                          shift_slope="ln2"), "relax", False),
    "chan":         (dict(softmax="contract", gelu="contract", ln="contract",
                          wq="channel"), "legacy", True),
    "i-vit-chan":   (dict(softmax="shiftmax", gelu="shiftgelu",
                          ln="newton", wq="channel",
                          shift_slope="half"), "relax", True),
}

CHANNEL_WEIGHTS = ("patch_w", *(f"b{n}_wqkv" for n in range(1, 13)),
                   *(f"b{n}_wproj" for n in range(1, 13)),
                   *(f"b{n}_w1" for n in range(1, 13)),
                   *(f"b{n}_w2" for n in range(1, 13)), "head_w")


def quantize_int8_chan(t):
    """Per-output-channel power-of-2 int8 quantization -> (q, exps)."""
    m = t.abs().amax(dim=0).clamp(min=1e-40)
    exps = torch.ceil(torch.log2(m / 127.0)).clamp(-32, 31).to(torch.int64)
    q = torch.clamp(torch.round(t / (2.0 ** exps.float())),
                    -128, 127).to(torch.int8)
    return q, exps


def quantize_bias_chan(t, a_exp, w_exps):
    step = 2.0 ** (a_exp + w_exps.float())
    return torch.clamp(torch.round(t / step), -(1 << 31),
                       (1 << 31) - 1).to(torch.int32)


def build_model(floats, table, device, use_chan):
    """Quantize weights/biases on the fly; per-channel or per-tensor."""
    s = table
    wchan = {} if use_chan else None
    blocks = []
    for n in range(1, 13):
        if use_chan:
            wqkv, e_qkv = quantize_int8_chan(floats[f"b{n}_wqkv"])
            wproj, e_proj = quantize_int8_chan(floats[f"b{n}_wproj"])
            w1, e_w1 = quantize_int8_chan(floats[f"b{n}_w1"])
            w2, e_w2 = quantize_int8_chan(floats[f"b{n}_w2"])
            wchan[f"b{n}_wqkv"] = e_qkv
            wchan[f"b{n}_wproj"] = e_proj
            wchan[f"b{n}_w1"] = e_w1
            wchan[f"b{n}_w2"] = e_w2
            a_ln1 = s.activation_exp(f"b{n}_ln1_out")
            a_ctx = s.activation_exp(f"b{n}_context_out")
            a_ln2 = s.activation_exp(f"b{n}_ln2_out")
            a_hid = s.activation_exp(f"b{n}_hidden")
            blocks.append(BlockP(
                gamma1=quantize_int8(floats[f"b{n}_gamma1"],
                                     s.weight_exp(f"b{n}_gamma1")),
                beta1=quantize_int8(floats[f"b{n}_beta1"],
                                    s.weight_exp(f"b{n}_beta1")),
                wqkv=wqkv,
                bqkv=quantize_bias_chan(floats[f"b{n}_bqkv"], a_ln1, e_qkv),
                wproj=wproj,
                bproj=quantize_bias_chan(floats[f"b{n}_bproj"], a_ctx,
                                         e_proj),
                gamma2=quantize_int8(floats[f"b{n}_gamma2"],
                                     s.weight_exp(f"b{n}_gamma2")),
                beta2=quantize_int8(floats[f"b{n}_beta2"],
                                    s.weight_exp(f"b{n}_beta2")),
                w1=w1,
                b1=quantize_bias_chan(floats[f"b{n}_b1"], a_ln2, e_w1),
                w2=w2,
                b2=quantize_bias_chan(floats[f"b{n}_b2"], a_hid, e_w2),
            ))
        else:
            a_ln1 = s.activation_exp(f"b{n}_ln1_out")
            a_hid = s.activation_exp(f"b{n}_hidden")
            blocks.append(BlockP(
                gamma1=quantize_int8(floats[f"b{n}_gamma1"],
                                     s.weight_exp(f"b{n}_gamma1")),
                beta1=quantize_int8(floats[f"b{n}_beta1"],
                                    s.weight_exp(f"b{n}_beta1")),
                wqkv=quantize_int8(floats[f"b{n}_wqkv"],
                                   s.weight_exp(f"b{n}_wqkv")),
                bqkv=quantize_bias(floats[f"b{n}_bqkv"],
                                   a_ln1 + s.weight_exp(f"b{n}_wqkv")),
                wproj=quantize_int8(floats[f"b{n}_wproj"],
                                    s.weight_exp(f"b{n}_wproj")),
                bproj=quantize_bias(
                    floats[f"b{n}_bproj"],
                    s.activation_exp(f"b{n}_context_out")
                    + s.weight_exp(f"b{n}_wproj")),
                gamma2=quantize_int8(floats[f"b{n}_gamma2"],
                                     s.weight_exp(f"b{n}_gamma2")),
                beta2=quantize_int8(floats[f"b{n}_beta2"],
                                    s.weight_exp(f"b{n}_beta2")),
                w1=quantize_int8(floats[f"b{n}_w1"],
                                 s.weight_exp(f"b{n}_w1")),
                b1=quantize_bias(floats[f"b{n}_b1"],
                                 s.activation_exp(f"b{n}_ln2_out")
                                 + s.weight_exp(f"b{n}_w1")),
                w2=quantize_int8(floats[f"b{n}_w2"],
                                 s.weight_exp(f"b{n}_w2")),
                b2=quantize_bias(floats[f"b{n}_b2"],
                                 a_hid + s.weight_exp(f"b{n}_w2")),
            ))
    selectors = []
    for idx in range(1, 4):
        selectors.append(SelectorP(
            local_w=torch.zeros(3, 64, 32, dtype=torch.int8),
            local_b=torch.zeros(3, 32, dtype=torch.int32),
            score_w1=torch.zeros(3, 64, 32, dtype=torch.int8),
            score_b1=torch.zeros(3, 32, dtype=torch.int32),
            score_w2=torch.zeros(3, 32, 16, dtype=torch.int8),
            score_b2=torch.zeros(3, 16, dtype=torch.int32),
            score_w3=torch.zeros(3, 16, 2, dtype=torch.int8),
            score_b3=torch.zeros(3, 2, dtype=torch.int32),
            hw_w1=torch.zeros(3, 3, dtype=torch.int8),
            hw_b1=torch.zeros(3, dtype=torch.int32),
            hw_w2=torch.zeros(3, 3, dtype=torch.int8),
            hw_b2=torch.zeros(3, dtype=torch.int32),
        ))
    if use_chan:
        patch_w, e_pw = quantize_int8_chan(floats["patch_w"])
        head_w, e_hw = quantize_int8_chan(floats["head_w"])
        wchan["patch_w"] = e_pw
        wchan["head_w"] = e_hw
        model = QuantDeiT(
            scales=s,
            patch_w=patch_w,
            patch_b=quantize_bias_chan(
                floats["patch_b"], s.activation_exp("input"), e_pw),
            cls=quantize_int8(floats["cls"], s.weight_exp("cls")),
            pos=quantize_int8(floats["pos"], s.weight_exp("pos")),
            blocks=blocks, selectors=selectors,
            final_gamma=quantize_int8(floats["final_gamma"],
                                      s.weight_exp("final_gamma")),
            final_beta=quantize_int8(floats["final_beta"],
                                     s.weight_exp("final_beta")),
            head_w=head_w,
            head_b=quantize_bias_chan(
                floats["head_b"], s.activation_exp("final_ln_out"), e_hw),
        )
    else:
        model = QuantDeiT(
            scales=s,
            patch_w=quantize_int8(floats["patch_w"], s.weight_exp("patch_w")),
            patch_b=quantize_bias(floats["patch_b"],
                                  s.activation_exp("input")
                                  + s.weight_exp("patch_w")),
            cls=quantize_int8(floats["cls"], s.weight_exp("cls")),
            pos=quantize_int8(floats["pos"], s.weight_exp("pos")),
            blocks=blocks, selectors=selectors,
            final_gamma=quantize_int8(floats["final_gamma"],
                                      s.weight_exp("final_gamma")),
            final_beta=quantize_int8(floats["final_beta"],
                                     s.weight_exp("final_beta")),
            head_w=quantize_int8(floats["head_w"], s.weight_exp("head_w")),
            head_b=quantize_bias(floats["head_b"],
                                 s.activation_exp("final_ln_out")
                                 + s.weight_exp("head_w")),
        )
    model = model.to(device)
    if use_chan:
        model.wchan = {k: v.to(device) for k, v in wchan.items()}
    return model


def eval_top1(model, cfg, loader, device, batch_eval=32):
    correct = total = 0
    t0 = time.time()
    with torch.no_grad():
        for img, label in loader:
            logits, _ = forward_batch_cfg(model, img.to(device), cfg)
            pred = logits.argmax(dim=1).cpu()
            correct += (pred == label).sum().item()
            total += label.size(0)
    elapsed = time.time() - t0
    return 100.0 * correct / total, elapsed, correct, total


def run_unit():
    """Sanity checks: contract-cfg bit-identity + nonlinearity accuracy."""
    torch.manual_seed(20260815)
    print("[unit] contract-cfg bit-identity vs p2_sim ...")
    # random small model (N=8 tokens) with a uniform table
    def ri(*shape):
        return torch.randint(-127, 128, shape, dtype=torch.int8)
    blocks = [BlockP(
        gamma1=ri(192), beta1=ri(192), wqkv=ri(192, 576),
        bqkv=torch.zeros(576, dtype=torch.int32),
        wproj=ri(192, 192), bproj=torch.zeros(192, dtype=torch.int32),
        gamma2=ri(192), beta2=ri(192), w1=ri(192, 768),
        b1=torch.zeros(768, dtype=torch.int32), w2=ri(768, 192),
        b2=torch.zeros(192, dtype=torch.int32)) for _ in range(12)]
    from tools.p2.scale_table import ACTIVATION_NAMES, WEIGHT_NAMES
    table = ScaleTable(
        weights={n: -7 for n in WEIGHT_NAMES},
        activations={n: -7 for n in ACTIVATION_NAMES})
    model = QuantDeiT(
        scales=table, patch_w=ri(768, 192),
        patch_b=torch.zeros(192, dtype=torch.int32), cls=ri(192), pos=ri(197, 192),
        blocks=blocks, selectors=[],
        final_gamma=ri(192), final_beta=ri(192), head_w=ri(192, 1000),
        head_b=torch.zeros(1000, dtype=torch.int32))
    img = (torch.rand(3, 224, 224) * 4 - 2)
    cfg = NonlinConfig()
    l1, c1, e1 = forward_image(model, img, prune=False)
    l2, c2, e2 = forward_image_cfg(model, img, cfg, prune=False)
    assert e1 == e2 and c1 == c2 and torch.equal(l1, l2), \
        "contract-cfg must be bit-identical to p2_sim"
    print("[unit]   OK: logits bit-identical")

    print("[unit] nonlinearity accuracy on random Q16 inputs ...")
    xs = (torch.randn(4096) * 12 * 65536).round().clamp(-(1 << 23),
                                                        (1 << 23) - 1).to(
        torch.int64)
    # ShiftGELU vs float GELU
    ref = torch.nn.functional.gelu(xs.float() / 65536) * 65536
    err = (shiftgelu_q16(xs, "half").float() - ref).abs().mean().item()
    print(f"[unit]   shiftgelu mean|err| = {err:.1f} Q16")
    # Shiftmax vs float softmax
    rows = (torch.randn(16, 197) * 8 * 65536).round().clamp(
        -(1 << 23), (1 << 23) - 1).to(torch.int64)
    refp = torch.softmax(rows.float() / 65536, dim=-1)
    p = shiftmax_attention(rows, "half").float() / 256
    errp = (p - refp).abs().mean().item()
    print(f"[unit]   shiftmax mean|err| = {errp:.6f}")

    print("[unit] per-channel GEMM vs float ...")
    a = torch.randint(-127, 128, (8, 192), dtype=torch.int8)
    w = (torch.randn(192, 576) * 0.05).float()
    q, exps = quantize_int8_chan(w)
    b = torch.randn(576) * 0.1
    bq = quantize_bias_chan(b, -4, exps)
    out = gemm_int8_chan(a, q, bq, -4, exps, 0, 8)
    refo = (a.float() * (2.0 ** -4)) @ (q.float() * (2.0 ** exps.float())) \
        + b
    errg = (out.float() - refo).abs().mean().item()
    print(f"[unit]   per-channel gemm mean|err| = {errg:.4f} (step 1.0)")
    assert errg < 2.0, errg

    print("[unit] LayerNorm variants ...")
    from tools.p2.p2_sim_ivit import layernorm_newton, layernorm_relaxed
    xs = torch.randint(-100, 101, (4, 192), dtype=torch.int8)
    g = torch.randint(20, 120, (192,), dtype=torch.int8)
    b = torch.randint(-120, 120, (192,), dtype=torch.int8)
    for x_exp in (-7, -3, 0, 2, 4, 6):
        out_r = layernorm_relaxed(xs, g, b, x_exp, -6, -6, -4, 6)
        out_n = layernorm_newton(xs, g, b, x_exp, -6, -6, -4, 6)
        assert torch.equal(out_r, out_n), \
            f"newton vs relaxed mismatch at x_exp={x_exp}"
        if x_exp <= 0:
            out_c = layernorm(xs, g, b, x_exp, -6, -6, -4)
            assert torch.equal(out_r, out_c), \
                f"relaxed vs contract mismatch at x_exp={x_exp}"
    print("[unit]   LN relaxed/newton consistent with contract")

    print("[unit] batched forward vs single-image forward ...")
    batch = (torch.rand(8, 3, 224, 224) * 4 - 2)
    lb, _ = forward_batch_cfg(model, batch, cfg)
    for i in range(8):
        li, _, _ = forward_image_cfg(model, batch[i], cfg, prune=False)
        assert torch.equal(lb[i], li), f"batch row {i} differs"
    print("[unit]   OK: batched logits bit-identical per image")
    print("[unit] all checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unit", action="store_true",
                        help="run sanity checks only")
    parser.add_argument("--calib", type=int, default=2048)
    parser.add_argument("--calib-batch", type=int, default=64)
    parser.add_argument("--eval", type=int, default=5000)
    parser.add_argument("--configs", default=",".join(CONFIGS),
                        help="comma-separated subset of the config matrix")
    parser.add_argument("--force-calib", action="store_true",
                        help="re-run float calibration even if tables exist")
    parser.add_argument("--out-dir", default="p2_out/ivit")
    args = parser.parse_args()

    if args.unit:
        run_unit()
        return

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    print(f"device={device}")

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_state_dict()
    floats = to_heatvit_tensors(state)

    weight_exps = build_weight_exps(floats)
    t_l_path = out_dir / "scale_table_legacy.json"
    t_w_path = out_dir / "scale_table_relax.json"
    if args.force_calib or not (t_l_path.exists() and t_w_path.exists()):
        loader = make_val_loader(args.calib, batch_size=args.calib_batch)
        images = [img for img, _ in loader]
        print(f"float calibration on {len(images)} batches "
              f"({len(images) * args.calib_batch} images) ...")
        t0 = time.time()
        hist, centers = collect_float_histograms(
            state, images, device, batch_size=args.calib_batch)
        print(f"  histograms done in {time.time() - t0:.1f}s")
        acts_l = pick_activation_exps(hist, centers, ln_clamp_max=0)
        acts_w = pick_activation_exps(hist, centers, ln_clamp_max=6)
        table_l = ScaleTable(weights=dict(weight_exps), activations=acts_l)
        table_w = ScaleTable(weights=dict(weight_exps), activations=acts_w)
        table_l.save(t_l_path)
        table_w.save(t_w_path)
        n_diff = sum(1 for k in acts_l if acts_l[k] != acts_w[k])
        for name in ("input", "act_tokens", "b12_y", "b12_out",
                     "b9_ln2_out", "final_ln_out"):
            print(f"  {name:12s} legacy={table_l.activations[name]:3d} "
                  f"relax={table_w.activations[name]:3d}")
        print(f"  activation exps differing: {n_diff}")
    else:
        print(f"reusing cached scale tables in {out_dir}")
        table_l = ScaleTable.load(t_l_path)
        table_w = ScaleTable.load(t_w_path)

    eval_loader = make_val_loader(args.eval, batch_size=32)
    results_path = out_dir / "results.json"
    results = {}
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results = {}
    for name in args.configs.split(","):
        name = name.strip()
        if not name or name not in CONFIGS:
            print(f"unknown config {name!r}, skipping")
            continue
        cfg_kwargs, table_kind, use_chan = CONFIGS[name]
        cfg = NonlinConfig(**cfg_kwargs)
        table = table_l if table_kind == "legacy" else table_w
        print(f"[{name}] building model ...")
        model = build_model(floats, table, device, use_chan)
        print(f"[{name}] evaluating on {args.eval} images ...")
        acc, elapsed, correct, total = eval_top1(model, cfg, eval_loader,
                                                 device)
        results[name] = {
            "top1": round(acc, 4), "correct": correct, "total": total,
            "elapsed_s": round(elapsed, 1),
            "config": cfg_kwargs, "table": table_kind,
        }
        print(f"[{name}] Top-1 = {acc:.2f}% ({correct}/{total}, "
              f"{elapsed:.0f}s)")
        results_path.write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\n=== summary ===")
    for name, r in results.items():
        print(f"{name:14s} {r['top1']:8.2f}%  "
              f"({r['correct']}/{r['total']})")
    print(f"results -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
