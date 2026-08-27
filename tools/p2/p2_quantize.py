#!/usr/bin/env python3
"""P2-B: per-tensor power-of-2 PTQ for the DeiT-T checkpoint.

Pipeline:

  1. Load the official DeiT-T snapshot and convert every float tensor into
     the HeatViT layout (Part 4 weight table shapes).
  2. Quantize weights per tensor: exp = ceil(log2(max_abs / 127)), int8.
  3. Calibrate static activation scales iteratively on a val subset using
     the bit-exact simulator (max_abs -> exp, clamped to the contract
     ranges; LayerNorm inputs are clamped to exp <= 0).
  4. Quantize biases at a_exp + w_exp into int32.
  5. Evaluate quantized Top-1 (no pruning) against the float baseline.

Usage (torch venv):

  .venv-torch\\Scripts\\python tools/p2/p2_quantize.py --calib 2048 --eval 5000

Artifacts: p2_out/scale_table.json (per-tensor exponents) and the eval
log (p2_out/ is gitignored; the sandbox denies writes under build/).
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_sim import (
    BlockP,
    QuantDeiT,
    SelectorP,
    forward_image,
)
from tools.p2.scale_table import (
    ACTIVATION_NAMES,
    WEIGHT_NAMES,
    ScaleTable,
)

# 数据与权重路径：环境变量优先（云端/AutoDL 用），默认保持本机路径不变。
_IMAGENET_DIR = os.environ.get(
    "HEATVIT_IMAGENET_DIR", r"D:\SEU_Liubo\prj\HeatViT\data\imagenet")
DATA_DIR = os.path.join(_IMAGENET_DIR, "val").replace("\\", "/")
CHECKPOINT = os.environ.get(
    "HEATVIT_DEIT_CHECKPOINT",
    r"C:\Users\Youxi\.cache\torch\hub\checkpoints"
    r"\deit_tiny_patch16_224-a1311bcf.pth")

LN_INPUT_NAMES = {"act_tokens"} | {f"b{n}_y" for n in range(1, 13)} \
    | {f"b{n}_out" for n in range(1, 13)}


def load_state_dict():
    raw = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    return raw.get("model", raw) if isinstance(raw, dict) else raw


def to_heatvit_tensors(state):
    """Extract float tensors in the HeatViT (Part 4) layout."""
    t = {}
    pe = state["patch_embed.proj.weight"]        # [192,3,16,16]
    # GEMM layout k = (h*16+w)*3+c (channel last, matching patchify).
    t["patch_w"] = pe.permute(0, 2, 3, 1).reshape(192, 768).t()
    t["patch_b"] = state["patch_embed.proj.bias"]
    t["cls"] = state["cls_token"].reshape(-1)      # [192]
    t["pos"] = state["pos_embed"].squeeze(0)     # [197,192]
    for n in range(12):
        p = f"blocks.{n}."
        t[f"b{n + 1}_gamma1"] = state[p + "norm1.weight"]
        t[f"b{n + 1}_beta1"] = state[p + "norm1.bias"]
        t[f"b{n + 1}_wqkv"] = state[p + "attn.qkv.weight"].t()   # [192,576]
        t[f"b{n + 1}_bqkv"] = state[p + "attn.qkv.bias"]
        t[f"b{n + 1}_wproj"] = state[p + "attn.proj.weight"].t()
        t[f"b{n + 1}_bproj"] = state[p + "attn.proj.bias"]
        t[f"b{n + 1}_gamma2"] = state[p + "norm2.weight"]
        t[f"b{n + 1}_beta2"] = state[p + "norm2.bias"]
        t[f"b{n + 1}_w1"] = state[p + "mlp.fc1.weight"].t()      # [192,768]
        t[f"b{n + 1}_b1"] = state[p + "mlp.fc1.bias"]
        t[f"b{n + 1}_w2"] = state[p + "mlp.fc2.weight"].t()      # [768,192]
        t[f"b{n + 1}_b2"] = state[p + "mlp.fc2.bias"]
    t["final_gamma"] = state["norm.weight"]
    t["final_beta"] = state["norm.bias"]
    t["head_w"] = state["head.weight"].t()                        # [192,1000]
    t["head_b"] = state["head.bias"]
    return t


def pick_weight_exp(tensor):
    max_abs = tensor.abs().max().item()
    if max_abs <= 0:
        return -32
    return max(-32, min(31, math.ceil(math.log2(max_abs / 127.0))))


def quantize_int8(tensor, exp):
    q = torch.round(tensor / (2.0 ** exp))
    return torch.clamp(q, -128, 127).to(torch.int8)


def quantize_bias(tensor, exp):
    q = torch.round(tensor / (2.0 ** exp))
    return torch.clamp(q, -(1 << 31), (1 << 31) - 1).to(torch.int32)


def build_weight_exps(floats):
    exps = {}
    for name in WEIGHT_NAMES:
        if name in floats:
            exps[name] = pick_weight_exp(floats[name])
        else:
            # Selector tensors are trained later (P2-C); placeholder exp.
            exps[name] = -7
    return exps


def build_model(floats, scale_table, device):
    """Quantize weights/biases on the fly from float tensors + scale table."""
    w = scale_table.weights
    blocks = []
    for n in range(1, 13):
        a_ln1 = scale_table.activation_exp(f"b{n}_ln1_out")
        a_hid = scale_table.activation_exp(f"b{n}_hidden")
        blocks.append(BlockP(
            gamma1=quantize_int8(floats[f"b{n}_gamma1"],
                                 w[f"b{n}_gamma1"]),
            beta1=quantize_int8(floats[f"b{n}_beta1"], w[f"b{n}_beta1"]),
            wqkv=quantize_int8(floats[f"b{n}_wqkv"], w[f"b{n}_wqkv"]),
            bqkv=quantize_bias(floats[f"b{n}_bqkv"],
                               a_ln1 + w[f"b{n}_wqkv"]),
            wproj=quantize_int8(floats[f"b{n}_wproj"], w[f"b{n}_wproj"]),
            bproj=quantize_bias(floats[f"b{n}_bproj"],
                                scale_table.activation_exp(
                                    f"b{n}_context_out")
                                + w[f"b{n}_wproj"]),
            gamma2=quantize_int8(floats[f"b{n}_gamma2"],
                                 w[f"b{n}_gamma2"]),
            beta2=quantize_int8(floats[f"b{n}_beta2"], w[f"b{n}_beta2"]),
            w1=quantize_int8(floats[f"b{n}_w1"], w[f"b{n}_w1"]),
            b1=quantize_bias(floats[f"b{n}_b1"],
                             scale_table.activation_exp(f"b{n}_ln2_out")
                             + w[f"b{n}_w1"]),
            w2=quantize_int8(floats[f"b{n}_w2"], w[f"b{n}_w2"]),
            b2=quantize_bias(floats[f"b{n}_b2"], a_hid + w[f"b{n}_w2"]),
        ))
    # Selectors: uniform synthetic defaults so the table stays complete; the
    # prune=False path never executes them.
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
    model = QuantDeiT(
        scales=scale_table,
        patch_w=quantize_int8(floats["patch_w"], w["patch_w"]),
        patch_b=quantize_bias(
            floats["patch_b"],
            scale_table.activation_exp("input") + w["patch_w"]),
        cls=quantize_int8(floats["cls"], w["cls"]),
        pos=quantize_int8(floats["pos"], w["pos"]),
        blocks=blocks, selectors=selectors,
        final_gamma=quantize_int8(floats["final_gamma"],
                                  w["final_gamma"]),
        final_beta=quantize_int8(floats["final_beta"], w["final_beta"]),
        head_w=quantize_int8(floats["head_w"], w["head_w"]),
        head_b=quantize_bias(
            floats["head_b"],
            scale_table.activation_exp("final_ln_out") + w["head_w"]),
    )
    return model.to(device)


def initial_scale_table(floats):
    weights = build_weight_exps(floats)
    acts = {}
    acts["input"] = -6
    acts["act_patch_matrix"] = -7
    acts["act_patch_embed"] = -7
    acts["act_tokens"] = -7
    for n in range(1, 13):
        for name in ("ln1_out", "qkv_out", "context_out", "msa_out", "y",
                     "ln2_out", "hidden", "ffn_out", "out"):
            acts[f"b{n}_{name}"] = -7
    for idx in range(1, 4):
        for name in ("local_out", "concat_out", "h1_out", "h2_out",
                     "logits_out", "stats_out", "hw_hidden_out"):
            acts[f"s{idx}_{name}"] = -7
    acts["final_ln_out"] = -7
    return ScaleTable(weights=weights, activations=acts)


def _pick_exp_mse(hist, centers, hi_exp=None):
    """Choose the scale exponent minimizing quantization MSE.

    ``hist`` is a histogram over the fixed grid ``centers``; for each
    candidate exponent e the MSE of round(clamp(x/2^e)) * 2^e against x is
    evaluated exactly (out-of-range mass is clamped). Returns the optimal
    exponent in [-32, 31]. ``hi_exp`` bounds the search: None = 4
    (non-LN tensors), an int = hard upper bound (0 for the original LN
    input contract, positive for the relaxed I-ViT variant).
    """
    hist = hist.to(torch.float64)
    best_exp, best_mse = None, None
    lo_e, hi_e = -32, 4 if hi_exp is None else hi_exp
    for e in range(lo_e, hi_e + 1):
        step = 2.0 ** e
        q = torch.clamp(torch.round(centers / step), -128, 127) * step
        mse = ((q - centers) ** 2 * hist).sum().item()
        if best_mse is None or mse < best_mse:
            best_mse, best_exp = mse, e
    return best_exp


HIST_BINS = 4096
HIST_LO, HIST_HI = -64.0, 64.0
HIST_CENTERS = None  # filled lazily


def _get_centers(hist_hi, hist_bins):
    return torch.linspace(-hist_hi, hist_hi, hist_bins, dtype=torch.float64)


def collect_float_histograms(state, images, device, batch_size=64,
                             hist_hi=64.0, hist_bins=4096):
    """Collect float activation histograms for all ACTIVATION_NAMES.

    Same hook positions as :func:`calibrate_float`; returns
    ``(histograms, centers)`` so several scale-table variants (e.g. the
    relaxed I-ViT LayerNorm input contract) can be derived from one float
    pass.
    """
    import timm
    model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    hist = {name: torch.zeros(hist_bins, dtype=torch.float64)
            for name in ACTIVATION_NAMES}
    block_inputs = {}

    def upd(name, tensor):
        h = tensor.detach().float().reshape(-1).histc(
            bins=hist_bins, min=-hist_hi, max=hist_hi).to(
            torch.float64).cpu()
        hist[name] += h

    hooks = []

    def add_post(module, name):
        hooks.append(module.register_forward_hook(
            lambda m, a, o, n=name: upd(n, o)))

    def add_pre(module, name):
        hooks.append(module.register_forward_pre_hook(
            lambda m, a, n=name: upd(n, a[0])))

    add_post(model.patch_embed, "act_patch_embed")
    # act_tokens = cls + pos[0] / embed + pos[1:], computed on the fly.
    def patch_hook(m, a, o):
        upd("act_patch_embed", o)
        embed = o
        pos = model.pos_embed
        cls_tok = model.cls_token
        batch = embed.shape[0]
        tokens0 = cls_tok.expand(batch, -1, -1) + pos[:, :1]
        tokens1 = embed + pos[:, 1:]
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
        add_post(blk.norm1, f"b{n}_ln1_out")
        add_post(blk.attn.qkv, f"b{n}_qkv_out")
        add_pre(blk.attn.proj, f"b{n}_context_out")
        add_post(blk.norm2, f"b{n}_ln2_out")
        add_post(blk.mlp.act, f"b{n}_hidden")
        add_post(blk.mlp.fc2, f"b{n}_ffn_out")
        add_post(blk, f"b{n}_out")
    add_post(model.norm, "final_ln_out")

    try:
        with torch.no_grad():
            for img in images:
                upd("input", img)
                upd("act_patch_matrix", img)
                model(img.to(device))
    finally:
        for h in hooks:
            h.remove()
    return hist, _get_centers(hist_hi, hist_bins)


def pick_activation_exps(hist, centers, ln_clamp_max=0):
    """Pick MSE-optimal activation exponents from collected histograms.

    ``ln_clamp_max`` bounds the chosen exponent of LayerNorm input
    tensors (``act_tokens``, ``b<N>_y``, ``b<N>_out``); the original P2-B
    contract uses 0 (range +-127), the relaxed I-ViT variant allows
    positive exponents so the residual streams are not clipped.
    """
    acts = {}
    for name in ACTIVATION_NAMES:
        if name.startswith("s") or hist[name].sum().item() <= 0:
            acts[name] = -7
            continue
        if name in LN_INPUT_NAMES:
            acts[name] = _pick_exp_mse(hist[name], centers, ln_clamp_max)
        elif name == "input":
            acts[name] = _pick_exp_mse(hist[name], centers, 0)
        else:
            acts[name] = _pick_exp_mse(hist[name], centers, None)
    return acts


def calibrate_float(state, images, device, batch_size=64):
    """Collect float activation histograms and pick MSE-optimal exponents.

    Measuring the *quantized* activations is biased (values saturate at
    127 and the range estimate collapses), so calibration runs the float
    timm model and hooks the exact tensor positions that the RTL contract
    names. One pass; the per-tensor exponent minimizes quantization MSE on
    the calibration histogram (max-based exponents waste bits on outliers).
    """
    hist, centers = collect_float_histograms(
        state, images, device, batch_size=batch_size)
    return pick_activation_exps(hist, centers, ln_clamp_max=0)


def make_val_loader(max_images, batch_size=1, shuffle=False):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize(256,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    if max_images and max_images < len(dataset):
        if shuffle:
            gen = torch.Generator().manual_seed(20260815)
            idx = torch.randperm(len(dataset), generator=gen)[:max_images]
            subset = Subset(dataset, idx.tolist())
        else:
            subset = Subset(dataset, range(max_images))
    else:
        subset = dataset
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calib", type=int, default=2048)
    parser.add_argument("--calib-batch", type=int, default=64)
    parser.add_argument("--eval", type=int, default=5000)
    parser.add_argument("--out", default="p2_out/scale_table.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    print(f"device={device}")

    state = load_state_dict()
    floats = to_heatvit_tensors(state)
    weight_exps = build_weight_exps(floats)

    loader = make_val_loader(args.calib, batch_size=args.calib_batch)
    images = [img for img, _ in loader]
    print(f"float calibration on {len(images)} images")
    t0 = time.time()
    acts = calibrate_float(state, images, device)
    print(f"calibration done in {time.time() - t0:.1f}s")
    table = ScaleTable(weights=weight_exps, activations=acts)
    for name in ("input", "act_tokens", "b1_ln1_out", "b12_out",
                 "final_ln_out"):
        print(f"  {name} = {table.activations[name]}")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.save(out_path)
    print(f"scale table -> {out_path}")

    model = build_model(floats, table, device)
    eval_loader = make_val_loader(args.eval)
    correct = 0
    total = 0
    t0 = time.time()
    with torch.no_grad():
        for img, label in eval_loader:
            logits, _, _ = forward_image(model, img[0].to(device),
                                         prune=False)
            correct += (logits.argmax().item() == label.item())
            total += 1
    elapsed = time.time() - t0
    print(f"quantized Top-1 on {total} val images: "
          f"{100.0 * correct / total:.2f}% ({elapsed:.1f}s, "
          f"{elapsed / max(total, 1):.3f}s/img)")


if __name__ == "__main__":
    main()
