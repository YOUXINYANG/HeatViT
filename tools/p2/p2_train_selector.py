#!/usr/bin/env python3
"""P2-C: train the three Token Selectors on the frozen quantized backbone.

Architecture (docs/heatvit.md Part 1 Section 12, exactly the RTL shape):

  per head h in 0..2 over the 64-dim slice:
    local  : Linear(64 -> 32) + GELU
    global : mean of local over candidates, broadcast
    score  : Linear(64 -> 32) -> GELU -> Linear(32 -> 16) -> GELU
             -> Linear(16 -> 2) -> Softmax(delta2=1.0), keep = col 1
  head stats : mean over each head's 64 lanes -> [C, 3]
  head weight: Linear(3 -> 3) -> GELU -> Linear(3 -> 3) -> PLAN sigmoid
  fused      : weighted mean of the three keep scores, threshold >= 0.5

Training (per the paper's spirit, adapted to the threshold-based RTL):

  * backbone is the EXACT integer simulator (batched primitives), frozen;
  * hard keep mask via straight-through threshold on the fused score;
  * pruned tokens form one package token weighted by the SOFT keep scores
    (gradient path), exactly like the RTL finalize;
  * sparsity loss (mean keep rate - target)^2 per stage, paper DeiT-T
    targets {0.45, 0.51, 0.71};
  * task loss = CE on the CLS logits (prune path), plus optional CE on the
    unpruned head output for stability.

Usage (torch venv):

  .venv-torch\\Scripts\\python tools/p2/p2_train_selector.py \
      --calib 512 --max-images 8192 --epochs 1 --lr 1e-4 --out p2_out/sel.pt

Exports int8/Q0.16 selector tensors + scale table entries and validates the
trained selectors inside the exact simulator (token counts per stage).
"""

import argparse
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from tools.p2.p2_quantize import (
    build_model,
    build_weight_exps,
    calibrate_float,
    load_state_dict,
    make_val_loader,
    quantize_bias,
    quantize_int8,
    pick_weight_exp,
    to_heatvit_tensors,
)
from tools.p2.scale_table import ScaleTable
from tools.p2 import p2_sim as S

STAGE_TARGETS = (0.45, 0.51, 0.71)


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a base dataset and yields (image, label, index) so the
    distillation teacher logits can be looked up per sample."""

    def __init__(self, base):
        self.base = base

    def __getitem__(self, i):
        img, label = self.base[i]
        return img, label, i

    def __len__(self):
        return len(self.base)


def plan_sigmoid_float(x):
    """Float mirror of the PLAN sigmoid (piecewise-linear, Q0.16 shape)."""
    ax = x.abs()
    y = torch.where(ax >= 5.0, torch.ones_like(x),
                    torch.where(ax >= 2.375, ax / 32.0 + 27.0 / 32.0,
                                torch.where(ax >= 1.0, ax / 8.0 + 5.0 / 8.0,
                                            ax / 4.0 + 0.5)))
    return torch.where(x < 0, 1.0 - y, y)


class TrainSelector(nn.Module):
    """One RTL-shaped Token Selector in float."""

    def __init__(self):
        super().__init__()
        self.local = nn.ModuleList(
            [nn.Sequential(nn.Linear(64, 32), nn.GELU()) for _ in range(3)])
        self.score = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 32), nn.GELU(),
                          nn.Linear(32, 16), nn.GELU(),
                          nn.Linear(16, 2))
            for _ in range(3)])
        self.head_weight = nn.Sequential(nn.Linear(3, 3), nn.GELU(),
                                         nn.Linear(3, 3))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        # Start keep-biased but inside the int8-at--7 contract range.
        for head in self.score:
            last = head[-1]
            last.bias.data[0] = 0.0
            last.bias.data[1] = 0.4

    def forward(self, tokens_deq, stats_scale=1.0):
        """tokens_deq: [B,N,192] float; returns (fused, keep_soft, range_loss).

        ``stats_scale`` mirrors the RTL head-weight branch: the stats are
        the per-head means in the INPUT's units but are fed to the
        head-weight GEMM at the ``stats_out`` scale (2^(stats_out -
        in_exp) relative to the input). ``range_loss`` penalizes selector
        activations/logits outside the int8-at--7 representable range
        (+-0.99): the RTL selector runs every intermediate through int8
        at scale exp -7, so anything beyond +-0.99 saturates and breaks
        the decision boundary.
        """
        b, n, d = tokens_deq.shape
        th = tokens_deq.reshape(b, n, 3, 64)
        range_loss = torch.tensor(0.0, device=tokens_deq.device)
        local_h = []
        for h in range(3):
            pre = self.local[h][0](th[:, :, h])
            act = self.local[h][1](pre)
            range_loss = range_loss + (act.abs() - 0.9).clamp(min=0).mean()
            local_h.append(act)
        local = torch.stack(local_h, dim=2)                     # [B,N,3,32]
        global_f = local.mean(dim=1, keepdim=True)              # [B,1,3,32]
        e = torch.cat([local, global_f.expand(-1, n, -1, -1)],
                      dim=-1)                                   # [B,N,3,64]
        keep_h = []
        logits_h = []
        for h in range(3):
            h1 = self.score[h][1](self.score[h][0](e[:, :, h]))
            range_loss = range_loss + (h1.abs() - 0.9).clamp(min=0).mean()
            h2 = self.score[h][3](self.score[h][2](h1))
            range_loss = range_loss + (h2.abs() - 0.9).clamp(min=0).mean()
            logits = self.score[h][4](h2)
            range_loss = range_loss \
                + (logits.abs() - 0.9).clamp(min=0).mean()
            logits_h.append(logits)
            keep_h.append(torch.softmax(logits, dim=-1)[..., 1])
        keep_scores = torch.stack(keep_h, dim=-1)               # [B,N,3]
        stats = th.mean(dim=-1) * stats_scale                   # [B,N,3]
        hw = plan_sigmoid_float(self.head_weight(stats))        # [B,N,3]
        if keep_scores.shape != hw.shape:
            raise RuntimeError(
                f"selector shape mismatch: tokens={tuple(tokens_deq.shape)} "
                f"keep={tuple(keep_scores.shape)} hw={tuple(hw.shape)} "
                f"local={tuple(local.shape)} stats={tuple(stats.shape)}")
        den = hw.sum(dim=-1).clamp(min=1e-6)                     # [B,N]
        fused = (keep_scores * hw).sum(dim=-1) / den            # [B,N]
        return fused, keep_scores, range_loss


def train_forward(model, selectors, img_q, targets, s, threshold=0.5):
    """One batched forward with soft-gradient/hard-mask selectors.

    Returns (logits, sparsity_loss, range_loss). The backbone is the exact
    integer simulator (gradients cannot flow through it); selectors see
    dequantized int8 features and their own activation/logit magnitudes
    are kept inside the int8-at--7 contract range via ``range_loss``.
    """
    inp_exp = s.activation_exp("input")
    img_q = torch.clamp(torch.round(img_q / (2.0 ** inp_exp)),
                        -128, 127).to(torch.int8)
    b = img_q.shape[0]
    patches = img_q.permute(0, 2, 3, 1).reshape(b, 14, 16, 14, 16, 3) \
        .permute(0, 1, 3, 2, 4, 5).reshape(b, 196, 768)
    embed = S.gemm_int8(patches, model.patch_w, model.patch_b, inp_exp,
                        s.weight_exp("patch_w"),
                        s.activation_exp("act_patch_embed"), 8)
    cls_exp = s.weight_exp("cls")
    pos_exp = s.weight_exp("pos")
    out_exp = s.activation_exp("act_tokens")
    row0 = S.residual_add(model.cls.unsqueeze(0).expand(b, -1, -1), cls_exp,
                          model.pos[:1].expand(b, -1, -1), pos_exp, out_exp)
    rows1 = S.residual_add(embed, s.activation_exp("act_patch_embed"),
                           model.pos[1:].unsqueeze(0).expand(b, -1, -1),
                           pos_exp, out_exp)
    tokens = torch.cat([row0, rows1], dim=1)                   # [B,197,192]

    valid = torch.ones(b, 197, 1, dtype=torch.int64,
                       device=tokens.device)
    sparsity_loss = torch.tensor(0.0, device=tokens.device)
    range_loss = torch.tensor(0.0, device=tokens.device)
    sel_idx = 0
    for n in range(1, 13):
        in_exp = s.activation_exp("act_tokens") if n == 1 \
            else s.activation_exp(f"b{n - 1}_out")
        if n in S.SELECTOR_BLOCKS:
            sel = selectors[sel_idx]
            scale = 2.0 ** in_exp
            deq = tokens.to(torch.float32) * scale
            # Mirror the RTL head-weight branch input scaling: the stats
            # (per-head means in the input's units) are consumed at the
            # stats_out scale.
            stats_scale = 2.0 ** (s.activation_exp(f"s{sel_idx + 1}_stats_out")
                                  - in_exp)
            fused, keep_soft, rl = sel(deq, stats_scale)
            range_loss = range_loss + rl
            fused = fused.clamp(0, 1)
            # CLS always kept.
            fused = torch.cat([torch.ones(b, 1, device=fused.device),
                               fused[:, 1:]], dim=1)
            keep_soft = torch.cat(
                [torch.ones(b, 1, 3, device=keep_soft.device),
                 keep_soft[:, 1:]], dim=1)
            hard = fused >= threshold
            mask = hard.float() + fused - fused.detach()      # STE
            # Per-sample pruning: package from the soft scores.
            parts = tokens * (1 - mask.unsqueeze(-1))
            wnum = (parts.to(torch.float32)
                    * keep_soft.mean(dim=-1, keepdim=True)
                    * (1 - mask.unsqueeze(-1))).sum(dim=1)
            den = (keep_soft.mean(dim=-1) * (1 - mask)).sum(
                dim=1, keepdim=True).clamp(min=1e-6)
            package_deq = wnum / den                             # [B,192]
            package_q = torch.clamp(
                torch.round(package_deq / scale), -128, 127).to(torch.int8)
            kept = tokens * mask.unsqueeze(-1)
            tokens = torch.cat([kept, package_q.unsqueeze(1)], dim=1)
            valid = torch.cat(
                [valid * mask.unsqueeze(-1),
                 torch.ones(b, 1, 1, device=valid.device)], dim=1)
            # Sparsity loss over the hard KEEP FRACTION (not the mean
            # score): the RTL threshold is fused >= 0.5, so the targets
            # {0.45, 0.51, 0.71} are kept-token fractions. The STE mask
            # carries the gradient; the boolean ``hard`` alone would not.
            rate = mask[:, 1:].mean()
            sparsity_loss = sparsity_loss + (rate - targets[sel_idx]) ** 2
            sel_idx += 1
        x_exp = s.activation_exp("act_tokens") if n == 1 \
            else s.activation_exp(f"b{n - 1}_out")
        tokens = transformer_block_masked(tokens, model.blocks[n - 1], s, n,
                                          x_exp, valid)

    final_ln = S.layernorm(tokens[:, 0:1], model.final_gamma,
                           model.final_beta, s.activation_exp("b12_out"),
                           s.weight_exp("final_gamma"),
                           s.weight_exp("final_beta"),
                           s.activation_exp("final_ln_out"))
    logits = S.gemm_int8(final_ln, model.head_w, model.head_b,
                         s.activation_exp("final_ln_out"),
                         s.weight_exp("head_w"), S.LOGIT_SCALE_EXP, 32)
    return logits[:, 0].float(), sparsity_loss, range_loss

def transformer_block_masked(tokens, p, s, n, x_exp, valid):
    """Exact integer block with an attention validity mask [B,N,1]."""
    msa_out = mhsa_masked(tokens, p, s, n, x_exp, valid)
    y = S.residual_add(tokens, x_exp, msa_out,
                       s.activation_exp(f"b{n}_msa_out"),
                       s.activation_exp(f"b{n}_y"))
    ffn_out = ffn_masked(y, p, s, n)
    return S.residual_add(y, s.activation_exp(f"b{n}_y"), ffn_out,
                          s.activation_exp(f"b{n}_ffn_out"),
                          s.activation_exp(f"b{n}_out"))


def mhsa_masked(x, p, s, n, x_exp, valid):
    ln1 = S.layernorm(x, p.gamma1, p.beta1, x_exp,
                      s.weight_exp(f"b{n}_gamma1"),
                      s.weight_exp(f"b{n}_beta1"),
                      s.activation_exp(f"b{n}_ln1_out"))
    fused = S.gemm_int8(ln1, p.wqkv, p.bqkv,
                        s.activation_exp(f"b{n}_ln1_out"),
                        s.weight_exp(f"b{n}_wqkv"),
                        s.activation_exp(f"b{n}_qkv_out"), 8)
    b, nt = fused.shape[0], fused.shape[1]
    q = fused[:, :, :S.D]
    k = fused[:, :, S.D:2 * S.D]
    v = fused[:, :, 2 * S.D:]
    score_exp = 2 * s.activation_exp(f"b{n}_qkv_out") - 3
    contexts = []
    for h in range(S.HEADS):
        qh = q[:, :, h * 64:(h + 1) * 64]
        kh = k[:, :, h * 64:(h + 1) * 64]
        vh = v[:, :, h * 64:(h + 1) * 64]
        acc = qh.to(torch.float64) @ kh.to(torch.float64).transpose(-2, -1)
        acc = acc + (1 - valid.to(torch.float64)) * (-2.0 ** 35)
        score = S.requant(acc.round().to(torch.int64),
                          2 * s.activation_exp(f"b{n}_qkv_out"),
                          score_exp, 32)
        # Contract conversion (1/sqrt(64) folded): shift = -(score_exp + 13);
        # matches forward_image for every per-tensor table (the old fixed
        # +4 shift only matched the synthetic score_exp = -17).
        q16 = S.sat(S.round_shift_away(score, -(score_exp + 13)), 24)
        prob = S.softmax_attention(q16)
        cacc = prob.to(torch.float64) @ vh.to(torch.float64)
        ctx = S.requant(cacc.round().to(torch.int64),
                        -8 + s.activation_exp(f"b{n}_qkv_out"),
                        s.activation_exp(f"b{n}_context_out"), 8).to(
                            torch.int8)
        contexts.append(ctx)
    concat = torch.cat(contexts, dim=-1)
    return S.gemm_int8(concat, p.wproj, p.bproj,
                       s.activation_exp(f"b{n}_context_out"),
                       s.weight_exp(f"b{n}_wproj"),
                       s.activation_exp(f"b{n}_msa_out"), 8)


def ffn_masked(y, p, s, n):
    ln2 = S.layernorm(y, p.gamma2, p.beta2, s.activation_exp(f"b{n}_y"),
                      s.weight_exp(f"b{n}_gamma2"),
                      s.weight_exp(f"b{n}_beta2"),
                      s.activation_exp(f"b{n}_ln2_out"))
    acc = ln2.to(torch.float64) @ p.w1.to(torch.float64) \
        + p.b1.to(torch.float64)
    src_exp = s.activation_exp(f"b{n}_ln2_out") + s.weight_exp(f"b{n}_w1")
    q16 = S.requant(acc.round().to(torch.int64), src_exp, -16, 24)
    hidden = S.requant(S.gelu_q16(q16), -16,
                       s.activation_exp(f"b{n}_hidden"), 8)
    return S.gemm_int8(hidden, p.w2, p.b2,
                       s.activation_exp(f"b{n}_hidden"),
                       s.weight_exp(f"b{n}_w2"),
                       s.activation_exp(f"b{n}_ffn_out"), 8)


def export_selector(sel, device, in_exp, act):
    """Quantize a trained selector into RTL int8/Q0.16 tensors.

    Returns ``(quantized, weight_exps)``. Weight tensors use per-tensor
    pick_weight_exp; biases quantize at (input activation exp + weight
    exp) from the selector scale-table entries, so the exported tensors
    pair with the calibrated selector scale table built by
    :func:`selector_scale_table`.

    PyTorch Linear stores weight [out, in]; the RTL/golden layout is
    [in, out], so every matrix is transposed here.
    """
    out = {}
    for h in range(3):
        out[f"local_w_{h}"] = sel.local[h][0].weight.data.detach().cpu().t()
        out[f"local_b_{h}"] = sel.local[h][0].bias.data.detach().cpu()
        out[f"score_w1_{h}"] = sel.score[h][0].weight.data.detach().cpu().t()
        out[f"score_b1_{h}"] = sel.score[h][0].bias.data.detach().cpu()
        out[f"score_w2_{h}"] = sel.score[h][2].weight.data.detach().cpu().t()
        out[f"score_b2_{h}"] = sel.score[h][2].bias.data.detach().cpu()
        out[f"score_w3_{h}"] = sel.score[h][4].weight.data.detach().cpu().t()
        out[f"score_b3_{h}"] = sel.score[h][4].bias.data.detach().cpu()
    out["hw_w1"] = sel.head_weight[0].weight.data.detach().cpu().t()
    out["hw_b1"] = sel.head_weight[0].bias.data.detach().cpu()
    out["hw_w2"] = sel.head_weight[2].weight.data.detach().cpu().t()
    out["hw_b2"] = sel.head_weight[2].bias.data.detach().cpu()

    # Per-tensor weight exponents + biases at (a_exp + w_exp), driven by
    # the selector scale-table entries.
    def qw(w_name, b_name, a_exp, stack_heads=True):
        if stack_heads:
            ws = [out[f"{w_name}_{h}"] for h in range(3)]
            bs = [out[f"{b_name}_{h}"] for h in range(3)]
            # One shared exponent per tensor (the RTL descriptor carries a
            # single scale for all three heads).
            exp = pick_weight_exp(torch.stack(ws))
            wq = torch.stack([quantize_int8(w, exp) for w in ws])
            bq = torch.stack(
                [quantize_bias(b, a_exp + exp) for b in bs])
            return wq, bq, exp
        exps = pick_weight_exp(out[w_name])
        wq = quantize_int8(out[w_name], exps)
        bq = quantize_bias(out[b_name], a_exp + exps)
        return wq, bq, exps

    q = {}
    w_exps = {}
    q["local_w"], q["local_b"], w_exps["local_w"] = \
        qw("local_w", "local_b", in_exp)
    q["score_w1"], q["score_b1"], w_exps["score_w1"] = \
        qw("score_w1", "score_b1", act["concat_out"])
    q["score_w2"], q["score_b2"], w_exps["score_w2"] = \
        qw("score_w2", "score_b2", act["h1_out"])
    q["score_w3"], q["score_b3"], w_exps["score_w3"] = \
        qw("score_w3", "score_b3", act["h2_out"])
    q["hw_w1"], q["hw_b1"], w_exps["hw_w1"] = \
        qw("hw_w1", "hw_b1", act["stats_out"], stack_heads=False)
    q["hw_w2"], q["hw_b2"], w_exps["hw_w2"] = \
        qw("hw_w2", "hw_b2", act["hw_hidden_out"], stack_heads=False)
    return q, w_exps


def selector_scale_table(table, in_exps, weight_exps):
    """Extend a backbone scale table with calibrated selector entries.

    ``in_exps`` = {stage: input activation exp}, ``weight_exps`` =
    {stage: {tensor: exp}} from :func:`export_selector`. Activation exps
    are the calibrated selector ranges (the uniform -7 placeholders
    saturate the GEMM requants under the real per-tensor backbone table);
    the stats live at the input's scale by construction.
    """
    acts = dict(table.activations)
    weights = dict(table.weights)
    for idx, in_exp in in_exps.items():
        for name, exp in weight_exps[idx].items():
            weights[f"s{idx}_{name}"] = exp
        acts[f"s{idx}_local_out"] = -6
        acts[f"s{idx}_concat_out"] = -6
        acts[f"s{idx}_h1_out"] = -6
        acts[f"s{idx}_h2_out"] = -6
        acts[f"s{idx}_logits_out"] = -7
        acts[f"s{idx}_stats_out"] = in_exp
        acts[f"s{idx}_hw_hidden_out"] = -3
    return ScaleTable(weights=weights, activations=acts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calib", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sparsity-weight", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="p2_out/selectors.pt")
    parser.add_argument("--table", default="p2_out/scale_table.json")
    parser.add_argument("--resume", default=None,
                        help="continue training from a selectors.pt "
                             "checkpoint (loads the float state dicts)")
    parser.add_argument("--teacher", default=None,
                        help="teacher logits .pt (from p2_teacher_logits.py) "
                             "for soft distillation from the unpruned "
                             "backbone")
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--range-weight", type=float, default=1.0,
                        help="penalty weight keeping selector activations/"
                             "logits inside the int8-at--7 contract range")
    parser.add_argument("--threshold-offset", type=float, default=0.012,
                        help="training threshold = 0.5 + offset; compensates "
                             "the systematic +~0.008 fused-score bias of the "
                             "Q16 softmax/round-div in the exact sim")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    state = load_state_dict()
    floats = to_heatvit_tensors(state)
    wexp = build_weight_exps(floats)
    table = ScaleTable.load(REPO_ROOT / args.table)

    teacher_logits = None
    if args.teacher:
        teacher_logits = torch.load(REPO_ROOT / args.teacher,
                                    map_location="cpu",
                                    weights_only=False)["logits"]
        assert teacher_logits.shape[0] >= args.max_images, \
            "teacher cache smaller than the training subset"

    model = build_model(floats, table, device)
    selectors = nn.ModuleList([TrainSelector() for _ in range(3)]).to(device)
    if args.resume:
        prev = torch.load(REPO_ROOT / args.resume, map_location="cpu",
                          weights_only=False)
        for sel, sd in zip(selectors, prev["selectors_float"]):
            sel.load_state_dict(sd)
        print(f"resumed from {args.resume}")

    # Extend the scale table with calibrated selector entries (the
    # uniform -7 placeholders saturate the selector GEMM requants under
    # the real backbone table): activation exps from the calibrated
    # ranges, per-tensor weight exps from the float selectors, stats at
    # the input's scale.
    in_exps = {1: table.activation_exp("b3_out"),
               2: table.activation_exp("b6_out"),
               3: table.activation_exp("b9_out")}
    sel_acts = {i: dict(local_out=-6, concat_out=-6, h1_out=-6,
                        h2_out=-6, logits_out=-7,
                        stats_out=in_exps[i], hw_hidden_out=-3)
                for i in (1, 2, 3)}
    sel_wexps = {}
    for i, sel in enumerate(selectors, start=1):
        _, wexp = export_selector(sel, device, in_exps[i], sel_acts[i])
        sel_wexps[i] = wexp
    table = selector_scale_table(table, in_exps, sel_wexps)

    from torchvision import datasets, transforms
    # Deterministic val subset (CenterCrop, first N images) as the training
    # set: identical to the teacher-logit pass, so the cache stays valid
    # across epochs (no augmentation mismatch).
    transform = transforms.Compose([
        transforms.Resize(256,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(
        r"D:\SEU_Liubo\prj\HeatViT\data\imagenet\val", transform=transform)
    subset = torch.utils.data.Subset(dataset, range(args.max_images))
    loader = torch.utils.data.DataLoader(
        IndexedDataset(subset), batch_size=args.batch_size, shuffle=True,
        num_workers=0)

    opt = torch.optim.AdamW(selectors.parameters(), lr=args.lr,
                            weight_decay=0.05)
    crit = nn.CrossEntropyLoss()
    t_scale = args.temperature

    steps = 0
    run_ce = 0.0
    run_sp = 0.0
    run_di = 0.0
    run_rg = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        for img, label, idx in loader:
            img = img.to(device)
            label = label.to(device)
            logits, sp_loss, rg_loss = train_forward(
                model, selectors, img, STAGE_TARGETS, table,
                threshold=0.5 + args.threshold_offset)
            ce = crit(logits, label)
            distill = torch.tensor(0.0, device=device)
            if teacher_logits is not None:
                teach = teacher_logits[idx].to(device).float() / t_scale
                distill = nn.functional.cross_entropy(
                    logits.float() / t_scale, teach.softmax(dim=1))
            loss = ce + args.distill_weight * distill \
                + args.sparsity_weight * sp_loss \
                + args.range_weight * rg_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selectors.parameters(), 1.0)
            opt.step()
            # Keep the decision heads inside the int8-at--7 contract:
            # weights +-0.99, score logit biases +-0.5 (the h2*w3 term
            # must still leave headroom inside +-0.99).
            with torch.no_grad():
                for sel in selectors:
                    for head in sel.score:
                        head[-1].bias.clamp_(-0.5, 0.5)
                    for m in sel.modules():
                        if isinstance(m, nn.Linear):
                            m.weight.clamp_(-0.9, 0.9)
            run_ce += ce.item()
            run_sp += sp_loss.item()
            run_di += distill.item()
            run_rg += rg_loss.item()
            steps += 1
            if steps % 20 == 0:
                print(f"step {steps}: ce {run_ce / 20:.2f} "
                      f"distill {run_di / 20:.2f} "
                      f"sparsity {run_sp / 20:.4f} "
                      f"range {run_rg / 20:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                run_ce = 0.0
                run_sp = 0.0
                run_di = 0.0
                run_rg = 0.0

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"selectors": [], "table": dict(table.activations),
               "weight_exps": {}, "sel_acts": {str(i): sel_acts[i]
                                                for i in (1, 2, 3)},
               "in_exps": {str(i): in_exps[i] for i in (1, 2, 3)},
               "selectors_float": [s.state_dict() for s in selectors]}
    for i, sel in enumerate(selectors, start=1):
        q, wexp = export_selector(sel, device, in_exps[i], sel_acts[i])
        payload["selectors"].append(q)
        payload["weight_exps"][str(i)] = wexp
    torch.save(payload, out_path)
    print(f"selectors -> {out_path}")


if __name__ == "__main__":
    main()
