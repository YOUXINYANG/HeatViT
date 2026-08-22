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
        # Start keep-biased (like the paper's MHTS init).
        for head in self.score:
            last = head[-1]
            last.bias.data[0] = 0.0
            last.bias.data[1] = 5.0

    def forward(self, tokens_deq):
        """tokens_deq: [B,N,192] float; returns (fused, keep_prob_soft)."""
        b, n, d = tokens_deq.shape
        th = tokens_deq.reshape(b, n, 3, 64)
        local_h = []
        for h in range(3):
            local_h.append(self.local[h](th[:, :, h]))          # [B,N,32]
        local = torch.stack(local_h, dim=2)                     # [B,N,3,32]
        global_f = local.mean(dim=1, keepdim=True)              # [B,1,3,32]
        e = torch.cat([local, global_f.expand(-1, n, -1, -1)],
                      dim=-1)                                   # [B,N,3,64]
        keep_h = []
        for h in range(3):
            logits = self.score[h](e[:, :, h])                  # [B,N,2]
            keep_h.append(torch.softmax(logits, dim=-1)[..., 1])
        keep_scores = torch.stack(keep_h, dim=-1)               # [B,N,3]
        stats = th.mean(dim=-1)                                 # [B,N,3]
        hw = plan_sigmoid_float(self.head_weight(stats))        # [B,N,3]
        if keep_scores.shape != hw.shape:
            raise RuntimeError(
                f"selector shape mismatch: tokens={tuple(tokens_deq.shape)} "
                f"keep={tuple(keep_scores.shape)} hw={tuple(hw.shape)} "
                f"local={tuple(local.shape)} stats={tuple(stats.shape)}")
        den = hw.sum(dim=-1).clamp(min=1e-6)                     # [B,N]
        fused = (keep_scores * hw).sum(dim=-1) / den            # [B,N]
        return fused, keep_scores


def train_forward(model, selectors, img_q, targets, s):
    """One batched forward with soft-gradient/hard-mask selectors.

    Returns (logits, sparsity_loss). The backbone is the exact integer
    simulator; selectors see dequantized int8 features; the package token
    uses the soft keep scores (gradient path) and the hard mask uses a
    straight-through threshold (RTL semantics).
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
    sel_idx = 0
    for n in range(1, 13):
        in_exp = s.activation_exp("act_tokens") if n == 1 \
            else s.activation_exp(f"b{n - 1}_out")
        if n in S.SELECTOR_BLOCKS:
            sel = selectors[sel_idx]
            scale = 2.0 ** in_exp
            deq = tokens.to(torch.float32) * scale
            fused, keep_soft = sel(deq)
            fused = fused.clamp(0, 1)
            # CLS always kept.
            fused = torch.cat([torch.ones(b, 1, device=fused.device),
                               fused[:, 1:]], dim=1)
            keep_soft = torch.cat(
                [torch.ones(b, 1, 3, device=keep_soft.device),
                 keep_soft[:, 1:]], dim=1)
            hard = fused >= 0.5
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
            # Sparsity loss over the normal candidates (excludes CLS and the
            # appended package slot).
            rate = fused[:, 1:].mean()
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
    return logits[:, 0].float(), sparsity_loss

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
        q16 = S.requant(score, score_exp, score_exp + 4, 24)
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


def export_selector(sel, device, stage):
    """Quantize a trained selector into RTL int8/Q0.16 tensors + scales.

    PyTorch Linear stores weight [out, in]; the RTL/golden layout is
    [in, out], so every matrix is transposed here.
    """
    out = {}
    for h in range(3):
        out[f"local_w{h}"] = sel.local[h][0].weight.data.detach().cpu().t()
        out[f"local_b{h}"] = sel.local[h][0].bias.data.detach().cpu()
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

    # The RTL/golden selector path uses uniform scale exp -7 for all
    # selector weights (descriptor SCALES), so quantize at -7 exactly
    # rather than per-tensor exponents.
    q = {}
    q[f"local_w"] = torch.stack(
        [quantize_int8(out[f"local_w{h}"], -7) for h in range(3)])
    q[f"local_b"] = torch.stack(
        [quantize_bias(out[f"local_b{h}"], -14) for h in range(3)])
    q["score_w1"] = torch.stack(
        [quantize_int8(out[f"score_w1_{h}"], -7) for h in range(3)])
    q["score_b1"] = torch.stack(
        [quantize_bias(out[f"score_b1_{h}"], -14) for h in range(3)])
    q["score_w2"] = torch.stack(
        [quantize_int8(out[f"score_w2_{h}"], -7) for h in range(3)])
    q["score_b2"] = torch.stack(
        [quantize_bias(out[f"score_b2_{h}"], -14) for h in range(3)])
    q["score_w3"] = torch.stack(
        [quantize_int8(out[f"score_w3_{h}"], -7) for h in range(3)])
    q["score_b3"] = torch.stack(
        [quantize_bias(out[f"score_b3_{h}"], -14) for h in range(3)])
    q["hw_w1"] = quantize_int8(out["hw_w1"], -7)
    q["hw_b1"] = quantize_bias(out["hw_b1"], -14)
    q["hw_w2"] = quantize_int8(out["hw_w2"], -7)
    q["hw_b2"] = quantize_bias(out["hw_b2"], -14)
    return q


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
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    state = load_state_dict()
    floats = to_heatvit_tensors(state)
    wexp = build_weight_exps(floats)
    table = ScaleTable.load(REPO_ROOT / args.table)

    model = build_model(floats, table, device)
    selectors = nn.ModuleList([TrainSelector() for _ in range(3)]).to(device)

    from torchvision import datasets, transforms
    train_dir = r"D:\SEU_Liubo\prj\HeatViT\data\imagenet\train"
    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    if args.max_images and args.max_images < len(dataset):
        gen = torch.Generator().manual_seed(20260815)
        idx = torch.randperm(len(dataset), generator=gen)[:args.max_images]
        dataset = torch.utils.data.Subset(dataset, idx.tolist())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW(selectors.parameters(), lr=args.lr,
                            weight_decay=0.05)
    crit = nn.CrossEntropyLoss()

    steps = 0
    run_ce = 0.0
    run_sp = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        for img, label in loader:
            img = img.to(device)
            label = label.to(device)
            logits, sp_loss = train_forward(model, selectors, img,
                                            STAGE_TARGETS, table)
            ce = crit(logits, label)
            loss = ce + args.sparsity_weight * sp_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selectors.parameters(), 1.0)
            opt.step()
            run_ce += ce.item()
            run_sp += sp_loss.item()
            steps += 1
            if steps % 20 == 0:
                print(f"step {steps}: ce {run_ce / 20:.2f} "
                      f"sparsity {run_sp / 20:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                run_ce = 0.0
                run_sp = 0.0

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"selectors": [], "table": dict(table.activations),
               "selectors_float": [s.state_dict() for s in selectors]}
    for sel in selectors:
        payload["selectors"].append(export_selector(sel, device, None))
    torch.save(payload, out_path)
    print(f"selectors -> {out_path}")


if __name__ == "__main__":
    main()
