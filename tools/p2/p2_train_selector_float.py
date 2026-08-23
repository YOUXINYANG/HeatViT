#!/usr/bin/env python3
"""P2-C selector training against the differentiable FLOAT backbone.

The exact integer simulator blocks the task gradient (its ops are int64),
so token choice cannot be learned from it — only the keep rate can.
Following the paper's methodology, this script trains the three Token
Selectors against the frozen FLOAT DeiT-T backbone with soft hard-mask
STE pruning + package token + distillation to the unpruned float
logits + keep-fraction sparsity loss. The trained selectors are then
quantized (per-tensor exponents, calibrated selector scale entries) and
evaluated in the EXACT integer simulator by p2_selector_eval.py.

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p2_train_selector_float.py \
      --max-images 8192 --epochs 6 --lr 3e-4 \
      --out p2_out/selectors_float_train.pt
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from tools.p2.p2_quantize import load_state_dict
from tools.p2.p2_train_selector import (
    STAGE_TARGETS,
    IndexedDataset,
    TrainSelector,
    export_selector,
    selector_scale_table,
)
from tools.p2.scale_table import ScaleTable

SELECTOR_BLOCKS = (4, 7, 10)


class MaskedAttention(nn.Module):
    """timm-style 3-head attention with a per-sample key validity mask."""

    def __init__(self, dim=192, heads=3):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, valid):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn + (1 - valid.unsqueeze(1).to(attn.dtype)) * (-1e9)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class PrunedBlock(nn.Module):
    """A DeiT block rebuilt from the frozen timm block with masked
    attention; the qkv/proj weights are SHARED with the timm module
    (frozen), so gradients only flow through the selectors."""

    def __init__(self, blk):
        super().__init__()
        self.norm1 = blk.norm1
        self.norm2 = blk.norm2
        self.mlp = blk.mlp
        self.attn = MaskedAttention()
        self.attn.qkv = blk.attn.qkv
        self.attn.proj = blk.attn.proj

    def forward(self, x, valid):
        x = x + self.attn(self.norm1(x), valid)
        x = x + self.mlp(self.norm2(x))
        return x


class PrunedDeiT(nn.Module):
    def __init__(self, timm_model):
        super().__init__()
        self.patch_embed = timm_model.patch_embed
        self.cls_token = timm_model.cls_token
        self.pos_embed = timm_model.pos_embed
        self.blocks = nn.ModuleList([PrunedBlock(b) for b in timm_model.blocks])
        self.norm = timm_model.norm
        self.head = timm_model.head

    def forward(self, img, selectors, threshold, targets):
        b = img.shape[0]
        x = self.patch_embed(img)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        valid = torch.ones(b, 197, 1, device=img.device)
        sp_loss = torch.tensor(0.0, device=img.device)
        range_loss = torch.tensor(0.0, device=img.device)
        sel_i = 0
        for i, blk in enumerate(self.blocks):
            if i + 1 in SELECTOR_BLOCKS:
                sel = selectors[sel_i]
                fused, keep_soft, rl = sel(x[:, 1:])
                range_loss = range_loss + rl
                fused = fused.clamp(0, 1)
                fused = torch.cat(
                    [torch.ones(b, 1, device=fused.device), fused], dim=1)
                keep_soft = torch.cat(
                    [torch.ones(b, 1, 3, device=keep_soft.device),
                     keep_soft], dim=1)
                hard = fused >= threshold
                mask = hard.float() + fused - fused.detach()      # STE
                parts = x * (1 - mask.unsqueeze(-1))
                wnum = (parts * keep_soft.mean(-1, keepdim=True)
                        * (1 - mask.unsqueeze(-1))).sum(dim=1)
                den = (keep_soft.mean(-1) * (1 - mask)).sum(
                    1, keepdim=True).clamp(min=1e-6)
                package = wnum / den                               # [B,192]
                kept = x * mask.unsqueeze(-1)
                x = torch.cat([kept, package.unsqueeze(1)], dim=1)
                valid = torch.cat(
                    [valid * mask.unsqueeze(-1),
                     torch.ones(b, 1, 1, device=valid.device)], dim=1)
                rate = mask[:, 1:].mean()
                sp_loss = sp_loss + (rate - targets[sel_i]) ** 2
                sel_i += 1
            x = blk(x, valid)
        logits = self.head(self.norm(x[:, 0]))
        return logits, sp_loss, range_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sparsity-weight", type=float, default=20.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--range-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--threshold-offset", type=float, default=-0.008,
                        help="training threshold = 0.5 + offset; compensates "
                             "the sim's systematic fused-score bias")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--table", default="p2_out/ivit/scale_table_legacy.json")
    parser.add_argument("--out", default="p2_out/selectors_float_train.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    import timm
    state = load_state_dict()
    fm = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    fm.load_state_dict(state, strict=True)
    fm = fm.to(device).eval()
    for p in fm.parameters():
        p.requires_grad_(False)
    pruned = PrunedDeiT(fm).to(device)

    selectors = nn.ModuleList([TrainSelector() for _ in range(3)]).to(device)

    table = ScaleTable.load(REPO_ROOT / args.table).validate()
    in_exps = {1: table.activation_exp("b3_out"),
               2: table.activation_exp("b6_out"),
               3: table.activation_exp("b9_out")}
    sel_acts = {i: dict(local_out=-6, concat_out=-6, h1_out=-6,
                        h2_out=-6, logits_out=-7,
                        stats_out=in_exps[i], hw_hidden_out=-3)
                for i in (1, 2, 3)}

    from torchvision import datasets, transforms
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
    threshold = 0.5 + args.threshold_offset
    t_scale = args.temperature

    steps = 0
    run_ce = 0.0
    run_di = 0.0
    run_sp = 0.0
    run_rg = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        for img, label, idx in loader:
            img = img.to(device)
            label = label.to(device)
            logits, sp_loss, rg_loss = pruned(
                img, selectors, threshold, STAGE_TARGETS)
            ce = crit(logits, label)
            with torch.no_grad():
                teach_logits = fm(img)
            distill = nn.functional.cross_entropy(
                logits / t_scale, (teach_logits / t_scale).softmax(dim=1))
            loss = ce + args.distill_weight * distill \
                + args.sparsity_weight * sp_loss \
                + args.range_weight * rg_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selectors.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for sel in selectors:
                    for head in sel.score:
                        head[-1].bias.clamp_(-0.5, 0.5)
                    for m in sel.modules():
                        if isinstance(m, nn.Linear):
                            m.weight.clamp_(-0.9, 0.9)
            run_ce += ce.item()
            run_di += distill.item()
            run_sp += sp_loss.item()
            run_rg += rg_loss.item()
            steps += 1
            if steps % 20 == 0:
                print(f"step {steps}: ce {run_ce / 20:.2f} "
                      f"distill {run_di / 20:.2f} "
                      f"sparsity {run_sp / 20:.4f} "
                      f"range {run_rg / 20:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                run_ce = 0.0
                run_di = 0.0
                run_sp = 0.0
                run_rg = 0.0

    sel_wexps = {}
    payload_selectors = []
    for i, sel in enumerate(selectors, start=1):
        q, wexp = export_selector(sel, device, in_exps[i], sel_acts[i])
        payload_selectors.append(q)
        sel_wexps[str(i)] = wexp
    out_path = REPO_ROOT / args.out
    payload = {
        "selectors": payload_selectors,
        "weight_exps": sel_wexps,
        "sel_acts": {str(i): sel_acts[i] for i in (1, 2, 3)},
        "in_exps": {str(i): in_exps[i] for i in (1, 2, 3)},
        "selectors_float": [s.state_dict() for s in selectors],
    }
    torch.save(payload, out_path)
    print(f"selectors -> {out_path}")


if __name__ == "__main__":
    main()
