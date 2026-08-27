#!/usr/bin/env python3
"""Collect stage-2/3 supervised data along the PRUNED path.

Stage 1 data (unpruned b3_out + labels) comes from the existing cache
(p2_selector_sup_data.py); stages 2 and 3 must see the PRUNED token set,
so this script replays the exact integer simulator with the
already-trained selectors and collects, per image:

  * features: the dequantized block outputs of the stage's candidate
    tokens (kept tokens plus the incoming package) at the selector
    position;
  * labels: the float teacher's CLS attention (aggregated over the
    remaining blocks) at the candidates' original indices, renormalized;
    package tokens get the mean importance of the tokens they carry.

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p2_selector_sup_data2.py \\
      --selectors p2_out/selectors_sup.pt --max-images 8192 \\
      --out p2_out/selector_sup_data2.pt
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_quantize import (
    build_model,
    load_state_dict,
    make_val_loader,
    to_heatvit_tensors,
)
from tools.p2.p2_selector_eval import load_selectors
from tools.p2.p2_sim import (
    KEEP_THRESHOLD,
    _head_gemm_gelu,
    gemm_int8,
    plan_sigmoid,
    requant,
    residual_add,
    round_div,
    sat,
    softmax_selector,
    token_selector,
    transformer_block,
)
from tools.p2.scale_table import ScaleTable


def selector_mask(toks, p, s, idx, in_exp, package_present):
    """Exact sim selector keep-mask over the normal candidate rows."""
    cand = toks[1:]
    c = cand.shape[0]
    reshaped = cand.reshape(c, 3, 64)
    locals_t = torch.stack([_head_gemm_gelu(
        reshaped[:, h, :], p.local_w[h], p.local_b[h], in_exp,
        s.weight_exp(f"s{idx}_local_w"),
        s.activation_exp(f"s{idx}_local_out")) for h in range(3)])
    count_t = torch.tensor(c, dtype=torch.int64, device=toks.device)
    glob = sat(round_div(locals_t.sum(dim=1), count_t), 8).to(torch.int8)
    lgl = torch.cat([locals_t, glob.unsqueeze(1).expand(3, c, 32)], -1)
    h1 = torch.stack([_head_gemm_gelu(
        lgl[h], p.score_w1[h], p.score_b1[h],
        s.activation_exp(f"s{idx}_concat_out"),
        s.weight_exp(f"s{idx}_score_w1"),
        s.activation_exp(f"s{idx}_h1_out")) for h in range(3)])
    h2 = torch.stack([_head_gemm_gelu(
        h1[h], p.score_w2[h], p.score_b2[h],
        s.activation_exp(f"s{idx}_h1_out"),
        s.weight_exp(f"s{idx}_score_w2"),
        s.activation_exp(f"s{idx}_h2_out")) for h in range(3)])
    logits = torch.stack([gemm_int8(
        h2[h], p.score_w3[h], p.score_b3[h],
        s.activation_exp(f"s{idx}_h2_out"),
        s.weight_exp(f"s{idx}_score_w3"),
        s.activation_exp(f"s{idx}_logits_out"), 8) for h in range(3)])
    scores_q16 = torch.stack([requant(
        logits[h].to(torch.int64), s.activation_exp(f"s{idx}_logits_out"),
        -16, 24) for h in range(3)])
    hs = torch.stack([softmax_selector(scores_q16[h]) for h in range(3)])
    lane_sum = reshaped.to(torch.int64).sum(dim=-1)
    stats = sat(round_div(lane_sum, torch.tensor(64, dtype=torch.int64,
                                                 device=toks.device)),
                8).to(torch.int8)
    hw_hidden = _head_gemm_gelu(
        stats, p.hw_w1, p.hw_b1, s.activation_exp(f"s{idx}_stats_out"),
        s.weight_exp(f"s{idx}_hw_w1"),
        s.activation_exp(f"s{idx}_hw_hidden_out"))
    hw_acc = hw_hidden.to(torch.float64) @ p.hw_w2.to(torch.float64).T \
        + p.hw_b2.to(torch.float64)
    hw_q16 = requant(hw_acc.round().to(torch.int64),
                     s.activation_exp(f"s{idx}_hw_hidden_out")
                     + s.weight_exp(f"s{idx}_hw_w2"), -16, 24)
    hw = plan_sigmoid(hw_q16)
    st = hs.T
    den = hw.sum(dim=-1)
    fused = torch.where(
        den == 0,
        round_div(st.sum(dim=-1),
                  torch.tensor(3, dtype=torch.int64, device=toks.device)),
        round_div((st * hw).sum(dim=-1), den.clamp(min=1))).clamp(0, 65536)
    normal = c - 1 if package_present else c
    keep = (fused >= KEEP_THRESHOLD)[:normal]
    return torch.nonzero(keep, as_tuple=False).flatten().tolist()


def collect_stage(model, table, loader, device, fm_attn, stage):
    """Replay the pruned sim up to selector ``stage``; collect the
    candidates' features + labels, zero-padded to a fixed per-stage
    width with a validity mask. Returns (features, labels, mask)."""
    s_ = table
    cap = 197
    feats = []
    labels = []
    masks = []
    with torch.no_grad():
        for img, _ in loader:
            img = img[0].to(device)
            inp_exp = s_.activation_exp("input")
            img_q = torch.clamp(torch.round(img / (2.0 ** inp_exp)),
                                -128, 127).to(torch.int8)
            img_nhwc = img_q.permute(1, 2, 0)
            patches = img_nhwc.reshape(14, 16, 14, 16, 3) \
                .permute(0, 2, 1, 3, 4).reshape(196, 768)
            embed = gemm_int8(patches, model.patch_w, model.patch_b,
                              inp_exp, s_.weight_exp("patch_w"),
                              s_.activation_exp("act_patch_embed"), 8)
            row0 = residual_add(model.cls.unsqueeze(0),
                                s_.weight_exp("cls"), model.pos[:1],
                                s_.weight_exp("pos"),
                                s_.activation_exp("act_tokens"))
            rows1 = residual_add(embed,
                                 s_.activation_exp("act_patch_embed"),
                                 model.pos[1:], s_.weight_exp("pos"),
                                 s_.activation_exp("act_tokens"))
            toks = torch.cat([row0, rows1], 0)
            for n in (1, 2, 3):
                toks = transformer_block(
                    toks, model.blocks[n - 1], s_, n,
                    s_.activation_exp("act_tokens") if n == 1 else
                    s_.activation_exp(f"b{n - 1}_out"))
            # selector 1
            keep1 = selector_mask(toks, model.selectors[0], s_, 1,
                                  s_.activation_exp("b3_out"), False)
            out, pkg1 = token_selector(toks, False, model.selectors[0], s_,
                                       1, s_.activation_exp("b3_out"))
            toks = out
            for n in (4, 5, 6):
                toks = transformer_block(
                    toks, model.blocks[n - 1], s_, n,
                    s_.activation_exp("b3_out") if n == 4 else
                    s_.activation_exp(f"b{n - 1}_out"))
            if stage == 2:
                exp = s_.activation_exp("b6_out")
                deq = toks[1:].float() * (2.0 ** exp)
                attn = fm_attn(img.unsqueeze(0))          # [12,197]
                agg = attn[6:].sum(dim=0)
                imp = agg[1:] / agg[1:].max().clamp(min=1e-6)
                lab = imp[keep1].clone()
                if pkg1:
                    pruned = [i for i in range(196) if i not in keep1]
                    lab = torch.cat([lab, imp[pruned].mean().reshape(1)],
                                    dim=0)
                n = lab.shape[0]
                f = torch.zeros(cap, 192, dtype=torch.float16)
                f[:n] = deq[:n].half().cpu()
                l = torch.zeros(cap, dtype=torch.float16)
                l[:n] = lab.half().cpu()
                m = torch.zeros(cap, dtype=torch.bool)
                m[:n] = True
                feats.append(f)
                labels.append(l)
                masks.append(m)
                continue
            # stage 3: selector 2 on the pruned set
            keep2 = selector_mask(toks, model.selectors[1], s_, 2,
                                  s_.activation_exp("b6_out"), pkg1)
            out, pkg2 = token_selector(toks, pkg1, model.selectors[1], s_,
                                       2, s_.activation_exp("b6_out"))
            toks = out
            for n in (7, 8, 9):
                toks = transformer_block(
                    toks, model.blocks[n - 1], s_, n,
                    s_.activation_exp("b6_out") if n == 7 else
                    s_.activation_exp(f"b{n - 1}_out"))
            exp = s_.activation_exp("b9_out")
            deq = toks[1:].float() * (2.0 ** exp)
            attn = fm_attn(img.unsqueeze(0))
            agg = attn[9:].sum(dim=0)
            imp = agg[1:] / agg[1:].max().clamp(min=1e-6)
            # stage-2 candidates: keep1 (orig idx) + package
            n1 = len(keep1)
            lab = []
            for j in keep2:
                if j < n1:
                    lab.append(imp[keep1[j]])
                else:
                    pruned = [i for i in range(196) if i not in keep1]
                    lab.append(imp[pruned].mean())
            lab = torch.stack(lab)
            if pkg2:
                pruned2 = [j for j in range(n1 + 1) if j not in keep2]
                vals = []
                for j in pruned2:
                    if j < n1:
                        vals.append(imp[keep1[j]])
                    else:
                        p_ = [i for i in range(196) if i not in keep1]
                        vals.append(imp[p_].mean())
                lab = torch.cat([lab, torch.stack(vals).mean().reshape(1)],
                                dim=0)
            n = lab.shape[0]
            f = torch.zeros(cap, 192, dtype=torch.float16)
            f[:n] = deq[:n].half().cpu()
            l = torch.zeros(cap, dtype=torch.float16)
            l[:n] = lab.half().cpu()
            m = torch.zeros(cap, dtype=torch.bool)
            m[:n] = True
            feats.append(f)
            labels.append(l)
            masks.append(m)
    return (torch.stack(feats, 0), torch.stack(labels, 0),
            torch.stack(masks, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selectors", required=True)
    parser.add_argument("--max-images", type=int, default=8192)
    parser.add_argument("--table", default="p2_out/ivit/scale_table_legacy.json")
    parser.add_argument("--backbone-checkpoint", default=None,
                        help="P4-B: use a QAT checkpoint's floats (HeatViT "
                             "layout) as the int8 backbone and its float "
                             "mirror as the attention teacher")
    parser.add_argument("--base", default="p2_out/selector_sup_data.pt",
                        help="stage-1 data cache (features/labels of the "
                             "same backbone)")
    parser.add_argument("--out", default="p2_out/selector_sup_data2.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    if args.backbone_checkpoint:
        ck = torch.load(REPO_ROOT / args.backbone_checkpoint,
                        map_location="cpu", weights_only=False)
        floats = {k: v for k, v in ck["floats"].items()}
        from tools.p2.qat_data import heatvit_to_timm_state
        state = heatvit_to_timm_state(floats)
        print(f"backbone: {args.backbone_checkpoint} (QAT floats + float "
              f"mirror teacher)")
    else:
        state = load_state_dict()
        floats = to_heatvit_tensors(state)
    table = ScaleTable.load(REPO_ROOT / args.table).validate()
    model = build_model(floats, table, device)
    load_selectors(REPO_ROOT / args.selectors, model, device, table)

    import timm
    fm = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    fm.load_state_dict(state, strict=True)
    fm = fm.to(device).eval()
    qkv = {}
    hooks = []
    for i, blk in enumerate(fm.blocks):
        def h(m, a, o, idx=i):
            qkv[idx] = o.detach()
        hooks.append(blk.attn.qkv.register_forward_hook(h))

    def fm_attn(img):
        qkv.clear()
        with torch.no_grad():
            fm(img.to(device))
        rows = []
        b = img.shape[0]
        for i in range(12):
            o = qkv[i]
            q, k, _ = o.reshape(b, 197, 3, 3, 64).permute(2, 3, 0, 1, 4)
            attn = torch.softmax((q @ k.transpose(-2, -1)) / 8.0, dim=-1)
            rows.append(attn[:, :, 0, :].mean(dim=0))     # [B,197]
        return torch.stack(rows, dim=1)[0]                 # [12,197]

    try:
        loader = make_val_loader(args.max_images, batch_size=1)
        print("collecting stage-2 data (pruned path) ...", flush=True)
        t0 = time.time()
        f2, l2, m2 = collect_stage(model, table, loader, device, fm_attn, 2)
        print(f"  stage-2 done in {time.time() - t0:.0f}s "
              f"{tuple(f2.shape)} / {tuple(l2.shape)}", flush=True)
        loader = make_val_loader(args.max_images, batch_size=1)
        print("collecting stage-3 data (pruned path) ...", flush=True)
        t0 = time.time()
        f3, l3, m3 = collect_stage(model, table, loader, device, fm_attn, 3)
        print(f"  stage-3 done in {time.time() - t0:.0f}s "
              f"{tuple(f3.shape)} / {tuple(l3.shape)}", flush=True)
    finally:
        for h in hooks:
            h.remove()

    base = torch.load(REPO_ROOT / args.base,
                      map_location="cpu", weights_only=False)
    m1 = torch.ones(base["features"]["1"].shape[0], 196, dtype=torch.bool)
    payload = {
        "features": {"1": base["features"]["1"], "2": f2, "3": f3},
        "labels": {"1": base["labels"]["1"], "2": l2, "3": l3},
        "masks": {"1": m1, "2": m2, "3": m3},
        "count": args.max_images,
    }
    out_path = REPO_ROOT / args.out
    torch.save(payload, out_path)
    print(f"data -> {out_path}")


if __name__ == "__main__":
    main()
