#!/usr/bin/env python3
"""P3 QAT driver: train / eval / recalib (decision D1-A pipeline).

train    QAT fine-tuning of the official DeiT-T checkpoint under the
         frozen per-tensor power-of-2 scale table: float forward +
         straight-through fake-quant at the RTL contract boundaries
         (tools/p2/qat_model.py). Checkpoints keep the HeatViT-layout
         float tensors, so every stage downstream (bit-exact eval, weight
         export) reuses the existing P2 machinery unchanged.
eval     bit-exact Top-1 of a QAT checkpoint on val images via
         exact_forward (p2_quantize.build_model + p2_sim_ivit.
         forward_batch_cfg, contract config), plus the train-path Top-1
         on the same images to expose the train/inference gap.
recalib  post-training activation recalibration (decision D3): collects
         float histograms on the QAT weights through the timm hooks and
         re-picks activation exponents with the legacy LN clamp
         (exp <= 0). Weight exponents stay frozen at the original table.

Usage (torch venv):
  .venv-torch\\Scripts\\python tools/p2/p2_qat.py train \
      --max-images 32768 --epochs 1 --eval-images 500
  .venv-torch\\Scripts\\python tools/p2/p2_qat.py eval \
      --checkpoint p2_out/qat/checkpoint.pt --images 5000
  .venv-torch\\Scripts\\python tools/p2/p2_qat.py recalib \
      --checkpoint p2_out/qat/checkpoint.pt --calib 2048 \
      --out p2_out/qat/scale_table_after.json
  # staged training: the next tier starts from the previous stage's weights
  # (fresh optimizer + fresh cosine schedule; see --init-checkpoint)
  .venv-torch\\Scripts\\python tools/p2/p2_qat.py train \
      --init-checkpoint p2_out/qat/quick32k/best.pt \
      --max-images 128000 --epochs 10 --eval-images 1000
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_quantize import (
    build_weight_exps,
    collect_float_histograms,
    load_state_dict,
    make_val_loader,
    pick_activation_exps,
    to_heatvit_tensors,
)
from tools.p2.qat_data import heatvit_to_timm_state, make_train_loader
from tools.p2.qat_model import QatDeiT, exact_forward
from tools.p2.qat_selector import attach_selectors, merge_selector_scales
from tools.p2.scale_table import ScaleTable

DEFAULT_TABLE = "p2_out/scale_table.json"
DEFAULT_OUT_DIR = "p2_out/qat"
SPOT_IMAGES = int(os.environ.get("HEATVIT_SPOT_IMAGES", "5000"))


def pick_device(arg):
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def build_qat(floats, table, device):
    return QatDeiT(floats, table).to(device)


def make_eval_loader(images, batch_size=1, sampling="head",
                     seed=20260815):
    """Build a validation loader with an explicit subset policy."""
    return make_val_loader(images, batch_size=batch_size,
                           sampling=sampling, seed=seed)


def eval_val(qat, loader, device, exact=True, prune=False):
    """Top-1 on val; ``exact`` runs the bit-exact integer path, otherwise
    the differentiable train path (no_grad). ``prune`` switches the train
    path to the P4 pruned forward."""
    correct = total = 0
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            if exact:
                logits, _ = exact_forward(qat, img)
            else:
                logits = qat(img, prune=prune)
            correct += (logits.argmax(dim=1).cpu() == label).sum().item()
            total += label.size(0)
    return 100.0 * correct / total, correct, total


def eval_pruned_exact(qat, selector_path, images, device, sampling="head",
                      seed=20260815):
    """Bit-exact PRUNED Top-1 (P4): the same tensors + table + frozen
    selectors through p2_sim_ivit.forward_image_cfg(prune=True), the
    deployment contract path used by tools/p2/qat_prune_eval.py."""
    from tools.p2.p2_quantize import build_model
    from tools.p2.qat_prune_eval import load_selectors
    from tools.p2.p2_sim_ivit import NonlinConfig, forward_image_cfg
    table = qat.table
    model = build_model(qat.tensors_dict(), table, device)
    load_selectors(selector_path, model, device, table)
    loader = make_eval_loader(images, sampling=sampling, seed=seed)
    correct = total = 0
    with torch.no_grad():
        for img, label in loader:
            logits, _, _ = forward_image_cfg(
                model, img[0].to(device), NonlinConfig(), prune=True)
            correct += (logits.argmax().item() == label.item())
            total += 1
    return 100.0 * correct / total, correct, total


# ---- train -----------------------------------------------------------------
def train(args):
    if args.resume and args.init_checkpoint:
        raise SystemExit("--resume and --init-checkpoint are mutually "
                         "exclusive (crash recovery vs staged training)")
    device = pick_device(args.device)
    print(f"device={device}")
    torch.manual_seed(args.seed)

    if args.amp == "bf16":
        print("amp=bf16 (autocast on the forward)")
    elif args.amp == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        print("amp=tf32 (matmul allow_tf32)")
    else:
        print("amp=off (fp32)")

    table = ScaleTable.load(REPO_ROOT / args.table)
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_path = None
    if args.selectors:
        sel_path = REPO_ROOT / args.selectors
        sel_payload = torch.load(sel_path, map_location="cpu",
                                 weights_only=False)
        merge_selector_scales(table, sel_payload)
        print(f"selectors: {sel_path} (frozen; s{{i}}_* scales merged)")

    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        floats = {k: v for k, v in ckpt["floats"].items()}
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("step", 0)
        best_acc = ckpt.get("best_exact", 0.0)
        print(f"resumed {ckpt_path} (epoch {start_epoch}, step {start_step}, "
              f"best exact {best_acc:.2f}%)")
    elif args.init_checkpoint:
        ckpt_path = Path(args.init_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        floats = {k: v for k, v in ckpt["floats"].items()}
        start_epoch, start_step, best_acc = 0, 0, 0.0
        print(f"init from {ckpt_path} (staged: fresh schedule + optimizer)")
    else:
        floats = to_heatvit_tensors(load_state_dict())
        start_epoch, start_step, best_acc = 0, 0, 0.0

    qat = build_qat(floats, table, device)
    if sel_path is not None:
        attach_selectors(qat, sel_payload, table)
    prune_mode = sel_path is not None
    if args.rate_weight > 0:
        if not prune_mode:
            raise SystemExit("--rate-weight requires --selectors")
        rate_targets = [float(t) for t in args.rate_targets.split(",")]
        if len(rate_targets) != 3:
            raise SystemExit("--rate-targets must have 3 values (88,45,32)")
        print(f"rate reg: weight={args.rate_weight} "
              f"targets={args.rate_targets}")
    else:
        rate_targets = None
    n_params = sum(p.numel() for p in qat.parameters())
    print(f"params: {n_params / 1e6:.2f}M")

    loader = make_train_loader(args.max_images, args.batch, args.workers,
                               randaug=args.randaug)
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    print(f"train images: {args.max_images or 'full'} -> "
          f"{steps_per_epoch} steps/epoch x {args.epochs} epochs")

    opt = torch.optim.AdamW(qat.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.999),
                            eps=1e-8)
    if args.resume and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    accum = max(1, args.accum)
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch / accum))

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(
            1, (total_steps // accum) - warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) \
            * (1.0 + math.cos(math.pi * progress))

    val_loader = make_eval_loader(
        args.eval_images, batch_size=32, sampling=args.eval_sampling,
        seed=args.seed)
    spot_loader = (make_eval_loader(SPOT_IMAGES, batch_size=32,
                                    sampling=args.eval_sampling,
                                    seed=args.seed)
                   if args.spot_every > 0 and not prune_mode else None)

    def evaluate(epoch):
        t0 = time.time()
        acc_e, c_e, n_e = eval_val(qat, val_loader, device, exact=True)
        t_e = time.time() - t0
        acc_t, c_t, n_t = eval_val(qat, val_loader, device, exact=False,
                                   prune=prune_mode)
        parts = (f"[eval e{epoch}] exact top1={acc_e:.2f}% ({c_e}/{n_e}, "
                 f"{t_e:.0f}s)  train-path top1={acc_t:.2f}% ({c_t}/{n_t})")
        if args.eval_prune > 0:
            t0 = time.time()
            acc_p, c_p, n_p = eval_pruned_exact(qat, sel_path,
                                                args.eval_prune, device,
                                                args.eval_sampling, args.seed)
            parts += (f"  pruned top1={acc_p:.2f}% ({c_p}/{n_p}, "
                      f"{time.time() - t0:.0f}s)")
            print(parts)
            return acc_p          # P4: best tracking follows the pruned path
        print(parts)
        return acc_e

    def save(path, epoch, step, best):
        torch.save({
            "floats": qat.tensors_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch, "step": step,
            "best_exact": best,
            "table": table.weights, "args": {k: str(v) for k, v in
                                             vars(args).items()},
        }, path)
        print(f"[save] {path}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        qat.train()
        running = 0.0
        running_rate = 0.0
        for i, (img, label) in enumerate(loader):
            step = epoch * steps_per_epoch + i + 1
            if step <= start_step:
                continue
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            if prune_mode and args.rate_weight > 0:
                logits, rates = qat(img, prune=True, return_rates=True)
                base = torch.nn.functional.cross_entropy(
                    logits, label, label_smoothing=args.smoothing)
                n_img = img.shape[0]
                rate_loss = sum(((r / n_img - t) / 197.0) ** 2
                                for r, t in zip(rates, rate_targets))
                loss = base + args.rate_weight * rate_loss
            elif args.amp == "bf16":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = qat(img, prune=prune_mode)
                loss = torch.nn.functional.cross_entropy(
                    logits, label, label_smoothing=args.smoothing)
            else:
                logits = qat(img, prune=prune_mode)
                loss = torch.nn.functional.cross_entropy(
                    logits, label, label_smoothing=args.smoothing)
            (loss / accum).backward()
            running += loss.item()
            if prune_mode and args.rate_weight > 0:
                running_rate += rate_loss.item()
            if (i + 1) % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step_done = (step - 1) // accum
                for g in opt.param_groups:
                    g["lr"] = lr_at(step_done)
            if (i + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                rate_part = f" rate={running_rate / args.log_every:.4f}" \
                    if args.rate_weight > 0 else ""
                print(f"e{epoch + 1}/{args.epochs} "
                      f"step {i + 1}/{steps_per_epoch} "
                      f"loss={running / args.log_every:.4f}{rate_part} "
                      f"lr={opt.param_groups[0]['lr']:.2e} "
                      f"{elapsed / (i + 1):.2f}s/step")
                running = 0.0
                running_rate = 0.0
        acc = evaluate(epoch + 1)
        save(out_dir / "checkpoint.pt", epoch + 1,
             epoch * steps_per_epoch + steps_per_epoch, acc)
        is_spot = (args.spot_every > 0
                   and ((epoch + 1) % args.spot_every == 0
                        or epoch + 1 == args.epochs))
        if is_spot:
            t0 = time.time()
            if prune_mode:
                acc_s, c_s, n_s = eval_pruned_exact(
                    qat, sel_path, SPOT_IMAGES, device,
                    args.eval_sampling, args.seed)
            else:
                acc_s, c_s, n_s = eval_val(qat, spot_loader, device,
                                           exact=True)
            print(f"[spot e{epoch + 1}] {SPOT_IMAGES} top1={acc_s:.2f}% "
                  f"({c_s}/{n_s}, {time.time() - t0:.0f}s)")
            if acc_s > best_acc:
                best_acc = acc_s
                save(out_dir / "best.pt", epoch + 1,
                     epoch * steps_per_epoch + steps_per_epoch, acc_s)
        elif acc > best_acc:
            best_acc = acc
            save(out_dir / "best.pt", epoch + 1,
                 epoch * steps_per_epoch + steps_per_epoch, acc)
    print(f"done. best exact top1={best_acc:.2f}% -> {out_dir}")


# ---- eval ------------------------------------------------------------------
def evaluate_cmd(args):
    device = pick_device(args.device)
    table = ScaleTable.load(REPO_ROOT / args.table)
    ckpt = torch.load(args.checkpoint, map_location="cpu",
                      weights_only=False)
    qat = build_qat(ckpt["floats"], table, device)
    loader = make_eval_loader(args.images, batch_size=32,
                              sampling=args.sampling, seed=args.seed)
    acc_e, c_e, n_e = eval_val(qat, loader, device, exact=True)
    print(f"[exact ] top1={acc_e:.2f}% ({c_e}/{n_e})")
    acc_t, c_t, n_t = eval_val(qat, loader, device, exact=False)
    print(f"[train ] top1={acc_t:.2f}% ({c_t}/{n_t})")


# ---- recalib ---------------------------------------------------------------
def recalib(args):
    device = pick_device(args.device)
    table = ScaleTable.load(REPO_ROOT / args.table)
    ckpt = torch.load(args.checkpoint, map_location="cpu",
                      weights_only=False)
    floats = ckpt["floats"]
    print("collecting float histograms on QAT weights ...")
    loader = make_val_loader(args.calib, batch_size=args.calib_batch)
    images = [img for img, _ in loader]
    t0 = time.time()
    hist, centers = collect_float_histograms(
        heatvit_to_timm_state(floats), images, device,
        batch_size=args.calib_batch)
    print(f"  histograms done in {time.time() - t0:.1f}s")
    acts = pick_activation_exps(hist, centers, ln_clamp_max=0)
    new_table = ScaleTable(weights=dict(table.weights), activations=acts)
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_table.save(out_path)
    changed = sum(1 for k in acts if acts[k] != table.activations.get(k))
    print(f"activation exps changed: {changed}; table -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="QAT fine-tuning")
    p.add_argument("--device", default="auto")
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--max-images", type=int, default=0,
                   help="train subset size (0 = full ImageNet)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--accum", type=int, default=1,
                   help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--amp", choices=["none", "bf16", "tf32"],
                   default="none")
    p.add_argument("--randaug", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-images", type=int, default=500)
    p.add_argument("--eval-sampling",
                   choices=("head", "random", "stratified"), default="head",
                   help="validation subset selection for all per-epoch evals")
    p.add_argument("--spot-every", type=int, default=0,
                   help="every N epochs also run the 5k bit-exact spot eval "
                        "(exact, or pruned in prune mode); best.pt tracking "
                        "then follows the spot metric only (0 = legacy "
                        "per-epoch metric tracking)")
    p.add_argument("--selectors", default=None,
                   help="P4: selector checkpoint (e.g. p2_out/"
                        "selectors_sup4.pt); attaches frozen float mirrors "
                        "and switches the train forward to prune=True")
    p.add_argument("--eval-prune", type=int, default=0,
                   help="P4: per-epoch bit-exact PRUNED eval on this many "
                        "val images (0 = off); best tracking follows it")
    p.add_argument("--rate-weight", type=float, default=0.0,
                   help="P4-2: keep-rate regularization weight lambda on "
                        "sum((count_k/197 - target_k/197)^2) over the three "
                        "stages (0 = off; requires --selectors)")
    p.add_argument("--rate-targets", default="88,45,32",
                   help="P4-2: comma-separated per-stage token targets")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", default=None,
                   help="checkpoint path to resume from")
    p.add_argument("--init-checkpoint", default=None,
                   help="start from a QAT checkpoint's weights with a fresh "
                        "optimizer and LR schedule (staged training); "
                        "mutually exclusive with --resume")
    p.add_argument("--seed", type=int, default=20260815)
    p.set_defaults(fn=train)

    p = sub.add_parser("eval", help="bit-exact + train-path Top-1")
    p.add_argument("--device", default="auto")
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", type=int, default=5000)
    p.add_argument("--sampling", choices=("head", "random", "stratified"),
                   default="head")
    p.add_argument("--seed", type=int, default=20260815)
    p.set_defaults(fn=evaluate_cmd)

    p = sub.add_parser("recalib", help="post-training activation recalib")
    p.add_argument("--device", default="auto")
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calib", type=int, default=2048)
    p.add_argument("--calib-batch", type=int, default=64)
    p.add_argument("--out", default="p2_out/qat/scale_table_after.json")
    p.set_defaults(fn=recalib)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
