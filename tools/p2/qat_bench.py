#!/usr/bin/env python3
"""P3 QAT step-time diagnosis: separate decode vs compute cost.

Three hypotheses for the ~100-115 s/step smoke result:

  H1 data decode (workers=0 serial PIL) dominates;
  H2 the QAT forward/backward graph itself is pathological on GPU;
  H3 the sandbox audits file reads outside the workspace (D:\\SEU_Liubo)
     so every JPEG open pays a per-file policy cost.

Modes:

  decode --dir train    time N batches of the real train loader
  decode --dir local    same, but from a workspace-local copy of a fixed
                        image sample (H3 test: same bytes, different tree)
  compute               synthetic GPU data, full train step (fwd+bwd+opt),
                        no DataLoader (H2 test; also reports the first-step
                        vs steady-state split)
  profile               torch.profiler on one fwd+bwd step

Run:
  .venv-torch\\Scripts\\python tools/p2/qat_bench.py decode --dir train --batches 3
  .venv-torch\\Scripts\\python tools/p2/qat_bench.py decode --dir local --batches 3
  .venv-torch\\Scripts\\python tools/p2/qat_bench.py compute --steps 5
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

LOCAL_DIR = REPO_ROOT / "p2_out" / "qat_bench_local"
TRAIN_DIR = r"D:\SEU_Liubo\prj\HeatViT\data\imagenet\train"


def pick_device(arg):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_local(n=1024):
    """Copy a fixed sample of train images into the workspace (H3 probe)."""
    from torchvision import datasets
    ds = datasets.ImageFolder(TRAIN_DIR, transform=None)
    gen = torch.Generator().manual_seed(20260815)
    idx = torch.randperm(len(ds), generator=gen)[:n].tolist()
    for i in idx:
        path, label = ds.samples[i]
        rel = Path(path).relative_to(TRAIN_DIR)
        dst = LOCAL_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(path, dst)
    print(f"local copy ready: {n} images -> {LOCAL_DIR}")


def cmd_decode(args):
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from tools.p2.qat_data import make_train_loader
    if args.dir == "local":
        if not LOCAL_DIR.exists():
            _prepare_local()
        # loader over the local dir with the same train transforms
        import torchvision.transforms as T
        from tools.p2.qat_data import MEAN, STD
        ops = [T.RandomResizedCrop(224, scale=(0.08, 1.0),
                                   interpolation=T.InterpolationMode.BICUBIC),
               T.RandomHorizontalFlip(), T.ToTensor(),
               T.Normalize(mean=MEAN, std=STD)]
        dataset = datasets.ImageFolder(str(LOCAL_DIR),
                                       transform=T.Compose(ops))
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0,
                            drop_last=True)
    else:
        loader = make_train_loader(0, args.batch, args.workers)
    t0 = time.time()
    n = 0
    for i, (img, label) in enumerate(loader):
        n += img.size(0)
        if i + 1 >= args.batches:
            break
    dt = time.time() - t0
    print(f"[decode {args.dir}] {n} images in {dt:.1f}s = "
          f"{1000.0 * dt / n:.1f} ms/img = {dt / args.batches:.1f} s/batch "
          f"(workers={args.workers})")


def _build_qat(device):
    from tools.p2.p2_quantize import load_state_dict, to_heatvit_tensors
    from tools.p2.qat_model import QatDeiT
    from tools.p2.scale_table import ScaleTable
    table = ScaleTable.load(REPO_ROOT / "p2_out" / "scale_table.json")
    floats = to_heatvit_tensors(load_state_dict())
    return QatDeiT(floats, table).to(device)


def _step(qat, img, label, opt, amp):
    if amp == "bf16":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = qat(img)
    else:
        logits = qat(img)
    loss = torch.nn.functional.cross_entropy(logits, label,
                                             label_smoothing=0.1)
    loss.backward()
    if opt is not None:
        opt.step()
        opt.zero_grad(set_to_none=True)
    return loss.item()


def cmd_compute(args):
    device = pick_device(args.device)
    if args.amp == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
    qat = _build_qat(device)
    opt = torch.optim.AdamW(qat.parameters(), lr=2e-5, weight_decay=0.05)
    img = torch.rand(args.batch, 3, 224, 224, device=device)
    label = torch.randint(0, 1000, (args.batch,), device=device)
    # first step (includes cuBLAS autotune / allocator warmup)
    torch.cuda.synchronize()
    t0 = time.time()
    _step(qat, img, label, opt, args.amp)
    torch.cuda.synchronize()
    first = time.time() - t0
    # steady state
    t0 = time.time()
    for _ in range(args.steps):
        _step(qat, img, label, opt, args.amp)
    torch.cuda.synchronize()
    steady = (time.time() - t0) / args.steps
    print(f"[compute] first step {first:.1f}s, steady {steady:.2f}s/step "
          f"(batch {args.batch}, amp={args.amp}, "
          f"mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB peak)")


def cmd_profile(args):
    device = pick_device(args.device)
    qat = _build_qat(device)
    qat.eval()
    img = torch.rand(args.batch, 3, 224, 224, device=device)
    label = torch.randint(0, 1000, (args.batch,), device=device)
    from torch.profiler import ProfilerActivity, profile, record_function
    if args.fwd_only:
        with torch.no_grad():
            qat(img)                      # warmup
            torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                qat(img)
        torch.cuda.synchronize()
    else:
        qat(img).sum().backward()          # warmup
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            with record_function("step"):
                loss = torch.nn.functional.cross_entropy(qat(img), label)
                loss.backward()
        torch.cuda.synchronize()
    print("== top by CUDA time ==")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=20))
    print("== top by self CPU time ==")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("decode")
    p.add_argument("--dir", choices=["train", "local"], default="train")
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=0)
    p.set_defaults(fn=cmd_decode)
    p = sub.add_parser("compute")
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--amp", choices=["none", "bf16", "tf32"], default="none")
    p.set_defaults(fn=cmd_compute)
    p = sub.add_parser("profile")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fwd-only", action="store_true")
    p.set_defaults(fn=cmd_profile)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
