#!/usr/bin/env python3
"""Throwaway: memory retention + backward batch-scaling probe (small first)."""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_quantize import load_state_dict, to_heatvit_tensors
from tools.p2.qat_model import QatDeiT
from tools.p2.scale_table import ScaleTable

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
table = ScaleTable.load(REPO_ROOT / "p2_out" / "scale_table.json")
floats = to_heatvit_tensors(load_state_dict())
qat = QatDeiT(floats, table).to(DEV)
qat.train()

for B in (16, 32, 64, 128):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    img = torch.rand(B, 3, 224, 224, device=DEV)
    label = torch.randint(0, 1000, (B,), device=DEV)
    logits = qat(img)
    torch.cuda.synchronize()
    after_fwd = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    loss = torch.nn.functional.cross_entropy(logits, label,
                                             label_smoothing=0.1)
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    bwd = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"batch {B:3d}: after-fwd {after_fwd:.2f}GiB (reserved "
          f"{reserved:.2f}GiB), peak {peak:.2f}GiB, bwd {bwd:.2f}s",
          flush=True)
    qat.zero_grad(set_to_none=True)
    del logits, loss, img, label
