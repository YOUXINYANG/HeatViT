#!/usr/bin/env python3
"""Throwaway probe: real-checkpoint train vs exact divergence.

Measures on real DeiT-T + legacy scale table + real val images:
  1. per-boundary bin distances (train vs exact)
  2. logits argmax agreement between the train path and the bit-exact
     integer path on real images
Run: .venv-torch\\Scripts\\python tools/p2/qat_probe_real.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.p2_quantize import load_state_dict, make_val_loader, \
    to_heatvit_tensors
from tools.p2.qat_model import QatDeiT, exact_forward
from tools.p2.scale_table import ScaleTable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = load_state_dict()
floats = to_heatvit_tensors(state)
table = ScaleTable.load(REPO_ROOT / "p2_out" / "scale_table.json")
qat = QatDeiT(floats, table).to(DEVICE)
loader = make_val_loader(8, batch_size=8, shuffle=False)
images, labels = next(iter(loader))
images = images.to(DEVICE)

with torch.no_grad():
    rec_t = {}
    logits_t = qat(images, rec_t)
    rec_e = {}
    logits_e, scale_e = exact_forward(qat, images, rec_e)

pred_t = logits_t.argmax(dim=1).cpu()
pred_e = logits_e.argmax(dim=1).cpu()
agree = (pred_t == pred_e).sum().item()
print(f"argmax agreement: {agree}/8  (labels {labels.tolist()})")
rel = ((logits_t.cpu() - logits_e.cpu().float() * (2.0 ** scale_e)).abs()
       / (logits_e.cpu().abs().float() * (2.0 ** scale_e) + 1.0))
print(f"logits mean|rel|={rel.mean():.4f} max={rel.max():.4f}")
# top-5 overlap
t5_t = logits_t.topk(5, dim=1).indices.cpu()
t5_e = logits_e.topk(5, dim=1).indices.cpu()
overlap = sum(len(set(a.tolist()) & set(b.tolist()))
              for a, b in zip(t5_t, t5_e)) / 8
print(f"top-5 mean overlap: {overlap:.2f}/5")

print("\nboundary bins (train vs exact):")
for name, t in rec_t.items():
    if name not in rec_e:
        continue
    exp = table.activations.get(name)
    if exp is None:
        continue
    e_deq = rec_e[name].to(torch.float32) * (2.0 ** exp)
    bins = ((t - e_deq).abs() / (2.0 ** exp))
    print(f"  {name:14s} mean={bins.mean():7.3f} max={bins.max():8.2f}")
