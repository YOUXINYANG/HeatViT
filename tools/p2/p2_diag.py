"""Diagnostic: per-block quantized vs float error with current calibration."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import timm

from tools.p2.p2_quantize import (
    build_model,
    build_weight_exps,
    calibrate_float,
    load_state_dict,
    make_val_loader,
    to_heatvit_tensors,
)
from tools.p2.scale_table import ScaleTable
from tools.p2.p2_sim import forward_image


def main():
    state = load_state_dict()
    floats = to_heatvit_tensors(state)
    wexp = build_weight_exps(floats)
    loader = make_val_loader(256, batch_size=8)
    imgs = [img for img, _ in loader]
    acts = calibrate_float(state, imgs, torch.device("cuda"))
    table = ScaleTable(weights=wexp, activations=acts)

    fm = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    fm.load_state_dict(state, strict=True)
    fm = fm.to("cuda").eval()

    img = imgs[0][0].to("cuda")
    model = build_model(floats, table, torch.device("cuda"))
    rec = {}
    logits, counts, logit_exp = forward_image(model, img, rec, prune=False)
    with torch.no_grad():
        fl = fm(img.unsqueeze(0))
    ql = logits.float() * (2.0 ** logit_exp)
    print("float top1", fl[0].argmax().item(), "quant top1",
          ql.argmax().item())
    print("logits corr:",
          torch.corrcoef(torch.stack([fl[0].cpu(), ql.cpu()]))[0, 1].item())

    cap = {}
    for i, blk in enumerate(fm.blocks):
        def h(_, a, o, idx=i):
            cap[f"b{idx+1}"] = o.detach().clone()
        blk.register_forward_hook(h)
    with torch.no_grad():
        fm(img.unsqueeze(0))

    for n in range(1, 13):
        f = cap[f"b{n}"][0].float().cpu()
        q = rec[f"b{n}_out"].float() * (2.0 ** table.activations[f"b{n}_out"])
        err = (f - q).abs().max().item()
        corr = torch.corrcoef(
            torch.stack([f.reshape(-1), q.reshape(-1)]))[0, 1].item()
        print(f"b{n:02d}: maxerr={err:7.3f} corr={corr:6.4f} "
              f"exp={table.activations[f'b{n}_out']}")


if __name__ == "__main__":
    main()
