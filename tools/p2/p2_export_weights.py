#!/usr/bin/env python3
"""P2-D: export real DeiT-T weights to golden params and e2e vectors.

Builds the golden HeatViTParams from the quantized float tensors, the
per-tensor scale table and the trained selectors, preprocesses real val
images into int8 NHWC inputs, runs the integer golden model and writes the
complete e2e vector directory (input/weights/scratch/output mems, golden
checkpoints, manifest, tb config) reusing the generate_e2e_vectors
serialization machinery.

Usage (torch venv, danger-full-access sandbox for build/ writes):

  .venv-torch\\Scripts\\python tools/p2/p2_export_weights.py \
      --table p2_out/scale_table.json --selectors p2_out/selectors_8k.pt \
      --images 3 --output build/vectors/e2e_real

Then run the bit-exact XSim e2e:

  powershell -File scripts/run_xsim.ps1 -Top tb_heatvit_e2e \
      -PlusArgs '+VECTOR_DIR=build/vectors/e2e_real +STALL_MASK=0'
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.generate_e2e_vectors import (
    INPUT_BASE,
    OUTPUT_BASE,
    SCRATCH_BASE,
    WEIGHT_BASE,
    _align8,
    _int32_le_bytes,
    checkpoint_layout,
    serialize_weights,
    sha256_file,
    watchdog_cycles,
    write_e2e_tb_config,
    write_error_roms,
    write_mem_lines,
)
from tools.generate_descriptors import build_memory_map, build_schedule
from tools.p2.p2_quantize import (
    load_state_dict,
    make_val_loader,
    to_heatvit_tensors,
)
from tools.p2.scale_table import ScaleTable
from verification.heatvit_ref.model import HeatViTModel, HeatViTParams
from verification.heatvit_ref.selector import (
    SelectorHeadWeightParams,
    SelectorLocalParams,
    SelectorParams,
    SelectorScoreParams,
)
from verification.heatvit_ref.transformer import (
    BlockParams,
    FfnParams,
    MhsaParams,
    PatchParams,
)


def _q8(floats, name, table):
    """Quantize a float tensor to int8 list rows at the table exponent."""
    t = floats[name]
    exp = table.weights[name]
    q = torch.clamp(torch.round(t / (2.0 ** exp)), -128, 127).to(torch.int8)
    if q.dim() == 1:
        return [int(v) for v in q]
    return [[int(v) for v in row] for row in q]


def _q32_bias(floats, name, exp):
    q = torch.clamp(torch.round(floats[name] / (2.0 ** exp)),
                    -(1 << 31), (1 << 31) - 1).to(torch.int32)
    return [int(v) for v in q]


def block_input_exp(table, n):
    return table.activation_exp("act_tokens") if n == 1 \
        else table.activation_exp(f"b{n - 1}_out")


def golden_params_from_real(floats, table, selector_payload):
    patch = PatchParams(
        patch_weight=_q8(floats, "patch_w", table),
        patch_bias=_q32_bias(
            floats, "patch_b",
            table.activation_exp("input") + table.weights["patch_w"]),
        cls=_q8(floats, "cls", table),
        pos=_q8(floats, "pos", table),
        width=224, height=224, patch=16, embed_dim=192, tokens=197,
        image_scale_exp=table.activation_exp("input"),
        weight_scale_exp=table.weights["patch_w"],
        activation_scale_exp=table.activation_exp("act_patch_embed"),
        cls_scale_exp=table.weights["cls"],
        pos_scale_exp=table.weights["pos"],
    )

    blocks = []
    for n in range(1, 13):
        act = table.activation_exp
        wt = table.weights
        mhsa = MhsaParams(
            ln_gamma=_q8(floats, f"b{n}_gamma1", table),
            ln_beta=_q8(floats, f"b{n}_beta1", table),
            wqkv=_q8(floats, f"b{n}_wqkv", table),
            bqkv=_q32_bias(floats, f"b{n}_bqkv",
                           act(f"b{n}_ln1_out") + wt[f"b{n}_wqkv"]),
            wproj=_q8(floats, f"b{n}_wproj", table),
            bproj=_q32_bias(floats, f"b{n}_bproj",
                            act(f"b{n}_context_out") + wt[f"b{n}_wproj"]),
            embed_dim=192, heads=3, head_dim=64,
            x_scale_exp=block_input_exp(table, n),
            gamma1_scale_exp=wt[f"b{n}_gamma1"],
            beta1_scale_exp=wt[f"b{n}_beta1"],
            ln1_out_scale_exp=act(f"b{n}_ln1_out"),
            wqkv_scale_exp=wt[f"b{n}_wqkv"],
            qkv_out_scale_exp=act(f"b{n}_qkv_out"),
            score_scale_exp=2 * act(f"b{n}_qkv_out") - 3,
            prob_scale_exp=-8,
            context_out_scale_exp=act(f"b{n}_context_out"),
            wproj_scale_exp=wt[f"b{n}_wproj"],
            msa_out_scale_exp=act(f"b{n}_msa_out"),
        )
        ffn = FfnParams(
            ln_gamma=_q8(floats, f"b{n}_gamma2", table),
            ln_beta=_q8(floats, f"b{n}_beta2", table),
            w1=_q8(floats, f"b{n}_w1", table),
            b1=_q32_bias(floats, f"b{n}_b1",
                         act(f"b{n}_ln2_out") + wt[f"b{n}_w1"]),
            w2=_q8(floats, f"b{n}_w2", table),
            b2=_q32_bias(floats, f"b{n}_b2",
                         act(f"b{n}_hidden") + wt[f"b{n}_w2"]),
            embed_dim=192, ffn_dim=768,
            x_scale_exp=act(f"b{n}_y"),
            gamma2_scale_exp=wt[f"b{n}_gamma2"],
            beta2_scale_exp=wt[f"b{n}_beta2"],
            ln2_out_scale_exp=act(f"b{n}_ln2_out"),
            w1_scale_exp=wt[f"b{n}_w1"],
            hidden_out_scale_exp=act(f"b{n}_hidden"),
            w2_scale_exp=wt[f"b{n}_w2"],
            ffn_out_scale_exp=act(f"b{n}_ffn_out"),
            out_scale_exp=act(f"b{n}_out"),
        )
        blocks.append(BlockParams(mhsa=mhsa, ffn=ffn))

    def sel_tensor(payload, key):
        return payload[key]

    def head_list(t, h):
        return [[int(v) for v in row] for row in t[h]]

    def bias_list(t, h):
        return [int(v) for v in t[h]]

    selectors = []
    for s_idx in range(3):
        p = selector_payload["selectors"][s_idx]
        local = SelectorLocalParams(
            w=tuple(head_list(p["local_w"], h) for h in range(3)),
            b=tuple(bias_list(p["local_b"], h) for h in range(3)),
        )
        score = SelectorScoreParams(
            w1=tuple(head_list(p["score_w1"], h) for h in range(3)),
            b1=tuple(bias_list(p["score_b1"], h) for h in range(3)),
            w2=tuple(head_list(p["score_w2"], h) for h in range(3)),
            b2=tuple(bias_list(p["score_b2"], h) for h in range(3)),
            w3=tuple(head_list(p["score_w3"], h) for h in range(3)),
            b3=tuple(bias_list(p["score_b3"], h) for h in range(3)),
        )
        head_weight = SelectorHeadWeightParams(
            w1=[[int(v) for v in row] for row in p["hw_w1"]],
            b1=[int(v) for v in p["hw_b1"]],
            w2=[[int(v) for v in row] for row in p["hw_w2"]],
            b2=[int(v) for v in p["hw_b2"]],
        )
        selectors.append(SelectorParams(local=local, score=score,
                                        head_weight=head_weight))

    params = HeatViTParams(
        patch=patch,
        blocks=tuple(blocks),
        selectors=tuple(selectors),
        final_gamma=_q8(floats, "final_gamma", table),
        final_beta=_q8(floats, "final_beta", table),
        head_w=_q8(floats, "head_w", table),
        head_b=_q32_bias(floats, "head_b",
                         table.activation_exp("final_ln_out")
                         + table.weights["head_w"]),
        final_gamma_scale_exp=table.weights["final_gamma"],
        final_beta_scale_exp=table.weights["final_beta"],
        final_ln_out_scale_exp=table.activation_exp("final_ln_out"),
        head_w_scale_exp=table.weights["head_w"],
    )
    return params


def real_image_list(img_float, table):
    """Preprocessed float CHW image -> quantized flat NHWC int8 list."""
    exp = table.activation_exp("input")
    q = torch.clamp(torch.round(img_float / (2.0 ** exp)), -128, 127) \
        .to(torch.int8)
    return [int(v) for v in q.permute(1, 2, 0).reshape(-1)]


def checkpoint_layout_real(mm, token_counts, table):
    """checkpoint_layout with per-tensor activation scale exponents."""
    layout = checkpoint_layout(mm, token_counts)
    act = table.activation_exp
    layout["patch"] = (layout["patch"][0], layout["patch"][1],
                       layout["patch"][2], act("act_tokens"),
                       layout["patch"][4])
    for b in range(1, 13):
        name = f"block_{b:02d}"
        layout[name] = (layout[name][0], layout[name][1], layout[name][2],
                        act(f"b{b}_out"), layout[name][4])
    for s in range(1, 4):
        name = f"selector_{s:02d}"
        layout[name] = (layout[name][0], layout[name][1], layout[name][2],
                        -7, layout[name][4])
    layout["final_ln"] = (layout["final_ln"][0], layout["final_ln"][1],
                          layout["final_ln"][2], act("final_ln_out"),
                          layout["final_ln"][4])
    return layout


def export_one(outdir, image, params, table, scale_table):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = HeatViTModel()
    result = model.infer(image, params)
    summary = result.selector_summary
    token_counts = [197] + [e["output_tokens"] for e in summary]

    mm, scratch_bytes, weight_bytes = build_memory_map()
    descs = build_schedule(mm, scale_table)
    layout = checkpoint_layout_real(mm, token_counts, table)
    watchdog = watchdog_cycles(descs, mm, token_counts)

    input_bytes = len(image)
    weight_raw = serialize_weights(params, mm["weight"])
    scratch_raw = b"\x00" * scratch_bytes
    output_raw = b"\x00" * 4000

    write_mem_lines(outdir / "input.mem",
                    bytes(v & 0xFF for v in image))
    write_mem_lines(outdir / "weights.mem", weight_raw)
    write_mem_lines(outdir / "scratch_init.mem", scratch_raw)
    write_mem_lines(outdir / "output_init.mem", output_raw)

    checkpoints_dir = outdir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cp_entries = []
    for name, (region, off, byte_count, scale, index) in layout.items():
        data = result.checkpoints[name]
        if name == "logits":
            raw = _int32_le_bytes(data)
        else:
            raw = bytes(v & 0xFF for row in data for v in row)
        if len(raw) < byte_count:
            raw = raw + b"\x00" * (byte_count - len(raw))
        path = checkpoints_dir / f"{name}.mem"
        write_mem_lines(path, raw)
        cp_entries.append({
            "name": name,
            "desc_index": index,
            "region": region,
            "offset": off,
            "bytes": byte_count,
            "scale_exp": scale,
            "sha256": sha256_file(path),
        })
    cp_entries.sort(key=lambda c: c["desc_index"])

    manifest = {
        "seed": 20260815,
        "part": "xc7k325tfbg900-3",
        "generator": "tools/p2/p2_export_weights.py",
        "descriptor_rom_sha256": sha256_file(
            REPO_ROOT / "rtl/generated/heatvit_descriptors.mem"),
        "regions": {
            "input": {"base": INPUT_BASE, "bytes": _align8(input_bytes),
                      "valid_bytes": input_bytes},
            "weight": {"base": WEIGHT_BASE, "bytes": _align8(weight_bytes),
                       "valid_bytes": weight_bytes},
            "scratch": {"base": SCRATCH_BASE, "bytes": scratch_bytes,
                        "valid_bytes": scratch_bytes},
            "output": {"base": OUTPUT_BASE, "bytes": 4000,
                       "valid_bytes": 4000},
        },
        "checkpoints": cp_entries,
        "selectors": [dict(e) for e in summary],
        "token_counts": token_counts,
        "output_scale_exp": result.output_scale_exp,
        "watchdog_cycles": watchdog,
        "files": {
            "input.mem": sha256_file(outdir / "input.mem"),
            "weights.mem": sha256_file(outdir / "weights.mem"),
            "scratch_init.mem": sha256_file(outdir / "scratch_init.mem"),
            "output_init.mem": sha256_file(outdir / "output_init.mem"),
        },
    }
    write_error_roms(outdir)
    with open(outdir / "manifest.json", "w", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    write_e2e_tb_config(mm, manifest, layout, token_counts)
    print(f"wrote {outdir}: token_counts={token_counts}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="p2_out/scale_table.json")
    parser.add_argument("--selectors", default="p2_out/selectors_8k.pt")
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--output", default="build/vectors/e2e_real")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-rom", action="store_true",
                        help="regenerate rtl/generated/heatvit_descriptors.mem "
                             "with the per-tensor scale table (do this only "
                             "when no other XSim run is pending, since the "
                             "ROM is read at simulation time 0)")
    args = parser.parse_args()

    table = ScaleTable.load(REPO_ROOT / args.table)
    table.validate()
    selector_payload = torch.load(REPO_ROOT / args.selectors,
                                  map_location="cpu", weights_only=False)

    floats = to_heatvit_tensors(load_state_dict())
    params = golden_params_from_real(floats, table, selector_payload)

    loader = make_val_loader(args.images)
    images = [img[0] for img, _ in loader]

    if args.write_rom:
        import tools.generate_descriptors as gd
        mm, _, _ = gd.build_memory_map()
        descs = gd.build_schedule(mm, table)
        for desc in descs:
            desc.validate()
        gd.emit(descs, mm, REPO_ROOT / "rtl/generated/heatvit_descriptors.mem",
                REPO_ROOT / "build/vectors/e2e_real/descriptor_listing.csv",
                REPO_ROOT / "build/vectors/e2e_real/memory_map.json")
        print("descriptor ROM regenerated with per-tensor scale table")

    for i, img in enumerate(images):
        image = real_image_list(img, table)
        export_one(Path(args.output) / f"img{i}",
                   image, params, table, table)
    print("done")


if __name__ == "__main__":
    main()
