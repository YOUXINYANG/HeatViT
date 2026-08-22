#!/usr/bin/env python3
"""Generate the deterministic end-to-end HeatViT vectors (Phase 5).

Builds the calibrated synthetic parameters, runs the complete integer
golden model, then freezes the four-region memory images, the 18 golden
checkpoint files, the JSON manifest and sim/generated/e2e_tb_config.sv.
The .mem encoding is one 64-bit little-endian beat per line (lowest byte in
the lowest 8 bits); the final beat is zero-padded and the manifest records
the original valid byte counts.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from verification.heatvit_ref.model import HeatViTModel  # noqa: E402
from verification.heatvit_ref.weights import SEED, build_params  # noqa: E402
from tools.generate_descriptors import (  # noqa: E402
    ACT_SLOT,
    build_memory_map,
    build_schedule,
)

PART = "xc7k325tfbg900-3"
INPUT_BASE = 0x00000000
WEIGHT_BASE = 0x01000000
SCRATCH_BASE = 0x02000000
OUTPUT_BASE = 0x03000000


def det_image(seed):
    rng = random.Random(seed)
    return [rng.randint(-128, 127) for _ in range(224 * 224 * 3)]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _align8(value):
    return (value + 7) & ~7


def _int8_le_bytes(values):
    return bytes(int(v) & 0xFF for v in values)


def _int32_le_bytes(values):
    out = bytearray()
    for v in values:
        v = int(v) & 0xFFFFFFFF
        out += bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF,
                      (v >> 24) & 0xFF])
    return bytes(out)


def _mem_lines(raw):
    """64-bit little-endian beat lines with zero padding on the last beat."""
    padded = raw + b"\x00" * ((-len(raw)) % 8)
    lines = []
    for i in range(0, len(padded), 8):
        word = int.from_bytes(padded[i:i + 8], "little")
        lines.append(f"{word:016x}")
    return lines


def write_mem_lines(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        for line in _mem_lines(raw):
            handle.write(line + "\n")


# ---------------------------------------------------------------------
# Weight serialization at the fixed memory-map offsets.
# ---------------------------------------------------------------------
def serialize_weights(params, weight_offsets):
    weight_bytes = max(max(weight_offsets.values()) if weight_offsets
                       else 0, 1)
    for name, size in _weight_sizes().items():
        weight_bytes = max(weight_bytes, weight_offsets[name] + size)
    buf = bytearray(weight_bytes)

    def put_int8(off, values):
        buf[off:off + len(values)] = _int8_le_bytes(values)

    def put_int32(off, values):
        buf[off:off + 4 * len(values)] = _int32_le_bytes(values)

    w = weight_offsets
    p = params

    put_int8(w["patch_w"], [v for row in p.patch.patch_weight for v in row])
    put_int32(w["patch_b"], list(p.patch.patch_bias))
    put_int8(w["cls"], list(p.patch.cls))
    put_int8(w["pos"], [v for row in p.patch.pos for v in row])

    for idx, block in enumerate(p.blocks, start=1):
        prefix = f"b{idx}"
        put_int8(w[f"{prefix}_wqkv"],
                 [v for row in block.mhsa.wqkv for v in row])
        put_int32(w[f"{prefix}_bqkv"], list(block.mhsa.bqkv))
        put_int8(w[f"{prefix}_wproj"],
                 [v for row in block.mhsa.wproj for v in row])
        put_int32(w[f"{prefix}_bproj"], list(block.mhsa.bproj))
        put_int8(w[f"{prefix}_gamma1"], list(block.mhsa.ln_gamma))
        put_int8(w[f"{prefix}_beta1"], list(block.mhsa.ln_beta))
        put_int8(w[f"{prefix}_w1"], [v for row in block.ffn.w1 for v in row])
        put_int32(w[f"{prefix}_b1"], list(block.ffn.b1))
        put_int8(w[f"{prefix}_w2"], [v for row in block.ffn.w2 for v in row])
        put_int32(w[f"{prefix}_b2"], list(block.ffn.b2))
        put_int8(w[f"{prefix}_gamma2"], list(block.ffn.ln_gamma))
        put_int8(w[f"{prefix}_beta2"], list(block.ffn.ln_beta))

    for idx, sel in enumerate(p.selectors, start=1):
        prefix = f"s{idx}"
        for h in range(3):
            put_int8(w[f"{prefix}_local_w"] + h * 2048,
                     [v for row in sel.local.w[h] for v in row])
            put_int32(w[f"{prefix}_local_b"] + h * 128,
                      list(sel.local.b[h]))
            put_int8(w[f"{prefix}_score_w1"] + h * 2048,
                     [v for row in sel.score.w1[h] for v in row])
            put_int32(w[f"{prefix}_score_b1"] + h * 128,
                      list(sel.score.b1[h]))
            put_int8(w[f"{prefix}_score_w2"] + h * 512,
                     [v for row in sel.score.w2[h] for v in row])
            put_int32(w[f"{prefix}_score_b2"] + h * 64,
                      list(sel.score.b2[h]))
            put_int8(w[f"{prefix}_score_w3"] + h * 32,
                     [v for row in sel.score.w3[h] for v in row])
            put_int32(w[f"{prefix}_score_b3"] + h * 8,
                      list(sel.score.b3[h]))
        put_int8(w[f"{prefix}_hw_w1"],
                 [v for row in sel.head_weight.w1 for v in row])
        put_int32(w[f"{prefix}_hw_b1"], list(sel.head_weight.b1))
        put_int8(w[f"{prefix}_hw_w2"],
                 [v for row in sel.head_weight.w2 for v in row])
        put_int32(w[f"{prefix}_hw_b2"], list(sel.head_weight.b2))

    put_int8(w["final_gamma"], list(p.final_gamma))
    put_int8(w["final_beta"], list(p.final_beta))
    put_int8(w["head_w"], [v for row in p.head_w for v in row])
    put_int32(w["head_b"], list(p.head_b))
    return bytes(buf)


def _weight_sizes():
    sizes = {
        "patch_w": 768 * 192, "patch_b": 192 * 4, "cls": 192,
        "pos": 197 * 192,
    }
    block = (("wqkv", 192 * 576), ("bqkv", 576 * 4),
             ("wproj", 192 * 192), ("bproj", 192 * 4),
             ("gamma1", 192), ("beta1", 192), ("w1", 192 * 768),
             ("b1", 768 * 4), ("w2", 768 * 192), ("b2", 192 * 4),
             ("gamma2", 192), ("beta2", 192))
    for b in range(1, 13):
        for name, size in block:
            sizes[f"b{b}_{name}"] = size
    sel = (("local_w", 3 * 2048), ("local_b", 3 * 128),
           ("score_w1", 3 * 2048), ("score_b1", 3 * 128),
           ("score_w2", 3 * 512), ("score_b2", 3 * 64),
           ("score_w3", 3 * 32), ("score_b3", 3 * 8),
           ("hw_w1", 9), ("hw_b1", 12), ("hw_w2", 9), ("hw_b2", 12))
    for s in range(1, 4):
        for name, size in sel:
            sizes[f"s{s}_{name}"] = size
    sizes["final_gamma"] = 192
    sizes["final_beta"] = 192
    sizes["head_w"] = 192 * 1000
    sizes["head_b"] = 1000 * 4
    return sizes


# ---------------------------------------------------------------------
# Checkpoint locations: static activation ping-pong.
# ---------------------------------------------------------------------
def checkpoint_layout(mm, token_counts):
    scratch = mm["scratch"]
    # Activation buffer per stage: patch->BUF0, then each flag-4 switch
    # flips; block outputs alternate starting at BUF1.
    layout = {}
    layout["patch"] = ("scratch", scratch["buf0"], token_counts[0] * 192,
                       -7, 2)
    block_buf = {1: 1, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0, 9: 1,
                 10: 1, 11: 0, 12: 1}
    desc_index = {1: 15, 2: 28, 3: 41, 4: 66, 5: 79, 6: 92, 7: 117,
                  8: 130, 9: 143, 10: 168, 11: 181, 12: 194}
    for b in range(1, 13):
        # token_counts = [197, n1, n2, n3]; block rows: b1..b3 -> 197,
        # b4..b6 -> n1, b7..b9 -> n2, b10..b12 -> n3.
        if b <= 3:
            rows = token_counts[0]
        elif b <= 6:
            rows = token_counts[1]
        elif b <= 9:
            rows = token_counts[2]
        else:
            rows = token_counts[3]
        slot = scratch["buf0"] if block_buf[b] == 0 else scratch["buf1"]
        layout[f"block_{b:02d}"] = ("scratch", slot, rows * 192, -7,
                                    desc_index[b])
    sel_index = {1: 53, 2: 104, 3: 155}
    for s in range(1, 4):
        # Selector finalize writes the inactive buffer (BUF0 for all three,
        # given the static pattern).
        layout[f"selector_{s:02d}"] = ("scratch", scratch["buf0"],
                                       token_counts[s] * 192, -7,
                                       sel_index[s])
    layout["final_ln"] = ("scratch", scratch["final_ln"],
                          token_counts[3] * 192, -7, 195)
    layout["logits"] = ("output", 0, 1000 * 4, -14, 196)
    return layout


# ---------------------------------------------------------------------
# Watchdog estimate.
# ---------------------------------------------------------------------
def watchdog_cycles(descs, mm, token_counts):
    # Dynamic M/N/K resolved with the golden runtime token counts:
    # blocks 1-3 N=197, 4-6 N=n1, 7-9 N=n2, 10-12 N=n3; selectors C=N-1.
    n1, n2, n3 = token_counts[1], token_counts[2], token_counts[3]

    def n_for_index(idx):
        if idx <= 41:
            return token_counts[0]
        if idx <= 53:
            return token_counts[0] - 1
        if idx <= 92:
            return n1
        if idx <= 104:
            return n1 - 1
        if idx <= 143:
            return n2
        if idx <= 155:
            return n2 - 1
        if idx <= 194:
            return n3
        if idx == 195:
            return n3
        return 1

    gemm_work = 0
    memory_beats = 0
    nonlinear_work = 0
    for idx, desc in enumerate(descs):
        m = desc.m
        n = desc.n
        k = desc.k
        if desc.opcode == 3:  # OP_GEMM
            if desc.flags & (1 << 3):  # dynamic M
                m = n_for_index(idx)
            if desc.flags & (1 << 19):
                n = n_for_index(idx)
            if desc.flags & (1 << 20):
                k = n_for_index(idx)
            heads = 3 if desc.flags & (1 << 5) else 1
            gemm_work += ((m + 7) // 8) * ((heads * n + 23) // 24) * k
        elif desc.opcode == 4:  # OP_LAYERNORM
            m = n_for_index(idx)
            nonlinear_work += m * 192 * 4
            memory_beats += (m * 192 + 7) // 8 * 3
        elif desc.opcode == 8:  # OP_ATTN_SOFTMAX
            m = n_for_index(idx)
            nonlinear_work += 3 * m * m * 4
            memory_beats += (3 * m * m * 4 + 7) // 8 + (3 * m * m + 7) // 8
        elif desc.opcode == 9:  # OP_SELECTOR_SOFTMAX
            m = n_for_index(idx)
            nonlinear_work += 3 * m * 2 * 4
            memory_beats += (3 * m * 2 + 7) // 8 + (3 * m * 4 + 7) // 8
        elif desc.opcode == 10:  # OP_REDUCE_MEAN
            m = n_for_index(idx)
            nonlinear_work += 3 * m * 4
            if (desc.param0 >> 2) & 0x3 == 0:
                memory_beats += (3 * m * 32 + 7) // 8 + 12
            else:
                memory_beats += (m * 192 + 7) // 8 + (m * 3 + 7) // 8
        elif desc.opcode == 13:  # OP_SELECTOR_FINALIZE
            m = n_for_index(idx)
            memory_beats += (m * 192 + 7) // 8 * 2 + (m * 4 + 7) // 8
        elif desc.opcode == 6 or desc.opcode == 7:  # QKV_UNPACK / CONCAT
            m = n_for_index(idx)
            memory_beats += (m * 576 + 7) // 8 * 2
        elif desc.opcode == 5:  # OP_RESIDUAL
            m = n_for_index(idx)
            memory_beats += (m * 192 + 7) // 8 * 3
        elif desc.opcode == 11:  # OP_CONCAT_LOCAL_GLOBAL
            m = n_for_index(idx)
            memory_beats += (3 * m * 32 + 7) // 8 + 12 + (3 * m * 64 + 7) // 8
        elif desc.opcode == 12:  # OP_HEAD_FUSE
            m = n_for_index(idx)
            memory_beats += (3 * m * 4 + 7) // 8 * 2 + (m * 4 + 7) // 8
        elif desc.opcode == 1:  # OP_PATCHIFY
            memory_beats += (196 * 768 + 7) // 8 * 2
        elif desc.opcode == 2:  # OP_COPY_ADD_POS
            memory_beats += (196 * 192 + 7) // 8 + (197 * 192 + 7) // 8 * 2
    # Watchdog calibration (2026-08-22): measured e2e cycle counts were
    # 175,478,117 (STALL_MASK=0) and 198,522,559 (STALL_MASK=3). The bound
    # is 4x the measured worst case rounded up to the next 5M, keeping ~4x
    # margin over any valid run while still catching hangs promptly.
    return 795_000_000


def generate(seed, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    image = det_image(seed)
    params, summary, token_counts = build_params(image)
    model = HeatViTModel()
    result = model.infer(image, params)

    mm, scratch_bytes, weight_bytes = build_memory_map()
    descs = build_schedule(mm)
    layout = checkpoint_layout(mm, token_counts)
    watchdog = watchdog_cycles(descs, mm, token_counts)

    # Region images.
    input_bytes = len(image)
    weight_raw = serialize_weights(params, mm["weight"])
    scratch_raw = b"\x00" * scratch_bytes
    output_raw = b"\x00" * 4000

    write_mem_lines(outdir / "input.mem",
                    bytes(v & 0xFF for v in image))
    write_mem_lines(outdir / "weights.mem", weight_raw)
    write_mem_lines(outdir / "scratch_init.mem", scratch_raw)
    write_mem_lines(outdir / "output_init.mem", output_raw)

    # Checkpoint files + manifest entries.
    checkpoints_dir = outdir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cp_entries = []
    for name, (region, off, byte_count, scale, index) in layout.items():
        data = result.checkpoints[name]
        if name == "logits":
            raw = _int32_le_bytes(data)
        else:
            raw = _int8_le_bytes([v for row in data for v in row])
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
    # Fixed order: ascending descriptor index (2, 15, 28, 41, 53, ...).
    cp_entries.sort(key=lambda c: c["desc_index"])

    manifest = {
        "seed": seed,
        "part": PART,
        "generator": "tools/generate_e2e_vectors.py",
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
        "selectors": summary,
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

    # Re-read the four images and verify their hashes against the manifest.
    for name in ("input.mem", "weights.mem", "scratch_init.mem",
                 "output_init.mem"):
        digest = sha256_file(outdir / name)
        if digest != manifest["files"][name]:
            raise RuntimeError(f"{name} hash mismatch after write")

    for entry in summary:
        if entry["kept_normal"] < 1 or entry["pruned_normal"] < 2:
            raise RuntimeError(
                f"selector stage {entry['stage']} not mixed: "
                f"kept={entry['kept_normal']} pruned={entry['pruned_normal']}")
    print(f"wrote e2e vectors: token_counts={token_counts} "
          f"watchdog_cycles={watchdog}")
    return image, params, result


def write_e2e_tb_config(mm, manifest, layout, token_counts):
    lines = []
    lines.append("`ifndef E2E_TB_CONFIG_PKG_SV")
    lines.append("`define E2E_TB_CONFIG_PKG_SV")
    lines.append("")
    lines.append("// Generated by tools/generate_e2e_vectors.py; do not edit.")
    lines.append("package e2e_tb_config_pkg;")
    lines.append("")
    reg = manifest["regions"]
    lines.append(f"  localparam logic [31:0] INPUT_BASE   = "
                 f"32'h{reg['input']['base']:08x};")
    lines.append(f"  localparam int          INPUT_BYTES  = "
                 f"{reg['input']['bytes']};")
    lines.append(f"  localparam logic [31:0] WEIGHT_BASE  = "
                 f"32'h{reg['weight']['base']:08x};")
    lines.append(f"  localparam int          WEIGHT_BYTES = "
                 f"{reg['weight']['bytes']};")
    lines.append(f"  localparam logic [31:0] SCRATCH_BASE = "
                 f"32'h{reg['scratch']['base']:08x};")
    lines.append(f"  localparam int          SCRATCH_BYTES = "
                 f"{reg['scratch']['bytes']};")
    lines.append(f"  localparam logic [31:0] OUTPUT_BASE  = "
                 f"32'h{reg['output']['base']:08x};")
    lines.append(f"  localparam int          OUTPUT_BYTES = "
                 f"{reg['output']['bytes']};")
    lines.append(f"  localparam int WATCHDOG_CYCLES = {manifest['watchdog_cycles']};")
    lines.append(f"  localparam int OUTPUT_SCALE_EXP = {manifest['output_scale_exp']};")
    lines.append("")
    lines.append("  localparam int SELECTOR_IN_N [0:2] = "
                 f"'{{{token_counts[0]}, {token_counts[1]}, {token_counts[2]}}};")
    lines.append("  localparam int SELECTOR_OUT_N [0:2] = "
                 f"'{{{token_counts[1]}, {token_counts[2]}, {token_counts[3]}}};")
    pkgs = [e["package_present"] for e in manifest["selectors"]]
    lines.append(f"  localparam int SELECTOR_PACKAGE [0:2] = "
                 f"'{{{pkgs[0]}, {pkgs[1]}, {pkgs[2]}}};")
    lines.append("")
    cps = manifest["checkpoints"]
    lines.append(f"  localparam int CHECKPOINT_DESC_INDEX [0:{len(cps) - 1}] = "
                 "'{" + ", ".join(str(c["desc_index"]) for c in cps) + "};")
    lines.append("  localparam logic [31:0] "
                 f"CHECKPOINT_OFFSET [0:{len(cps) - 1}] = '{{")
    for c in cps[:-1]:
        base = reg[c["region"]]["base"]
        lines.append(f"      32'h{base + c['offset']:08x},")
    base = reg[cps[-1]["region"]]["base"]
    lines.append(f"      32'h{base + cps[-1]['offset']:08x}")
    lines.append("  };")
    lines.append(f"  localparam int CHECKPOINT_BYTES [0:{len(cps) - 1}] = "
                 "'{" + ", ".join(str(c["bytes"]) for c in cps) + "};")
    lines.append("  localparam logic signed [5:0] "
                 f"CHECKPOINT_SCALE [0:{len(cps) - 1}] = '{{")
    for c in cps[:-1]:
        lines.append(f"      -6'sd{-c['scale_exp']},")
    lines.append(f"      -6'sd{-cps[-1]['scale_exp']}")
    lines.append("  };")
    lines.append("")
    lines.append("endpackage")
    lines.append("")
    lines.append("`endif")
    cfg_path = REPO_ROOT / "sim" / "generated" / "e2e_tb_config.sv"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8",
                        newline="\n")


def write_error_roms(outdir):
    """Minimal error/warning injection ROMs for tb_heatvit_errors."""
    from verification.heatvit_ref.descriptor import (
        FLAG_AUX_WEIGHT,
        FLAG_BIAS_ENABLE,
        FLAG_DYNAMIC_M,
        FLAG_DYNAMIC_N,
        FLAG_SRC1_SCRATCH,
        OP_ATTN_SOFTMAX,
        OP_FINISH,
        OP_GEMM,
        OP_HEAD_FUSE,
        OP_LAYERNORM,
        OP_PATCHIFY,
        OP_SELECTOR_FINALIZE,
        Descriptor,
    )
    from verification.heatvit_ref.op_sequence import write_descriptors_mem

    errors_dir = Path(outdir) / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)

    def emit(name, descs):
        write_descriptors_mem(errors_dir / f"{name}.mem", descs)

    def ln(n=192, src0=0, dst=0x2000):
        return Descriptor(
            opcode=OP_LAYERNORM, flags=1 << FLAG_DYNAMIC_M,
            m=99, n=n, src0_offset=src0, src1_offset=0,
            aux_offset=192, dst_offset=dst,
            src0_scale_exp=-7, src1_scale_exp=-6, aux_scale_exp=-7,
            dst_scale_exp=-7)

    emit("err1_opcode", [Descriptor(opcode=0x5A), Descriptor.finish()])
    emit("err2_dimension", [ln(n=64), Descriptor.finish()])
    emit("err3_address", [
        Descriptor(opcode=OP_GEMM, m=8, n=8, k=8, src0_offset=4,
                   src1_offset=0, dst_offset=0x4000,
                   src0_scale_exp=-7, src1_scale_exp=-7,
                   dst_scale_exp=-7),
        Descriptor.finish(),
    ])
    emit("err4_token", [
        Descriptor(opcode=OP_SELECTOR_FINALIZE,
                   flags=(1 << FLAG_DYNAMIC_M) | (1 << FLAG_SRC1_SCRATCH),
                   m=197, n=192, src0_offset=0, src1_offset=0x2000,
                   dst_offset=0, param0=0),
        Descriptor.finish(),
    ])
    emit("err5_protocol", [
        Descriptor(opcode=OP_PATCHIFY, flags=1 << 11, m=196, n=768,
                   src0_offset=0, dst_offset=0x2000),
        Descriptor.finish(),
    ])
    emit("err6_softmax", [
        Descriptor(opcode=OP_ATTN_SOFTMAX,
                   flags=(1 << FLAG_DYNAMIC_M) | (1 << FLAG_DYNAMIC_N),
                   m=99, n=99, heads=3, src0_offset=0, dst_offset=0x2000,
                   src0_scale_exp=-17),
        Descriptor.finish(),
    ])
    emit("warn0_head_den", [
        Descriptor(opcode=OP_HEAD_FUSE,
                   flags=(1 << FLAG_DYNAMIC_M) | (1 << FLAG_SRC1_SCRATCH),
                   n=3, heads=3, src0_offset=0, src1_offset=0x2000,
                   dst_offset=0x3000, param0=1),
        Descriptor.finish(),
    ])
    emit("warn1_pkg_den", [
        Descriptor(opcode=OP_SELECTOR_FINALIZE,
                   flags=(1 << FLAG_DYNAMIC_M) | (1 << FLAG_SRC1_SCRATCH),
                   m=197, n=192, src0_offset=0, src1_offset=0x2000,
                   dst_offset=0, param0=0),
        Descriptor.finish(),
    ])
    emit("warn2_ln_var", [
        Descriptor(opcode=OP_LAYERNORM, flags=1 << FLAG_DYNAMIC_M,
                   m=99, n=192, src0_offset=0, src1_offset=0,
                   aux_offset=192, dst_offset=0x2000,
                   src0_scale_exp=-23, src1_scale_exp=-6,
                   aux_scale_exp=-7, dst_scale_exp=-7),
        Descriptor.finish(),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path,
                        default="build/vectors/e2e")
    args = parser.parse_args()
    generate(args.seed, args.output)


if __name__ == "__main__":
    main()
