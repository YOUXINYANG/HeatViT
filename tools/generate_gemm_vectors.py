#!/usr/bin/env python3
"""Generate deterministic GEMM-related .mem vectors and a JSON manifest.

Every .mem line holds one 64-bit word as an unprefixed little-endian hex
value. The MAC-bank suite packs 35 words per case: A lanes, B lanes, a
control word (row/column masks, unsigned flag, K), then 64 int32 expected
accumulators two per word. Eight memory scenarios cover the six required
matrix shapes plus transpose and unsigned-src0 flag variants, and the
testbench configuration is emitted as sim/generated/gemm_cases.sv.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from verification.heatvit_ref.gemm import gemm, gemm_writeback  # noqa: E402


FLAG_RHS_TRANSPOSE = 1 << 0
FLAG_BIAS_ENABLE = 1 << 1
FLAG_HEAD_MODE = 1 << 5
FLAG_OUTPUT_INT32 = 1 << 7
FLAG_SRC0_INPUT = 1 << 11
FLAG_DST_OUTPUT = 1 << 15
FLAG_SRC0_UNSIGNED = 1 << 18

INPUT_BASE = 0x00000000
WEIGHT_BASE = 0x01000000
SCRATCH_BASE = 0x02000000
OUTPUT_BASE = 0x03000000

A5 = 0xA5


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_mem(path, words):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        for word in words:
            handle.write(f"{word:x}\n")


def write_manifest_full(outdir, seed, files, scenarios):
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": "gemm",
        "seed": seed,
        "generator": "tools/generate_gemm_vectors.py",
        "files": files,
        "scenarios": scenarios,
    }
    with open(outdir / "manifest.json", "w", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def s8(value):
    """Reinterpret an integer as an int8 two's-complement value."""
    value = int(value)
    value &= 0xFF
    return value - 256 if value >= 128 else value


def pad8(count):
    return (-count) % 8


def pack_mac_case(a, b, a_unsigned, row_mask, col_mask, k):
    """Pack one MAC-bank case as 35 little-endian 64-bit words."""
    words = []
    w0 = 0
    for lane in range(8):
        w0 |= (int(a[lane]) & 0xFF) << (8 * lane)
    w1 = 0
    for lane in range(8):
        w1 |= (int(b[lane]) & 0xFF) << (8 * lane)
    w2 = int(row_mask) & 0xFF
    w2 |= (int(col_mask) & 0xFF) << 8
    w2 |= (1 if a_unsigned else 0) << 16
    w2 |= (int(k) & 0x3FFF) << 17
    words.extend((w0, w1, w2))

    accumulators = []
    for r in range(8):
        for c in range(8):
            if (row_mask >> r) & 1 and (col_mask >> c) & 1:
                if a_unsigned:
                    product = (int(a[r]) & 0xFF) * s8(b[c])
                else:
                    product = s8(a[r]) * s8(b[c])
                accumulators.append(int(k) * product)
            else:
                accumulators.append(0)

    for idx in range(0, 64, 2):
        word = accumulators[idx] & 0xFFFFFFFF
        word |= (accumulators[idx + 1] & 0xFFFFFFFF) << 32
        words.append(word)
    return words


def generate_mac_bank(seed, outdir):
    rng = np.random.default_rng(seed)
    cases = [
        dict(a=[1, 2, 3, 4, 5, 6, 7, 8],
             b=[1, -1, 2, -2, 3, -3, 4, -4], u=0, rm=0xFF, cm=0xFF, k=1),
        dict(a=[-128] * 8, b=[-128] * 8, u=0, rm=0xFF, cm=0xFF, k=1),
        dict(a=[127] * 8, b=[127] * 8, u=0, rm=0xFF, cm=0xFF, k=1),
        dict(a=[128] * 8, b=[-128] * 8, u=1, rm=0xFF, cm=0xFF, k=1),
        dict(a=[255] * 8, b=[127] * 8, u=1, rm=0xFF, cm=0xFF, k=1),
        dict(a=[3] * 8, b=[4] * 8, u=0, rm=0x0F, cm=0x03, k=5),
        dict(a=[127] * 8, b=[127] * 8, u=0, rm=0xFF, cm=0xFF, k=768),
        dict(a=[255] * 8, b=[127] * 8, u=1, rm=0xFF, cm=0xFF, k=768),
    ]
    for _ in range(32):
        a = rng.integers(-128, 128, size=8, dtype=np.int64)
        b = rng.integers(-128, 128, size=8, dtype=np.int64)
        u = int(rng.integers(0, 2))
        rm = int(rng.integers(0, 256))
        cm = int(rng.integers(0, 256))
        k = int(rng.integers(1, 17))
        cases.append(dict(
            a=[int(value) for value in a],
            b=[int(value) for value in b],
            u=u, rm=rm, cm=cm, k=k,
        ))

    words = []
    for case in cases:
        words.extend(pack_mac_case(
            case["a"], case["b"], case["u"], case["rm"], case["cm"], case["k"]))

    mem_path = outdir / "mac_bank.mem"
    write_mem(mem_path, [len(words)] + words)
    files = [
        {
            "name": "mac_bank.mem",
            "width_bits": 64,
            "records": len(words),
            "header_words": 1,
            "cases": len(cases),
            "sha256": sha256_file(mem_path),
        }
    ]
    return files


def pack_int8_bytes(values):
    """Pack signed int8 values row-major, little-endian, padded with 0xA5."""
    byte_values = [int(value) & 0xFF for value in values]
    byte_values += [A5] * pad8(len(byte_values))
    words = []
    for i in range(0, len(byte_values), 8):
        word = 0
        for j in range(8):
            word |= byte_values[i + j] << (8 * j)
        words.append(word)
    return words, len(byte_values)


def pack_int32_bytes(values):
    """Pack int32 values row-major, little-endian, padded with 0xA5."""
    byte_values = []
    for value in values:
        value = int(value) & 0xFFFFFFFF
        byte_values += [
            value & 0xFF,
            (value >> 8) & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 24) & 0xFF,
        ]
    byte_values += [A5] * pad8(len(byte_values))
    words = []
    for i in range(0, len(byte_values), 8):
        word = 0
        for j in range(8):
            word |= byte_values[i + j] << (8 * j)
        words.append(word)
    return words, len(byte_values)


def bytes_to_words(byte_values):
    words = []
    for i in range(0, len(byte_values), 8):
        word = 0
        for j in range(8):
            word |= byte_values[i + j] << (8 * j)
        words.append(word)
    return words


def build_case(outdir, name, case_seed, m, n, k, heads, flags,
               src0_scale, src1_scale, dst_scale, bias_en,
               transpose_b, a_unsigned, out_int32):
    n_heads = heads if heads else 1
    nph = n
    a_blocks = []
    b_blocks = []
    for h in range(n_heads):
        hrng = np.random.default_rng(case_seed + 1009 * h + 7)
        if a_unsigned:
            a_blk = hrng.integers(0, 256, size=(m, k), dtype=np.int64)
        else:
            a_blk = hrng.integers(-128, 128, size=(m, k), dtype=np.int64)
        if transpose_b:
            b_blk = hrng.integers(-128, 128, size=(nph, k), dtype=np.int64)
        else:
            b_blk = hrng.integers(-128, 128, size=(k, nph), dtype=np.int64)
        a_blocks.append(a_blk)
        b_blocks.append(b_blk)

    if a_unsigned:
        # Anchor the unsigned x signed product 128 * -128 = -16384.
        a_blocks[0][0][0] = 128
        b_blocks[0][0][0] = -128

    bias = None
    if bias_en:
        brng = np.random.default_rng(case_seed + 50021)
        bias = [int(value) for value in brng.integers(-64, 64, size=n)]

    src_exp = src0_scale + src1_scale
    out_blocks = []
    for h in range(n_heads):
        bias_h = bias[h * nph:(h + 1) * nph] if bias is not None else None
        accum = gemm(
            a_blocks[h].tolist(),
            b_blocks[h].tolist(),
            bias_h,
            transpose_b,
            a_unsigned,
        )
        out_blocks.append(gemm_writeback(
            accum, src_exp, dst_scale, 32 if out_int32 else 8))

    a_flat = [int(v) for blk in a_blocks for row in blk for v in row]
    a_words, a_bytes = pack_int8_bytes(a_flat)
    b_flat = [int(v) for blk in b_blocks for row in blk for v in row]
    b_words, b_bytes = pack_int8_bytes(b_flat)
    bias_words, bias_bytes = pack_int32_bytes(bias) if bias is not None else ([], 0)
    bias_offset = b_bytes
    w_words = b_words + bias_words
    weight_bytes = bias_offset + bias_bytes

    esize = 4 if out_int32 else 1
    dst_bytes_total = m * n_heads * n * esize
    output_bytes = dst_bytes_total + pad8(dst_bytes_total)
    expected_image = [A5] * output_bytes
    for h in range(n_heads):
        for r in range(m):
            for c in range(nph):
                value = out_blocks[h][r][c]
                base = (h * m * nph + r * nph + c) * esize
                if out_int32:
                    expected_image[base:base + 4] = [
                        value & 0xFF,
                        (value >> 8) & 0xFF,
                        (value >> 16) & 0xFF,
                        (value >> 24) & 0xFF,
                    ]
                else:
                    expected_image[base] = value & 0xFF
    dst_init_words = bytes_to_words([A5] * output_bytes)
    expected_words = bytes_to_words(expected_image)

    def write_case_file(suffix, words, extra):
        path = outdir / f"{name}_{suffix}.mem"
        write_mem(path, words)
        entry = {
            "name": f"{name}_{suffix}.mem",
            "width_bits": 64,
            "records": len(words),
            "sha256": sha256_file(path),
        }
        entry.update(extra)
        return entry

    files = [
        write_case_file("a", a_words, {"tensor": "A", "bytes": a_bytes}),
        write_case_file("w", w_words, {"tensor": "B+bias", "bytes": weight_bytes}),
        write_case_file("dst_init", dst_init_words, {"tensor": "dst_init"}),
        write_case_file("expected", expected_words, {"tensor": "dst_expected"}),
    ]

    return {
        "name": name,
        "m": m,
        "n": n,
        "k": k,
        "heads": heads,
        "n_per_head": nph if heads else 0,
        "flags": flags,
        "transpose": transpose_b,
        "unsigned": a_unsigned,
        "output_bits": 32 if out_int32 else 8,
        "bias": bias_en,
        "src0_offset": 0,
        "src1_offset": 0,
        "bias_offset": bias_offset,
        "dst_offset": 0,
        "src0_scale": src0_scale,
        "src1_scale": src1_scale,
        "dst_scale": dst_scale,
        "input_bytes": a_bytes,
        "weight_bytes": weight_bytes,
        "output_bytes": output_bytes,
        "files": files,
    }


def write_sv_config(seed, cases):
    lines = []
    lines.append("`ifndef GEMM_CASES_PKG_SV")
    lines.append("`define GEMM_CASES_PKG_SV")
    lines.append("")
    lines.append("// Generated by tools/generate_gemm_vectors.py; do not edit.")
    lines.append("package gemm_cases_pkg;")
    lines.append("")
    lines.append(f"  localparam logic [31:0] INPUT_BASE   = 32'h{INPUT_BASE:08x};")
    lines.append(f"  localparam logic [31:0] WEIGHT_BASE  = 32'h{WEIGHT_BASE:08x};")
    lines.append(f"  localparam logic [31:0] SCRATCH_BASE = 32'h{SCRATCH_BASE:08x};")
    lines.append(f"  localparam logic [31:0] OUTPUT_BASE  = 32'h{OUTPUT_BASE:08x};")

    def scale_text(value):
        if value < 0:
            return f"-6'sd{abs(value)}"
        return f"6'sd{value}"

    for case in cases:
        p = case["name"].upper()
        lines.append("")
        lines.append(f"  // {case['name']}")
        lines.append(f"  localparam logic [15:0] {p}_M = 16'd{case['m']};")
        lines.append(f"  localparam logic [15:0] {p}_N = 16'd{case['n']};")
        lines.append(f"  localparam logic [15:0] {p}_K = 16'd{case['k']};")
        lines.append(f"  localparam logic [3:0]  {p}_HEADS = 4'd{case['heads']};")
        lines.append(f"  localparam logic [23:0] {p}_FLAGS = 24'h{case['flags']:06x};")
        lines.append(
            f"  localparam logic [31:0] {p}_SRC0_OFFSET = 32'h{case['src0_offset']:08x};")
        lines.append(
            f"  localparam logic [31:0] {p}_SRC1_OFFSET = 32'h{case['src1_offset']:08x};")
        lines.append(
            f"  localparam logic [31:0] {p}_BIAS_OFFSET = 32'h{case['bias_offset']:08x};")
        lines.append(
            f"  localparam logic [31:0] {p}_DST_OFFSET = 32'h{case['dst_offset']:08x};")
        lines.append(
            f"  localparam logic signed [5:0] {p}_SRC0_SCALE = "
            f"{scale_text(case['src0_scale'])};")
        lines.append(
            f"  localparam logic signed [5:0] {p}_SRC1_SCALE = "
            f"{scale_text(case['src1_scale'])};")
        lines.append(
            f"  localparam logic signed [5:0] {p}_DST_SCALE = "
            f"{scale_text(case['dst_scale'])};")
        lines.append(
            f"  localparam logic [31:0] {p}_INPUT_BYTES = 32'd{case['input_bytes']};")
        lines.append(
            f"  localparam logic [31:0] {p}_WEIGHT_BYTES = 32'd{case['weight_bytes']};")
        lines.append(
            f"  localparam logic [31:0] {p}_OUTPUT_BYTES = 32'd{case['output_bytes']};")
        lines.append(
            f"  localparam logic [15:0] {p}_N_PER_HEAD = 16'd{case['n_per_head']};")
        lines.append(
            f'  localparam string {p}_A_FILE = "sim/vectors/gemm/{case["name"]}_a.mem";')
        lines.append(
            f'  localparam string {p}_W_FILE = "sim/vectors/gemm/{case["name"]}_w.mem";')
        lines.append(
            f'  localparam string {p}_DST_FILE = "sim/vectors/gemm/{case["name"]}_dst_init.mem";')
        lines.append(
            f'  localparam string {p}_EXP_FILE = "sim/vectors/gemm/{case["name"]}_expected.mem";')
    lines.append("")
    lines.append("endpackage")
    lines.append("")
    lines.append("`endif")

    config_path = REPO_ROOT / "sim" / "generated" / "gemm_cases.sv"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def generate_scenarios(seed, outdir):
    base_flags = FLAG_SRC0_INPUT | FLAG_DST_OUTPUT
    specs = [
        ("mini", 1, 1, 1, 0, base_flags, (-7, -7, -7),
         False, False, False, False),
        ("ordinary", 7, 9, 5, 0, base_flags, (-7, -7, -7),
         False, False, False, False),
        ("full", 8, 24, 8, 0, base_flags, (-7, -7, -7),
         False, False, False, False),
        ("tail", 9, 25, 17, 0,
         base_flags | FLAG_BIAS_ENABLE | FLAG_OUTPUT_INT32, (-7, -7, -14),
         True, False, False, True),
        ("large", 197, 192, 192, 0, base_flags, (-7, -7, -7),
         False, False, False, False),
        ("head", 8, 64, 8, 3, base_flags | FLAG_HEAD_MODE, (-7, -7, -7),
         False, False, False, False),
        ("transpose", 8, 8, 64, 0, base_flags | FLAG_RHS_TRANSPOSE, (-7, -7, -7),
         False, True, False, False),
        ("unsigned", 8, 6, 3, 3,
         base_flags | FLAG_HEAD_MODE | FLAG_OUTPUT_INT32 | FLAG_SRC0_UNSIGNED,
         (-8, -7, -15), False, False, True, True),
    ]
    cases = []
    for idx, (name, m, n, k, heads, flags, scales, bias_en,
              transpose_b, a_unsigned, out_int32) in enumerate(specs):
        src0_scale, src1_scale, dst_scale = scales
        cases.append(build_case(
            outdir, name, seed + 1000003 * idx, m, n, k, heads, flags,
            src0_scale, src1_scale, dst_scale, bias_en,
            transpose_b, a_unsigned, out_int32))

    write_sv_config(seed, cases)

    files = []
    scenarios = []
    for case in cases:
        files.extend(case["files"])
        scenarios.append({
            "name": case["name"],
            "m": case["m"],
            "n": case["n"],
            "k": case["k"],
            "heads": case["heads"],
            "n_per_head": case["n_per_head"],
            "flags": case["flags"],
            "transpose": case["transpose"],
            "unsigned": case["unsigned"],
            "output_bits": case["output_bits"],
            "bias": case["bias"],
            "sha256": {
                entry["name"]: entry["sha256"] for entry in case["files"]
            },
        })
    return files, scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    outdir = args.output
    if not outdir.is_absolute():
        outdir = REPO_ROOT / outdir
    mac_files = generate_mac_bank(args.seed, outdir)
    scenario_files, scenarios = generate_scenarios(args.seed, outdir)
    write_manifest_full(outdir, args.seed, mac_files + scenario_files, scenarios)
    print(f"generated gemm vectors -> {outdir}")


if __name__ == "__main__":
    main()
