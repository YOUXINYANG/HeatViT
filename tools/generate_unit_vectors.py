#!/usr/bin/env python3
"""Generate deterministic .mem unit vectors and JSON manifests.

Each .mem line holds one word as an unprefixed little-endian hex value.
The manifest records the seed, word width, record count, and SHA-256 so
regressions can verify that regeneration is reproducible.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from verification.heatvit_ref.fixed import isqrt, requant, udiv  # noqa: E402
from verification.heatvit_ref.nonlinear import (  # noqa: E402
    gelu,
    layernorm,
    plan_sigmoid,
    softmax_attention,
    softmax_selector,
)


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


def write_manifest(outdir, suite, seed, files):
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": suite,
        "seed": seed,
        "generator": "tools/generate_unit_vectors.py",
        "files": files,
    }
    with open(outdir / "manifest.json", "w", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def pack_requant(value, src_exp, dst_exp, expected):
    """Pack a 48-bit value, two 6-bit exponents, and int8 result (68 bits)."""
    word = int(value) & ((1 << 48) - 1)
    word |= (int(src_exp) & 0x3F) << 48
    word |= (int(dst_exp) & 0x3F) << 54
    word |= (int(expected) & 0xFF) << 60
    return word


def generate_fixed(seed, outdir):
    rng = np.random.default_rng(seed)
    count = 128

    values = rng.integers(-(1 << 47), 1 << 47, size=count, dtype=np.int64)
    src_exps = rng.integers(-32, 32, size=count, dtype=np.int64)
    dst_exps = rng.integers(-32, 32, size=count, dtype=np.int64)

    edge_values = [-(1 << 47), (1 << 47) - 1, 0, 1, -1, 3, -3, 255, -255]
    edge_src = [-32, -32, -8, 0, 0, 0, 0, -8, -8]
    edge_dst = [31, 31, -7, 1, 1, 1, 1, -7, -7]

    values = np.concatenate([values, np.array(edge_values, dtype=np.int64)])
    src_exps = np.concatenate([src_exps, np.array(edge_src, dtype=np.int64)])
    dst_exps = np.concatenate([dst_exps, np.array(edge_dst, dtype=np.int64)])

    words = []
    for value, src_exp, dst_exp in zip(values, src_exps, dst_exps):
        expected = requant(int(value), int(src_exp), int(dst_exp), 8)
        words.append(pack_requant(value, src_exp, dst_exp, expected))

    mem_path = outdir / "requant.mem"
    write_mem(mem_path, words)
    files = [
        {
            "name": "requant.mem",
            "width_bits": 68,
            "records": len(words),
            "sha256": sha256_file(mem_path),
        }
    ]
    write_manifest(outdir, "fixed", seed, files)


def generate_requant(seed, outdir):
    rng = np.random.default_rng(seed)
    count = 1024

    values = rng.integers(-(1 << 47), 1 << 47, size=count, dtype=np.int64)
    src_exps = rng.integers(-32, 32, size=count, dtype=np.int64)
    dst_exps = rng.integers(-32, 32, size=count, dtype=np.int64)

    edge_values = [
        1, -1, 3, -3, 1024, -1025, 127, -128, 64, 65, -65,
        -(1 << 47), (1 << 47) - 1,
    ]
    edge_src = [0, 0, 0, 0, 0, 0, -7, -7, -7, -7, -7, -32, -32]
    edge_dst = [1, 1, 1, 1, 0, 0, -6, -6, -8, -8, -8, 31, 31]

    values = np.concatenate([values, np.array(edge_values, dtype=np.int64)])
    src_exps = np.concatenate([src_exps, np.array(edge_src, dtype=np.int64)])
    dst_exps = np.concatenate([dst_exps, np.array(edge_dst, dtype=np.int64)])

    words = []
    for value, src_exp, dst_exp in zip(values, src_exps, dst_exps):
        expected = requant(int(value), int(src_exp), int(dst_exp), 8)
        words.append(pack_requant(value, src_exp, dst_exp, expected))

    mem_path = outdir / "requant.mem"
    write_mem(mem_path, [len(words)] + words)
    files = [
        {
            "name": "requant.mem",
            "width_bits": 68,
            "records": len(words),
            "header_words": 1,
            "sha256": sha256_file(mem_path),
        }
    ]
    write_manifest(outdir, "requant", seed, files)


def generate_divsqrt(seed, outdir):
    rng = np.random.default_rng(seed)
    count = 1024

    nums = rng.integers(0, 1 << 64, size=count, dtype=np.uint64)
    dens = rng.integers(1, 1 << 64, size=count, dtype=np.uint64)
    rads = rng.integers(0, 1 << 48, size=count, dtype=np.uint64)

    udiv_edges = [
        (10, 3),
        (0, 5),
        (5, 10),
        ((1 << 64) - 1, 1),
        ((1 << 64) - 1, (1 << 64) - 1),
        (1, (1 << 64) - 1),
        ((1 << 63), 7),
    ]
    isqrt_edges = [
        0, 1, 2, 3, 15, 16, 255, 256,
        (1 << 48) - 1, (1 << 48) - 2,
        ((1 << 24) - 1) ** 2,
        (1 << 47) + 12345,
    ]

    nums = np.concatenate([nums, np.array([e[0] for e in udiv_edges], dtype=np.uint64)])
    dens = np.concatenate([dens, np.array([e[1] for e in udiv_edges], dtype=np.uint64)])
    rads = np.concatenate([rads, np.array(isqrt_edges, dtype=np.uint64)])

    udiv_words = []
    for num, den in zip(nums, dens):
        quotient, remainder = udiv(int(num), int(den))
        word = int(num)
        word |= int(den) << 64
        word |= quotient << 128
        word |= remainder << 192
        udiv_words.append(word)

    isqrt_words = []
    for rad in rads:
        root, remainder = isqrt(int(rad))
        word = int(rad)
        word |= root << 48
        word |= remainder << 80
        isqrt_words.append(word)

    udiv_path = outdir / "udiv.mem"
    isqrt_path = outdir / "isqrt.mem"
    write_mem(udiv_path, [len(udiv_words)] + udiv_words)
    write_mem(isqrt_path, [len(isqrt_words)] + isqrt_words)

    files = [
        {
            "name": "udiv.mem",
            "width_bits": 256,
            "records": len(udiv_words),
            "header_words": 1,
            "sha256": sha256_file(udiv_path),
        },
        {
            "name": "isqrt.mem",
            "width_bits": 128,
            "records": len(isqrt_words),
            "header_words": 1,
            "sha256": sha256_file(isqrt_path),
        },
    ]
    write_manifest(outdir, "divsqrt", seed, files)


def generate_nonlinear(seed, outdir):
    rng = np.random.default_rng(seed)
    count = 1024

    inputs = rng.integers(-(1 << 23), 1 << 23, size=count, dtype=np.int64)
    # Edge inputs: ShiftGELU shift-exp boundaries. i_p2 = x * 621/256
    # (1.702 * log2(e) with (1.1011)b * (1.0111)b = 621/256 exactly), so
    # the q = |i_p2| >> 16 breakpoints sit at x = k * 65536 * 256 / 621.
    # Covered: q in {1, 2} (near zero), {7, 8} (e saturation edge),
    # {16, 17} (negative-side underflow edge), 24 (deep saturation),
    # plus the fractional boundary r = 0xFFFF/0x0000 and sign flips.
    edge_inputs = [
        0, 1, -1, 2, -2,
        32767, 32768, 32769, -32767, -32768, -32769,
        65535, 65536, 65537, -65535, -65536, -65537,
        155647, 155648, 155649, -155647, -155648, -155649,
        327679, 327680, 327681, -327679, -327680, -327681,
        -(1 << 23), (1 << 23) - 1,
    ]
    for k in (1, 2, 7, 8, 16, 17, 24):
        x0 = k * 65536 * 256 // 621
        for dx in (-1, 0, 1):
            edge_inputs += [x0 + dx, -(x0 + dx)]
    # r = 0xFFFF fractional boundary: i_p2 = k*65536 - 1 -> x ~ k*65536*256/621 - 256/621
    for k in (2, 8):
        x0 = (k * 65536 - 1) * 256 // 621
        for dx in (-1, 0, 1):
            edge_inputs += [x0 + dx, -(x0 + dx)]
    inputs = np.concatenate([inputs, np.array(edge_inputs, dtype=np.int64)])

    words = []
    for x in inputs:
        expected_gelu = gelu(int(x))
        expected_plan = plan_sigmoid(int(x))
        word = int(x) & ((1 << 24) - 1)
        word |= (int(expected_gelu) & ((1 << 24) - 1)) << 24
        word |= (int(expected_plan) & ((1 << 17) - 1)) << 48
        words.append(word)

    mem_path = outdir / "nonlinear.mem"
    write_mem(mem_path, [len(words)] + words)
    files = [
        {
            "name": "nonlinear.mem",
            "width_bits": 65,
            "records": len(words),
            "header_words": 1,
            "sha256": sha256_file(mem_path),
        }
    ]
    write_manifest(outdir, "nonlinear", seed, files)


def generate_softmax(seed, outdir):
    rng = np.random.default_rng(seed)
    rows = [
        [0],
        [0, 0],
        [0, 0, 0],
        [12345],
        [0] * 197,
        [(1 << 23) - 1, 0, -(1 << 23)],
        [-(1 << 23), -(1 << 23), (1 << 23) - 1, (1 << 23) - 1],
        [100000, 0, -100000],
        [1, 2, 3, 4, 5],
        [(1 << 23) - 1] + [0] * 196,
        [-(1 << 23)] * 197,
        [0, 1],
        [16384, -16384],
        [0, 100, 200, 300],
        [327680, 327680, 0, -327680],
        [-1, 0, 1],
    ]
    for _ in range(256):
        count = int(rng.integers(1, 198))
        row = rng.integers(-(1 << 23), 1 << 23, size=count, dtype=np.int64)
        rows.append([int(value) for value in row])
    rows.append([
        int(value)
        for value in rng.integers(-(1 << 23), 1 << 23, size=197, dtype=np.int64)
    ])

    words = []
    for row in rows:
        expected_sel = softmax_selector(row)
        expected_att = softmax_attention(row)
        for x, expected_selector, expected_attention in zip(row, expected_sel, expected_att):
            word = len(row)
            word |= (int(x) & ((1 << 24) - 1)) << 8
            word |= (int(expected_attention) & 0xFF) << 32
            word |= (int(expected_selector) & ((1 << 17) - 1)) << 40
            words.append(word)

    mem_path = outdir / "softmax.mem"
    write_mem(mem_path, [len(words)] + words)
    files = [
        {
            "name": "softmax.mem",
            "width_bits": 57,
            "records": len(words),
            "header_words": 1,
            "sha256": sha256_file(mem_path),
        }
    ]
    write_manifest(outdir, "softmax", seed, files)


def generate_layernorm(seed, outdir):
    rng = np.random.default_rng(seed)
    gamma64 = [64] * 192
    beta0 = [0] * 192
    tokens = [
        ([0] * 192, gamma64, beta0, (-7, -6, -7, -7)),
        ([42] * 192, gamma64, beta0, (-7, -6, -7, -7)),
        ([42] * 192, gamma64, [5] * 192, (-7, -6, -7, -7)),
        (list(range(-128, 64)), gamma64, beta0, (-7, -6, -7, -7)),
        (list(range(63, -129, -1)), gamma64, beta0, (-7, -6, -7, -7)),
        ([(i % 37) * ((i % 2) * 2 - 1) for i in range(192)], gamma64, beta0, (-7, -6, -7, -7)),
        ([127] * 95 + [90] * 97, gamma64, beta0, (-23, -6, -7, -7)),
    ]
    for _ in range(64):
        inputs = rng.integers(-128, 128, size=192, dtype=np.int64)
        gammas = rng.integers(-128, 128, size=192, dtype=np.int64)
        betas = rng.integers(-128, 128, size=192, dtype=np.int64)
        scales = (
            int(rng.integers(-32, 1)),
            int(rng.integers(-32, 32)),
            int(rng.integers(-32, 32)),
            int(rng.integers(-32, 32)),
        )
        tokens.append((
            [int(value) for value in inputs],
            [int(value) for value in gammas],
            [int(value) for value in betas],
            scales,
        ))

    words = []
    for inputs, gammas, betas, scales in tokens:
        outputs, warn = layernorm(inputs, gammas, betas, *scales)
        cfg_word = int(scales[0]) & 0x3F
        cfg_word |= (int(scales[1]) & 0x3F) << 6
        cfg_word |= (int(scales[2]) & 0x3F) << 12
        cfg_word |= (int(scales[3]) & 0x3F) << 18
        cfg_word |= (1 if warn else 0) << 24
        words.append(cfg_word)
        for x, gamma, beta, expected in zip(inputs, gammas, betas, outputs):
            word = int(x) & 0xFF
            word |= (int(gamma) & 0xFF) << 8
            word |= (int(beta) & 0xFF) << 16
            word |= (int(expected) & 0xFF) << 24
            words.append(word)

    mem_path = outdir / "layernorm.mem"
    write_mem(mem_path, [len(tokens)] + words)
    files = [
        {
            "name": "layernorm.mem",
            "width_bits": 32,
            "records": len(tokens),
            "header_words": 1,
            "channels": 192,
            "sha256": sha256_file(mem_path),
        }
    ]
    write_manifest(outdir, "layernorm", seed, files)


SUITES = {
    "fixed": generate_fixed,
    "requant": generate_requant,
    "divsqrt": generate_divsqrt,
    "nonlinear": generate_nonlinear,
    "softmax": generate_softmax,
    "layernorm": generate_layernorm,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    outdir = args.output
    if not outdir.is_absolute():
        outdir = REPO_ROOT / outdir
    SUITES[args.suite](args.seed, outdir)
    print(f"generated {args.suite} vectors -> {outdir}")


if __name__ == "__main__":
    main()
