#!/usr/bin/env python3
"""Generate deterministic Token Selector vectors (Phase 4).

``--suite unit`` emits the six deterministic finalize/fuse boundary cases:
all-keep without Package, all-pruned without Package, mixed pruning,
threshold-equality, incoming Package plus pruning, and the two zero
denominators (head fuse and package fallback). Each case directory holds the
input tokens, fused keep scores, expected output tokens and state; the
top-level manifest lists every case with its expected warning bits.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from verification.heatvit_ref.selector import (  # noqa: E402
    WARN_HEAD_DEN_ZERO,
    WARN_PACKAGE_DEN_ZERO,
    finalize_tokens,
    fuse_head_scores,
)

EMBED_DIM = 192


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_pattern(t):
    """Distinct deterministic int8 fill for token index ``t``."""
    return ((t * 53 + 7) % 251) - 125


def make_tokens(n):
    """``n`` tokens; token t fills all 192 channels with ``token_pattern(t)``."""
    return [[token_pattern(t)] * EMBED_DIM for t in range(n)]


def pack_int8_words(values):
    """Pack int8 bytes into 64-bit little-endian words, 0xA5 padded."""
    byte_values = [int(v) & 0xFF for v in values]
    byte_values += [0xA5] * ((-len(byte_values)) % 8)
    words = []
    for i in range(0, len(byte_values), 8):
        word = 0
        for j in range(8):
            word |= byte_values[i + j] << (8 * j)
        words.append(word)
    return words


def pack_q016_words(values):
    """Pack Q0.16 values as 4-byte little-endian into 64-bit words."""
    byte_values = []
    for value in values:
        value = int(value) & 0xFFFFFFFF
        byte_values += [
            value & 0xFF,
            (value >> 8) & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 24) & 0xFF,
        ]
    byte_values += [0xA5] * ((-len(byte_values)) % 8)
    words = []
    for i in range(0, len(byte_values), 8):
        word = 0
        for j in range(8):
            word |= byte_values[i + j] << (8 * j)
        words.append(word)
    return words


def write_mem(path, words):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        for word in words:
            handle.write(f"{word:x}\n")


def write_finalize_case(case_dir, tokens, scores, input_package_present):
    """Write one finalize case; returns ``(files, case_meta)``."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    files = []

    normals = list(tokens[1:])
    incoming = None
    if input_package_present:
        incoming = normals[-1]
        normals = normals[:-1]
    result = finalize_tokens(tokens[0], normals, incoming, list(scores))

    paths = {
        "tokens.mem": pack_int8_words(
            [v for row in tokens for v in row]),
        "scores.mem": pack_q016_words(scores),
        "output.mem": pack_int8_words(
            [v for row in result.tokens for v in row]),
    }
    for name, words in paths.items():
        write_mem(case_dir / name, words)
        files.append(str(case_dir / name))

    meta = {
        "tokens": len(tokens),
        "candidates": len(tokens) - 1,
        "input_package_present": input_package_present,
        "next_token_count": len(result.tokens),
        "next_package_present": int(result.package_present),
        "warning_flags": result.warnings,
    }
    return files, meta


def write_fuse_case(case_dir, triples):
    """Write head-fuse vectors; returns ``(files, expected_warns)``."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    files = []

    scores_rows = [t[0] for t in triples]
    weights_rows = [t[1] for t in triples]
    expected = []
    warns = []
    for scores, weights in triples:
        # head_scores is [3 heads][1 candidate]; head_weights [1 cand][3].
        fused, warn = fuse_head_scores([[s] for s in scores], [weights])
        expected.append(fused[0])
        warns.append(1 if warn else 0)

    paths = {
        "fuse_scores.mem": pack_q016_words(
            [v for row in scores_rows for v in row]),
        "fuse_weights.mem": pack_q016_words(
            [v for row in weights_rows for v in row]),
        "fuse_expected.mem": pack_q016_words(expected),
        "fuse_warn.mem": pack_int8_words(warns),
    }
    for name, words in paths.items():
        write_mem(case_dir / name, words)
        files.append(str(case_dir / name))
    return files, warns


def generate_unit(seed, outdir):
    outdir = Path(outdir)
    cases = {}

    # 1. All keep, no incoming Package.
    tokens = make_tokens(8)
    scores = [40000 + 10 * i for i in range(7)]
    cases["keep_all"] = (tokens, scores, 0)

    # 2. All pruned, no incoming Package: one Package accumulates everything.
    tokens = make_tokens(8)
    scores = [1000 + 100 * i for i in range(7)]
    cases["prune_all"] = (tokens, scores, 0)

    # 3. Mixed pruning: alternating keep/prune.
    tokens = make_tokens(12)
    scores = [50000 if i % 2 == 0 else 1000 for i in range(11)]
    cases["mixed"] = (tokens, scores, 0)

    # 4. Threshold equality: exactly 32768 keeps, 32767 prunes.
    tokens = make_tokens(6)
    scores = [32768, 32767, 32768, 32767, 32768]
    cases["threshold_eq"] = (tokens, scores, 0)

    # 5. Incoming Package (last candidate) plus additional pruning; the
    #    Package must accumulate and never become a normal output token.
    tokens = make_tokens(8)
    scores = [50000, 50000, 1000, 1000, 1000, 1000, 65536]
    cases["incoming_package"] = (tokens, scores, 1)

    # 6. Zero denominators: all-zero scores fall back to the unweighted
    #    feature mean (package warning), plus a head-fuse zero-denominator
    #    vector set (head warning).
    tokens = make_tokens(6)
    scores = [0] * 5
    cases["zero_den"] = (tokens, scores, 0)

    all_files = []
    manifest_cases = {}
    for name, (tokens, scores, pkg) in cases.items():
        files, meta = write_finalize_case(outdir / name, tokens, scores, pkg)
        all_files += files
        manifest_cases[name] = meta

    fuse_triples = [
        ([0, 32768, 65536], [65536, 65536, 0]),     # normal -> 16384
        ([1000, 2000, 3000], [0, 0, 0]),            # zero den -> 2000, warn
        ([65536, 65536, 65536], [65536, 65536, 65536]),  # saturate -> 65536
        ([0, 0, 65536], [1, 1, 2]),                 # exact 32768
    ]
    fuse_files, fuse_warns = write_fuse_case(outdir / "zero_den", fuse_triples)
    all_files += fuse_files
    manifest_cases["zero_den"]["fuse_candidates"] = len(fuse_triples)
    manifest_cases["zero_den"]["fuse_warn_bits"] = fuse_warns

    file_entries = []
    for path in sorted(set(all_files)):
        rel = str(Path(path).relative_to(outdir))
        file_entries.append({
            "path": rel,
            "bytes": Path(path).stat().st_size,
            "sha256": sha256_file(path),
        })

    manifest = {
        "suite": "unit",
        "case": "selector",
        "seed": seed,
        "generator": "tools/generate_selector_vectors.py",
        "warning_bits": {
            "head_den_zero": WARN_HEAD_DEN_ZERO,
            "package_den_zero": WARN_PACKAGE_DEN_ZERO,
        },
        "cases": manifest_cases,
        "files": file_entries,
    }
    with open(outdir / "manifest.json", "w", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return len(manifest_cases)


def _align8(value):
    return (value + 7) & ~7


def generate_full(case, tokens, seed, outdir):
    """Full N-token selector: synthetic weights, input, 12 descriptors,
    golden checkpoints and the tb config package."""
    import numpy as np

    from verification.heatvit_ref.descriptor import (
        FLAG_BIAS_ENABLE,
        FLAG_DYNAMIC_M,
        FLAG_HEAD_MODE,
        FLAG_SRC0_CAND_MAJOR,
        FLAG_SRC1_SCRATCH,
        OP_CONCAT_LOCAL_GLOBAL,
        OP_GEMM,
        OP_HEAD_FUSE,
        OP_REDUCE_MEAN,
        OP_SELECTOR_FINALIZE,
        OP_SELECTOR_SOFTMAX,
        Descriptor,
    )
    from verification.heatvit_ref.op_sequence import write_descriptors_mem
    from verification.heatvit_ref.selector import (
        SelectorHeadWeightParams,
        SelectorLocalParams,
        SelectorParams,
        SelectorScoreParams,
        concat_local_global,
        score_mlp_layers,
        token_selector,
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = int(tokens)
    c_count = n - 1
    rng = np.random.default_rng(seed)

    def int8_matrix(rows, cols):
        return [[int(v) for v in row]
                for row in rng.integers(-8, 8, size=(rows, cols),
                                        dtype=np.int64)]

    def int32_vec(count):
        return [int(v) for v in rng.integers(-64, 64, size=count)]

    # ---------------------------------------------------------------
    # Synthetic tensors.
    # ---------------------------------------------------------------
    tokens_in = [
        [int(v) for v in row]
        for row in rng.integers(-128, 128, size=(n, 192), dtype=np.int64)
    ]

    local_w = [int8_matrix(64, 32) for _ in range(3)]
    local_b = [int32_vec(32) for _ in range(3)]
    score_w1 = [int8_matrix(64, 32) for _ in range(3)]
    score_b1 = [int32_vec(32) for _ in range(3)]
    score_w2 = [int8_matrix(32, 16) for _ in range(3)]
    score_b2 = [int32_vec(16) for _ in range(3)]
    score_w3 = [int8_matrix(16, 2) for _ in range(3)]
    score_b3 = [int32_vec(2) for _ in range(3)]
    hw_w1 = int8_matrix(3, 3)
    hw_b1 = int32_vec(3)
    hw_w2 = int8_matrix(3, 3)
    hw_b2 = int32_vec(3)

    # Deterministic calibration: adjust the logit keep bias until the
    # pruning is mixed (>=2 pruned normals, >=1 kept normal).
    calibration = []
    for _step in range(32):
        params = SelectorParams(
            local=SelectorLocalParams(w=local_w, b=local_b),
            score=SelectorScoreParams(
                w1=score_w1, b1=score_b1, w2=score_w2, b2=score_b2,
                w3=score_w3, b3=score_b3),
            head_weight=SelectorHeadWeightParams(
                w1=hw_w1, b1=hw_b1, w2=hw_w2, b2=hw_b2),
        )
        result = token_selector(tokens_in, False, params)
        if result.pruned_normal_count >= 2 and result.kept_normal_count >= 1:
            break
        if result.pruned_normal_count < 2:
            for h in range(3):
                score_b3[h] = [b - 32 for b in score_b3[h]]
            calibration.append("drop")
        else:
            for h in range(3):
                score_b3[h] = [b + 32 for b in score_b3[h]]
            calibration.append("keep")
    else:
        raise RuntimeError("selector calibration failed to reach mixed pruning")

    h1, h2, logits = score_mlp_layers(
        concat_local_global(result.local, result.global_features),
        params.score)

    # ---------------------------------------------------------------
    # Memory layout.
    # ---------------------------------------------------------------
    off_input = 0                       # N*192 int8
    off_cand = 192                      # candidates = input rows 1..N-1
    off_local = _align8(n * 192)        # [3][C][32] int8, C*96 bytes
    off_global = _align8(off_local + c_count * 96)      # 96 bytes
    off_concat = _align8(off_global + 96)               # C*192 bytes
    off_h1 = _align8(off_concat + c_count * 192)        # C*96
    off_h2 = _align8(off_h1 + c_count * 96)             # C*48
    off_logits = _align8(off_h2 + c_count * 48)         # C*6
    off_keep = _align8(off_logits + c_count * 6)        # C*12 Q0.16
    off_stats = _align8(off_keep + c_count * 12)        # C*3
    off_hw_hidden = _align8(off_stats + c_count * 3)    # C*3
    off_hw = _align8(off_hw_hidden + c_count * 3)       # C*12 Q0.16
    off_fused = _align8(off_hw + c_count * 12)          # C*4 Q0.16
    off_output = _align8(off_fused + c_count * 4)       # N*192
    scratch_bytes = _align8(off_output + n * 192)

    w_off = {}
    cursor = 0

    def alloc_w(bytes_):
        nonlocal cursor
        offset = _align8(cursor)
        cursor = offset + bytes_
        return offset

    w_off["local_w"] = alloc_w(3 * 2048)
    w_off["local_b"] = alloc_w(3 * 128)
    w_off["score_w1"] = alloc_w(3 * 2048)
    w_off["score_b1"] = alloc_w(3 * 128)
    w_off["score_w2"] = alloc_w(3 * 512)
    w_off["score_b2"] = alloc_w(3 * 64)
    w_off["score_w3"] = alloc_w(3 * 32)
    w_off["score_b3"] = alloc_w(3 * 8)
    w_off["hw_w1"] = alloc_w(9)
    w_off["hw_b1"] = alloc_w(12)
    w_off["hw_w2"] = alloc_w(9)
    w_off["hw_b2"] = alloc_w(12)
    weight_bytes = _align8(cursor)

    # ---------------------------------------------------------------
    # Descriptor sequence.
    # ---------------------------------------------------------------
    def flag(*bits):
        return sum(1 << b for b in bits)

    POST_GELU = 2
    POST_PLAN = 5

    def make(opcode, flags=0, m=0, n=0, k=0, heads=0, src0=0, src1=0,
             bias=0, dst=0, s0=-7, s1=-7, dsts=-7, param0=0):
        return Descriptor(opcode=opcode, flags=flags, m=m, n=n, k=k,
                          heads=heads, src0_offset=src0, src1_offset=src1,
                          bias_offset=bias, dst_offset=dst, src0_scale_exp=s0,
                          src1_scale_exp=s1, dst_scale_exp=dsts, param0=param0)

    descs = [
        make(OP_GEMM,
             flag(FLAG_HEAD_MODE, FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M,
                  FLAG_SRC0_CAND_MAJOR) | (POST_GELU << 8),
             m=c_count, n=32, k=64, heads=3,
             src0=off_cand, src1=w_off["local_w"], bias=w_off["local_b"],
             dst=off_local, param0=1),
        make(OP_REDUCE_MEAN, flag(FLAG_DYNAMIC_M),
             n=32, heads=3, src0=off_local, dst=off_global, param0=1),
        make(OP_CONCAT_LOCAL_GLOBAL, flag(FLAG_DYNAMIC_M, FLAG_SRC1_SCRATCH),
             n=64, heads=3, src0=off_local, src1=off_global, dst=off_concat,
             param0=1),
        make(OP_GEMM,
             flag(FLAG_HEAD_MODE, FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M) |
                 (POST_GELU << 8),
             m=c_count, n=32, k=64, heads=3,
             src0=off_concat, src1=w_off["score_w1"], bias=w_off["score_b1"],
             dst=off_h1, param0=1),
        make(OP_GEMM,
             flag(FLAG_HEAD_MODE, FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M) |
                 (POST_GELU << 8),
             m=c_count, n=16, k=32, heads=3,
             src0=off_h1, src1=w_off["score_w2"], bias=w_off["score_b2"],
             dst=off_h2, param0=1),
        make(OP_GEMM, flag(FLAG_HEAD_MODE, FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M),
             m=c_count, n=2, k=16, heads=3,
             src0=off_h2, src1=w_off["score_w3"], bias=w_off["score_b3"],
             dst=off_logits, param0=1),
        make(OP_SELECTOR_SOFTMAX, flag(FLAG_DYNAMIC_M),
             n=2, heads=3, src0=off_logits, dst=off_keep, s0=-7, param0=1),
        make(OP_REDUCE_MEAN, flag(FLAG_DYNAMIC_M),
             n=64, heads=3, src0=off_cand, dst=off_stats, param0=5),
        make(OP_GEMM,
             flag(FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M) | (POST_GELU << 8),
             m=c_count, n=3, k=3,
             src0=off_stats, src1=w_off["hw_w1"], bias=w_off["hw_b1"],
             dst=off_hw_hidden, param0=1),
        make(OP_GEMM,
             flag(FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M) | (POST_PLAN << 8),
             m=c_count, n=3, k=3,
             src0=off_hw_hidden, src1=w_off["hw_w2"], bias=w_off["hw_b2"],
             dst=off_hw, param0=1),
        make(OP_HEAD_FUSE, flag(FLAG_DYNAMIC_M, FLAG_SRC1_SCRATCH),
             n=3, heads=3, src0=off_keep, src1=off_hw, dst=off_fused,
             param0=1),
        make(OP_SELECTOR_FINALIZE, flag(FLAG_DYNAMIC_M, FLAG_SRC1_SCRATCH),
             n=192, src0=off_input, src1=off_fused, dst=off_output, param0=0),
    ]

    # ---------------------------------------------------------------
    # Weight region bytes.
    # ---------------------------------------------------------------
    files = []

    def emit(name, words):
        path = outdir / name
        write_mem(path, words)
        files.append(str(path))
        return path

    wbytes = bytearray(weight_bytes)
    for h in range(3):
        for r in range(64):
            for col in range(32):
                wbytes[w_off["local_w"] + h * 2048 + r * 32 + col] = \
                    local_w[h][r][col] & 0xFF
    for h in range(3):
        for col in range(32):
            v = local_b[h][col] & 0xFFFFFFFF
            base = w_off["local_b"] + h * 128 + col * 4
            wbytes[base:base + 4] = v.to_bytes(4, "little")
    for h in range(3):
        for r in range(64):
            for col in range(32):
                wbytes[w_off["score_w1"] + h * 2048 + r * 32 + col] = \
                    score_w1[h][r][col] & 0xFF
    for h in range(3):
        for col in range(32):
            v = score_b1[h][col] & 0xFFFFFFFF
            base = w_off["score_b1"] + h * 128 + col * 4
            wbytes[base:base + 4] = v.to_bytes(4, "little")
    for h in range(3):
        for r in range(32):
            for col in range(16):
                wbytes[w_off["score_w2"] + h * 512 + r * 16 + col] = \
                    score_w2[h][r][col] & 0xFF
    for h in range(3):
        for col in range(16):
            v = score_b2[h][col] & 0xFFFFFFFF
            base = w_off["score_b2"] + h * 64 + col * 4
            wbytes[base:base + 4] = v.to_bytes(4, "little")
    for h in range(3):
        for r in range(16):
            for col in range(2):
                wbytes[w_off["score_w3"] + h * 32 + r * 2 + col] = \
                    score_w3[h][r][col] & 0xFF
    for h in range(3):
        for col in range(2):
            v = score_b3[h][col] & 0xFFFFFFFF
            base = w_off["score_b3"] + h * 8 + col * 4
            wbytes[base:base + 4] = v.to_bytes(4, "little")
    for r in range(3):
        for col in range(3):
            wbytes[w_off["hw_w1"] + r * 3 + col] = hw_w1[r][col] & 0xFF
    for col in range(3):
        v = hw_b1[col] & 0xFFFFFFFF
        base = w_off["hw_b1"] + col * 4
        wbytes[base:base + 4] = v.to_bytes(4, "little")
    for r in range(3):
        for col in range(3):
            wbytes[w_off["hw_w2"] + r * 3 + col] = hw_w2[r][col] & 0xFF
    for col in range(3):
        v = hw_b2[col] & 0xFFFFFFFF
        base = w_off["hw_b2"] + col * 4
        wbytes[base:base + 4] = v.to_bytes(4, "little")

    # ---------------------------------------------------------------
    # Byte emission (64-bit little-endian hex words).
    # ---------------------------------------------------------------
    emit("weight.mem", pack_int8_words(list(wbytes)))
    emit("input.mem", pack_int8_words([v for row in tokens_in for v in row]))
    write_descriptors_mem(outdir / "descriptors.mem", descs)
    files.append(str(outdir / "descriptors.mem"))

    # Golden checkpoints.
    emit("local_expected.mem",
         pack_int8_words([v for h in range(3) for row in result.local[h]
                          for v in row]))
    emit("global_expected.mem",
         pack_int8_words([v for h in range(3)
                          for v in result.global_features[h]]))
    local_global = concat_local_global(result.local, result.global_features)
    emit("concat_expected.mem",
         pack_int8_words([v for h in range(3) for row in local_global[h]
                          for v in row]))
    emit("h1_expected.mem",
         pack_int8_words([v for h in range(3) for row in h1[h] for v in row]))
    emit("h2_expected.mem",
         pack_int8_words([v for h in range(3) for row in h2[h] for v in row]))
    emit("logits_expected.mem",
         pack_int8_words([v for h in range(3) for row in logits[h]
                          for v in row]))
    emit("keep_expected.mem",
         pack_q016_words([v for h in range(3)
                          for v in result.head_scores[h]]))
    emit("stats_expected.mem",
         pack_int8_words([v for row in result.head_stats for v in row]))
    emit("hw_expected.mem",
         pack_q016_words([v for row in result.head_weights for v in row]))
    emit("fused_expected.mem", pack_q016_words(list(result.fused_scores)))
    emit("output_expected.mem",
         pack_int8_words([v for row in result.tokens for v in row]))

    # The head-weight hidden layer equals the golden int8 GELU layer; the
    # golden only stores the Q0.16 weights, so recompute the hidden layer
    # from the same MLP inputs (stats -> w_h1 GELU) deterministically.
    from verification.heatvit_ref.fixed import requant
    from verification.heatvit_ref.gemm import gemm_numpy
    from verification.heatvit_ref.nonlinear import gelu

    hw_hidden = []
    for row in gemm_numpy([list(r) for r in result.head_stats],
                          [list(r) for r in hw_w1], list(hw_b1), False):
        hw_hidden.append([
            requant(gelu(requant(int(v), -14, -16, 24)), -16, -7, 8)
            for v in row
        ])
    emit("hw_hidden_expected.mem",
         pack_int8_words([v for row in hw_hidden for v in row]))

    # ---------------------------------------------------------------
    # TB config package.
    # ---------------------------------------------------------------
    lines = []
    lines.append("`ifndef SELECTOR_TB_CONFIG_PKG_SV")
    lines.append("`define SELECTOR_TB_CONFIG_PKG_SV")
    lines.append("")
    lines.append("// Generated by tools/generate_selector_vectors.py; do not edit.")
    lines.append("package selector_tb_config_pkg;")
    lines.append("")
    lines.append("  localparam logic [31:0] INPUT_BASE   = 32'h02000000;")
    lines.append(f"  localparam int          N             = {n};")
    lines.append(f"  localparam int          C             = {c_count};")
    lines.append(f"  localparam logic [31:0] WEIGHT_BASE  = 32'h01000000;")
    lines.append(f"  localparam int          WEIGHT_BYTES = {weight_bytes};")
    lines.append("  localparam logic [31:0] SCRATCH_BASE = 32'h02000000;")
    lines.append(f"  localparam int          SCRATCH_BYTES = {scratch_bytes};")
    lines.append("  localparam logic [31:0] OUTPUT_BASE  = 32'h03000000;")
    lines.append("  localparam int          OUTPUT_BYTES = 0;")
    lines.append("")
    offsets = {
        "INPUT": off_input, "CAND": off_cand, "LOCAL": off_local,
        "GLOBAL": off_global, "CONCAT": off_concat, "H1": off_h1,
        "H2": off_h2, "LOGITS": off_logits, "KEEP": off_keep,
        "STATS": off_stats, "HW_HIDDEN": off_hw_hidden, "HW": off_hw,
        "FUSED": off_fused, "OUTPUT": off_output,
    }
    for name, off in offsets.items():
        lines.append(f"  localparam logic [31:0] {name}_OFF = 32'h{off:08x};")
    lines.append(f"  localparam int INPUT_BYTES = {n * 192};")
    lines.append(f"  localparam int LOCAL_BYTES = {c_count * 96};")
    lines.append("  localparam int GLOBAL_BYTES = 96;")
    lines.append(f"  localparam int CONCAT_BYTES = {c_count * 192};")
    lines.append(f"  localparam int H1_BYTES = {c_count * 96};")
    lines.append(f"  localparam int H2_BYTES = {c_count * 48};")
    lines.append(f"  localparam int LOGITS_BYTES = {c_count * 6};")
    lines.append(f"  localparam int KEEP_BYTES = {c_count * 12};")
    lines.append(f"  localparam int STATS_BYTES = {c_count * 3};")
    lines.append(f"  localparam int HW_HIDDEN_BYTES = {c_count * 3};")
    lines.append(f"  localparam int HW_BYTES = {c_count * 12};")
    lines.append(f"  localparam int FUSED_BYTES = {c_count * 4};")
    lines.append(f"  localparam int OUT_BYTES = {len(result.tokens) * 192};")
    lines.append(f"  localparam logic [7:0] NEXT_TOKEN_COUNT = "
                 f"8'd{result.token_count};")
    lines.append(f"  localparam logic       NEXT_PACKAGE_PRESENT = "
                 f"1'b{int(result.package_present)};")
    lines.append("")
    for idx, desc in enumerate(descs):
        lines.append(
            f"  localparam logic [319:0] DESC{idx} = 320'h{desc.pack():080x};")
    lines.append("")
    trace = [
        (0, [(0x02000000 + off_cand, c_count * 192),
             (0x01000000 + w_off["local_w"], 3 * 2048),
             (0x01000000 + w_off["local_b"], 3 * 128),
             (0x02000000 + off_local, c_count * 96)]),
        (1, [(0x02000000 + off_local, c_count * 96),
             (0x02000000 + off_global, 96)]),
        (2, [(0x02000000 + off_local, c_count * 96),
             (0x02000000 + off_global, 96),
             (0x02000000 + off_concat, c_count * 192)]),
        (3, [(0x02000000 + off_concat, c_count * 192),
             (0x01000000 + w_off["score_w1"], 3 * 2048),
             (0x01000000 + w_off["score_b1"], 3 * 128),
             (0x02000000 + off_h1, c_count * 96)]),
        (4, [(0x02000000 + off_h1, c_count * 96),
             (0x01000000 + w_off["score_w2"], 3 * 512),
             (0x01000000 + w_off["score_b2"], 3 * 64),
             (0x02000000 + off_h2, c_count * 48)]),
        (5, [(0x02000000 + off_h2, c_count * 48),
             (0x01000000 + w_off["score_w3"], 3 * 32),
             (0x01000000 + w_off["score_b3"], 3 * 8),
             (0x02000000 + off_logits, c_count * 6)]),
        (6, [(0x02000000 + off_logits, c_count * 6),
             (0x02000000 + off_keep, c_count * 12)]),
        (7, [(0x02000000 + off_cand, c_count * 192),
             (0x02000000 + off_stats, c_count * 3)]),
        (8, [(0x02000000 + off_stats, c_count * 3),
             (0x01000000 + w_off["hw_w1"], 9),
             (0x01000000 + w_off["hw_b1"], 12),
             (0x02000000 + off_hw_hidden, c_count * 3)]),
        (9, [(0x02000000 + off_hw_hidden, c_count * 3),
             (0x01000000 + w_off["hw_w2"], 9),
             (0x01000000 + w_off["hw_b2"], 12),
             (0x02000000 + off_hw, c_count * 12)]),
        (10, [(0x02000000 + off_keep, c_count * 12),
              (0x02000000 + off_hw, c_count * 12),
              (0x02000000 + off_fused, c_count * 4)]),
        (11, [(0x02000000 + off_input, n * 192),
              (0x02000000 + off_fused, c_count * 4),
              (0x02000000 + off_output, n * 192)]),
    ]
    for prefix, regions in trace:
        for idx, (base, size) in enumerate(regions):
            lines.append(
                f"  localparam logic [31:0] D{prefix}_R{idx}_BASE  = "
                f"32'h{base:08x};")
            lines.append(
                f"  localparam int          D{prefix}_R{idx}_BYTES = "
                f"{_align8(size)};")
    lines.append("")
    lines.append("endpackage")
    lines.append("")
    lines.append("`endif")
    cfg_path = REPO_ROOT / "sim" / "generated" / "selector_tb_config.sv"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8",
                        newline="\n")

    file_entries = []
    for path in sorted(set(files)):
        rel = str(Path(path).relative_to(outdir))
        file_entries.append({
            "path": rel,
            "bytes": Path(path).stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "suite": "full",
        "case": case,
        "seed": seed,
        "tokens": n,
        "candidates": c_count,
        "generator": "tools/generate_selector_vectors.py",
        "scratch_bytes": scratch_bytes,
        "weight_bytes": weight_bytes,
        "next_token_count": result.token_count,
        "next_package_present": int(result.package_present),
        "kept_normal_count": result.kept_normal_count,
        "pruned_normal_count": result.pruned_normal_count,
        "warnings": result.warnings,
        "calibration": calibration,
        "scales": {"activation": -7, "weight": -7, "logit": -7,
                   "q0_16": -16},
        "descriptors": [
            {"opcode": d.opcode, "hex": f"{d.pack():080x}"} for d in descs
        ],
        "files": file_entries,
    }
    with open(outdir / "manifest.json", "w", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=["unit", "full"],
                        help="vector suite to generate")
    parser.add_argument("--case", choices=["mixed"], default="mixed",
                        help="full-suite case name")
    parser.add_argument("--tokens", type=int, default=197,
                        help="full-suite token count")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.suite == "unit":
        count = generate_unit(args.seed, args.output)
        print(f"wrote {count} selector unit cases to {args.output}")
    elif args.suite == "full":
        generate_full(args.case, args.tokens, args.seed, args.output)
        print(f"wrote full selector case {args.case} to {args.output}")


if __name__ == "__main__":
    main()
