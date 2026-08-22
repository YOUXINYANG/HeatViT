#!/usr/bin/env python3
"""Compile the fixed 198-descriptor HeatViT schedule (Phase 5 Task 1).

Builds the four-region memory map (input / weight / scratch / output), the
five descriptor sequence builders (patch, block, selector, final LayerNorm,
classifier), and emits:

  * rtl/generated/heatvit_descriptors.mem  - 198 lines of 80 hex digits
  * build/vectors/e2e/descriptor_listing.csv - one human-readable row per
    descriptor with a label
  * build/vectors/e2e/memory_map.json     - every tensor offset/size/scale

Activation double-buffering convention: the two activation slots sit at the
start of the Scratch region (slot size = 197*192 = 37824 bytes). Descriptors
encode activation references as slot-relative offsets (< slot size); the
scheduler rebases them onto the current activation buffer at issue time.
The 15 buffer-switch points (each Block Residual2 and each Selector
Finalize) set flag 4 and write the inactive slot. The static schedule
therefore ping-pongs: patch -> BUF0, block 1 -> BUF1, block 2 -> BUF0, ...
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from verification.heatvit_ref.descriptor import (  # noqa: E402
    FLAG_AUX_WEIGHT,
    FLAG_BIAS_ENABLE,
    FLAG_DST_OUTPUT,
    FLAG_DYNAMIC_K,
    FLAG_DYNAMIC_M,
    FLAG_DYNAMIC_N,
    FLAG_HEAD_MODE,
    FLAG_OUTPUT_INT32,
    FLAG_RHS_TRANSPOSE,
    FLAG_SRC0_CAND_MAJOR,
    FLAG_SRC0_INPUT,
    FLAG_SRC0_UNSIGNED,
    FLAG_SRC1_SCRATCH,
    FLAG_SWAP_ACTIVATION,
    OP_ATTN_SOFTMAX,
    OP_CONCAT_LOCAL_GLOBAL,
    OP_COPY_ADD_POS,
    OP_FINISH,
    OP_GEMM,
    OP_HEAD_CONCAT,
    OP_HEAD_FUSE,
    OP_LAYERNORM,
    OP_PATCHIFY,
    OP_QKV_UNPACK,
    OP_REDUCE_MEAN,
    OP_RESIDUAL,
    OP_SELECTOR_FINALIZE,
    OP_SELECTOR_SOFTMAX,
    Descriptor,
)
from verification.heatvit_ref.op_sequence import write_descriptors_mem  # noqa: E402

D = 192
HEADS = 3
HEAD_DIM = 64
FFN_DIM = 768
N_MAX = 197
CLASSES = 1000
ACT_SLOT = N_MAX * D          # 37824 bytes per activation buffer

POST_GELU = 2
POST_PLAN = 5

SCALES = {
    "input": -7, "activation": -7, "weight": -7, "cls_pos_beta": -7,
    "gamma": -6, "q8_16": -16, "attn_uq0_8": -8, "selector_q0_16": -16,
    "logit": -14,
}


def _align8(value):
    return (value + 7) & ~7


def _flag(*bits):
    return sum(1 << b for b in bits)


def _desc(opcode, flags=0, m=0, n=0, k=0, heads=0, src0=0, src1=0,
          bias=0, aux=0, dst=0, s0=-7, s1=-7, saux=-7, sdst=-7,
          param0=0):
    return Descriptor(
        opcode=opcode, flags=flags, m=m, n=n, k=k, heads=heads,
        src0_offset=src0, src1_offset=src1, bias_offset=bias,
        aux_offset=aux, dst_offset=dst,
        src0_scale_exp=s0, src1_scale_exp=s1, aux_scale_exp=saux,
        dst_scale_exp=sdst, param0=param0,
    )


# ---------------------------------------------------------------------
# Memory map.
# ---------------------------------------------------------------------
def build_memory_map():
    mm = {"scratch": {}, "weight": {}}
    scratch = mm["scratch"]
    scratch["activation_slot_bytes"] = ACT_SLOT
    scratch["buf0"] = 0
    scratch["buf1"] = ACT_SLOT
    cur = 2 * ACT_SLOT

    def alloc(name, size):
        nonlocal cur
        off = _align8(cur)
        cur = off + size
        scratch[name] = off
        return off

    scratch["patch_matrix"] = alloc("patch_matrix", 196 * 768)
    scratch["patch_embed"] = alloc("patch_embed", 196 * 192)

    b = {}
    b["ln1"] = alloc("block_ln1", N_MAX * 192)
    b["fused"] = alloc("block_fused", N_MAX * 576)
    b["qkv"] = alloc("block_qkv", 3 * 3 * N_MAX * 64)
    b["score"] = alloc("block_score", 3 * N_MAX * N_MAX * 4)
    b["prob"] = alloc("block_prob", 3 * N_MAX * N_MAX)
    b["context"] = alloc("block_context", 3 * N_MAX * 64)
    b["concat"] = alloc("block_concat", N_MAX * 192)
    b["msa"] = alloc("block_msa", N_MAX * 192)
    b["y"] = alloc("block_y", N_MAX * 192)
    b["ln2"] = alloc("block_ln2", N_MAX * 192)
    b["hidden"] = alloc("block_hidden", N_MAX * 768)
    b["ffn_out"] = alloc("block_ffn_out", N_MAX * 192)
    scratch["block"] = b

    s = {}
    s["local"] = alloc("sel_local", 3 * 196 * 32)
    s["global"] = alloc("sel_global", 96)
    s["concat"] = alloc("sel_concat", 3 * 196 * 64)
    s["h1"] = alloc("sel_h1", 3 * 196 * 32)
    s["h2"] = alloc("sel_h2", 3 * 196 * 16)
    s["logits"] = alloc("sel_logits", 3 * 196 * 2)
    s["keep"] = alloc("sel_keep", 3 * 196 * 4)
    s["stats"] = alloc("sel_stats", 196 * 3)
    s["hw_hidden"] = alloc("sel_hw_hidden", 196 * 3)
    s["hw"] = alloc("sel_hw", 196 * 3 * 4)
    s["fused"] = alloc("sel_fused", 196 * 4)
    scratch["selector"] = s

    scratch["final_ln"] = alloc("final_ln", N_MAX * 192)
    scratch_bytes = _align8(cur)

    weight = mm["weight"]
    wcur = 0

    def alloc_w(name, size):
        nonlocal wcur
        off = _align8(wcur)
        wcur = off + size
        weight[name] = off
        return off

    weight["patch_w"] = alloc_w("patch_w", 768 * 192)
    weight["patch_b"] = alloc_w("patch_b", 192 * 4)
    weight["cls"] = alloc_w("cls", 192)
    weight["pos"] = alloc_w("pos", 197 * 192)

    block_layout = (
        ("wqkv", 192 * 576), ("bqkv", 576 * 4),
        ("wproj", 192 * 192), ("bproj", 192 * 4),
        ("gamma1", 192), ("beta1", 192),
        ("w1", 192 * 768), ("b1", 768 * 4),
        ("w2", 768 * 192), ("b2", 192 * 4),
        ("gamma2", 192), ("beta2", 192),
    )
    for block in range(1, 13):
        for name, size in block_layout:
            alloc_w(f"b{block}_{name}", size)

    selector_layout = (
        ("local_w", 3 * 64 * 32), ("local_b", 3 * 32 * 4),
        ("score_w1", 3 * 64 * 32), ("score_b1", 3 * 32 * 4),
        ("score_w2", 3 * 32 * 16), ("score_b2", 3 * 16 * 4),
        ("score_w3", 3 * 16 * 2), ("score_b3", 3 * 2 * 4),
        ("hw_w1", 9), ("hw_b1", 12), ("hw_w2", 9), ("hw_b2", 12),
    )
    for sel in range(1, 4):
        for name, size in selector_layout:
            alloc_w(f"s{sel}_{name}", size)

    weight["final_gamma"] = alloc_w("final_gamma", 192)
    weight["final_beta"] = alloc_w("final_beta", 192)
    weight["head_w"] = alloc_w("head_w", 192 * 1000)
    weight["head_b"] = alloc_w("head_b", 1000 * 4)
    weight_bytes = _align8(wcur)

    return mm, scratch_bytes, weight_bytes


# ---------------------------------------------------------------------
# Sequence builders. Activation references use slot-relative offsets
# (0..ACT_SLOT-1); the scheduler rebases them at issue time.
# ---------------------------------------------------------------------
def patch_sequence(mm):
    s = mm["scratch"]
    w = mm["weight"]
    return [
        _desc(OP_PATCHIFY, _flag(FLAG_SRC0_INPUT), m=196, n=768,
              src0=0, dst=s["patch_matrix"],
              s0=SCALES["input"], sdst=SCALES["activation"]),
        _desc(OP_GEMM, _flag(FLAG_BIAS_ENABLE), m=196, n=192, k=768,
              src0=s["patch_matrix"], src1=w["patch_w"],
              bias=w["patch_b"], dst=s["patch_embed"],
              s0=SCALES["activation"], s1=SCALES["weight"],
              sdst=SCALES["activation"]),
        _desc(OP_COPY_ADD_POS, _flag(FLAG_AUX_WEIGHT), m=197, n=192,
              src0=s["patch_embed"], src1=w["pos"], aux=w["cls"],
              dst=0, s0=SCALES["activation"], s1=SCALES["cls_pos_beta"],
              saux=SCALES["cls_pos_beta"], sdst=SCALES["activation"]),
    ]


def block_sequence(block_index, mm, flag4):
    """13 descriptors; x/z reference the activation slots, flag4 on
    Residual2 marks the buffer switch."""
    s = mm["scratch"]
    w = mm["weight"]
    b = s["block"]
    prefix = f"b{block_index}"
    base_flags = _flag(FLAG_DYNAMIC_M)
    base4 = base_flags | _flag(FLAG_SWAP_ACTIVATION)
    return [
        _desc(OP_LAYERNORM, base_flags | _flag(FLAG_AUX_WEIGHT),
              m=99, n=192, src0=0, src1=w[f"{prefix}_gamma1"],
              aux=w[f"{prefix}_beta1"], dst=b["ln1"],
              s0=SCALES["activation"], s1=SCALES["gamma"],
              saux=SCALES["cls_pos_beta"], sdst=SCALES["activation"]),
        _desc(OP_GEMM, _flag(FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M),
              m=99, n=576, k=192, src0=b["ln1"],
              src1=w[f"{prefix}_wqkv"], bias=w[f"{prefix}_bqkv"],
              dst=b["fused"], s0=SCALES["activation"],
              s1=SCALES["weight"], sdst=SCALES["activation"]),
        _desc(OP_QKV_UNPACK, base_flags, m=99, n=576, heads=3,
              src0=b["fused"], dst=b["qkv"],
              s0=SCALES["activation"], sdst=SCALES["activation"]),
        _desc(OP_GEMM,
              _flag(FLAG_RHS_TRANSPOSE, FLAG_HEAD_MODE, FLAG_OUTPUT_INT32,
                    FLAG_DYNAMIC_M, FLAG_DYNAMIC_N, FLAG_SRC1_SCRATCH),
              m=99, n=99, k=64, heads=3, src0=b["qkv"],
              src1=b["qkv"], dst=b["score"],
              s0=SCALES["activation"], s1=SCALES["activation"],
              sdst=-17),
        _desc(OP_ATTN_SOFTMAX, _flag(FLAG_DYNAMIC_M, FLAG_DYNAMIC_N),
              m=99, n=99, heads=3, src0=b["score"], dst=b["prob"],
              s0=-17, sdst=SCALES["attn_uq0_8"]),
        _desc(OP_GEMM,
              _flag(FLAG_HEAD_MODE, FLAG_SRC0_UNSIGNED, FLAG_DYNAMIC_M,
                    FLAG_DYNAMIC_K, FLAG_SRC1_SCRATCH),
              m=99, n=64, k=99, heads=3, src0=b["prob"],
              src1=b["qkv"], dst=b["context"],
              s0=SCALES["attn_uq0_8"], s1=SCALES["activation"],
              sdst=SCALES["activation"]),
        _desc(OP_HEAD_CONCAT, base_flags, m=99, n=192, heads=3,
              src0=b["context"], dst=b["concat"],
              s0=SCALES["activation"], sdst=SCALES["activation"]),
        _desc(OP_GEMM, _flag(FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M),
              m=99, n=192, k=192, src0=b["concat"],
              src1=w[f"{prefix}_wproj"], bias=w[f"{prefix}_bproj"],
              dst=b["msa"], s0=SCALES["activation"],
              s1=SCALES["weight"], sdst=SCALES["activation"]),
        _desc(OP_RESIDUAL, base_flags, m=99, n=192,
              src0=0, aux=b["msa"], dst=b["y"],
              s0=SCALES["activation"], saux=SCALES["activation"],
              sdst=SCALES["activation"]),
        _desc(OP_LAYERNORM, base_flags | _flag(FLAG_AUX_WEIGHT),
              m=99, n=192, src0=b["y"], src1=w[f"{prefix}_gamma2"],
              aux=w[f"{prefix}_beta2"], dst=b["ln2"],
              s0=SCALES["activation"], s1=SCALES["gamma"],
              saux=SCALES["cls_pos_beta"], sdst=SCALES["activation"]),
        _desc(OP_GEMM,
              _flag(FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M) | (POST_GELU << 8),
              m=99, n=768, k=192, src0=b["ln2"],
              src1=w[f"{prefix}_w1"], bias=w[f"{prefix}_b1"],
              dst=b["hidden"], s0=SCALES["activation"],
              s1=SCALES["weight"], sdst=SCALES["activation"]),
        _desc(OP_GEMM, _flag(FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M),
              m=99, n=192, k=768, src0=b["hidden"],
              src1=w[f"{prefix}_w2"], bias=w[f"{prefix}_b2"],
              dst=b["ffn_out"], s0=SCALES["activation"],
              s1=SCALES["weight"], sdst=SCALES["activation"]),
        _desc(OP_RESIDUAL, base4 if flag4 else base_flags, m=99, n=192,
              src0=b["y"], aux=b["ffn_out"], dst=0,
              s0=SCALES["activation"], saux=SCALES["activation"],
              sdst=SCALES["activation"]),
    ]


def selector_sequence(selector_index, mm, flag4):
    """12 descriptors; the finalize writes the inactive activation slot."""
    s = mm["scratch"]
    w = mm["weight"]
    sel = s["selector"]
    prefix = f"s{selector_index}"
    dyn_c = _flag(FLAG_DYNAMIC_M)
    return [
        _desc(OP_GEMM,
              _flag(FLAG_HEAD_MODE, FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M,
                    FLAG_SRC0_CAND_MAJOR) | (POST_GELU << 8),
              m=196, n=32, k=64, heads=3, src0=192,
              src1=w[f"{prefix}_local_w"], bias=w[f"{prefix}_local_b"],
              dst=sel["local"], s0=SCALES["activation"],
              s1=SCALES["weight"], sdst=SCALES["activation"], param0=1),
        _desc(OP_REDUCE_MEAN, dyn_c, n=32, heads=3,
              src0=sel["local"], dst=sel["global"],
              s0=SCALES["activation"], sdst=SCALES["activation"], param0=1),
        _desc(OP_CONCAT_LOCAL_GLOBAL,
              _flag(FLAG_DYNAMIC_M, FLAG_SRC1_SCRATCH),
              n=64, heads=3, src0=sel["local"], src1=sel["global"],
              dst=sel["concat"], s0=SCALES["activation"],
              sdst=SCALES["activation"], param0=1),
        _desc(OP_GEMM,
              _flag(FLAG_HEAD_MODE, FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M) |
                  (POST_GELU << 8),
              m=196, n=32, k=64, heads=3, src0=sel["concat"],
              src1=w[f"{prefix}_score_w1"], bias=w[f"{prefix}_score_b1"],
              dst=sel["h1"], s0=SCALES["activation"],
              s1=SCALES["weight"], sdst=SCALES["activation"], param0=1),
        _desc(OP_GEMM,
              _flag(FLAG_HEAD_MODE, FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M) |
                  (POST_GELU << 8),
              m=196, n=16, k=32, heads=3, src0=sel["h1"],
              src1=w[f"{prefix}_score_w2"], bias=w[f"{prefix}_score_b2"],
              dst=sel["h2"], s0=SCALES["activation"],
              s1=SCALES["weight"], sdst=SCALES["activation"], param0=1),
        _desc(OP_GEMM,
              _flag(FLAG_HEAD_MODE, FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M),
              m=196, n=2, k=16, heads=3, src0=sel["h2"],
              src1=w[f"{prefix}_score_w3"], bias=w[f"{prefix}_score_b3"],
              dst=sel["logits"], s0=SCALES["activation"],
              s1=SCALES["weight"], sdst=SCALES["activation"], param0=1),
        _desc(OP_SELECTOR_SOFTMAX, dyn_c, n=2, heads=3,
              src0=sel["logits"], dst=sel["keep"],
              s0=SCALES["activation"], sdst=SCALES["selector_q0_16"],
              param0=1),
        _desc(OP_REDUCE_MEAN, dyn_c, n=64, heads=3,
              src0=192, dst=sel["stats"],
              s0=SCALES["activation"], sdst=SCALES["activation"], param0=5),
        _desc(OP_GEMM,
              _flag(FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M) | (POST_GELU << 8),
              m=196, n=3, k=3, src0=sel["stats"],
              src1=w[f"{prefix}_hw_w1"], bias=w[f"{prefix}_hw_b1"],
              dst=sel["hw_hidden"], s0=SCALES["activation"],
              s1=SCALES["weight"], sdst=SCALES["activation"], param0=1),
        _desc(OP_GEMM,
              _flag(FLAG_BIAS_ENABLE, FLAG_DYNAMIC_M) | (POST_PLAN << 8),
              m=196, n=3, k=3, src0=sel["hw_hidden"],
              src1=w[f"{prefix}_hw_w2"], bias=w[f"{prefix}_hw_b2"],
              dst=sel["hw"], s0=SCALES["activation"],
              s1=SCALES["weight"], sdst=SCALES["selector_q0_16"],
              param0=1),
        _desc(OP_HEAD_FUSE, _flag(FLAG_DYNAMIC_M, FLAG_SRC1_SCRATCH),
              n=3, heads=3, src0=sel["keep"], src1=sel["hw"],
              dst=sel["fused"], s0=SCALES["selector_q0_16"],
              s1=SCALES["selector_q0_16"], sdst=SCALES["selector_q0_16"],
              param0=1),
        _desc(OP_SELECTOR_FINALIZE,
              _flag(FLAG_DYNAMIC_M, FLAG_SRC1_SCRATCH) |
                  (_flag(FLAG_SWAP_ACTIVATION) if flag4 else 0),
              m=197, n=192, src0=0, src1=sel["fused"], dst=0,
              s0=SCALES["activation"], s1=SCALES["selector_q0_16"],
              sdst=SCALES["activation"], param0=0),
    ]


def final_layernorm_sequence(mm):
    s = mm["scratch"]
    w = mm["weight"]
    return [
        _desc(OP_LAYERNORM, _flag(FLAG_DYNAMIC_M, FLAG_AUX_WEIGHT),
              m=99, n=192, src0=0, src1=w["final_gamma"],
              aux=w["final_beta"], dst=s["final_ln"],
              s0=SCALES["activation"], s1=SCALES["gamma"],
              saux=SCALES["cls_pos_beta"], sdst=SCALES["activation"]),
    ]


def classifier_sequence(mm):
    s = mm["scratch"]
    w = mm["weight"]
    return [
        _desc(OP_GEMM, _flag(FLAG_BIAS_ENABLE, FLAG_OUTPUT_INT32,
                             FLAG_DST_OUTPUT),
              m=1, n=CLASSES, k=192, src0=s["final_ln"],
              src1=w["head_w"], bias=w["head_b"], dst=0,
              s0=SCALES["activation"], s1=SCALES["weight"],
              sdst=SCALES["logit"]),
    ]


def build_schedule(memory_map):
    descs = list(patch_sequence(memory_map))
    # Static ping-pong bookkeeping: the active buffer flips after every
    # Residual2 / Finalize (flag 4), which the scheduler mirrors at runtime.
    for block_index in range(1, 13):
        if block_index in (4, 7, 10):
            selector_index = {4: 1, 7: 2, 10: 3}[block_index]
            descs.extend(selector_sequence(selector_index, memory_map, True))
        descs.extend(block_sequence(block_index, memory_map, True))
    descs.extend(final_layernorm_sequence(memory_map))
    descs.extend(classifier_sequence(memory_map))
    descs.append(Descriptor.finish())
    if len(descs) != 198:
        raise ValueError(f"descriptor count {len(descs)} != 198")
    return descs


# ---------------------------------------------------------------------
# Emission.
# ---------------------------------------------------------------------
def emit(descs, mm, rom_path, listing_path, map_path):
    write_descriptors_mem(rom_path, descs)

    # Labels follow the fixed descriptor index map:
    # 0..2 patch, 3..41 blocks 1-3, 42..53 selector 1, 54..92 blocks 4-6,
    # 93..104 selector 2, 105..143 blocks 7-9, 144..155 selector 3,
    # 156..194 blocks 10-12, 195 final LN, 196 head, 197 finish.
    def block_of(index):
        if 3 <= index <= 41:
            return (index - 3) // 13 + 1
        if 54 <= index <= 92:
            return (index - 54) // 13 + 4
        if 105 <= index <= 143:
            return (index - 105) // 13 + 7
        if 156 <= index <= 194:
            return (index - 156) // 13 + 10
        return 0

    def block_pos_of(index):
        return (index - 3) % 13 + 1

    def selector_of(index):
        if 42 <= index <= 53:
            return 1
        if 93 <= index <= 104:
            return 2
        if 144 <= index <= 155:
            return 3
        return 0

    def selector_pos_of(index):
        return (index - 42) % 12 + 1

    labels = []
    for idx, desc in enumerate(descs):
        if idx <= 2:
            names = ("patch_patchify", "patch_gemm", "patch_copy_add_pos")
            labels.append(names[idx])
        elif idx == 195:
            labels.append("final_layernorm")
        elif idx == 196:
            labels.append("head_classifier")
        elif idx == 197:
            labels.append("finish")
        elif selector_of(idx):
            labels.append(f"selector_{selector_of(idx):02d}_op"
                          f"{selector_pos_of(idx)}")
        else:
            labels.append(f"block_{block_of(idx):02d}_op"
                          f"{block_pos_of(idx)}")

    with open(listing_path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("index,label,opcode,m,n,k,heads,flags,src0,src1,bias,"
                     "aux,dst,s0,s1,saux,sdst,param0\n")
        for idx, (desc, label) in enumerate(zip(descs, labels)):
            handle.write(
                f"{idx},{label},{desc.opcode},{desc.m},{desc.n},{desc.k},"
                f"{desc.heads},0x{desc.flags:06x},{desc.src0_offset},"
                f"{desc.src1_offset},{desc.bias_offset},{desc.aux_offset},"
                f"{desc.dst_offset},{desc.src0_scale_exp},"
                f"{desc.src1_scale_exp},{desc.aux_scale_exp},"
                f"{desc.dst_scale_exp},{desc.param0}\n")

    with open(map_path, "w", newline="\n", encoding="utf-8") as handle:
        json.dump(mm, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default="config/heatvit_t.json")
    parser.add_argument("--rom", type=Path,
                        default="rtl/generated/heatvit_descriptors.mem")
    parser.add_argument("--listing", type=Path,
                        default="build/vectors/e2e/descriptor_listing.csv")
    parser.add_argument("--map", type=Path,
                        default="build/vectors/e2e/memory_map.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    assert config["tokens"] == 197 and config["blocks"] == 12

    mm, scratch_bytes, weight_bytes = build_memory_map()
    descs = build_schedule(mm)
    for desc in descs:
        desc.validate()

    args.rom.parent.mkdir(parents=True, exist_ok=True)
    args.listing.parent.mkdir(parents=True, exist_ok=True)
    args.map.parent.mkdir(parents=True, exist_ok=True)
    emit(descs, mm, args.rom, args.listing, args.map)
    print(f"wrote {len(descs)} descriptors: rom={args.rom} "
          f"listing={args.listing} map={args.map}")
    print(f"scratch_bytes={scratch_bytes} weight_bytes={weight_bytes}")


if __name__ == "__main__":
    main()
