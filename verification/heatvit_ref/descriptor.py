"""Packed 320-bit descriptor matching ``heatvit_pkg::heatvit_desc_t``.

The packed bit order is explicit: ``reserved`` occupies the lowest 4 bits and
``opcode`` the highest 8 bits, mirroring the SystemVerilog packed struct. No
Python ``struct`` alignment or byte order is involved.
"""

from dataclasses import dataclass


OP_NOP = 0
OP_PATCHIFY = 1
OP_COPY_ADD_POS = 2
OP_GEMM = 3
OP_LAYERNORM = 4
OP_RESIDUAL = 5
OP_QKV_UNPACK = 6
OP_HEAD_CONCAT = 7
OP_ATTN_SOFTMAX = 8
OP_SELECTOR_SOFTMAX = 9
OP_REDUCE_MEAN = 10
OP_CONCAT_LOCAL_GLOBAL = 11
OP_HEAD_FUSE = 12
OP_SELECTOR_FINALIZE = 13
OP_FINISH = 14

FLAG_RHS_TRANSPOSE = 0
FLAG_BIAS_ENABLE = 1
FLAG_AUX_ENABLE = 2
FLAG_DYNAMIC_M = 3
FLAG_SWAP_ACTIVATION = 4
FLAG_HEAD_MODE = 5
FLAG_HEAD_CONCAT = 6
FLAG_OUTPUT_INT32 = 7
FLAG_SRC0_INPUT = 11
FLAG_SRC1_SCRATCH = 12
FLAG_BIAS_SCRATCH = 13
FLAG_AUX_WEIGHT = 14
FLAG_DST_OUTPUT = 15
FLAG_TOKEN_TAIL = 16
FLAG_CHANNEL_TAIL = 17
FLAG_SRC0_UNSIGNED = 18
FLAG_DYNAMIC_N = 19
FLAG_DYNAMIC_K = 20
FLAG_SRC0_CAND_MAJOR = 21


def _check_int(name, value):
    if isinstance(value, float):
        raise TypeError(f"{name} must be an integer, got float")


def _signed6(value):
    _check_int("scale_exp", value)
    value = int(value)
    if value < -32 or value > 31:
        raise ValueError("scale_exp outside [-32, 31]")
    return value & 0x3F


def _from_signed6(bits):
    return bits - 64 if bits >= 32 else bits


@dataclass(frozen=True)
class Descriptor:
    opcode: int = 0
    flags: int = 0
    m: int = 0
    n: int = 0
    k: int = 0
    heads: int = 0
    src0_offset: int = 0
    src1_offset: int = 0
    bias_offset: int = 0
    aux_offset: int = 0
    dst_offset: int = 0
    src0_scale_exp: int = 0
    src1_scale_exp: int = 0
    aux_scale_exp: int = 0
    dst_scale_exp: int = 0
    next_index: int = 0
    param0: int = 0
    param1: int = 0
    reserved: int = 0

    def pack(self):
        for name, value, bits in (
            ("opcode", self.opcode, 8),
            ("flags", self.flags, 24),
            ("m", self.m, 16),
            ("n", self.n, 16),
            ("k", self.k, 16),
            ("heads", self.heads, 4),
            ("src0_offset", self.src0_offset, 32),
            ("src1_offset", self.src1_offset, 32),
            ("bias_offset", self.bias_offset, 32),
            ("aux_offset", self.aux_offset, 32),
            ("dst_offset", self.dst_offset, 32),
            ("next_index", self.next_index, 16),
            ("param0", self.param0, 16),
            ("param1", self.param1, 16),
            ("reserved", self.reserved, 4),
        ):
            _check_int(name, value)
            if value < 0 or value >= (1 << bits):
                raise ValueError(f"{name} outside {bits}-bit range")

        word = 0
        word |= self.reserved & 0xF
        word |= (self.param1 & 0xFFFF) << 4
        word |= (self.param0 & 0xFFFF) << 20
        word |= (self.next_index & 0xFFFF) << 36
        word |= _signed6(self.dst_scale_exp) << 52
        word |= _signed6(self.aux_scale_exp) << 58
        word |= _signed6(self.src1_scale_exp) << 64
        word |= _signed6(self.src0_scale_exp) << 70
        word |= (self.dst_offset & 0xFFFFFFFF) << 76
        word |= (self.aux_offset & 0xFFFFFFFF) << 108
        word |= (self.bias_offset & 0xFFFFFFFF) << 140
        word |= (self.src1_offset & 0xFFFFFFFF) << 172
        word |= (self.src0_offset & 0xFFFFFFFF) << 204
        word |= (self.heads & 0xF) << 236
        word |= (self.k & 0xFFFF) << 240
        word |= (self.n & 0xFFFF) << 256
        word |= (self.m & 0xFFFF) << 272
        word |= (self.flags & 0xFFFFFF) << 288
        word |= (self.opcode & 0xFF) << 312
        return word

    @classmethod
    def unpack(cls, word):
        _check_int("word", word)
        word = int(word)
        if word < 0 or word >= 1 << 320:
            raise ValueError("word outside 320-bit range")

        def bits(msb, lsb):
            return (word >> lsb) & ((1 << (msb - lsb + 1)) - 1)

        return cls(
            opcode=bits(319, 312),
            flags=bits(311, 288),
            m=bits(287, 272),
            n=bits(271, 256),
            k=bits(255, 240),
            heads=bits(239, 236),
            src0_offset=bits(235, 204),
            src1_offset=bits(203, 172),
            bias_offset=bits(171, 140),
            aux_offset=bits(139, 108),
            dst_offset=bits(107, 76),
            src0_scale_exp=_from_signed6(bits(75, 70)),
            src1_scale_exp=_from_signed6(bits(69, 64)),
            aux_scale_exp=_from_signed6(bits(63, 58)),
            dst_scale_exp=_from_signed6(bits(57, 52)),
            next_index=bits(51, 36),
            param0=bits(35, 20),
            param1=bits(19, 4),
            reserved=bits(3, 0),
        )

    @staticmethod
    def finish():
        return Descriptor(opcode=OP_FINISH)

    def validate(self):
        """Schema validation for the compiled Phase 5 schedule.

        Raises ValueError on: unknown opcode, non-zero reserved, scale
        exponents outside [-32, 31], flag 18 outside the Attention*V GEMM
        pattern, flag 3 on non-dynamic opcodes, zero-dimension GEMMs and
        any tensor offset that is not 8-byte aligned.
        """
        known = {OP_NOP, OP_PATCHIFY, OP_COPY_ADD_POS, OP_GEMM,
                 OP_LAYERNORM, OP_RESIDUAL, OP_QKV_UNPACK, OP_HEAD_CONCAT,
                 OP_ATTN_SOFTMAX, OP_SELECTOR_SOFTMAX, OP_REDUCE_MEAN,
                 OP_CONCAT_LOCAL_GLOBAL, OP_HEAD_FUSE,
                 OP_SELECTOR_FINALIZE, OP_FINISH}
        if self.opcode not in known:
            raise ValueError(f"unknown opcode {self.opcode}")
        if self.reserved != 0:
            raise ValueError("reserved must be zero")
        for name in ("src0_scale_exp", "src1_scale_exp", "aux_scale_exp",
                     "dst_scale_exp"):
            value = getattr(self, name)
            if value < -32 or value > 31:
                raise ValueError(f"{name} outside [-32, 31]: {value}")

        # Flag 18 (unsigned src0) is only legal for the Attention*V GEMM:
        # head mode with three heads and no post-op.
        if self.flags & (1 << FLAG_SRC0_UNSIGNED):
            if not (self.opcode == OP_GEMM and
                    self.flags & (1 << FLAG_HEAD_MODE) and
                    self.heads == 3 and (self.flags >> 8) & 0x7 == 0):
                raise ValueError("flag 18 only legal on Attention*V GEMM")

        # Flag 3 (dynamic M) is only legal on the dynamic opcodes.
        dynamic_ops = {OP_GEMM, OP_LAYERNORM, OP_RESIDUAL, OP_QKV_UNPACK,
                       OP_HEAD_CONCAT, OP_ATTN_SOFTMAX,
                       OP_SELECTOR_SOFTMAX, OP_REDUCE_MEAN,
                       OP_CONCAT_LOCAL_GLOBAL, OP_HEAD_FUSE,
                       OP_SELECTOR_FINALIZE}
        if self.flags & (1 << FLAG_DYNAMIC_M):
            if self.opcode not in dynamic_ops:
                raise ValueError(f"flag 3 not legal on opcode {self.opcode}")
            if (self.param0 & 0x3) == 2 or (self.param0 & 0x3) == 3:
                raise ValueError("dynamic-M param0[1:0] must be 00 or 01")

        if self.opcode == OP_GEMM:
            if self.m == 0 or self.n == 0 or self.k == 0:
                raise ValueError("GEMM dimensions must be non-zero")

        for name in ("src0_offset", "src1_offset", "bias_offset",
                     "aux_offset", "dst_offset"):
            if getattr(self, name) & 0x7:
                raise ValueError(f"{name} must be 8-byte aligned")

        return self


def format_desc_hex(word):
    """Format a 320-bit word as exactly 80 most-significant-first hex chars."""
    _check_int("word", word)
    word = int(word)
    if word < 0 or word >= 1 << 320:
        raise ValueError("word outside 320-bit range")
    return f"{word:080x}"
