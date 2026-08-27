#!/usr/bin/env python3
"""P3 QAT: straight-through fake-quantization primitives (value space).

The RTL datapath is pure integer; the QAT training path (decision D1,
option A) instead carries dequantized float values and reproduces every
quantization boundary of the contract with a fake-quant step:

    out = round(clip(x / 2**exp, lo, hi)) * 2**exp,   backward = identity

Grids (docs/heatvit.md Part 1 Section 9):

    fake_quant_int8    signed int8    [-128, 127]        * 2**exp
    fake_quant_int32   signed int32   [-2^31, 2^31 - 1]  * 2**exp
    fake_quant_q816    signed Q8.16   [-2^23, 2^23 - 1]  * 2**-16
    fake_quant_uq08    unsigned UQ0.8 [0, 255]           * 2**-8

Rounding uses torch.round (ties-to-even). The integer contract rounds
ties away from zero; ties have measure zero on float data (and never
occur for dyadic inputs), so the choice does not affect training
dynamics. Bit-exact evaluation always goes through the integer simulators
(tools/p2/p2_sim.py / p2_sim_ivit.py), never through this module.

``shiftgelu_float`` is the faithful float mirror of the RTL ShiftGELU-ln2
(rtl/common/heatvit_gelu.sv): it reproduces the shift-exp core's sign
branch, 11/16 fractional slope and q clamps in float, so it tracks the
integer contract GELU to the level of the dropped Q16 roundings
(mean |err| ~2e-5 over the Q8.16 grid, see test_qat.py).
"""

import torch

__all__ = [
    "fake_quant",
    "fake_quant_int8",
    "fake_quant_int32",
    "fake_quant_q816",
    "fake_quant_uq08",
    "shiftgelu_float",
]

# RTL ShiftGELU shift sequence (heatvit_gelu.sv):
#   i_p  = x + (x>>1) + (x>>3) + (x>>4)   -> 1.6875 = (1.1011)b
#   i_p2 = i_p + (i_p>>1) - (i_p>>4)      -> *1.4375 = (1.0111)b
#   q = |i_p2| >> 16,  r = |i_p2| & 0xFFFF
#   frac = (r*11 + 15) >> 4               -> r * 11/16 (shift-exp slope)
#   e = i_p2 < 0 ? 2^-q * (1 - frac/2^16) [q <= 16 else 0]
#                : 2^+q * (1 + frac/2^16) [q <= 7 else 2^23-1]
#   gelu = x * sigmoid-style e/(1 + e)
GELU_IP_LOG2E = (1.0 + 1.0 / 2.0 + 1.0 / 8.0 + 1.0 / 16.0) \
    * (1.0 + 1.0 / 2.0 - 1.0 / 16.0)      # 2.42578x
_GELU_SLOPE = 11.0 / 16.0
_GELU_SAT = float((1 << 23) - 1) / 65536.0   # 2^23-1 in Q16 value


def fake_quant(x: torch.Tensor, exp: int, lo: float, hi: float) \
        -> torch.Tensor:
    """Round x to the grid 2**exp and clip to [lo, hi] * 2**exp; STE.

    The quantization arithmetic runs under ``torch.no_grad()``: the
    autograd graph only sees ``x + constant`` (identity backward), and
    none of the round/clamp/mul intermediates are retained for the
    backward pass. This is one half of the memory fix (the other half is
    block-level gradient checkpointing in qat_model): the eager QAT
    graph otherwise retains ~110 MiB/image and spills past the 8 GiB
    VRAM at batch 128, making the backward pass PCIe-bound
    (measured 20-90 s/step before, ~0.9 s/step after).
    """
    scale = 2.0 ** exp
    with torch.no_grad():
        q = torch.clamp(torch.round(x / scale), lo, hi)
        xq = q * scale
    return x + (xq - x).detach()


def fake_quant_int8(x: torch.Tensor, exp: int) -> torch.Tensor:
    """Signed int8 boundary: round/clip to [-128, 127] * 2**exp."""
    return fake_quant(x, exp, -128.0, 127.0)


def fake_quant_int32(x: torch.Tensor, exp: int) -> torch.Tensor:
    """Signed int32 boundary (bias): [-2^31, 2^31 - 1] * 2**exp."""
    return fake_quant(x, exp, float(-(1 << 31)), float((1 << 31) - 1))


def fake_quant_q816(x: torch.Tensor) -> torch.Tensor:
    """Signed 24-bit Q8.16: value = int * 2**-16, int in [-2^23, 2^23 - 1]."""
    return fake_quant(x, -16, float(-(1 << 23)), float((1 << 23) - 1))


def fake_quant_uq08(x: torch.Tensor) -> torch.Tensor:
    """Unsigned UQ0.8: value = int * 2**-8, int in [0, 255]."""
    return fake_quant(x, -8, 0.0, 255.0)


def shiftgelu_float(x: torch.Tensor) -> torch.Tensor:
    """Float mirror of the RTL ShiftGELU-ln2 shift-exp core.

    Faithfully mirrors the RTL algebra (rtl/common/heatvit_gelu.sv),
    including the sign branch, the 11/16 linear slope of the shift-exp
    core and the q clamps (negative underflow at q > 16, positive
    saturation at 2^23-1). Only the Q16-level integer roundings of the
    intermediate steps are dropped (errors ~1e-3 relative).

    The backward pass flows through the fractional term ``r`` only;
    ``q = floor(|u|)`` carries no gradient (a.e. zero, standard STE).
    """
    u = GELU_IP_LOG2E * x
    neg = u < 0
    a = u.abs()
    q = torch.floor(a)
    r = a - q
    frac = r * _GELU_SLOPE
    i_b = torch.where(neg, 1.0 - frac, 1.0 + frac)
    q_c = torch.where(neg, q.clamp(max=16.0), q.clamp(max=7.0))
    shift = torch.where(neg, -q_c, q_c)          # i_b >> q  vs  i_b << q
    e = i_b * (2.0 ** shift)
    e = torch.where(neg & (q > 16.0), torch.zeros_like(e), e)
    e = e.clamp(max=_GELU_SAT)
    return x * (e / (1.0 + e))
