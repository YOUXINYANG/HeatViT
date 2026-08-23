#!/usr/bin/env python3
"""I-ViT variant simulator for HeatViT PTQ accuracy experiments.

Implements the I-ViT paper's integer-only nonlinearities
(arXiv:2207.01405) on top of the contract-faithful P2 simulator
(tools/p2/p2_sim.py), selectable per component:

  softmax : ``contract``  = golden quadratic exp-approx softmax
            ``shiftmax``  = I-ViT Shiftmax (Alg. 1): shift-only exp with a
                            linear 2^x approximation over the fractional
                            part, then one integer division (IntDiv k=9).
  gelu    : ``contract``  = RTL contract (I-ViT ShiftGELU, ln2 slope; the
                            retired erf polynomial is available as
                            ``poly`` for ablation reference)
            ``shiftgelu`` = I-ViT ShiftGELU (Alg. 2): x * sigmoid(1.702x)
                            through the same shift-exp core.
  ln      : ``contract``  = golden two-pass LN, input scale exp in [-32, 0]
            ``relaxed``   = same algorithm, input scale exp up to
                            ``ln_max_exp`` (I-ViT style min-max calibration
                            philosophy: the residual stream is not clipped)
            ``newton``    = relaxed range + I-LayerNorm fixed 10-iteration
                            Newton integer sqrt instead of restoring isqrt.
  wq      : ``tensor``    = per-tensor power-of-2 weight exponents
            ``channel``   = per-output-channel weight exponents for GEMM
                            weights (dyadic per-channel requantization;
                            a contract-extension candidate, not strictly
                            an I-ViT component).

``shift_slope`` selects the linear 2^x slope used by Shiftmax/ShiftGELU:
``half`` is the paper's slope 1/2 (pure shift), ``ln2`` uses 11/16 (~ln 2,
one small multiply, about 6x smaller approximation error).

With all components set to ``contract``/``tensor`` this module reproduces
tools/p2/p2_sim.py bit-exactly (checked by --unit), so it can serve as the
single forward path for both the baseline and the I-ViT ablations.

The simulator performs no float arithmetic on weights or activations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from tools.p2.p2_sim import (
    BlockP,
    D,
    DELTA2_ATTENTION,
    FFN_DIM,
    HEADS,
    HEAD_DIM,
    KEEP_THRESHOLD,
    LN_D,
    LN_EPS_Q32,
    LOGIT_SCALE_EXP,
    QuantDeiT,
    SelectorP,
    SELECTOR_BLOCKS,
    ScaleTable,
    gemm_int8,
    gelu_q16,
    layernorm,
    plan_sigmoid,
    requant,
    residual_add,
    round_div,
    round_shift_away,
    rsa_sq32,
    sat,
    softmax_attention,
    softmax_selector,
    token_selector,
)

# ---- I-ViT constants ------------------------------------------------------
SHIFT_I0 = 65536          # 1/S for the Q16 logit scale (paper Alg. 1)
SHIFT_EXP_QUARTERS = 16   # underflow cutoff: 2^-16 relative -> prob 0


@dataclass
class NonlinConfig:
    """Selects the nonlinearity/quantization variants for one ablation."""
    softmax: str = "contract"     # contract | shiftmax
    shift_slope: str = "half"     # half | ln2 (Shiftmax / ShiftGELU)
    gelu: str = "contract"        # contract(=shiftgelu-ln2) | shiftgelu | plan | poly
    ln: str = "contract"          # contract | relaxed | newton
    ln_max_exp: int = 6           # upper bound for relaxed/newton LN inputs
    wq: str = "tensor"            # tensor | channel

    def validate(self):
        assert self.softmax in ("contract", "shiftmax"), self.softmax
        assert self.shift_slope in ("half", "ln2"), self.shift_slope
        assert self.gelu in ("contract", "shiftgelu", "plan", "poly"), \
            self.gelu
        assert self.ln in ("contract", "relaxed", "newton"), self.ln
        assert 0 <= self.ln_max_exp <= 6, self.ln_max_exp
        assert self.wq in ("tensor", "channel"), self.wq


# ---- I-ViT ShiftExp core --------------------------------------------------
def _shift_frac(r: torch.Tensor, slope: str) -> torch.Tensor:
    """Linear approx of 2^(r * 2^-16) - 1 for r in [0, 2^16): ``frac``
    with 2^(r*2^-16) ~= 1 + frac/2^16. Paper slope 1/2, or 11/16 ~= ln2."""
    if slope == "half":
        return (r + 1) >> 1          # ceil(r/2)  -> slope 1/2
    return (r * 11 + 15) >> 4        # round(r*11/16) -> slope 11/16


def shift_exp(n: torch.Tensor, slope: str) -> torch.Tensor:
    """2^(n * 2^-16) in Q16 for arbitrary signed int64 ``n`` (sat 24)."""
    neg = n < 0
    a = n.abs()
    q = a // SHIFT_I0
    r = a - q * SHIFT_I0
    frac = _shift_frac(r, slope)
    i_b_neg = (SHIFT_I0 - frac).clamp(min=1)
    e_neg = torch.where(q <= SHIFT_EXP_QUARTERS, i_b_neg >> q,
                        torch.zeros_like(i_b_neg))
    i_b_pos = SHIFT_I0 + frac
    e_pos = torch.where(q <= 7, i_b_pos << q,
                        torch.full_like(i_b_pos, 1 << 23))
    return torch.where(neg, e_neg, e_pos).clamp(0, (1 << 23) - 1)


def shiftmax_attention(score_q16: torch.Tensor, slope: str = "half") \
        -> torch.Tensor:
    """I-ViT Shiftmax (paper Alg. 1), Q8.16 rows -> UQ0.8 probabilities.

    exp is replaced by: Ip = Id + (Id>>1) - (Id>>4) (x log2 e), then
    2^(Ip*2^-16) = 2^(-q) * 2^(-r*2^-16) with the fractional 2^x linearly
    approximated, and IntDiv(k=9) normalizes the row. The only non-shift
    arithmetic is one max-subtraction, one summation and one division.
    """
    m = score_q16.max(dim=-1, keepdim=True).values
    i_d = score_q16 - m                      # <= 0
    i_p = i_d + (i_d >> 1) - (i_d >> 4)      # * log2(e) ~ 1.4375
    q = (-i_p) // SHIFT_I0
    r = (-i_p) - q * SHIFT_I0                # [0, 2^16)
    frac = _shift_frac(r, slope)
    i_b = (SHIFT_I0 - frac).clamp(min=1)
    e = torch.where(q <= SHIFT_EXP_QUARTERS, i_b >> q,
                    torch.zeros_like(i_b))
    s = e.sum(dim=-1, keepdim=True).clamp(min=1)
    recip = round_div(torch.tensor(1 << 32, dtype=torch.int64,
                                   device=e.device), s)
    ratio = round_shift_away(e * recip, 16)
    return round_shift_away(ratio, 8).clamp(0, 255).to(torch.uint8)


def shiftgelu_q16(x_q16: torch.Tensor, slope: str = "half") -> torch.Tensor:
    """I-ViT ShiftGELU, Q8.16 -> Q8.16 sat 24 bits.

    GELU(x) ~= x * sigmoid(1.702x) with 1.702 ~= (1.1011)b so the argument
    is x + (x>>1) + (x>>3) + (x>>4). The sigmoid is evaluated through the
    same shift-exp core as Shiftmax. Deviation from paper Alg. 2: instead
    of the global-max normalization (two ShiftExps, which underflows for
    PTQ-scale pre-activations when both numerator and denominator are far
    below 1), the mathematically identical form sigmoid(z) = 1/(1+e^-z)
    is used (one ShiftExp, one division) — same shift-based arithmetic,
    robust for any input range.
    """
    i_p = x_q16 + (x_q16 >> 1) + (x_q16 >> 3) + (x_q16 >> 4)  # 1.702x
    i_p2 = i_p + (i_p >> 1) - (i_p >> 4)                       # * log2 e
    e = shift_exp(i_p2, slope)          # e^{1.702x} in Q16, sat 24 bits
    one = 65536
    den = (one + e).clamp(min=1)
    sig = (((e.to(torch.int64) << 16) + (den >> 1)) // den).clamp(0, 65536)
    return sat(round_shift_away(x_q16.to(torch.int64) * sig, 16), 24)


# ---- I-LayerNorm variants -------------------------------------------------
def mul_rsa48_wide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact round_shift_away(a*b, 48) for |a| < 2^46, 0 <= b < 2^48.

    The relaxed LN input contract (x_exp up to +6) makes |x_q32 - mean|
    reach 2^40, beyond the plain mul_rsa48 split (|a| < 2^33). Both ``a``
    and ``b`` are split into 16-bit words; the 94-bit product is
    accumulated word-wise with carries, then rounded once at bit 48.
    Identical to mul_rsa48 on their shared input range.
    """
    sign = torch.where(a < 0, -1, 1)
    x = a.abs()
    a0 = x & 0xFFFF
    a1 = (x >> 16) & 0xFFFF
    a2 = (x >> 32) & 0x3FFF            # x < 2^46 -> a2 < 2^14
    b0 = b & 0xFFFF
    b1 = (b >> 16) & 0xFFFF
    b2 = (b >> 32) & 0xFFFF
    t0 = a0 * b0                       # 2^0
    t1 = a1 * b0 + a0 * b1             # 2^16
    t2 = a2 * b0 + a1 * b1 + a0 * b2   # 2^32
    t3 = a2 * b1 + a1 * b2             # 2^48
    t4 = a2 * b2                       # 2^64
    w0 = t0 & 0xFFFF
    c1 = t0 >> 16
    s1 = t1 + c1
    w1 = s1 & 0xFFFF
    c2 = s1 >> 16
    s2 = t2 + c2
    w2 = s2 & 0xFFFF
    c3 = s2 >> 16
    s3 = t3 + c3
    w3 = s3 & 0xFFFF
    c4 = s3 >> 16
    s4 = t4 + c4
    w4 = s4 & 0xFFFF
    c5 = s4 >> 16
    lo_part = (w2 << 32) + (w1 << 16) + w0
    hi_part = (((c5 << 16) + w4) << 16) + w3
    q = hi_part + ((lo_part + (1 << 47)) >> 48)
    return sign * q


def layernorm_newton(xs: torch.Tensor, gammas: torch.Tensor,
                     betas: torch.Tensor, x_exp: int, g_exp: int,
                     b_exp: int, out_exp: int,
                     ln_max_exp: int) -> torch.Tensor:
    """I-LayerNorm style: relaxed input scale + fixed 10-iteration Newton
    integer sqrt (paper Eq. 20) replacing the restoring isqrt. The two-pass
    mean/variance accumulation and the final affine/requant are identical
    to the golden contract, so all differences are attributable to the
    sqrt engine (expected to be <= 1 LSB) and the wider input range."""
    if not (-32 <= x_exp <= ln_max_exp):
        raise ValueError(f"layernorm input scale out of [-32, {ln_max_exp}]")
    x64 = xs.to(torch.int64)
    x_q32 = x64 << (x_exp + 32)
    sum_x = x_q32.sum(dim=-1, keepdim=True)
    sum_sq = round_shift_away(x64 * x64, -(2 * x_exp + 32)).sum(
        dim=-1, keepdim=True)
    d = torch.tensor(LN_D, dtype=torch.int64, device=x64.device)
    mean = round_div(sum_x, d)
    e2 = round_div(sum_sq, d)
    mean_sq = rsa_sq32(mean)
    variance = (e2 - mean_sq).clamp(min=0)
    v = variance + LN_EPS_Q32
    # Newton: I_{i+1} = (I_i + floor(v / I_i)) >> 1, fixed 10 iterations
    # (paper Eq. 20), I0 = 2^floor(bit_length(v)/2).
    length = torch.log2(v.clamp(min=1).to(torch.float64)).floor() \
        .to(torch.int64) + 1
    i_n = torch.pow(2, (length >> 1).to(torch.float64)).round().to(
        torch.int64)
    for _ in range(10):
        i_n = (i_n + v // i_n.clamp(min=1)) >> 1
    std_q16 = i_n
    inv_std = round_div(torch.tensor(1 << 48, dtype=torch.int64,
                                     device=x64.device), std_q16)
    norm_q16 = sat(mul_rsa48_wide(x_q32 - mean, inv_std), 24)
    common = min(g_exp - 16, b_exp)
    if g_exp - 16 - common > 24 or b_exp - common > 24:
        raise ValueError("layernorm affine scale gap exceeds sim int64 "
                         "safety bound (24 bits)")
    aligned = ((norm_q16 * gammas.to(torch.int64))
               << (g_exp - 16 - common)) \
        + (betas.to(torch.int64) << (b_exp - common))
    return sat(round_shift_away(aligned, out_exp - common), 8).to(torch.int8)


def layernorm_relaxed(xs: torch.Tensor, gammas: torch.Tensor,
                      betas: torch.Tensor, x_exp: int, g_exp: int,
                      b_exp: int, out_exp: int,
                      ln_max_exp: int) -> torch.Tensor:
    """Golden two-pass LN with the input scale contract widened from
    [-32, 0] to [-32, ln_max_exp]; the algorithm is otherwise untouched
    (restoring isqrt + round_div)."""
    if not (-32 <= x_exp <= ln_max_exp):
        raise ValueError(f"layernorm input scale out of [-32, {ln_max_exp}]")
    x64 = xs.to(torch.int64)
    x_q32 = x64 << (x_exp + 32)
    sum_x = x_q32.sum(dim=-1, keepdim=True)
    sum_sq = round_shift_away(x64 * x64, -(2 * x_exp + 32)).sum(
        dim=-1, keepdim=True)
    d = torch.tensor(LN_D, dtype=torch.int64, device=x64.device)
    mean = round_div(sum_x, d)
    e2 = round_div(sum_sq, d)
    mean_sq = rsa_sq32(mean)
    variance = (e2 - mean_sq).clamp(min=0)
    std_q16 = torch.sqrt((variance + LN_EPS_Q32).to(torch.float64)).floor() \
        .to(torch.int64)
    inv_std = round_div(torch.tensor(1 << 48, dtype=torch.int64,
                                     device=x64.device), std_q16)
    norm_q16 = sat(mul_rsa48_wide(x_q32 - mean, inv_std), 24)
    common = min(g_exp - 16, b_exp)
    if g_exp - 16 - common > 24 or b_exp - common > 24:
        raise ValueError("layernorm affine scale gap exceeds sim int64 "
                         "safety bound (24 bits)")
    aligned = ((norm_q16 * gammas.to(torch.int64))
               << (g_exp - 16 - common)) \
        + (betas.to(torch.int64) << (b_exp - common))
    return sat(round_shift_away(aligned, out_exp - common), 8).to(torch.int8)


def ln_cfg(xs, gammas, betas, x_exp, g_exp, b_exp, out_exp, cfg):
    if cfg.ln == "contract":
        return layernorm(xs, gammas, betas, x_exp, g_exp, b_exp, out_exp)
    if cfg.ln == "newton":
        return layernorm_newton(xs, gammas, betas, x_exp, g_exp, b_exp,
                                out_exp, cfg.ln_max_exp)
    return layernorm_relaxed(xs, gammas, betas, x_exp, g_exp, b_exp,
                             out_exp, cfg.ln_max_exp)


# ---- legacy erf-polynomial GELU (pre-I-ViT contract, kept for reference) --
GELU_A_Q16 = -18927
GELU_B_Q16 = -115933
GELU_DELTA_Q16 = 32768
INV_SQRT2_Q16 = 46341


def legacy_gelu_q16(x_q16: torch.Tensor) -> torch.Tensor:
    """The retired HeatViT-paper erf polynomial (delta1 = 0.5). Kept only
    as the ``poly`` ablation reference; the RTL contract now uses the
    ShiftGELU shift-exp core (see tools/p2/p2_sim.gelu_q16)."""
    u = round_shift_away(x_q16.to(torch.int64) * INV_SQRT2_Q16, 16)
    clip = u.abs().clamp(max=-GELU_B_Q16)
    t = clip + GELU_B_Q16
    t2 = round_shift_away(t * t, 16)
    poly = round_shift_away(GELU_A_Q16 * t2, 16) + 65536
    erf_mag = round_shift_away(GELU_DELTA_Q16 * poly, 16)
    sign = (u > 0).to(torch.int64) - (u < 0).to(torch.int64)
    l_erf = sign * erf_mag
    y = round_shift_away(x_q16.to(torch.int64) * (65536 + l_erf), 17)
    return sat(y, 24)


def plangelu_q16(x_q16: torch.Tensor) -> torch.Tensor:
    """Division-free integer GELU: x * PLAN_sigmoid(1.702x), Q8.16 -> Q8.16.

    Same identity as ShiftGELU but the sigmoid uses the existing PLAN
    piecewise-linear unit (shift-only, no divider) instead of the
    shift-exp core + integer division. 1.702 ~= (1.1011)b is built from
    three shift-adds. Candidate for the RTL replacement of the golden
    poly GELU: reuses heatvit_plan_sigmoid verbatim, one multiply.
    """
    i_p = x_q16 + (x_q16 >> 1) + (x_q16 >> 3) + (x_q16 >> 4)
    sig = plan_sigmoid(i_p)                      # Q0.16 unsigned
    y = round_shift_away(x_q16.to(torch.int64) * sig.to(torch.int64), 16)
    return sat(y, 24)


def gelu_cfg(x_q16, cfg):
    if cfg.gelu == "shiftgelu":
        return shiftgelu_q16(x_q16, cfg.shift_slope)
    if cfg.gelu == "plan":
        return plangelu_q16(x_q16)
    if cfg.gelu == "poly":
        return legacy_gelu_q16(x_q16)
    return gelu_q16(x_q16)   # contract = ShiftGELU-ln2 (p2_sim)


def softmax_cfg(score_q16, cfg):
    return shiftmax_attention(score_q16, cfg.shift_slope) \
        if cfg.softmax == "shiftmax" else softmax_attention(score_q16)


# ---- per-channel weight requantization -----------------------------------
def gemm_int8_chan(a: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor],
                   a_exp: int, w_exp_chan: torch.Tensor, dst_exp: int,
                   bits: int) -> torch.Tensor:
    """int8 GEMM with per-output-channel weight exponents (dyadic requant,
    one shift per output channel); int32 bias per channel; exact float64
    accumulation (same exactness argument as gemm_int8)."""
    acc = a.to(torch.float64) @ w.to(torch.float64)
    if b is not None:
        acc = acc + b.to(torch.float64)
    acc_i = acc.round().to(torch.int64)                  # [..., C]
    shift = dst_exp - a_exp - w_exp_chan.to(torch.int64)  # [C] src->dst
    mag = acc_i.abs()
    shift_safe = shift.clamp(min=0, max=62)
    add = (1 << shift_safe) >> 1        # round-half offset (0 for shift<=0)
    shift_pos = shift.clamp(min=0)
    shift_neg = (-shift).clamp(min=0, max=32)
    out = torch.where(shift >= 0, (mag + add) >> shift_pos,
                      mag << shift_neg)
    out = torch.where(acc_i < 0, -out, out)
    out = sat(out, bits)
    if bits == 8:
        return out.to(torch.int8)
    if bits == 32:
        return out.to(torch.int32)
    return out  # int64 (Q8.16 intermediates)


# ---- operator blocks (I-ViT configurable) ---------------------------------
def _rec(rec, name, tensor):
    if rec is not None:
        rec[name] = tensor.detach().cpu().clone()


def mhsa_cfg(x: torch.Tensor, p: BlockP, s: ScaleTable, n: int, x_exp: int,
             cfg: NonlinConfig, wchan: Optional[Dict[str, torch.Tensor]],
             rec=None) -> torch.Tensor:
    ln1 = ln_cfg(x, p.gamma1, p.beta1, x_exp,
                 s.weight_exp(f"b{n}_gamma1"), s.weight_exp(f"b{n}_beta1"),
                 s.activation_exp(f"b{n}_ln1_out"), cfg)
    _rec(rec, f"b{n}_ln1_out", ln1)
    if cfg.wq == "channel":
        fused = gemm_int8_chan(
            ln1, p.wqkv, p.bqkv, s.activation_exp(f"b{n}_ln1_out"),
            wchan[f"b{n}_wqkv"], s.activation_exp(f"b{n}_qkv_out"), 8)
    else:
        fused = gemm_int8(
            ln1, p.wqkv, p.bqkv, s.activation_exp(f"b{n}_ln1_out"),
            s.weight_exp(f"b{n}_wqkv"), s.activation_exp(f"b{n}_qkv_out"), 8)
    _rec(rec, f"b{n}_qkv_out", fused)
    q = fused[..., :D]
    k = fused[..., D:2 * D]
    v = fused[..., 2 * D:]
    score_exp = 2 * s.activation_exp(f"b{n}_qkv_out") - 3
    contexts = []
    for h in range(HEADS):
        qh = q[..., h * HEAD_DIM:(h + 1) * HEAD_DIM]
        kh = k[..., h * HEAD_DIM:(h + 1) * HEAD_DIM]
        vh = v[..., h * HEAD_DIM:(h + 1) * HEAD_DIM]
        acc = qh.to(torch.float64) @ kh.to(torch.float64).transpose(-1, -2)
        score = requant(acc.round().to(torch.int64),
                        2 * s.activation_exp(f"b{n}_qkv_out"),
                        score_exp, 32)
        q16 = sat(round_shift_away(score, -(score_exp + 13)), 24)
        prob = softmax_cfg(q16, cfg)
        cacc = prob.to(torch.float64) @ vh.to(torch.float64)
        ctx = requant(cacc.round().to(torch.int64),
                      -8 + s.activation_exp(f"b{n}_qkv_out"),
                      s.activation_exp(f"b{n}_context_out"), 8).to(torch.int8)
        contexts.append(ctx)
    concat = torch.cat(contexts, dim=-1)
    _rec(rec, f"b{n}_context_out", concat)
    if cfg.wq == "channel":
        out = gemm_int8_chan(
            concat, p.wproj, p.bproj,
            s.activation_exp(f"b{n}_context_out"), wchan[f"b{n}_wproj"],
            s.activation_exp(f"b{n}_msa_out"), 8)
    else:
        out = gemm_int8(concat, p.wproj, p.bproj,
                        s.activation_exp(f"b{n}_context_out"),
                        s.weight_exp(f"b{n}_wproj"),
                        s.activation_exp(f"b{n}_msa_out"), 8)
    _rec(rec, f"b{n}_msa_out", out)
    return out


def ffn_cfg(y: torch.Tensor, p: BlockP, s: ScaleTable, n: int,
            cfg: NonlinConfig, wchan: Optional[Dict[str, torch.Tensor]],
            rec=None) -> torch.Tensor:
    ln2 = ln_cfg(y, p.gamma2, p.beta2, s.activation_exp(f"b{n}_y"),
                 s.weight_exp(f"b{n}_gamma2"), s.weight_exp(f"b{n}_beta2"),
                 s.activation_exp(f"b{n}_ln2_out"), cfg)
    _rec(rec, f"b{n}_ln2_out", ln2)
    if cfg.wq == "channel":
        q16 = gemm_int8_chan(
            ln2, p.w1, p.b1, s.activation_exp(f"b{n}_ln2_out"),
            wchan[f"b{n}_w1"], -16, 24)
    else:
        acc = ln2.to(torch.float64) @ p.w1.to(torch.float64) \
            + p.b1.to(torch.float64)
        src_exp = s.activation_exp(f"b{n}_ln2_out") + s.weight_exp(f"b{n}_w1")
        q16 = requant(acc.round().to(torch.int64), src_exp, -16, 24)
    hidden = requant(gelu_cfg(q16, cfg), -16,
                     s.activation_exp(f"b{n}_hidden"), 8)
    _rec(rec, f"b{n}_hidden", hidden)
    if cfg.wq == "channel":
        out = gemm_int8_chan(hidden, p.w2, p.b2,
                             s.activation_exp(f"b{n}_hidden"),
                             wchan[f"b{n}_w2"],
                             s.activation_exp(f"b{n}_ffn_out"), 8)
    else:
        out = gemm_int8(hidden, p.w2, p.b2, s.activation_exp(f"b{n}_hidden"),
                        s.weight_exp(f"b{n}_w2"),
                        s.activation_exp(f"b{n}_ffn_out"), 8)
    _rec(rec, f"b{n}_ffn_out", out)
    return out


def transformer_block_cfg(x: torch.Tensor, p: BlockP, s: ScaleTable, n: int,
                          x_exp: int, cfg: NonlinConfig,
                          wchan: Optional[Dict[str, torch.Tensor]],
                          rec=None) -> torch.Tensor:
    msa_out = mhsa_cfg(x, p, s, n, x_exp, cfg, wchan, rec)
    y = residual_add(x, x_exp, msa_out, s.activation_exp(f"b{n}_msa_out"),
                     s.activation_exp(f"b{n}_y"))
    _rec(rec, f"b{n}_y", y)
    ffn_out = ffn_cfg(y, p, s, n, cfg, wchan, rec)
    z = residual_add(y, s.activation_exp(f"b{n}_y"), ffn_out,
                     s.activation_exp(f"b{n}_ffn_out"),
                     s.activation_exp(f"b{n}_out"))
    _rec(rec, f"b{n}_out", z)
    return z


def forward_image_cfg(model: QuantDeiT, image_float: torch.Tensor,
                      cfg: Optional[NonlinConfig] = None, rec=None,
                      prune: bool = True):
    """Full inference under ``cfg`` (default: contract = p2_sim behavior).

    Returns (logits_int32 [1000], token_counts, logits_scale_exp), same
    signature as p2_sim.forward_image.
    """
    if cfg is None:
        cfg = NonlinConfig()
    cfg.validate()
    wchan = getattr(model, "wchan", None)
    if cfg.wq == "channel" and wchan is None:
        raise ValueError("cfg.wq='channel' requires model.wchan dict")
    s = model.scales
    inp_exp = s.activation_exp("input")
    img_q = torch.clamp(torch.round(image_float / (2.0 ** inp_exp)),
                        -128, 127).to(torch.int8)
    img_nhwc = img_q.permute(1, 2, 0)
    patches = img_nhwc.reshape(14, 16, 14, 16, 3) \
        .permute(0, 2, 1, 3, 4).reshape(196, 768)
    _rec(rec, "act_patch_matrix", patches)

    if cfg.wq == "channel":
        embed = gemm_int8_chan(patches, model.patch_w, model.patch_b,
                               inp_exp, wchan["patch_w"],
                               s.activation_exp("act_patch_embed"), 8)
    else:
        embed = gemm_int8(patches, model.patch_w, model.patch_b, inp_exp,
                          s.weight_exp("patch_w"),
                          s.activation_exp("act_patch_embed"), 8)
    _rec(rec, "act_patch_embed", embed)
    cls_exp = s.weight_exp("cls")
    pos_exp = s.weight_exp("pos")
    out_exp = s.activation_exp("act_tokens")
    row0 = residual_add(model.cls.unsqueeze(0), cls_exp,
                        model.pos[:1], pos_exp, out_exp)
    rows1 = residual_add(embed, s.activation_exp("act_patch_embed"),
                         model.pos[1:], pos_exp, out_exp)
    tokens = torch.cat([row0, rows1], dim=0)
    _rec(rec, "act_tokens", tokens)

    package_present = False
    token_counts = [tokens.shape[0]]
    sel_idx = 0
    for n in range(1, 13):
        in_exp = s.activation_exp("act_tokens") if n == 1 \
            else s.activation_exp(f"b{n - 1}_out")
        if n in SELECTOR_BLOCKS and prune:
            sel_idx += 1
            tokens, package_present = token_selector(
                tokens, package_present, model.selectors[sel_idx - 1], s,
                sel_idx, in_exp)
            token_counts.append(tokens.shape[0])
        tokens = transformer_block_cfg(tokens, model.blocks[n - 1], s, n,
                                       in_exp, cfg, wchan, rec)

    final_ln = ln_cfg(tokens, model.final_gamma, model.final_beta,
                      s.activation_exp("b12_out"),
                      s.weight_exp("final_gamma"),
                      s.weight_exp("final_beta"),
                      s.activation_exp("final_ln_out"), cfg)
    _rec(rec, "final_ln_out", final_ln)
    if cfg.wq == "channel":
        logits = gemm_int8_chan(final_ln[:1], model.head_w, model.head_b,
                                s.activation_exp("final_ln_out"),
                                wchan["head_w"], LOGIT_SCALE_EXP, 32)
    else:
        logits = gemm_int8(final_ln[:1], model.head_w, model.head_b,
                           s.activation_exp("final_ln_out"),
                           s.weight_exp("head_w"), LOGIT_SCALE_EXP, 32)
    return logits[0], token_counts, LOGIT_SCALE_EXP


def forward_batch_cfg(model: QuantDeiT, images: torch.Tensor,
                      cfg: Optional[NonlinConfig] = None):
    """Batched full inference: float [B,3,224,224] -> int32 [B,1000].

    Same semantics as :func:`forward_image_cfg` (prune=False) but processes
    a batch, avoiding the ~100ms/image kernel-dispatch overhead of the
    single-image path. All tensor ops are batch-generic; returns
    (logits_int32 [B,1000], logits_scale_exp).
    """
    if cfg is None:
        cfg = NonlinConfig()
    cfg.validate()
    wchan = getattr(model, "wchan", None)
    if cfg.wq == "channel" and wchan is None:
        raise ValueError("cfg.wq='channel' requires model.wchan dict")
    s = model.scales
    inp_exp = s.activation_exp("input")
    img_q = torch.clamp(torch.round(images / (2.0 ** inp_exp)),
                        -128, 127).to(torch.int8)
    img_nhwc = img_q.permute(0, 2, 3, 1)
    patches = img_nhwc.reshape(img_q.shape[0], 14, 16, 14, 16, 3) \
        .permute(0, 1, 3, 2, 4, 5).reshape(img_q.shape[0], 196, 768)
    if cfg.wq == "channel":
        embed = gemm_int8_chan(patches, model.patch_w, model.patch_b,
                               inp_exp, wchan["patch_w"],
                               s.activation_exp("act_patch_embed"), 8)
    else:
        embed = gemm_int8(patches, model.patch_w, model.patch_b, inp_exp,
                          s.weight_exp("patch_w"),
                          s.activation_exp("act_patch_embed"), 8)
    cls_exp = s.weight_exp("cls")
    pos_exp = s.weight_exp("pos")
    out_exp = s.activation_exp("act_tokens")
    bsz = img_q.shape[0]
    row0 = residual_add(model.cls.unsqueeze(0).expand(bsz, 1, -1), cls_exp,
                        model.pos[:1].expand(bsz, 1, -1), pos_exp, out_exp)
    rows1 = residual_add(embed, s.activation_exp("act_patch_embed"),
                         model.pos[1:].expand(bsz, 196, -1), pos_exp, out_exp)
    tokens = torch.cat([row0, rows1], dim=1)

    for n in range(1, 13):
        in_exp = s.activation_exp("act_tokens") if n == 1 \
            else s.activation_exp(f"b{n - 1}_out")
        tokens = transformer_block_cfg(tokens, model.blocks[n - 1], s, n,
                                       in_exp, cfg, wchan)

    final_ln = ln_cfg(tokens, model.final_gamma, model.final_beta,
                      s.activation_exp("b12_out"),
                      s.weight_exp("final_gamma"),
                      s.weight_exp("final_beta"),
                      s.activation_exp("final_ln_out"), cfg)
    if cfg.wq == "channel":
        logits = gemm_int8_chan(final_ln[:, :1, :], model.head_w,
                                model.head_b,
                                s.activation_exp("final_ln_out"),
                                wchan["head_w"], LOGIT_SCALE_EXP, 32)
    else:
        logits = gemm_int8(final_ln[:, :1, :], model.head_w, model.head_b,
                           s.activation_exp("final_ln_out"),
                           s.weight_exp("head_w"), LOGIT_SCALE_EXP, 32)
    return logits[:, 0, :], LOGIT_SCALE_EXP
