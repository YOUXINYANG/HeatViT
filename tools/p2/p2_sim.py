"""P2-B: contract-faithful quantized DeiT-T simulator (torch).

A tensorized port of the pure-integer golden model semantics
(verification/heatvit_ref/*) that runs on GPU. It is used for:

  * fast PTQ accuracy iteration and activation-scale calibration (P2-B),
  * evaluation of trained selectors under the exact RTL contract (P2-C),
  * cross-checking logits/checkpoints against the golden model before
    exporting vectors for XSim (P2-D).

Faithfulness notes:

  * All nonlinear sequences (GELU, Softmax exp, LayerNorm two-pass, PLAN
    sigmoid) follow nonlinear.py exactly, vectorized on int64 tensors.
  * GEMM accumulation runs in float64. Every int8*int8 product is < 2^28
    and a sum of at most 768 terms is < 2^38, so float64 accumulates are
    EXACT (IEEE double represents all integers up to 2^53).
  * Only deviation from the golden model: integer sqrt of
    ``variance + eps`` uses floor(sqrt(float64)) instead of the restoring
    isqrt; values are < 2^40 so float64 sqrt is correctly rounded, and a
    floor discrepancy of at most 1 is possible only in adversarial cases.
    Final bit-exactness is guaranteed by the golden model / XSim in P2-D.

The simulator takes int8 tensors and a ``ScaleTable`` (tools/p2/scale_table.py);
it never performs float arithmetic on weights or activations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from tools.p2.scale_table import ScaleTable

# ---- contract constants (nonlinear.py / docs Part 1 Section 9.5) ----------
GELU_A_Q16 = -18927
GELU_B_Q16 = -115933
GELU_DELTA_Q16 = 32768
INV_SQRT2_Q16 = 46341
PLAN_ONE = 65536
PLAN_BP1, PLAN_BP2, PLAN_BP3 = 65536, 155648, 327680
PLAN_C0, PLAN_C1, PLAN_C2 = 32768, 40960, 55296
SOFTMAX_LN2_Q16 = 45426
SOFTMAX_QUAD_Q16 = 23495
SOFTMAX_OFFSET_Q16 = 88670
SOFTMAX_CONST_Q16 = 22544
DELTA2_ATTENTION = 65536
DELTA2_SELECTOR = 65536
LN_D = 192
LN_EPS_Q32 = 4295
KEEP_THRESHOLD = 32768
LOGIT_SCALE_EXP = -14
HEADS = 3
HEAD_DIM = 64
D = 192
FFN_DIM = 768
CLASSES = 1000
SELECTOR_BLOCKS = (4, 7, 10)


# ---- fixed-point helpers (exact int64 semantics) --------------------------
def round_shift_away(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Nearest round-shift, ties away from zero (nonlinear.fixed)."""
    if shift == 0:
        return x
    if shift < 0:
        return x << (-shift)
    mag = x.abs()
    rounded = (mag + (1 << (shift - 1))) >> shift
    return torch.where(x < 0, -rounded, rounded)


def sat(x: torch.Tensor, bits: int) -> torch.Tensor:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return x.clamp(lo, hi)


def round_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Nearest integer division, ties away from zero (fixed.round_div).

    ``den`` must be positive; negative numerators round on their magnitude
    and restore the sign.
    """
    sign = torch.where(num < 0, -1, 1)
    q, r = num.abs() // den, num.abs() % den
    q = torch.where(2 * r >= den, q + 1, q)
    return sign * q


def requant(x: torch.Tensor, src_exp: int, dst_exp: int,
            bits: int) -> torch.Tensor:
    """scale_to_exp with saturation to ``bits`` (no wrap-around)."""
    return sat(round_shift_away(x, dst_exp - src_exp), bits)


def gelu_q16(x_q16: torch.Tensor) -> torch.Tensor:
    """Exact Q8.16 GELU approximation (nonlinear.gelu), int64 tensors."""
    u = round_shift_away(x_q16 * INV_SQRT2_Q16, 16)
    clip = u.abs().clamp(max=-GELU_B_Q16)
    t = clip + GELU_B_Q16
    t2 = round_shift_away(t * t, 16)
    poly = round_shift_away(GELU_A_Q16 * t2, 16) + 65536
    erf_mag = round_shift_away(GELU_DELTA_Q16 * poly, 16)
    sign = (u > 0).to(torch.int64) - (u < 0).to(torch.int64)
    l_erf = sign * erf_mag
    y = round_shift_away(x_q16 * (65536 + l_erf), 17)
    return sat(y, 24)


def plan_sigmoid(x_q16: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear PLAN sigmoid, Q8.16 -> Q0.16 (unsigned)."""
    abs_x = x_q16.abs()
    y_abs = torch.where(
        abs_x >= PLAN_BP3, torch.full_like(abs_x, PLAN_ONE),
        torch.where(abs_x >= PLAN_BP2, (abs_x >> 5) + PLAN_C2,
                    torch.where(abs_x >= PLAN_BP1, (abs_x >> 3) + PLAN_C1,
                                (abs_x >> 2) + PLAN_C0)))
    return torch.where(x_q16 < 0, PLAN_ONE - y_abs, y_abs)


def _softmax_exp_scaled(score_q16: torch.Tensor,
                        delta2: int) -> torch.Tensor:
    """Shared exp-approx softmax core: Q8.16 rows -> scaled Q0.16 rows."""
    row_max = score_q16.max(dim=-1, keepdim=True).values
    x_tilde = score_q16 - row_max
    z = (-x_tilde) // SOFTMAX_LN2_Q16
    p = x_tilde + z * SOFTMAX_LN2_Q16
    sq = round_shift_away((p + SOFTMAX_OFFSET_Q16) ** 2, 16)
    e = round_shift_away(SOFTMAX_QUAD_Q16 * sq, 16) + SOFTMAX_CONST_Q16
    e = e >> z
    s = e.sum(dim=-1, keepdim=True)
    recip = round_div(torch.tensor(1 << 32, dtype=torch.int64,
                                   device=e.device), s)
    ratio = round_shift_away(e * recip, 16)
    return round_shift_away(ratio * delta2, 16)


def softmax_attention(score_q16: torch.Tensor) -> torch.Tensor:
    """Q8.16 rows -> UQ0.8 attention probabilities (delta2 = 0.5)."""
    scaled = _softmax_exp_scaled(score_q16, DELTA2_ATTENTION)
    return round_shift_away(scaled, 8).clamp(0, 255).to(torch.uint8)


def softmax_selector(logits_q16: torch.Tensor) -> torch.Tensor:
    """Q8.16 2-class rows -> keep probability Q0.16 (delta2 = 1.0)."""
    scaled = _softmax_exp_scaled(logits_q16, DELTA2_SELECTOR)
    return scaled[..., 1].clamp(0, PLAN_ONE)


def mul_rsa48(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact round_shift_away(a*b, 48) for |a| < 2^33, 0 <= b < 2^48.

    The product needs up to 81 bits, exceeding int64; split b into 24-bit
    halves so every intermediate stays inside int64. Ties round away from
    zero; the sign comes from a (b is non-negative).
    """
    sign = torch.where(a < 0, -1, 1)
    a_abs = a.abs()
    hi = b >> 24
    lo = b & ((1 << 24) - 1)
    q_hi = a_abs * hi                      # < 2^57
    q_lo = a_abs * lo                      # < 2^57
    r = ((q_hi & ((1 << 24) - 1)) << 24) + (q_lo & ((1 << 48) - 1))
    q = (q_hi >> 24) + (q_lo >> 48) + (r >> 48)
    r = r & ((1 << 48) - 1)
    q = torch.where(2 * r >= (1 << 48), q + 1, q)
    return sign * q


def rsa_sq32(m: torch.Tensor) -> torch.Tensor:
    """Exact round_shift_away(m*m, 32) for |m| < 2^32 (no int64 overflow).

    The result is always non-negative (it rounds a square).
    """
    a = m.abs()
    mh = a >> 17
    ml = a & ((1 << 17) - 1)
    t1 = mh * mh
    t2 = 2 * mh * ml
    t3 = ml * ml
    r = ((t1 << 2) & 0xFFFFFFFF) + ((t2 << 17) & 0xFFFFFFFF) \
        + (t3 & 0xFFFFFFFF)
    q = (t1 << 2) + (t2 >> 15) + (t3 >> 32) + (r >> 32)
    r = r & 0xFFFFFFFF
    return torch.where(2 * r >= (1 << 32), q + 1, q)


def layernorm(xs: torch.Tensor, gammas: torch.Tensor, betas: torch.Tensor,
              x_exp: int, g_exp: int, b_exp: int,
              out_exp: int) -> torch.Tensor:
    """Exact two-pass D=192 LayerNorm (nonlinear.layernorm), int8 -> int8."""
    if not (-32 <= x_exp <= 0):
        raise ValueError("layernorm input scale must be in [-32, 0]")
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
    std_q16 = torch.sqrt((variance + LN_EPS_Q32).to(torch.float64)).floor().to(
        torch.int64)
    inv_std = round_div(torch.tensor(1 << 48, dtype=torch.int64,
                                     device=x64.device), std_q16)
    norm_q16 = sat(mul_rsa48(x_q32 - mean, inv_std), 24)
    common = min(g_exp - 16, b_exp)
    if g_exp - 16 - common > 24 or b_exp - common > 24:
        raise ValueError("layernorm affine scale gap exceeds sim int64 "
                         "safety bound (24 bits)")
    aligned = ((norm_q16 * gammas.to(torch.int64))
               << (g_exp - 16 - common)) \
        + (betas.to(torch.int64) << (b_exp - common))
    return sat(round_shift_away(aligned, out_exp - common), 8).to(torch.int8)


def gemm_int8(a: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor],
              a_exp: int, w_exp: int, dst_exp: int,
              bits: int) -> torch.Tensor:
    """int8 GEMM with int32 bias; exact float64 accumulation."""
    acc = a.to(torch.float64) @ w.to(torch.float64)
    if b is not None:
        acc = acc + b.to(torch.float64)
    out = requant(acc.round().to(torch.int64), a_exp + w_exp, dst_exp, bits)
    return out.to(torch.int8 if bits == 8 else torch.int32)


def residual_add(a: torch.Tensor, a_exp: int, b: torch.Tensor, b_exp: int,
                 out_exp: int) -> torch.Tensor:
    """Scale-aligned int8 residual sum (transformer._align_add_requant)."""
    if abs(a_exp - b_exp) > 24:
        raise ValueError("residual scale gap exceeds sim int64 safety "
                         "bound (24 bits)")
    common = min(a_exp, b_exp)
    a_q = a.to(torch.int64) << (a_exp - common)
    b_q = b.to(torch.int64) << (b_exp - common)
    return sat(round_shift_away(a_q + b_q, out_exp - common), 8).to(torch.int8)


# ---- parameter containers -------------------------------------------------
@dataclass
class BlockP:
    gamma1: torch.Tensor
    beta1: torch.Tensor
    wqkv: torch.Tensor
    bqkv: torch.Tensor
    wproj: torch.Tensor
    bproj: torch.Tensor
    gamma2: torch.Tensor
    beta2: torch.Tensor
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor


@dataclass
class SelectorP:
    local_w: torch.Tensor
    local_b: torch.Tensor
    score_w1: torch.Tensor
    score_b1: torch.Tensor
    score_w2: torch.Tensor
    score_b2: torch.Tensor
    score_w3: torch.Tensor
    score_b3: torch.Tensor
    hw_w1: torch.Tensor
    hw_b1: torch.Tensor
    hw_w2: torch.Tensor
    hw_b2: torch.Tensor


@dataclass
class QuantDeiT:
    scales: ScaleTable
    patch_w: torch.Tensor
    patch_b: torch.Tensor
    cls: torch.Tensor
    pos: torch.Tensor
    blocks: List[BlockP]
    selectors: List[SelectorP]
    final_gamma: torch.Tensor
    final_beta: torch.Tensor
    head_w: torch.Tensor
    head_b: torch.Tensor

    def to(self, device):
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))
            elif isinstance(value, list):
                for item in value:
                    for f2 in item.__dataclass_fields__:
                        t = getattr(item, f2)
                        if isinstance(t, torch.Tensor):
                            setattr(item, f2, t.to(device))
        return self


# ---- operator blocks ------------------------------------------------------
def _rec(rec, name, tensor):
    """Record a named activation into the optional collector dict."""
    if rec is not None:
        rec[name] = tensor.detach().cpu().clone()


def mhsa(x: torch.Tensor, p: BlockP, s: ScaleTable, n: int,
         x_exp: int, rec=None) -> torch.Tensor:
    """Pre-LN MHSA under per-tensor scales; x is int8 [N,192]."""
    ln1 = layernorm(x, p.gamma1, p.beta1, x_exp,
                    s.weight_exp(f"b{n}_gamma1"), s.weight_exp(f"b{n}_beta1"),
                    s.activation_exp(f"b{n}_ln1_out"))
    _rec(rec, f"b{n}_ln1_out", ln1)
    fused = gemm_int8(ln1, p.wqkv, p.bqkv, s.activation_exp(f"b{n}_ln1_out"),
                      s.weight_exp(f"b{n}_wqkv"),
                      s.activation_exp(f"b{n}_qkv_out"), 8)
    _rec(rec, f"b{n}_qkv_out", fused)
    q = fused[:, :D]
    k = fused[:, D:2 * D]
    v = fused[:, 2 * D:]
    score_exp = 2 * s.activation_exp(f"b{n}_qkv_out") - 3
    contexts = []
    for h in range(HEADS):
        qh = q[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]
        kh = k[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]
        vh = v[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]
        acc = qh.to(torch.float64) @ kh.to(torch.float64).T
        # Two golden steps: int32 writeback at score_exp (2*act - 3), then
        # Q8.16 conversion with the 1/sqrt(64) folded in: the shift is
        # score_exp + 13 (right for synthetic -17, left for per-tensor
        # scales above -13). Matches the RTL ATTN_SOFTMAX s0 = score-3.
        score = requant(acc.round().to(torch.int64),
                        2 * s.activation_exp(f"b{n}_qkv_out"),
                        score_exp, 32)
        q16 = sat(round_shift_away(score, -(score_exp + 13)), 24)
        prob = softmax_attention(q16)
        cacc = prob.to(torch.float64) @ vh.to(torch.float64)
        ctx = requant(cacc.round().to(torch.int64),
                      -8 + s.activation_exp(f"b{n}_qkv_out"),
                      s.activation_exp(f"b{n}_context_out"), 8).to(torch.int8)
        contexts.append(ctx)
    concat = torch.cat(contexts, dim=-1)
    _rec(rec, f"b{n}_context_out", concat)
    out = gemm_int8(concat, p.wproj, p.bproj,
                    s.activation_exp(f"b{n}_context_out"),
                    s.weight_exp(f"b{n}_wproj"),
                    s.activation_exp(f"b{n}_msa_out"), 8)
    _rec(rec, f"b{n}_msa_out", out)
    return out


def ffn(y: torch.Tensor, p: BlockP, s: ScaleTable, n: int,
        rec=None) -> torch.Tensor:
    ln2 = layernorm(y, p.gamma2, p.beta2, s.activation_exp(f"b{n}_y"),
                    s.weight_exp(f"b{n}_gamma2"), s.weight_exp(f"b{n}_beta2"),
                    s.activation_exp(f"b{n}_ln2_out"))
    _rec(rec, f"b{n}_ln2_out", ln2)
    acc = ln2.to(torch.float64) @ p.w1.to(torch.float64) \
        + p.b1.to(torch.float64)
    src_exp = s.activation_exp(f"b{n}_ln2_out") + s.weight_exp(f"b{n}_w1")
    q16 = requant(acc.round().to(torch.int64), src_exp, -16, 24)
    hidden = requant(gelu_q16(q16), -16, s.activation_exp(f"b{n}_hidden"), 8)
    _rec(rec, f"b{n}_hidden", hidden)
    out = gemm_int8(hidden, p.w2, p.b2, s.activation_exp(f"b{n}_hidden"),
                    s.weight_exp(f"b{n}_w2"),
                    s.activation_exp(f"b{n}_ffn_out"), 8)
    _rec(rec, f"b{n}_ffn_out", out)
    return out


def transformer_block(x: torch.Tensor, p: BlockP, s: ScaleTable, n: int,
                      x_exp: int, rec=None) -> torch.Tensor:
    """Y = X + MSA(LN1(X)); Z = Y + FFN(LN2(Y)). Returns Z int8."""
    msa_out = mhsa(x, p, s, n, x_exp, rec)
    y = residual_add(x, x_exp, msa_out, s.activation_exp(f"b{n}_msa_out"),
                     s.activation_exp(f"b{n}_y"))
    _rec(rec, f"b{n}_y", y)
    ffn_out = ffn(y, p, s, n, rec)
    z = residual_add(y, s.activation_exp(f"b{n}_y"), ffn_out,
                     s.activation_exp(f"b{n}_ffn_out"),
                     s.activation_exp(f"b{n}_out"))
    _rec(rec, f"b{n}_out", z)
    return z


def _head_gemm_gelu(a: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                    act_exp: int, wt_exp: int,
                    dst_exp: int) -> torch.Tensor:
    acc = a.to(torch.float64) @ w.to(torch.float64) + b.to(torch.float64)
    q16 = requant(acc.round().to(torch.int64), act_exp + wt_exp, -16, 24)
    return requant(gelu_q16(q16), -16, dst_exp, 8).to(torch.int8)


def token_selector(tokens: torch.Tensor, package_present: bool,
                   p: SelectorP, s: ScaleTable, idx: int,
                   in_exp: int) -> Tuple[torch.Tensor, bool]:
    """Integer Token Selector under the RTL contract.

    tokens: int8 [N,192]; returns (next_tokens int8 [N',192], package flag).
    """
    cand = tokens[1:]
    c = cand.shape[0]
    reshaped = cand.reshape(c, HEADS, HEAD_DIM)
    act = in_exp

    locals_h = []
    for h in range(HEADS):
        locals_h.append(_head_gemm_gelu(
            reshaped[:, h, :], p.local_w[h], p.local_b[h], act,
            s.weight_exp(f"s{idx}_local_w"),
            s.activation_exp(f"s{idx}_local_out")))
    locals_t = torch.stack(locals_h, dim=0)                    # [3,C,32]
    count_t = torch.tensor(c, dtype=torch.int64, device=tokens.device)
    glob = sat(round_div(locals_t.sum(dim=1), count_t), 8).to(torch.int8)
    lgl = torch.cat([locals_t, glob.unsqueeze(1).expand(HEADS, c, 32)],
                    dim=-1)                                    # [3,C,64]

    h1 = torch.stack([_head_gemm_gelu(
        lgl[h], p.score_w1[h], p.score_b1[h],
        s.activation_exp(f"s{idx}_concat_out"),
        s.weight_exp(f"s{idx}_score_w1"),
        s.activation_exp(f"s{idx}_h1_out")) for h in range(HEADS)])
    h2 = torch.stack([_head_gemm_gelu(
        h1[h], p.score_w2[h], p.score_b2[h],
        s.activation_exp(f"s{idx}_h1_out"),
        s.weight_exp(f"s{idx}_score_w2"),
        s.activation_exp(f"s{idx}_h2_out")) for h in range(HEADS)])
    logits_exp = s.activation_exp(f"s{idx}_logits_out")
    logits = torch.stack([gemm_int8(
        h2[h], p.score_w3[h], p.score_b3[h],
        s.activation_exp(f"s{idx}_h2_out"),
        s.weight_exp(f"s{idx}_score_w3"), logits_exp, 8)
        for h in range(HEADS)])
    scores_q16 = torch.stack([requant(logits[h].to(torch.int64), logits_exp,
                                      -16, 24) for h in range(HEADS)])
    head_scores = torch.stack([softmax_selector(scores_q16[h])
                               for h in range(HEADS)])         # [3,C]

    lane_sum = reshaped.to(torch.int64).sum(dim=-1)
    stats = sat(round_div(lane_sum, torch.tensor(HEAD_DIM, dtype=torch.int64,
                                                 device=tokens.device)),
                8).to(torch.int8)
    hw_hidden = _head_gemm_gelu(
        stats, p.hw_w1, p.hw_b1, s.activation_exp(f"s{idx}_stats_out"),
        s.weight_exp(f"s{idx}_hw_w1"),
        s.activation_exp(f"s{idx}_hw_hidden_out"))
    hw_acc = hw_hidden.to(torch.float64) @ p.hw_w2.to(torch.float64).T \
        + p.hw_b2.to(torch.float64)
    hw_q16 = requant(hw_acc.round().to(torch.int64),
                     s.activation_exp(f"s{idx}_hw_hidden_out")
                     + s.weight_exp(f"s{idx}_hw_w2"), -16, 24)
    head_weights = plan_sigmoid(hw_q16)                        # [C,3]

    scores_t = head_scores.T                                  # [C,3]
    num = (scores_t * head_weights).sum(dim=-1)
    den = head_weights.sum(dim=-1)
    fused = torch.where(den == 0,
                        round_div(scores_t.sum(dim=-1),
                                  torch.tensor(3, dtype=torch.int64,
                                               device=tokens.device)),
                        round_div(num, den.clamp(min=1)))
    fused = fused.clamp(0, PLAN_ONE)

    normal_rows = cand[:-1] if package_present else cand
    incoming = cand[-1:] if package_present else None
    keep = fused >= KEEP_THRESHOLD
    keep_norm = keep[:normal_rows.shape[0]]
    kept = normal_rows[keep_norm]
    pruned = normal_rows[~keep_norm]
    participants = [pruned]
    participant_scores = [fused[:normal_rows.shape[0]][~keep_norm]]
    if incoming is not None:
        participants.append(incoming)
        participant_scores.append(fused[-1:])
    package = None
    if sum(p.shape[0] for p in participants) > 0:
        parts = torch.cat(participants, dim=0)
        sc = torch.cat(participant_scores, dim=0)
        den2 = sc.sum()
        if den2 == 0:
            package = sat(round_div(
                parts.to(torch.int64).sum(dim=0),
                torch.tensor(parts.shape[0], dtype=torch.int64,
                             device=tokens.device)), 8).to(torch.int8)
        else:
            package = sat(round_div(
                (parts.to(torch.int64) * sc.unsqueeze(1)).sum(dim=0), den2),
                8).to(torch.int8)
    out = [tokens[:1], kept]
    if package is not None:
        out.append(package.unsqueeze(0))
    return torch.cat(out, dim=0), package is not None


def forward_image(model: QuantDeiT, image_float: torch.Tensor, rec=None,
                  prune: bool = True):
    """Full inference: normalized float [3,224,224] -> int32 logits.

    Returns (logits_int32 [1000], token_counts, logits_scale_exp).
    ``prune=False`` skips all Token Selectors (full 197-token path, used
    for PTQ accuracy evaluation of the quantized backbone).
    """
    s = model.scales
    inp_exp = s.activation_exp("input")
    img_q = torch.clamp(torch.round(image_float / (2.0 ** inp_exp)),
                        -128, 127).to(torch.int8)
    # NHWC patchify: [224,224,3] -> 14x14 grid of 16x16 patches, each
    # flattened as (in_row, in_col, channel) in raster patch order.
    img_nhwc = img_q.permute(1, 2, 0)
    patches = img_nhwc.reshape(14, 16, 14, 16, 3) \
        .permute(0, 2, 1, 3, 4).reshape(196, 768)
    _rec(rec, "act_patch_matrix", patches)

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
        tokens = transformer_block(tokens, model.blocks[n - 1], s, n, in_exp,
                                   rec)

    final_ln = layernorm(tokens, model.final_gamma, model.final_beta,
                         s.activation_exp("b12_out"),
                         s.weight_exp("final_gamma"),
                         s.weight_exp("final_beta"),
                         s.activation_exp("final_ln_out"))
    _rec(rec, "final_ln_out", final_ln)
    logits = gemm_int8(final_ln[:1], model.head_w, model.head_b,
                       s.activation_exp("final_ln_out"),
                       s.weight_exp("head_w"), LOGIT_SCALE_EXP, 32)
    return logits[0], token_counts, LOGIT_SCALE_EXP
