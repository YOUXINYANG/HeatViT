"""Bit-exact integer references for GELU and PLAN sigmoid.

Both functions accept signed Q8.16 integers and never touch Python
floats, so the golden model cannot drift from the RTL fixed-point
contract. GELU returns signed Q8.16; PLAN returns unsigned Q0.16.
"""

from .fixed import isqrt, requant, round_div, round_shift_away, sat_signed


GELU_A_Q16 = -18927
GELU_B_Q16 = -115933
GELU_DELTA_Q16 = 32768
INV_SQRT2_Q16 = 46341

PLAN_BP_1_Q16 = 65536    # 1.0
PLAN_BP_2_Q16 = 155648   # 2.375
PLAN_BP_3_Q16 = 327680   # 5.0

PLAN_C0_Q16 = 32768      # 1/2
PLAN_C1_Q16 = 40960      # 5/8
PLAN_C2_Q16 = 55296      # 27/32
PLAN_ONE_Q16 = 65536

SOFTMAX_LN2_Q16 = 45426
SOFTMAX_QUAD_Q16 = 23495
SOFTMAX_OFFSET_Q16 = 88670
SOFTMAX_CONST_Q16 = 22544
SOFTMAX_DELTA2_Q16_ATTENTION = 32768
SOFTMAX_DELTA2_Q16_SELECTOR = 65536

LAYERNORM_D = 192
LN_EPS_Q32 = 4295


def _check_int(name, value):
    if isinstance(value, float):
        raise TypeError(f"{name} must be an integer, got float")


def gelu(x: int) -> int:
    """HeatViT Q8.16 GELU approximation, saturated to signed 24 bits."""
    _check_int("x", x)
    u = round_shift_away(x * INV_SQRT2_Q16, 16)
    clip = min(abs(u), -GELU_B_Q16)
    t = clip + GELU_B_Q16
    t2 = round_shift_away(t * t, 16)
    poly = round_shift_away(GELU_A_Q16 * t2, 16) + 65536
    erf_mag = round_shift_away(GELU_DELTA_Q16 * poly, 16)
    sign = (u > 0) - (u < 0)
    l_erf = sign * erf_mag
    y = round_shift_away(x * (65536 + l_erf), 17)
    return sat_signed(y, 24)


def plan_sigmoid(x: int) -> int:
    """Piecewise-linear PLAN sigmoid approximation in unsigned Q0.16."""
    _check_int("x", x)
    abs_x = abs(x)
    if abs_x >= PLAN_BP_3_Q16:
        y_abs = PLAN_ONE_Q16
    elif abs_x >= PLAN_BP_2_Q16:
        y_abs = (abs_x >> 5) + PLAN_C2_Q16
    elif abs_x >= PLAN_BP_1_Q16:
        y_abs = (abs_x >> 3) + PLAN_C1_Q16
    else:
        y_abs = (abs_x >> 2) + PLAN_C0_Q16
    return PLAN_ONE_Q16 - y_abs if x < 0 else y_abs


def _softmax_exp(x_tilde: int) -> int:
    """Range-reduced Q8.16 exponential of a non-positive argument."""
    _check_int("x_tilde", x_tilde)
    z = (-x_tilde) // SOFTMAX_LN2_Q16
    p = x_tilde + z * SOFTMAX_LN2_Q16
    shifted = p + SOFTMAX_OFFSET_Q16
    square = round_shift_away(shifted * shifted, 16)
    exp_q16 = round_shift_away(SOFTMAX_QUAD_Q16 * square, 16) + SOFTMAX_CONST_Q16
    return exp_q16 >> z


def softmax_scaled(inputs, delta2_q16: int):
    """Q0.16 scaled softmax row before wrapper narrowing."""
    _check_int("delta2_q16", delta2_q16)
    if not inputs:
        raise ValueError("softmax row must not be empty")
    row_max = max(inputs)
    exps = [_softmax_exp(x - row_max) for x in inputs]
    row_sum = sum(exps)
    if row_sum == 0:
        raise ValueError("softmax row sum is zero")
    recip = round_div(1 << 32, row_sum)
    ratios = [round_shift_away(exp_i * recip, 16) for exp_i in exps]
    return [round_shift_away(ratio * delta2_q16, 16) for ratio in ratios]


def softmax_selector(inputs):
    """Selector softmax: scaled Q0.16 saturated to 17 bits."""
    scaled = softmax_scaled(inputs, SOFTMAX_DELTA2_Q16_SELECTOR)
    return [min(value, PLAN_ONE_Q16) for value in scaled]


def softmax_attention(inputs):
    """Attention softmax: scaled Q0.16 rounded to UQ0.8."""
    scaled = softmax_scaled(inputs, SOFTMAX_DELTA2_Q16_ATTENTION)
    return [min(round_shift_away(value, 8), 255) for value in scaled]


def layernorm(xs, gammas, betas, x_scale, gamma_scale, beta_scale, out_scale):
    """D=192 two-pass fixed-point LayerNorm.

    Returns ``(outputs, negative_variance)``. Gamma is aligned to the Q8.16
    normalized value at a common exponent, Beta is added, and the exact sum
    is rounded once to the output int8 scale and saturated.
    """
    for name, value in (
        ("x_scale", x_scale),
        ("gamma_scale", gamma_scale),
        ("beta_scale", beta_scale),
        ("out_scale", out_scale),
    ):
        _check_int(name, value)
    if not (-32 <= x_scale <= 0):
        raise ValueError("layernorm input scale must be in [-32, 0]")
    if not (len(xs) == len(gammas) == len(betas) == LAYERNORM_D):
        raise ValueError(f"layernorm rows must have exactly {LAYERNORM_D} channels")
    for values in (xs, gammas, betas):
        for value in values:
            _check_int("channel value", value)
            if not (-128 <= value <= 127):
                raise ValueError("layernorm channels must be signed int8")

    x_q32 = [x << (x_scale + 32) for x in xs]
    sum_x = sum(x_q32)
    sum_square = sum(
        round_shift_away(x * x, -(2 * x_scale + 32)) for x in xs
    )
    mean = round_div(sum_x, LAYERNORM_D)
    e2 = round_div(sum_square, LAYERNORM_D)
    mean_square = round_shift_away(mean * mean, 32)
    variance = e2 - mean_square
    negative_variance = variance < 0
    variance = max(0, variance)
    std_q16 = isqrt(variance + LN_EPS_Q32)[0]
    inv_std_q32 = round_div(1 << 48, std_q16)

    common_exp = min(gamma_scale - 16, beta_scale)
    outputs = []
    for x, x_q32_value, gamma, beta in zip(xs, x_q32, gammas, betas):
        norm_q16 = sat_signed(
            round_shift_away((x_q32_value - mean) * inv_std_q32, 48), 24
        )
        product = norm_q16 * gamma
        aligned = (
            (product << (gamma_scale - 16 - common_exp))
            + (beta << (beta_scale - common_exp))
        )
        outputs.append(requant(aligned, common_exp, out_scale, 8))
    return outputs, negative_variance
