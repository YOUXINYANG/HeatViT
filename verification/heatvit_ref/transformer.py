"""Integer golden operations for the Transformer data path.

Patch embedding is the first operator: NHWC patchify, one [196][768] x
[768][192] GEMM with Bias, then CLS/position addition. Every tensor keeps an
explicit scale exponent; the addition requantizes to a common scale before
summing, mirroring the RTL residual unit.
"""

from dataclasses import dataclass

from .fixed import requant
from .gemm import gemm_numpy, gemm_writeback
from .layout import head_concat, patchify, qkv_unpack
from .nonlinear import gelu, layernorm, softmax_attention


def _int8_row(name, row, cols):
    if not isinstance(row, (list, tuple)) or len(row) != cols:
        raise ValueError(f"{name} must have {cols} elements")
    out = []
    for i, value in enumerate(row):
        if isinstance(value, float):
            raise TypeError(f"{name}[{i}] must be an integer")
        value = int(value)
        if value < -128 or value > 127:
            raise ValueError(f"{name}[{i}] outside int8")
        out.append(value)
    return tuple(out)


def _int8_matrix(name, rows, n_rows, n_cols):
    if not isinstance(rows, (list, tuple)) or len(rows) != n_rows:
        raise ValueError(f"{name} must have {n_rows} rows")
    return tuple(_int8_row(f"{name}[{r}]", rows[r], n_cols) for r in range(n_rows))


def _int32_row(name, row, cols):
    if not isinstance(row, (list, tuple)) or len(row) != cols:
        raise ValueError(f"{name} must have {cols} elements")
    out = []
    for i, value in enumerate(row):
        if isinstance(value, float):
            raise TypeError(f"{name}[{i}] must be an integer")
        value = int(value)
        if value < -(1 << 31) or value > (1 << 31) - 1:
            raise ValueError(f"{name}[{i}] outside int32")
        out.append(value)
    return tuple(out)


@dataclass(frozen=True)
class PatchParams:
    """Immutable patch-embedding tensors, shapes and scale exponents."""

    patch_weight: tuple = ()
    patch_bias: tuple = ()
    cls: tuple = ()
    pos: tuple = ()
    width: int = 224
    height: int = 224
    patch: int = 16
    embed_dim: int = 192
    tokens: int = 197
    image_scale_exp: int = -7
    weight_scale_exp: int = -7
    activation_scale_exp: int = -7

    def __post_init__(self):
        if self.embed_dim != 192:
            raise ValueError("embed_dim must be 192")
        if self.tokens != 197:
            raise ValueError("tokens must be 197")
        if self.width % self.patch != 0 or self.height % self.patch != 0:
            raise ValueError("image size must be a multiple of patch")
        patch_rows = (self.width // self.patch) * (self.height // self.patch)
        patch_cols = self.patch * self.patch * 3
        object.__setattr__(
            self, "patch_weight",
            _int8_matrix("patch_weight", self.patch_weight, patch_cols, self.embed_dim),
        )
        object.__setattr__(
            self, "patch_bias", _int32_row("patch_bias", self.patch_bias, self.embed_dim)
        )
        object.__setattr__(self, "cls", _int8_row("cls", self.cls, self.embed_dim))
        object.__setattr__(
            self, "pos",
            _int8_matrix("pos", self.pos, self.tokens, self.embed_dim),
        )
        if patch_rows + 1 != self.tokens:
            raise ValueError("tokens must be num_patches + 1")


def _align_add_requant(main, main_exp, aux, aux_exp, out_exp):
    """Align two int8 values to the finer scale, add, requantize, saturate."""
    common_exp = min(main_exp, aux_exp)
    main_q = int(main) << (main_exp - common_exp)
    aux_q = int(aux) << (aux_exp - common_exp)
    return requant(main_q + aux_q, common_exp, out_exp, 8)


def patch_embedding(image, params):
    """NHWC image -> [197][192] int8 activations at ``activation_scale_exp``."""
    if not isinstance(params, PatchParams):
        raise TypeError("params must be PatchParams")
    if isinstance(image, (bytes, bytearray)) or not isinstance(image, (list, tuple)):
        raise TypeError("image must be a flat list")
    if len(image) != params.height * params.width * 3:
        raise ValueError("image length mismatch")

    patches = patchify(list(image), params.width, params.patch)
    accum = gemm_numpy(
        patches,
        [list(row) for row in params.patch_weight],
        list(params.patch_bias),
        False,
    )
    patch_embed = gemm_writeback(
        accum,
        2 * params.weight_scale_exp,
        params.activation_scale_exp,
        8,
    )

    activation = []
    activation.append([
        _align_add_requant(
            params.cls[c], params.activation_scale_exp,
            params.pos[0][c], params.activation_scale_exp,
            params.activation_scale_exp,
        )
        for c in range(params.embed_dim)
    ])
    for i, embed_row in enumerate(patch_embed):
        activation.append([
            _align_add_requant(
                embed_row[c], params.activation_scale_exp,
                params.pos[i + 1][c], params.activation_scale_exp,
                params.activation_scale_exp,
            )
            for c in range(params.embed_dim)
        ])
    return activation


@dataclass(frozen=True)
class MhsaParams:
    """Immutable three-head MHSA tensors and fixed-point scales."""

    ln_gamma: tuple = ()
    ln_beta: tuple = ()
    wqkv: tuple = ()
    bqkv: tuple = ()
    wproj: tuple = ()
    bproj: tuple = ()
    embed_dim: int = 192
    heads: int = 3
    head_dim: int = 64
    x_scale_exp: int = -7
    gamma_scale_exp: int = -6
    beta_scale_exp: int = -7
    ln_out_scale_exp: int = -7
    weight_scale_exp: int = -7
    score_scale_exp: int = -17
    prob_scale_exp: int = -8
    activation_scale_exp: int = -7

    def __post_init__(self):
        if self.embed_dim != 192 or self.heads != 3 or self.head_dim != 64:
            raise ValueError("MHSA dims must be 192/3/64")
        object.__setattr__(
            self, "ln_gamma", _int8_row("ln_gamma", self.ln_gamma, self.embed_dim)
        )
        object.__setattr__(
            self, "ln_beta", _int8_row("ln_beta", self.ln_beta, self.embed_dim)
        )
        object.__setattr__(
            self, "wqkv", _int8_matrix("wqkv", self.wqkv, self.embed_dim, 3 * self.embed_dim)
        )
        object.__setattr__(
            self, "bqkv", _int32_row("bqkv", self.bqkv, 3 * self.embed_dim)
        )
        object.__setattr__(
            self, "wproj", _int8_matrix("wproj", self.wproj, self.embed_dim, self.embed_dim)
        )
        object.__setattr__(
            self, "bproj", _int32_row("bproj", self.bproj, self.embed_dim)
        )


def mhsa(x, params):
    """Three-head MHSA: LN -> QKV -> per-head QK^T -> Softmax -> Attention*V
    -> Head concat -> projection. Returns ``(output, checkpoints)``."""
    if not isinstance(params, MhsaParams):
        raise TypeError("params must be MhsaParams")
    x = _int8_matrix("x", x, len(x), params.embed_dim)
    n = len(x)

    ln1 = []
    for row in x:
        out, _warn = layernorm(
            list(row),
            list(params.ln_gamma),
            list(params.ln_beta),
            params.x_scale_exp,
            params.gamma_scale_exp,
            params.beta_scale_exp,
            params.ln_out_scale_exp,
        )
        ln1.append(out)

    fused = gemm_writeback(
        gemm_numpy(ln1, [list(r) for r in params.wqkv], list(params.bqkv), False),
        params.x_scale_exp + params.weight_scale_exp,
        params.activation_scale_exp,
        8,
    )
    qkv = qkv_unpack(fused, n)

    score = []
    for h in range(params.heads):
        acc = gemm_numpy(qkv[0][h], qkv[1][h], None, True)
        score.append(gemm_writeback(acc, 2 * params.activation_scale_exp,
                                    params.score_scale_exp, 32))

    prob = []
    for h in range(params.heads):
        rows = []
        for r in range(n):
            q16 = [
                requant(score[h][r][c], params.score_scale_exp, -16, 24)
                for c in range(n)
            ]
            rows.append(softmax_attention(q16))
        prob.append(rows)

    context = []
    for h in range(params.heads):
        acc = gemm_numpy(prob[h], qkv[2][h], None, False, a_unsigned=True)
        context.append(gemm_writeback(
            acc,
            params.prob_scale_exp + params.activation_scale_exp,
            params.activation_scale_exp,
            8,
        ))

    concat = head_concat(context, n)
    output = gemm_writeback(
        gemm_numpy(concat, [list(r) for r in params.wproj], list(params.bproj), False),
        params.activation_scale_exp + params.weight_scale_exp,
        params.activation_scale_exp,
        8,
    )

    checkpoints = {
        "ln1": ln1,
        "fused_qkv": fused,
        "qkv": qkv,
        "score": score,
        "prob": prob,
        "context": context,
        "concat": concat,
        "msa": output,
    }
    return output, checkpoints


@dataclass(frozen=True)
class FfnParams:
    """Immutable two-layer FFN tensors and fixed-point scales."""

    ln_gamma: tuple = ()
    ln_beta: tuple = ()
    w1: tuple = ()
    b1: tuple = ()
    w2: tuple = ()
    b2: tuple = ()
    embed_dim: int = 192
    ffn_dim: int = 768
    x_scale_exp: int = -7
    gamma_scale_exp: int = -6
    beta_scale_exp: int = -7
    ln_out_scale_exp: int = -7
    weight_scale_exp: int = -7
    activation_scale_exp: int = -7

    def __post_init__(self):
        if self.embed_dim != 192 or self.ffn_dim != 768:
            raise ValueError("FFN dims must be 192/768")
        object.__setattr__(
            self, "ln_gamma", _int8_row("ln_gamma", self.ln_gamma, self.embed_dim)
        )
        object.__setattr__(
            self, "ln_beta", _int8_row("ln_beta", self.ln_beta, self.embed_dim)
        )
        object.__setattr__(
            self, "w1", _int8_matrix("w1", self.w1, self.embed_dim, self.ffn_dim)
        )
        object.__setattr__(self, "b1", _int32_row("b1", self.b1, self.ffn_dim))
        object.__setattr__(
            self, "w2", _int8_matrix("w2", self.w2, self.ffn_dim, self.embed_dim)
        )
        object.__setattr__(self, "b2", _int32_row("b2", self.b2, self.embed_dim))


def ffn(y, params):
    """Pre-LN FFN: LN2 -> GEMM+GELU (192->768) -> GEMM (768->192) ->
    residual. Returns ``(z, checkpoints)``."""
    if not isinstance(params, FfnParams):
        raise TypeError("params must be FfnParams")
    y = _int8_matrix("y", y, len(y), params.embed_dim)
    n = len(y)

    ln2 = []
    for row in y:
        out, _warn = layernorm(
            list(row),
            list(params.ln_gamma),
            list(params.ln_beta),
            params.x_scale_exp,
            params.gamma_scale_exp,
            params.beta_scale_exp,
            params.ln_out_scale_exp,
        )
        ln2.append(out)

    hidden_acc = gemm_numpy(ln2, [list(r) for r in params.w1], list(params.b1), False)
    src_exp = params.x_scale_exp + params.weight_scale_exp
    hidden = [
        [
            requant(gelu(requant(v, src_exp, -16, 24)), -16,
                    params.activation_scale_exp, 8)
            for v in row
        ]
        for row in hidden_acc
    ]

    ffn_out = gemm_writeback(
        gemm_numpy(hidden, [list(r) for r in params.w2], list(params.b2), False),
        params.activation_scale_exp + params.weight_scale_exp,
        params.activation_scale_exp,
        8,
    )

    z = []
    for r in range(n):
        z.append([
            _align_add_requant(
                y[r][c], params.activation_scale_exp,
                ffn_out[r][c], params.activation_scale_exp,
                params.activation_scale_exp,
            )
            for c in range(params.embed_dim)
        ])

    checkpoints = {
        "ln2": ln2,
        "hidden": hidden,
        "ffn_out": ffn_out,
        "z": z,
    }
    return z, checkpoints


@dataclass(frozen=True)
class BlockParams:
    """Immutable complete Pre-LN Transformer block (MHSA plus FFN)."""

    mhsa: MhsaParams
    ffn: FfnParams

    def __post_init__(self):
        if not isinstance(self.mhsa, MhsaParams):
            raise TypeError("mhsa must be MhsaParams")
        if not isinstance(self.ffn, FfnParams):
            raise TypeError("ffn must be FfnParams")


def transformer_block(x, params):
    """One complete Pre-LN block.

    Fixed order: ``Y = X + MSA(LN1(X))`` then ``Z = Y + FFN(LN2(Y))``.
    Returns ``(z, checkpoints)`` with the MHSA keys, the intermediate ``y``
    and the FFN keys, so two blocks can be chained on the previous ``z``.
    """
    if not isinstance(params, BlockParams):
        raise TypeError("params must be BlockParams")
    x = _int8_matrix("x", x, len(x), params.mhsa.embed_dim)
    n = len(x)
    act_exp = params.mhsa.activation_scale_exp

    msa_out, msa_checkpoints = mhsa(x, params.mhsa)
    y = [
        [
            _align_add_requant(
                x[r][c], act_exp, msa_out[r][c], act_exp, act_exp
            )
            for c in range(params.mhsa.embed_dim)
        ]
        for r in range(n)
    ]
    z, ffn_checkpoints = ffn(y, params.ffn)

    checkpoints = dict(msa_checkpoints)
    checkpoints["y"] = y
    checkpoints.update(ffn_checkpoints)
    return z, checkpoints
