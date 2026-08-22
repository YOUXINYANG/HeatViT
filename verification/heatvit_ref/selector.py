"""Integer golden model for the HeatViT dynamic Token Selector.

Phase 4 Task 1. Fixed layouts (docs/heatvit.md "锁定 Selector Tensor 布局"):

    input tokens       [N][192] signed int8
    candidate tokens   [C][3][64] signed int8
    local features     [3][C][32] signed int8
    global features    [3][32] signed int8
    local_global       [3][C][64] signed int8
    head keep scores   [3][C] Q0.16
    head statistics    [C][3] signed int8
    head weights       [C][3] Q0.16
    fused keep scores  [C] Q0.16
    output tokens      [N_next][192] signed int8

Q0.16 values are 17-bit unsigned (0..65536, 1.0 == 65536). The keep
threshold is inclusive: ``keep_score >= 32768``. All divisions round to
nearest with ties away from zero. The head-weight denominator and the
package denominator each have a zero fallback with its own warning bit
(WARN_HEAD_DEN_ZERO / WARN_PACKAGE_DEN_ZERO). CLS sits at index 0 and is
copied through unchanged; an incoming Package is the last candidate and
never becomes a normal output token.
"""

from dataclasses import dataclass

from .fixed import requant, round_div, sat_signed
from .gemm import gemm_numpy, gemm_writeback
from .nonlinear import gelu, plan_sigmoid, softmax_selector

WARN_HEAD_DEN_ZERO = 1 << 0
WARN_PACKAGE_DEN_ZERO = 1 << 1

KEEP_THRESHOLD = 32768
Q016_ONE = 65536


def _check_int(name, value):
    if isinstance(value, float):
        raise TypeError(f"{name} must be an integer, got float")


def _int8_row(name, row, cols):
    if not isinstance(row, (list, tuple)) or len(row) != cols:
        raise ValueError(f"{name} must have {cols} elements")
    out = []
    for i, value in enumerate(row):
        _check_int(f"{name}[{i}]", value)
        value = int(value)
        if value < -128 or value > 127:
            raise ValueError(f"{name}[{i}]={value} outside int8")
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
        _check_int(f"{name}[{i}]", value)
        value = int(value)
        if value < -(1 << 31) or value > (1 << 31) - 1:
            raise ValueError(f"{name}[{i}] outside int32")
        out.append(value)
    return tuple(out)


def _q016(name, value):
    _check_int(name, value)
    value = int(value)
    if value < 0 or value > Q016_ONE:
        raise ValueError(f"{name}={value} outside Q0.16 [0, {Q016_ONE}]")
    return value


def sat_q016(value):
    """Saturate an integer to the 17-bit unsigned Q0.16 range."""
    _check_int("value", value)
    return min(Q016_ONE, max(0, int(value)))


def _gemm_gelu(a, w, b, act_exp, wt_exp, dst_exp):
    """GEMM with bias, then GELU post-op: int8 writeback at ``dst_exp``."""
    acc = gemm_numpy(a, w, b, False)
    src_exp = act_exp + wt_exp
    return [
        [
            requant(gelu(requant(int(v), src_exp, -16, 24)), -16, dst_exp, 8)
            for v in row
        ]
        for row in acc
    ]


def _reshape_candidates(candidates):
    """Reshape [C][192] candidate tokens into [C][3][64] head-major rows."""
    out = []
    for cand in candidates:
        out.append(tuple(_int8_row("candidate", cand[64 * h:64 * (h + 1)], 64)
                         for h in range(3)))
    return tuple(out)


@dataclass(frozen=True)
class SelectorLocalParams:
    """Per-head 64->32 local MLP weights (GELU post-op, int8 writeback)."""

    w: tuple = ()   # [3][64][32] int8
    b: tuple = ()   # [3][32] int32

    def __post_init__(self):
        if not isinstance(self.w, (list, tuple)) or len(self.w) != 3:
            raise ValueError("local w must have 3 heads")
        if not isinstance(self.b, (list, tuple)) or len(self.b) != 3:
            raise ValueError("local b must have 3 heads")
        object.__setattr__(self, "w", tuple(
            _int8_matrix(f"local w[{h}]", self.w[h], 64, 32) for h in range(3)))
        object.__setattr__(self, "b", tuple(
            _int32_row(f"local b[{h}]", self.b[h], 32) for h in range(3)))


@dataclass(frozen=True)
class SelectorScoreParams:
    """Per-head 64->32->16->2 score MLP weights; logits are int8."""

    w1: tuple = ()  # [3][64][32] int8
    b1: tuple = ()  # [3][32] int32
    w2: tuple = ()  # [3][32][16] int8
    b2: tuple = ()  # [3][16] int32
    w3: tuple = ()  # [3][16][2] int8
    b3: tuple = ()  # [3][2] int32

    def __post_init__(self):
        for name in ("w1", "w2", "w3", "b1", "b2", "b3"):
            if not isinstance(getattr(self, name), (list, tuple)) or \
                    len(getattr(self, name)) != 3:
                raise ValueError(f"score {name} must have 3 heads")
        object.__setattr__(self, "w1", tuple(
            _int8_matrix(f"score w1[{h}]", self.w1[h], 64, 32) for h in range(3)))
        object.__setattr__(self, "b1", tuple(
            _int32_row(f"score b1[{h}]", self.b1[h], 32) for h in range(3)))
        object.__setattr__(self, "w2", tuple(
            _int8_matrix(f"score w2[{h}]", self.w2[h], 32, 16) for h in range(3)))
        object.__setattr__(self, "b2", tuple(
            _int32_row(f"score b2[{h}]", self.b2[h], 16) for h in range(3)))
        object.__setattr__(self, "w3", tuple(
            _int8_matrix(f"score w3[{h}]", self.w3[h], 16, 2) for h in range(3)))
        object.__setattr__(self, "b3", tuple(
            _int32_row(f"score b3[{h}]", self.b3[h], 2) for h in range(3)))


@dataclass(frozen=True)
class SelectorHeadWeightParams:
    """Shared 3->3 (GELU, int8) then 3->3 (PLAN, Q0.16) head-weight MLP."""

    w1: tuple = ()  # [3][3] int8
    b1: tuple = ()  # [3] int32
    w2: tuple = ()  # [3][3] int8
    b2: tuple = ()  # [3] int32

    def __post_init__(self):
        object.__setattr__(self, "w1",
                           _int8_matrix("head w1", self.w1, 3, 3))
        object.__setattr__(self, "b1", _int32_row("head b1", self.b1, 3))
        object.__setattr__(self, "w2",
                           _int8_matrix("head w2", self.w2, 3, 3))
        object.__setattr__(self, "b2", _int32_row("head b2", self.b2, 3))


@dataclass(frozen=True)
class SelectorParams:
    """Immutable selector tensors, shapes and fixed-point scales."""

    local: SelectorLocalParams
    score: SelectorScoreParams
    head_weight: SelectorHeadWeightParams
    heads: int = 3
    head_dim: int = 64
    local_dim: int = 32
    activation_scale_exp: int = -7
    weight_scale_exp: int = -7
    logits_scale_exp: int = -7

    def __post_init__(self):
        if self.heads != 3 or self.head_dim != 64 or self.local_dim != 32:
            raise ValueError("selector dims must be 3 heads / 64 / 32")
        if not isinstance(self.local, SelectorLocalParams):
            raise TypeError("local must be SelectorLocalParams")
        if not isinstance(self.score, SelectorScoreParams):
            raise TypeError("score must be SelectorScoreParams")
        if not isinstance(self.head_weight, SelectorHeadWeightParams):
            raise TypeError("head_weight must be SelectorHeadWeightParams")


@dataclass(frozen=True)
class FinalizeResult:
    """Atomic finalize: tokens, counts, package presence and warning bits."""

    tokens: tuple = ()
    package_present: bool = False
    kept_normal_count: int = 0
    pruned_normal_count: int = 0
    warnings: int = 0


@dataclass(frozen=True)
class SelectorResult:
    """Complete selector output: state, warnings and every checkpoint."""

    tokens: tuple = ()
    token_count: int = 0
    package_present: bool = False
    kept_normal_count: int = 0
    pruned_normal_count: int = 0
    warnings: int = 0
    local: tuple = ()
    global_features: tuple = ()
    head_scores: tuple = ()
    head_stats: tuple = ()
    head_weights: tuple = ()
    fused_scores: tuple = ()


def per_head_local_mlp(candidates, params):
    """Per-head 64->32 GEMM + GELU: [C][3][64] -> local [3][C][32] int8."""
    if not isinstance(params, SelectorLocalParams):
        raise TypeError("params must be SelectorLocalParams")
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("local MLP needs at least one candidate")
    local = []
    for h in range(3):
        a = [list(cand[h]) for cand in candidates]
        local.append(tuple(
            tuple(row) for row in _gemm_gelu(
                a, [list(r) for r in params.w[h]], list(params.b[h]),
                -7, -7, -7,
            )
        ))
    return tuple(local)


def mean_over_candidates(local):
    """Mean over the candidate axis: [3][C][32] -> global [3][32] int8."""
    local = tuple(local)
    if len(local) != 3:
        raise ValueError("local must have 3 heads")
    count = len(local[0])
    if count == 0:
        raise ValueError("mean needs at least one candidate")
    cols = len(local[0][0])
    out = []
    for h in range(3):
        row = []
        for j in range(cols):
            total = sum(local[h][c][j] for c in range(count))
            row.append(sat_signed(round_div(total, count), 8))
        out.append(tuple(row))
    return tuple(out)


def concat_local_global(local, global_features):
    """[3][C][32] + broadcast [3][32] -> local_global [3][C][64] int8."""
    local = tuple(local)
    global_features = tuple(global_features)
    if len(local) != 3 or len(global_features) != 3:
        raise ValueError("concat needs 3 heads")
    out = []
    for h in range(3):
        rows = []
        for c in range(len(local[h])):
            rows.append(tuple(list(local[h][c]) + list(global_features[h])))
        out.append(tuple(rows))
    return tuple(out)


def score_mlp_layers(local_global, params):
    """Per-head 64->32->16->2 score MLP: returns (h1, h2, logits).

    Each list is [3 heads][C][width]: int8 GELU activations for the 32 and
    16-wide layers and int8 logits for the 2-wide layer.
    """
    if not isinstance(params, SelectorScoreParams):
        raise TypeError("params must be SelectorScoreParams")
    local_global = tuple(local_global)
    if len(local_global) != 3:
        raise ValueError("local_global must have 3 heads")
    act_exp = -7
    wt_exp = -7
    logits_exp = -7
    h1_all, h2_all, logits_all = [], [], []
    for h in range(3):
        a = [list(row) for row in local_global[h]]
        h1 = _gemm_gelu(a, [list(r) for r in params.w1[h]], list(params.b1[h]),
                        act_exp, wt_exp, act_exp)
        h2 = _gemm_gelu(h1, [list(r) for r in params.w2[h]], list(params.b2[h]),
                        act_exp, wt_exp, act_exp)
        logits = gemm_writeback(
            gemm_numpy(h2, [list(r) for r in params.w3[h]],
                       list(params.b3[h]), False),
            act_exp + wt_exp, logits_exp, 8,
        )
        h1_all.append(tuple(tuple(row) for row in h1))
        h2_all.append(tuple(tuple(row) for row in h2))
        logits_all.append(tuple(tuple(row) for row in logits))
    return tuple(h1_all), tuple(h2_all), tuple(logits_all)


def per_head_score_mlp(local_global, params):
    """Per-head 64->32->16->2 MLP: [3][C][64] -> head keep scores [3][C] Q0.16."""
    h1, h2, logits = score_mlp_layers(local_global, params)
    head_scores = []
    for h in range(3):
        rows = []
        for c in range(len(logits[h])):
            q16 = [
                requant(int(logits[h][c][0]), -7, -16, 24),
                requant(int(logits[h][c][1]), -7, -16, 24),
            ]
            rows.append(softmax_selector(q16)[1])
        head_scores.append(tuple(rows))
    return tuple(head_scores)


def mean_over_head_lanes(candidates):
    """Mean over each head's 64 lanes: [C][3][64] -> head stats [C][3] int8."""
    candidates = tuple(candidates)
    out = []
    for cand in candidates:
        row = []
        for h in range(3):
            row.append(sat_signed(round_div(sum(cand[h]), 64), 8))
        out.append(tuple(row))
    return tuple(out)


def head_weight_mlp(stats, params):
    """Shared 3->3 GELU then 3->3 PLAN: [C][3] -> head weights [C][3] Q0.16."""
    if not isinstance(params, SelectorHeadWeightParams):
        raise TypeError("params must be SelectorHeadWeightParams")
    stats = tuple(stats)
    hidden = _gemm_gelu(
        [list(row) for row in stats],
        [list(r) for r in params.w1], list(params.b1),
        -7, -7, -7,
    )
    acc = gemm_numpy(hidden, [list(r) for r in params.w2],
                     list(params.b2), False)
    weights = []
    for row in acc:
        weights.append(tuple(
            sat_q016(plan_sigmoid(requant(int(v), -14, -16, 24))) for v in row
        ))
    return tuple(weights)


def fuse_head_scores(head_scores, head_weights):
    """Weighted three-head fusion: [3][C] x [C][3] -> fused [C] Q0.16.

    Returns ``(fused, warn_head)`` where ``warn_head`` is the
    WARN_HEAD_DEN_ZERO bit. Zero total weight falls back to the equal-weight
    mean of the three keep scores with denominator 3.
    """
    head_scores = tuple(head_scores)
    head_weights = tuple(head_weights)
    if len(head_scores) != 3:
        raise ValueError("head_scores must have 3 heads")
    count = len(head_scores[0])
    if len(head_weights) != count:
        raise ValueError("head_weights row count must equal candidates")
    fused = []
    warn_head = 0
    for c in range(count):
        scores = [head_scores[h][c] for h in range(3)]
        weights = [head_weights[c][h] for h in range(3)]
        num = sum(scores[h] * weights[h] for h in range(3))
        den = sum(weights)
        if den == 0:
            num = sum(scores)
            den = 3
            warn_head = WARN_HEAD_DEN_ZERO
        fused.append(sat_q016(round_div(num, den)))
    return tuple(fused), warn_head


def finalize_tokens(cls, normal, incoming_package, scores):
    """Atomic stable compaction plus single Package accumulation.

    ``normal`` are the normal candidates in input order; ``incoming_package``
    is the previous Package token (the last candidate) or None; ``scores``
    holds one fused keep score per normal candidate plus, when a Package is
    present, one trailing score for it. Candidates with ``score >= 32768``
    are kept in stable order. Every pruned normal and the incoming Package
    (regardless of its score) accumulate into the Package: numerator
    ``sum(score * feature[d])`` over denominator ``sum(score)``, rounded per
    channel; a zero denominator falls back to the unweighted feature mean
    and sets WARN_PACKAGE_DEN_ZERO.
    """
    width = len(cls) if isinstance(cls, (list, tuple)) else 0
    if width <= 0:
        raise ValueError("cls must be a non-empty row")
    cls = _int8_row("cls", cls, width)
    normal = [
        _int8_row(f"normal[{r}]", row, width) for r, row in enumerate(normal)
    ]
    if not isinstance(scores, (list, tuple)):
        raise TypeError("scores must be a list")
    scores = [_q016(f"scores[{i}]", v) for i, v in enumerate(scores)]

    package_token = None
    package_score = 0
    if incoming_package is not None:
        if not normal:
            raise ValueError("incoming package requires at least one candidate")
        package_token = _int8_row("incoming_package", incoming_package, width)
        package_score = scores[-1]
        scores = scores[:-1]
    if len(scores) != len(normal):
        raise ValueError("scores length must match normal candidates")

    kept = []
    participants = []
    for token, score in zip(normal, scores):
        if score >= KEEP_THRESHOLD:
            kept.append(token)
        else:
            participants.append((score, token))
    if package_token is not None:
        participants.append((package_score, package_token))

    pruned_normal_count = len(participants) - (1 if package_token is not None
                                               else 0)
    package_will_exist = len(participants) > 0
    warnings = 0
    package = None
    if package_will_exist:
        den = sum(score for score, _ in participants)
        if den == 0:
            warnings |= WARN_PACKAGE_DEN_ZERO
            package = tuple(
                sat_signed(
                    round_div(sum(token[d] for _, token in participants),
                              len(participants)),
                    8,
                )
                for d in range(width)
            )
        else:
            package = tuple(
                sat_signed(
                    round_div(sum(score * token[d] for score, token
                                  in participants), den),
                    8,
                )
                for d in range(width)
            )

    tokens = [cls] + kept + ([package] if package_will_exist else [])
    return FinalizeResult(
        tokens=tuple(tokens),
        package_present=package_will_exist,
        kept_normal_count=len(kept),
        pruned_normal_count=pruned_normal_count,
        warnings=warnings,
    )


def token_selector(tokens, package_present, params):
    """Complete HeatViT Token Selector.

    ``tokens`` is the [N][192] int8 activation; ``package_present`` marks the
    last candidate as the incoming Package. Returns a ``SelectorResult`` with
    the next activation, Token/Package state, warning bits and every
    intermediate checkpoint.
    """
    if not isinstance(params, SelectorParams):
        raise TypeError("params must be SelectorParams")
    if not isinstance(tokens, (list, tuple)) or len(tokens) < 2:
        raise ValueError("selector input needs CLS plus at least one candidate")
    tokens = _int8_matrix("tokens", tokens, len(tokens), 192)
    package_present = int(bool(package_present))

    candidates = _reshape_candidates(tokens[1:])
    local = per_head_local_mlp(candidates, params.local)
    global_features = mean_over_candidates(local)
    local_global = concat_local_global(local, global_features)
    head_scores = per_head_score_mlp(local_global, params.score)
    head_stats = mean_over_head_lanes(candidates)
    head_weights = head_weight_mlp(head_stats, params.head_weight)
    fused_scores, warn_head = fuse_head_scores(head_scores, head_weights)

    normals = list(tokens[1:])
    incoming = None
    if package_present:
        incoming = normals[-1]
        normals = normals[:-1]
    finalized = finalize_tokens(tokens[0], normals, incoming, fused_scores)

    return SelectorResult(
        tokens=finalized.tokens,
        token_count=len(finalized.tokens),
        package_present=finalized.package_present,
        kept_normal_count=finalized.kept_normal_count,
        pruned_normal_count=finalized.pruned_normal_count,
        warnings=warn_head | finalized.warnings,
        local=local,
        global_features=global_features,
        head_scores=head_scores,
        head_stats=head_stats,
        head_weights=head_weights,
        fused_scores=fused_scores,
    )
