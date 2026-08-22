"""Deterministic synthetic HeatViT-T parameters with Selector calibration.

Phase 5 Task 4. All ordinary parameters come from a single
``random.Random(20260815)`` stream in the fixed order patch -> block 1..12 ->
selector 1..3 -> final LayerNorm -> head. Main weights use [-8, 7], biases
[-64, 63], gamma is centered on 64 and clamped to int8, beta uses [-4, 4].

Each Selector's final score layer is calibrated per the approved algorithm:
for stage s and attempt a the score-layer seed is
``20260815 + s*1000 + a``; the head-weight branch is deterministic and equal
so the fusion reduces to the equal-weight mean of the three head scores.
The per-candidate bias-free keep-drop logit difference is computed; if all
differences are equal the attempt is skipped, otherwise the keep/drop bias
difference is set to the negative median difference and the full
Softmax/fuse/finalize path is evaluated. An attempt is accepted when it
keeps at least one normal and prunes at least two; the first accepted
attempt is recorded.
"""

import random
from dataclasses import replace

from .model import HeatViTParams
from .selector import (
    SelectorHeadWeightParams,
    SelectorLocalParams,
    SelectorParams,
    SelectorScoreParams,
    concat_local_global,
    mean_over_candidates,
    per_head_local_mlp,
    score_mlp_layers,
    token_selector,
)
from .transformer import (
    BlockParams,
    FfnParams,
    MhsaParams,
    PatchParams,
    patch_embedding,
    transformer_block,
)

SEED = 20260815
SELECTOR_ATTEMPTS = 256


def _int8_matrix(rng, rows, cols, lo=-8, hi=7):
    return [[rng.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]


def _int32_vec(rng, cols, lo=-64, hi=63):
    return [rng.randint(lo, hi) for _ in range(cols)]


def _gamma_vec(rng, cols):
    # Gamma centered on 64, clamped to int8.
    return [min(127, max(0, 64 + rng.randint(-24, 24))) for _ in range(cols)]


def _beta_vec(rng, cols):
    return [rng.randint(-4, 4) for _ in range(cols)]


def build_patch_params(rng):
    return PatchParams(
        patch_weight=_int8_matrix(rng, 768, 192),
        patch_bias=_int32_vec(rng, 192),
        cls=[rng.randint(-128, 127) for _ in range(192)],
        pos=[[rng.randint(-128, 127) for _ in range(192)]
             for _ in range(197)],
        width=224, height=224, patch=16, embed_dim=192, tokens=197,
        image_scale_exp=-7, weight_scale_exp=-7, activation_scale_exp=-7,
        cls_scale_exp=-7, pos_scale_exp=-7,
    )


def build_block_params(rng):
    mhsa = MhsaParams(
        ln_gamma=_gamma_vec(rng, 192),
        ln_beta=_beta_vec(rng, 192),
        wqkv=_int8_matrix(rng, 192, 576),
        bqkv=_int32_vec(rng, 576),
        wproj=_int8_matrix(rng, 192, 192),
        bproj=_int32_vec(rng, 192),
        x_scale_exp=-7, gamma1_scale_exp=-6, beta1_scale_exp=-7,
        ln1_out_scale_exp=-7, wqkv_scale_exp=-7, qkv_out_scale_exp=-7,
        score_scale_exp=-17, prob_scale_exp=-8,
        context_out_scale_exp=-7, wproj_scale_exp=-7,
        msa_out_scale_exp=-7,
    )
    ffn = FfnParams(
        ln_gamma=_gamma_vec(rng, 192),
        ln_beta=_beta_vec(rng, 192),
        w1=_int8_matrix(rng, 192, 768),
        b1=_int32_vec(rng, 768),
        w2=_int8_matrix(rng, 768, 192),
        b2=_int32_vec(rng, 192),
        x_scale_exp=-7, gamma2_scale_exp=-6, beta2_scale_exp=-7,
        ln2_out_scale_exp=-7, w1_scale_exp=-7,
        hidden_out_scale_exp=-7, w2_scale_exp=-7,
        ffn_out_scale_exp=-7, out_scale_exp=-7,
    )
    return BlockParams(mhsa=mhsa, ffn=ffn)


def build_selector_base(rng):
    """Selector parameters with the ordinary layers from the master stream;
    the final score layer weights/bias are filled in by calibration.

    As-built note: the local and score MLP weights use [-64, 63] (wider
    than the main block weights) so the double-GELU feature chain keeps
    signal; the GEMM accumulators stay far inside int32.
    """
    def head_matrix(rows, cols):
        return [_int8_matrix(rng, rows, cols, lo=-64, hi=63)
                for _ in range(3)]

    local = SelectorLocalParams(
        w=head_matrix(64, 32),
        b=[_int32_vec(rng, 32) for _ in range(3)],
    )
    score = SelectorScoreParams(
        w1=head_matrix(64, 32),
        b1=[_int32_vec(rng, 32) for _ in range(3)],
        w2=head_matrix(32, 16),
        b2=[_int32_vec(rng, 16) for _ in range(3)],
        w3=head_matrix(16, 2),
        b3=[[0, 0] for _ in range(3)],
    )
    # Deterministic equal head-weight branch: identity weights and zero
    # biases make every candidate's three head weights equal, so the fusion
    # is the equal-weight mean of the three head keep scores.
    head_weight = SelectorHeadWeightParams(
        w1=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        b1=[0, 0, 0],
        w2=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        b2=[0, 0, 0],
    )
    return SelectorParams(local=local, score=score, head_weight=head_weight)


def _selector_h2(tokens, params):
    """Compute the per-head second hidden layer for the given activation."""
    candidates = tokens[1:]
    reshaped = tuple(
        tuple(tuple(int(v) for v in cand[64 * h:64 * (h + 1)])
              for h in range(3))
        for cand in candidates
    )
    local = per_head_local_mlp(reshaped, params.local)
    global_features = mean_over_candidates(local)
    local_global = concat_local_global(local, global_features)
    _h1, h2, _logits = score_mlp_layers(local_global, params.score)
    return h2


def calibrate_selector(stage, tokens, package_present, base_params):
    """Calibrate one Selector's final score layer.

    Returns ``(params, summary)`` with the first accepted attempt, or raises
    RuntimeError after 256 failed attempts.
    """
    h2 = _selector_h2(tokens, base_params)

    for attempt in range(SELECTOR_ATTEMPTS):
        rng = random.Random(SEED + stage * 1000 + attempt)
        w3 = [[[rng.randint(-8, 7) for _ in range(2)] for _ in range(16)]
              for _ in range(3)]

        # Bias-free keep-drop logit difference per head/candidate.
        diffs = []
        for h in range(3):
            for c in range(len(h2[h])):
                diffs.append(sum(
                    h2[h][c][k] * (w3[h][k][1] - w3[h][k][0])
                    for k in range(16)))
        if len(set(diffs)) == 1:
            continue

        diffs_sorted = sorted(diffs)
        median = diffs_sorted[len(diffs_sorted) // 2]
        b3 = [[0, -median] for _ in range(3)]

        score = replace(base_params.score, w3=tuple(
            tuple(tuple(row) for row in w3[h]) for h in range(3)),
            b3=tuple(tuple(row) for row in b3))
        params = replace(base_params, score=score)
        result = token_selector(tokens, package_present, params)
        if result.kept_normal_count >= 1 and result.pruned_normal_count >= 2:
            summary = {
                "stage": stage,
                "attempt": attempt,
                "seed": SEED + stage * 1000 + attempt,
                "median_logit_diff": median,
                "input_tokens": len(tokens),
                "output_tokens": result.token_count,
                "kept_normal": result.kept_normal_count,
                "pruned_normal": result.pruned_normal_count,
                "package_present": int(result.package_present),
            }
            return params, summary

    raise RuntimeError(f"selector stage {stage} calibration failed")


def build_params(image):
    """Full deterministic parameter set with calibrated Selectors.

    The calibration needs the real stage inputs, so the patch and block
    pipeline runs here; returns ``(HeatViTParams, selector_summary,
    token_counts)``.
    """
    rng = random.Random(SEED)
    patch = build_patch_params(rng)
    blocks = [build_block_params(rng) for _ in range(12)]
    selectors = [build_selector_base(rng) for _ in range(3)]
    final_gamma = _gamma_vec(rng, 192)
    final_beta = _beta_vec(rng, 192)
    head_w = _int8_matrix(rng, 192, 1000)
    head_b = _int32_vec(rng, 1000)

    x = patch_embedding(image, patch)
    package_present = False
    summary = []
    selector_params = []
    stage_no = 0
    token_counts = [197]
    for block_number in range(1, 13):
        if block_number in (4, 7, 10):
            stage_no += 1
            params, s = calibrate_selector(stage_no, x, package_present,
                                           selectors[stage_no - 1])
            selector_params.append(params)
            summary.append(s)
            result = token_selector(x, package_present, params)
            x = [list(row) for row in result.tokens]
            package_present = result.package_present
            token_counts.append(result.token_count)
        x, _ = transformer_block(x, blocks[block_number - 1])

    params = HeatViTParams(
        patch=patch,
        blocks=tuple(blocks),
        selectors=tuple(selector_params),
        final_gamma=tuple(final_gamma),
        final_beta=tuple(final_beta),
        head_w=tuple(tuple(row) for row in head_w),
        head_b=tuple(head_b),
    )
    return params, summary, token_counts
