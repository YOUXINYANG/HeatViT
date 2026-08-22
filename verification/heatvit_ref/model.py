"""Complete integer HeatViT-T golden model (Phase 5 Task 4).

Fixed flow: Patch -> Block 1..3 -> Selector 1 -> Block 4..6 -> Selector 2 ->
Block 7..9 -> Selector 3 -> Block 10..12 -> Final LayerNorm -> 1000-class
head. Every checkpoint is a plain Python integer list; NumPy arrays are
never used, matching the bit-exact integer reference discipline.
"""

from dataclasses import dataclass

from .gemm import gemm
from .nonlinear import layernorm
from .selector import SelectorParams, token_selector
from .transformer import (
    BlockParams,
    PatchParams,
    patch_embedding,
    transformer_block,
)

LOGIT_SCALE_EXP = -14


@dataclass(frozen=True)
class HeatViTParams:
    """Immutable complete model parameters (synthetic, deterministic)."""

    patch: PatchParams
    blocks: tuple            # BlockParams (12 in the real model)
    selectors: tuple         # calibrated SelectorParams (3 in the real model)
    final_gamma: tuple       # 192 int8
    final_beta: tuple        # 192 int8
    head_w: tuple            # [192][1000] int8
    head_b: tuple            # 1000 int32
    selector_blocks: tuple = (4, 7, 10)  # block numbers preceded by a
                                         # selector (config: selector_before_blocks)

    def __post_init__(self):
        if len(self.blocks) != 12:
            raise ValueError("HeatViT-T needs exactly 12 blocks")
        if len(self.selectors) != len(self.selector_blocks):
            raise ValueError("selector count must match selector_blocks")


@dataclass(frozen=True)
class ModelResult:
    """Inference output, state and all mandatory checkpoints."""

    logits: tuple
    output_scale_exp: int
    checkpoints: dict
    selector_summary: tuple


def _layer_norm_rows(x, gamma, beta):
    out = []
    for row in x:
        out_row, _warn = layernorm(
            list(row), list(gamma), list(beta),
            -7, -6, -7, -7)
        out.append(out_row)
    return out


class HeatViTModel:
    """Integer HeatViT-T inference over fixed synthetic parameters."""

    def infer(self, image, params):
        if not isinstance(params, HeatViTParams):
            raise TypeError("params must be HeatViTParams")
        package_present = False
        x = patch_embedding(image, params.patch)
        checkpoints = {"patch": [list(row) for row in x]}
        selector_summary = []
        selector_number = 0
        for block_number in range(1, len(params.blocks) + 1):
            if block_number in params.selector_blocks:
                selector_number += 1
                input_count = len(x)
                selected = token_selector(
                    x, package_present,
                    params.selectors[selector_number - 1])
                x = [list(row) for row in selected.tokens]
                package_present = selected.package_present
                selector_summary.append({
                    "input_tokens": input_count,
                    "output_tokens": len(x),
                    "kept_normal": selected.kept_normal_count,
                    "pruned_normal": selected.pruned_normal_count,
                    "package_present": int(package_present),
                })
                checkpoints[f"selector_{selector_number:02d}"] = \
                    [list(row) for row in x]
            x, _ = transformer_block(x, params.blocks[block_number - 1])
            checkpoints[f"block_{block_number:02d}"] = [list(row) for row in x]

        final_ln = _layer_norm_rows(x, params.final_gamma, params.final_beta)
        logits = gemm([final_ln[0]], [list(row) for row in params.head_w],
                      list(params.head_b), False)[0]
        checkpoints["final_ln"] = final_ln
        checkpoints["logits"] = list(logits)
        return ModelResult(
            logits=tuple(logits),
            output_scale_exp=LOGIT_SCALE_EXP,
            checkpoints=checkpoints,
            selector_summary=tuple(selector_summary),
        )
