"""Per-tensor scale exponent table for real-weight P2 runs.

The RTL descriptor format already carries per-tensor ``scale_exp`` fields
(6-bit signed, [-32, 31]); the synthetic-weight flow hardcodes a uniform
SCALES dict instead. Real DeiT-T tensors need per-tensor exponents, so P2
defines this JSON schema as the single source of truth shared by:

  * tools/p2/p2_quantize.py        (producer: PTQ calibration)
  * tools/generate_descriptors.py  (consumer: descriptor scale fields)
  * tools/p2/p2_export_weights.py  (consumer: golden HeatViTParams scales)
  * verification/heatvit_ref/*     (consumer: golden per-tensor scales)

Name space (fixed by docs/heatvit.md Part 4 weight table and the descriptor
sequence builders):

  weights : patch_w, cls, pos,
            b<N>_wqkv/b<N>_wproj/b<N>_gamma1/b<N>_beta1/b<N>_w1/b<N>_w2/
            b<N>_gamma2/b<N>_beta2 (N = 1..12),
            s<S>_local_w, s<S>_score_w1/w2/w3, s<S>_hw_w1/w2 (S = 1..3),
            final_gamma, final_beta, head_w
  activations : input,
            act_patch_matrix, act_patch_embed, act_tokens,
            b<N>_ln1_out, b<N>_qkv_out, b<N>_msa_out, b<N>_y,
            b<N>_ln2_out, b<N>_hidden, b<N>_ffn_out, b<N>_out (N = 1..12),
            s<S>_local_out, s<S>_concat_out, s<S>_h1_out, s<S>_h2_out,
            s<S>_logits_out, s<S>_stats_out, s<S>_hw_hidden_out (S = 1..3),
            final_ln_out

Bias scale exponents are always derived (s0 + s1) and are not stored.
Fixed internal exponents (Q8.16 = -16, attention UQ0.8 = -8, selector
Q0.16 = -16, attention score = -17, logit = -14) are contract constants
and are not part of this table.
"""

import json
from pathlib import Path

EXP_MIN, EXP_MAX = -32, 31

_BLOCK_WEIGHTS = ("wqkv", "wproj", "gamma1", "beta1", "w1", "w2",
                  "gamma2", "beta2")
_SELECTOR_WEIGHTS = ("local_w", "score_w1", "score_w2", "score_w3",
                     "hw_w1", "hw_w2")
_BLOCK_ACTIVATIONS = ("ln1_out", "qkv_out", "msa_out", "y", "ln2_out",
                      "hidden", "ffn_out", "out")
_SELECTOR_ACTIVATIONS = ("local_out", "concat_out", "h1_out", "h2_out",
                         "logits_out", "stats_out", "hw_hidden_out")

WEIGHT_NAMES = [
    "patch_w", "cls", "pos",
    *[f"b{n}_{name}" for n in range(1, 13) for name in _BLOCK_WEIGHTS],
    *[f"s{s}_{name}" for s in range(1, 4) for name in _SELECTOR_WEIGHTS],
    "final_gamma", "final_beta", "head_w",
]

ACTIVATION_NAMES = [
    "input",
    "act_patch_matrix", "act_patch_embed", "act_tokens",
    *[f"b{n}_{name}" for n in range(1, 13) for name in _BLOCK_ACTIVATIONS],
    *[f"s{s}_{name}" for s in range(1, 4) for name in _SELECTOR_ACTIVATIONS],
    "final_ln_out",
]


def check_exp(name, value):
    """Validate one scale exponent; raise ValueError on violation."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"scale_exp {name} must be an integer, got {value!r}")
    if not (EXP_MIN <= value <= EXP_MAX):
        raise ValueError(f"scale_exp {name}={value} outside [{EXP_MIN}, {EXP_MAX}]")
    return value


class ScaleTable:
    """JSON-backed per-tensor scale exponent table."""

    def __init__(self, weights=None, activations=None):
        self.weights = dict(weights or {})
        self.activations = dict(activations or {})

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("scale table root must be an object")
        return cls(weights=raw.get("weights"), activations=raw.get("activations"))

    def save(self, path):
        payload = {
            "weights": {k: self.weights[k] for k in sorted(self.weights)},
            "activations": {
                k: self.activations[k] for k in sorted(self.activations)},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n",
                        encoding="utf-8", newline="\n")

    def validate(self):
        """Full name-set and range validation against the fixed schema."""
        for name in WEIGHT_NAMES:
            if name not in self.weights:
                raise ValueError(f"missing weight scale_exp: {name}")
        for name in ACTIVATION_NAMES:
            if name not in self.activations:
                raise ValueError(f"missing activation scale_exp: {name}")
        for name, value in self.weights.items():
            if name not in WEIGHT_NAMES:
                raise ValueError(f"unknown weight tensor: {name}")
            check_exp(name, value)
        for name, value in self.activations.items():
            if name not in ACTIVATION_NAMES:
                raise ValueError(f"unknown activation tensor: {name}")
            check_exp(name, value)
        return self

    def weight_exp(self, name):
        if name not in self.weights:
            raise KeyError(f"no scale_exp for weight tensor {name}")
        return self.weights[name]

    def activation_exp(self, name):
        if name not in self.activations:
            raise KeyError(f"no scale_exp for activation tensor {name}")
        return self.activations[name]
