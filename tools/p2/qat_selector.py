#!/usr/bin/env python3
"""P4: frozen Token Selectors as differentiable float mirrors.

``QatSelector`` mirrors tools/p2/p2_sim.token_selector (the integer RTL
contract, shared by the bit-exact eval path) in float value space for
the QAT train path:

  * weights/biases are the dequantized deployment int8/int32 tensors
    from the P2-C selector checkpoint, fake-quantized at every use with
    the s{idx}_* scale entries (so the train path uses exactly the
    deployed values);
  * the selector GELU is the shared deployment ShiftGELU-ln2
    (qat_fakeq.shiftgelu_float), the 2-class softmax and the PLAN
    sigmoid are float mirrors quantized to Q0.16 at the contract points;
  * the keep decision is the hard threshold 0.5 (KEEP_THRESHOLD/65536)
    taken on the fake-quantized fused score with the mask detached
    (straight-through threshold: the decision itself carries no
    gradient, the score path and the package aggregation do);
  * the package token is the fused-score-weighted mean of the pruned
    rows (plus the incoming package when present), re-fake-quantized to
    int8 at the token scale — gradients flow back to the backbone
    through both the kept rows and the package.

The selectors are registered as buffers (requires_grad=False): P4 step 1
freezes them by construction.

Payload layout (p2_out/selectors_sup4.pt):
  selectors[i]      deployment tensors: local_w [3,64,32] int8,
                    local_b [3,32] int32, score_w1 [3,64,32],
                    score_b1 [3,32], score_w2 [3,32,16], score_b2 [3,16],
                    score_w3 [3,16,2], score_b3 [3,2], hw_w1 [3,3],
                    hw_b1 [3], hw_w2 [3,3], hw_b2 [3]
  weight_exps[i]    s{i}_<name> weight exponents
  sel_acts[i]       s{i}_<name> activation exponents
"""

import torch
import torch.nn as nn

from tools.p2.qat_fakeq import (
    fake_quant,
    fake_quant_int8,
    fake_quant_int32,
    fake_quant_q816,
    shiftgelu_float,
)

HEADS = 3
HEAD_DIM = 64
KEEP_THRESHOLD = 0.5          # Q0.16 value of KEEP_THRESHOLD (32768)

_Q16 = -16


def fake_quant_q016(x):
    """Unsigned Q0.16 boundary: round to 2**-16, clip to the integer
    grid [0, 65536] (PLAN_ONE) — lo/hi are in quantized-integer space,
    not value space (the same convention as p2_sim.plan / softmax)."""
    return fake_quant(x, _Q16, 0.0, 65536.0)

_WT = ("local_w", "score_w1", "score_w2", "score_w3", "hw_w1", "hw_w2")
# (bias name, input activation exp key)
_BS = (("local_b", "b{blk}_out"), ("score_b1", "concat_out"),
       ("score_b2", "h1_out"), ("score_b3", "h2_out"),
       ("hw_b1", "stats_out"), ("hw_b2", "hw_hidden_out"))


def merge_selector_scales(table, payload):
    """Merge the s{idx}_* scale entries from a selector checkpoint into
    the backbone table (idempotent; mirrors qat_prune_eval.load_selectors
    but for the train-path table)."""
    for i in (1, 2, 3):
        for name, exp in payload["weight_exps"][str(i)].items():
            table.weights[f"s{i}_{name}"] = exp
        for name, exp in payload["sel_acts"][str(i)].items():
            table.activations[f"s{i}_{name}"] = exp
    table.validate()


def plan_sigmoid_float(x):
    """Float mirror of p2_sim.plan_sigmoid (piecewise-linear PLAN sigmoid,
    Q8.16 -> Q0.16): |x|>=5 -> 1; >=2.375 -> x/32+0.84375;
    >=1 -> x/8+0.625; else x/4+0.5."""
    a = x.abs()
    y = torch.where(a >= 5.0, torch.ones_like(a),
                    torch.where(a >= 155648.0 / 65536.0,
                                a / 32.0 + 55296.0 / 65536.0,
                                torch.where(a >= 1.0,
                                            a / 8.0 + 40960.0 / 65536.0,
                                            a / 4.0 + 32768.0 / 65536.0)))
    return torch.where(x < 0, 1.0 - y, y)


# p2_sim softmax contract constants (Q16 values)
_LN2 = 45426.0 / 65536.0
_OFFSET = 88670.0 / 65536.0
_QUAD = 23495.0 / 65536.0
_CONST = 22544.0 / 65536.0


def softmax_selector_float(score_q16):
    """Float mirror of p2_sim.softmax_selector (2-class Q8.16 -> Q0.16,
    quadratic 2^x approximation, delta2 = 1.0). Reproduces the integer
    algebra in value space so the keep decisions track the deployed
    path bin-for-bin (the analytic softmax deviates ~1e-3 and flips
    tokens near the 0.5 threshold)."""
    row_max = score_q16.max(dim=-1, keepdim=True).values
    x_tilde = score_q16 - row_max
    z = torch.floor(-x_tilde / _LN2)
    p = x_tilde + z * _LN2
    sq = fake_quant_q816((p + _OFFSET) ** 2)
    e = fake_quant_q816(_QUAD * sq) + _CONST
    e = e * (2.0 ** (-z))
    s = e.sum(dim=-1, keepdim=True)
    recip = fake_quant_q816(1.0 / s.clamp(min=1e-12))
    ratio = fake_quant_q816(e * recip)
    return fake_quant_q016(ratio)[..., 1]


class QatSelector(nn.Module):
    """Frozen float mirror of one integer Token Selector (idx = 1..3).

    Input ``x`` is the dequantized token tensor [1, N, 192] at exponent
    ``a_exp`` (value space). Returns (next_tokens [1, N', 192],
    package_present, fused [1, C], keep_mask [1, Nn]).
    """

    def __init__(self, payload_sel, table, idx):
        super().__init__()
        self.idx = idx
        s = table
        in_exp = table.activation_exp(f"b{3 * idx}_out")
        for name in _WT:
            exp = s.weight_exp(f"s{idx}_{name}")
            self.register_buffer(name, payload_sel[name].to(torch.float32)
                                 * (2.0 ** exp))
        for name, act_key in _BS:
            a_exp = in_exp if act_key.startswith("b") \
                else s.activation_exp(f"s{idx}_{act_key}")
            w_name = "local_w" if name == "local_b" \
                else "score_w1" if name == "score_b1" \
                else "score_w2" if name == "score_b2" \
                else "score_w3" if name == "score_b3" \
                else "hw_w1" if name == "hw_b1" else "hw_w2"
            exp = a_exp + s.weight_exp(f"s{idx}_{w_name}")
            self.register_buffer(name, payload_sel[name].to(torch.float32)
                                 * (2.0 ** exp))

    def _k(self, name):
        return f"s{self.idx}_{name}"

    def forward(self, x, table, a_exp, package_present):
        s = table
        cand = x[:, 1:]                          # [1, C, 192]
        c = cand.shape[1]
        resh = cand.reshape(1, c, HEADS, HEAD_DIM)

        w_local = fake_quant_int8(self.local_w,
                                  s.weight_exp(self._k("local_w")))
        b_local = fake_quant_int32(
            self.local_b, a_exp + s.weight_exp(self._k("local_w")))
        acc = torch.einsum("bchd,hdn->bchn", resh, w_local) + b_local
        locs = fake_quant_int8(
            shiftgelu_float(fake_quant_q816(acc)),
            s.activation_exp(self._k("local_out")))
        locals_t = locs.transpose(1, 2)                    # [1,3,C,32]
        # glob = per-head mean over TOKENS (p2_sim: round_div(sum over
        # C, C)), expanded back over C -- the global context per head.
        glob = fake_quant_int8(locals_t.mean(dim=2),
                               s.activation_exp(self._k("local_out")))
        lgl = torch.cat(
            [locals_t, glob.unsqueeze(2).expand(-1, -1, c, -1)],
            dim=-1)                                        # [1,3,C,64]

        w1 = fake_quant_int8(self.score_w1, s.weight_exp(self._k("score_w1")))
        b1 = fake_quant_int32(
            self.score_b1,
            s.activation_exp(self._k("concat_out"))
            + s.weight_exp(self._k("score_w1")))
        h1 = fake_quant_int8(
            shiftgelu_float(fake_quant_q816(
                torch.einsum("bhcd,hdn->bhcn", lgl, w1) + b1.unsqueeze(1))),
            s.activation_exp(self._k("h1_out")))           # [1,3,C,32]
        w2 = fake_quant_int8(self.score_w2, s.weight_exp(self._k("score_w2")))
        b2 = fake_quant_int32(
            self.score_b2,
            s.activation_exp(self._k("h1_out"))
            + s.weight_exp(self._k("score_w2")))
        h2 = fake_quant_int8(
            shiftgelu_float(fake_quant_q816(
                torch.einsum("bhcn,hno->bhco", h1, w2) + b2.unsqueeze(1))),
            s.activation_exp(self._k("h2_out")))           # [1,3,C,16]
        w3 = fake_quant_int8(self.score_w3, s.weight_exp(self._k("score_w3")))
        b3 = fake_quant_int32(
            self.score_b3,
            s.activation_exp(self._k("h2_out"))
            + s.weight_exp(self._k("score_w3")))
        logits = fake_quant_int8(
            torch.einsum("bhco,hop->bhcp", h2, w3) + b3.unsqueeze(1),
            s.activation_exp(self._k("logits_out")))       # [1,3,C,2]
        scores = fake_quant_q816(logits)
        head_scores = softmax_selector_float(scores)       # [1,3,C]

        stats = fake_quant_int8(resh.mean(dim=-1),
                                s.activation_exp(self._k("stats_out")))
        hw_w1 = fake_quant_int8(self.hw_w1,
                                s.weight_exp(self._k("hw_w1")))
        hw_b1 = fake_quant_int32(
            self.hw_b1, s.activation_exp(self._k("stats_out"))
            + s.weight_exp(self._k("hw_w1")))
        hw_hidden = fake_quant_int8(
            shiftgelu_float(fake_quant_q816(stats @ hw_w1 + hw_b1)),
            s.activation_exp(self._k("hw_hidden_out")))    # [1,C,3]
        hw_w2 = fake_quant_int8(self.hw_w2,
                                s.weight_exp(self._k("hw_w2")))
        hw_b2 = fake_quant_int32(
            self.hw_b2, s.activation_exp(self._k("hw_hidden_out"))
            + s.weight_exp(self._k("hw_w2")))
        hw_q16 = fake_quant_q816(hw_hidden @ hw_w2 + hw_b2)
        head_weights = fake_quant_q016(plan_sigmoid_float(hw_q16))  # [1,C,3]

        scores_t = head_scores.transpose(1, 2)            # [1,C,3]
        num = (scores_t * head_weights).sum(dim=-1)
        den = head_weights.sum(dim=-1)
        fused = torch.where(den == 0, scores_t.mean(dim=-1),
                            num / den.clamp(min=1e-12))
        fused = fake_quant_q016(fused)                      # [1,C]

        normal_rows = cand[:, :-1] if package_present else cand
        incoming = cand[:, -1:] if package_present else None
        keep = (fused >= KEEP_THRESHOLD).detach()
        keep_norm = keep[:, :normal_rows.shape[1]]           # [1,Nn]
        kept = normal_rows[:, keep_norm[0]]
        pruned = normal_rows[:, ~keep_norm[0]]
        parts = [pruned]
        sc = [fused[:, :normal_rows.shape[1]][:, ~keep_norm[0]]]
        if incoming is not None:
            parts.append(incoming)
            sc.append(fused[:, -1:])
        package = None
        if sum(p.shape[1] for p in parts) > 0:
            parts_t = torch.cat(parts, dim=1)
            sc_t = torch.cat(sc, dim=1)
            den2 = sc_t.sum(dim=1, keepdim=True)
            if (den2 == 0).any():
                package = fake_quant_int8(
                    parts_t.mean(dim=1, keepdim=True), a_exp)
            else:
                package = fake_quant_int8(
                    (parts_t * sc_t.unsqueeze(-1)).sum(dim=1, keepdim=True)
                    / den2.clamp(min=1e-12).unsqueeze(-1), a_exp)
        out = [x[:, :1], kept]
        if package is not None:
            out.append(package)
        return torch.cat(out, dim=1), package is not None, fused, keep_norm


def attach_selectors(qat_model, payload, table):
    """Attach the three frozen float mirrors to a QatDeiT (P4 step 1)."""
    device = next(qat_model.parameters()).device
    qat_model.selectors = nn.ModuleList(
        [QatSelector(payload["selectors"][i], table, i + 1)
         for i in range(3)]).to(device)
    return qat_model
