#!/usr/bin/env python3
"""P3 QAT: differentiable DeiT-T forward mirroring the RTL contract.

``QatDeiT`` holds the HeatViT-layout float tensors (the output of
tools/p2/p2_quantize.to_heatvit_tensors) as nn.Parameters and runs the
train path (decision D1, option A: float analytic nonlinearities +
straight-through fake-quant at the exact named contract boundaries):

  * weights/biases are fake-quantized to int8/int32 with the frozen
    ScaleTable exponents at every use;
  * activations are fake-quantized to int8 at the named contract points
    (act_patch_embed, act_tokens, b<n>_ln1_out, b<n>_qkv_out,
    b<n>_context_out, b<n>_msa_out, b<n>_y, b<n>_ln2_out, b<n>_hidden,
    b<n>_ffn_out, b<n>_out, final_ln_out);
  * attention scores are fake-quantized to Q8.16 with the 1/sqrt(64)
    folded in (score / 8), probabilities to UQ0.8;
  * FFN pre-activations are fake-quantized to Q8.16 and GELU is the
    float mirror x*sigmoid(1.6814x) of the RTL ShiftGELU-ln2;
  * LayerNorm is the exact two-pass D=192 float form with eps=1e-6
    (LN_EPS_Q32 = 10^-6), gamma/beta fake-quantized to int8;
  * logits stay float (CE loss); the int32 logit rounding at -14 is
    negligible for training.

The train path mirrors tools/p2/p2_sim_ivit.forward_batch_cfg op by op.
Bit-exact evaluation does NOT go through this module: ``exact_forward``
quantizes the same float tensors with the same table via
p2_quantize.build_model and runs p2_sim_ivit.forward_batch_cfg with the
contract config (the deployed RTL configuration). The structural
conformance tests in verification/tests/test_qat.py pin the wiring of the
two paths against each other.

Selectors are not part of this module (P1-P3 QAT runs prune=False; the
STE threshold/package path arrives in P4).
"""

import torch
import torch.nn as nn

from tools.p2.p2_sim import D, HEAD_DIM, HEADS, SELECTOR_BLOCKS
from tools.p2.qat_fakeq import (
    fake_quant_int8,
    fake_quant_int32,
    fake_quant_q816,
    fake_quant_uq08,
    shiftgelu_float,
)

LN_EPS = 1e-6        # RTL LN_EPS_Q32 = 10^-6 (Q16.32)
ATTN_SCALE = 8.0      # sqrt(64), folded into the Q8.16 conversion

_BLOCK_NAMES = ("gamma1", "beta1", "wqkv", "bqkv", "wproj", "bproj",
                "gamma2", "beta2", "w1", "b1", "w2", "b2")

PARAM_NAMES = [
    "patch_w", "patch_b", "cls", "pos",
    *[f"b{n}_{name}" for n in range(1, 13) for name in _BLOCK_NAMES],
    "final_gamma", "final_beta", "head_w", "head_b",
]


def _rec(rec, name, tensor):
    if rec is not None:
        rec[name] = tensor.detach().cpu().clone()


class QatDeiT(nn.Module):
    """Differentiable QAT forward in HeatViT layout (no selectors)."""

    def __init__(self, floats, table):
        super().__init__()
        self.table = table
        self.selectors = None        # nn.ModuleList attached by P4 (optional)
        missing = [n for n in PARAM_NAMES if n not in floats]
        if missing:
            raise ValueError(f"missing tensors for QatDeiT: {missing}")
        for name in PARAM_NAMES:
            self.register_parameter(name, nn.Parameter(
                floats[name].detach().float().clone()))

    # ---- helpers -----------------------------------------------------------
    def tensors_dict(self):
        """Detached CPU dict in HeatViT layout (checkpoint / exact eval)."""
        return {name: p.detach().cpu() for name, p in self.named_parameters()}

    def _w(self, name):
        """fake-quantized weight, dequantized (value space)."""
        return fake_quant_int8(getattr(self, name),
                               self.table.weight_exp(name))

    def _gemm(self, a, w_name, b_name, a_exp, dst_name=None):
        """int8-fake-quantized GEMM + int32-fake-quantized bias; the
        result is fake-quantized to ``dst_name`` unless it is None (the
        head keeps float logits, the FFN keeps the float accumulator)."""
        w_exp = self.table.weight_exp(w_name)
        acc = a @ self._w(w_name)
        if b_name is not None:
            acc = acc + fake_quant_int32(getattr(self, b_name),
                                         a_exp + w_exp)
        if dst_name is None:
            return acc
        return fake_quant_int8(acc, self.table.activation_exp(dst_name))

    def _ln(self, x, gamma_name, beta_name, out_name):
        """Float two-pass D=192 LayerNorm with eps=1e-6 (RTL mirror).

        The RTL computes mean/variance on the int8 input in Q32 and
        applies the int8-quantized gamma/beta; the float mirror is the
        same algebra in value space with fake-quantized gamma/beta and a
        fake-quantized int8 output.
        """
        g = self._w(gamma_name)
        b = self._w(beta_name)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + LN_EPS)
        return fake_quant_int8(norm * g + b,
                               self.table.activation_exp(out_name))

    def _mhsa(self, x, n, rec):
        s = self.table
        ln1 = self._ln(x, f"b{n}_gamma1", f"b{n}_beta1", f"b{n}_ln1_out")
        _rec(rec, f"b{n}_ln1_out", ln1)
        fused = self._gemm(ln1, f"b{n}_wqkv", f"b{n}_bqkv",
                           s.activation_exp(f"b{n}_ln1_out"),
                           f"b{n}_qkv_out")
        _rec(rec, f"b{n}_qkv_out", fused)
        q = fused[..., :D]
        k = fused[..., D:2 * D]
        v = fused[..., 2 * D:]
        ctx_exp = s.activation_exp(f"b{n}_context_out")
        q3 = q.reshape(*q.shape[:-1], HEADS, HEAD_DIM)
        k3 = k.reshape(*k.shape[:-1], HEADS, HEAD_DIM)
        v3 = v.reshape(*v.shape[:-1], HEADS, HEAD_DIM)
        score = torch.einsum("bnhd,bmhd->bhnm", q3, k3) / ATTN_SCALE
        prob = fake_quant_uq08(torch.softmax(fake_quant_q816(score),
                                             dim=-1))
        ctx = fake_quant_int8(torch.einsum("bhnm,bmhd->bnhd", prob, v3),
                              ctx_exp)
        concat = ctx.reshape(*ctx.shape[:2], D)
        _rec(rec, f"b{n}_context_out", concat)
        msa_out = self._gemm(concat, f"b{n}_wproj", f"b{n}_bproj",
                             ctx_exp, f"b{n}_msa_out")
        _rec(rec, f"b{n}_msa_out", msa_out)
        return msa_out

    def _ffn(self, y, n, rec):
        s = self.table
        ln2 = self._ln(y, f"b{n}_gamma2", f"b{n}_beta2", f"b{n}_ln2_out")
        _rec(rec, f"b{n}_ln2_out", ln2)
        acc = self._gemm(ln2, f"b{n}_w1", f"b{n}_b1",
                         s.activation_exp(f"b{n}_ln2_out"), None)
        hidden = fake_quant_int8(
            fake_quant_q816(shiftgelu_float(fake_quant_q816(acc))),
            s.activation_exp(f"b{n}_hidden"))
        _rec(rec, f"b{n}_hidden", hidden)
        ffn_out = self._gemm(hidden, f"b{n}_w2", f"b{n}_b2",
                             s.activation_exp(f"b{n}_hidden"),
                             f"b{n}_ffn_out")
        _rec(rec, f"b{n}_ffn_out", ffn_out)
        return ffn_out

    def _block(self, x, n, x_exp, rec):
        s = self.table
        msa_out = self._mhsa(x, n, rec)
        y = fake_quant_int8(x + msa_out, s.activation_exp(f"b{n}_y"))
        _rec(rec, f"b{n}_y", y)
        ffn_out = self._ffn(y, n, rec)
        z = fake_quant_int8(y + ffn_out, s.activation_exp(f"b{n}_out"))
        _rec(rec, f"b{n}_out", z)
        return z

    # ---- forward -----------------------------------------------------------
    def forward(self, images, rec=None, prune=False, return_rates=False):
        """Train path: float value space + STE fake-quant at contract points.

        images: float [B, 3, 224, 224]. Returns float logits [B, 1000].
        ``rec`` optionally collects the dequantized named activations
        (float) for structural conformance diagnostics.

        ``prune=True`` (P4) inserts the frozen Token Selector float mirrors
        before blocks 4/7/10: hard 0.5 keep threshold (detached, STE),
        pruned rows compressed into the fused-score-weighted package
        token. Token counts become per-image dynamic, so a batch is
        processed image by image. ``return_rates=True`` additionally
        returns the per-stage soft token counts [3] (summed over the
        batch; STE via the fused scores) for keep-rate regularization.
        """
        if prune:
            if self.selectors is None:
                raise ValueError("prune=True requires attached selectors "
                                 "(qat_selector.attach_selectors)")
            if images.shape[0] > 1:
                outs, all_rates = [], []
                for i in range(images.shape[0]):
                    if return_rates:
                        o, r = self._forward_body(images[i:i + 1], rec,
                                                  True, True)
                        outs.append(o)
                        all_rates.append(r)
                    else:
                        outs.append(self._forward_body(images[i:i + 1], rec,
                                                       True, False))
                logits = torch.cat(outs, dim=0)
                if return_rates:
                    return logits, [sum(r[k] for r in all_rates)
                                    for k in range(3)]
                return logits
            return self._forward_body(images, rec, prune, return_rates)
        return self._forward_body(images, rec, prune, return_rates)

    def _forward_body(self, images, rec, prune, return_rates=False):
        s = self.table
        rates = []
        inp_exp = s.activation_exp("input")
        img_q = fake_quant_int8(images, inp_exp)
        _rec(rec, "input", img_q)
        nhwc = img_q.permute(0, 2, 3, 1)
        patches = nhwc.reshape(img_q.shape[0], 14, 16, 14, 16, 3) \
            .permute(0, 1, 3, 2, 4, 5).reshape(img_q.shape[0], 196, 768)
        _rec(rec, "act_patch_matrix", patches)
        embed = self._gemm(patches, "patch_w", "patch_b", inp_exp,
                           "act_patch_embed")
        _rec(rec, "act_patch_embed", embed)
        cls_q = self._w("cls")
        pos_q = self._w("pos")
        out_exp = s.activation_exp("act_tokens")
        bsz = images.shape[0]
        row0 = fake_quant_int8(
            cls_q.expand(bsz, 1, -1) + pos_q[:1].expand(bsz, 1, -1),
            out_exp)
        rows1 = fake_quant_int8(
            embed + pos_q[1:].expand(bsz, 196, -1), out_exp)
        tokens = torch.cat([row0, rows1], dim=1)
        _rec(rec, "act_tokens", tokens)
        package_present = False
        sel_idx = 0
        for n in range(1, 13):
            x_exp = s.activation_exp("act_tokens") if n == 1 \
                else s.activation_exp(f"b{n - 1}_out")
            if prune and n in SELECTOR_BLOCKS:
                sel_idx += 1
                tokens, package_present, fused, keep_mask = \
                    self.selectors[sel_idx - 1](
                        tokens, self.table, x_exp, package_present)
                if return_rates:
                    # soft token count (STE through fused): CLS + kept +
                    # package; gradients flow via the fused scores.
                    soft = keep_mask.float() \
                        + fused[:, :keep_mask.shape[1]] \
                        - fused[:, :keep_mask.shape[1]].detach()
                    rates.append(1.0 + soft.sum()
                                 + (1.0 if package_present else 0.0))
            if self.training and torch.is_grad_enabled() \
                    and tokens.requires_grad and not prune:
                # Gradient checkpointing (unpruned batched path only):
                # the eager graph retains ~110 MiB per image (~14 GiB at
                # batch 128 — see tools/p2/qat_memprobe.py), spilling past
                # the 8 GiB VRAM into shared system memory and making the
                # backward pass PCIe-bound (measured ~20-90 s/step).
                # Checkpointing keeps only the block outputs (O(1) blocks)
                # and recomputes one block's forward during backward
                # (~0.5 s total; batch 128 backward 22.96 s -> 0.89 s).
                # The pruned path processes one image at a time, so the
                # per-image graph is tiny and checkpointing would only
                # double the forward work.
                tokens = torch.utils.checkpoint.checkpoint(
                    self._block, tokens, n, x_exp, None,
                    use_reentrant=False)
            else:
                tokens = self._block(tokens, n, x_exp, rec)
        final_ln = self._ln(tokens, "final_gamma", "final_beta",
                            "final_ln_out")
        _rec(rec, "final_ln_out", final_ln)
        logits = self._gemm(final_ln[:, :1], "head_w", "head_b",
                            s.activation_exp("final_ln_out"), None)
        if return_rates:
            return logits[:, 0, :], rates
        return logits[:, 0, :]


def exact_forward(qat_model, images, rec=None):
    """Bit-exact integer forward of the same tensors + table.

    Quantizes ``qat_model``'s float tensors with p2_quantize.build_model
    and runs p2_sim_ivit.forward_batch_cfg with the contract NonlinConfig
    (the deployed RTL configuration). Returns (logits_int32 [B, 1000],
    logits_scale_exp).
    """
    from tools.p2.p2_quantize import build_model
    from tools.p2.p2_sim_ivit import NonlinConfig, forward_batch_cfg
    model = build_model(qat_model.tensors_dict(), qat_model.table,
                        images.device)
    return forward_batch_cfg(model, images.detach().float(), NonlinConfig(),
                             rec)
