#!/usr/bin/env python3
"""P3 QAT unit tests (torch venv).

Covers the P0 acceptance criteria:

  1. STE fake-quant primitives (grid, clip, straight-through gradient);
  2. the float shift-exp GELU mirror vs the integer contract GELU;
  3. the exact-eval pipeline (QatDeiT float tensors -> p2_quantize.
     build_model -> p2_sim_ivit.forward_batch_cfg) is bit-exact by
     construction, and the train forward is structurally conformant with
     it at every named contract point (same shapes, values on the exact
     contract grids, bounded bin distance);
  4. gradients flow through every parameter (STE applied everywhere).

The train path is NOT bit-exact against the integer path by design
(decision D1, option A: float analytic ops + fp32 accumulation vs fp64
exact integer accumulation), so the conformance test compares values on
the contract grid with depth-scaled tolerances measured on the fixed
random seed. Bit-exactness itself is guaranteed by exact_forward().

Run (torch venv):
  .venv-torch\\Scripts\\python -m unittest verification.tests.test_qat -v
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.p2.qat_fakeq import (
    fake_quant_int8,
    fake_quant_int32,
    fake_quant_q816,
    fake_quant_uq08,
    shiftgelu_float,
)
from tools.p2.qat_model import PARAM_NAMES, QatDeiT, exact_forward
from tools.p2.scale_table import ACTIVATION_NAMES, WEIGHT_NAMES, ScaleTable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Measured on the fixed seed (tools/p2/qat_probe.py): the train path drifts
# from the exact path by a growing number of contract-grid bins per block
# (fp32 vs fp64 accumulation + float LN + softmax approximation), reaching
# ~2.4 mean / ~26 max bins at block 12. The bounds below carry ~50% margin.
MEAN_BINS_LIMIT = 3.5
MAX_BINS_LIMIT = 45.0
LOGITS_MEAN_REL_LIMIT = 0.05
LOGITS_MAX_REL_LIMIT = 0.20


def make_floats(seed=20260815):
    """Random HeatViT-layout float tensors (small-signal, like run_unit)."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, std=0.05):
        return torch.randn(*shape, generator=g) * std

    floats = {
        "patch_w": r(768, 192, std=0.04), "patch_b": r(192, std=0.1),
        "cls": r(192, std=0.05), "pos": r(197, 192, std=0.05),
    }
    for n in range(1, 13):
        floats[f"b{n}_gamma1"] = 0.8 + r(192, std=0.05)
        floats[f"b{n}_beta1"] = r(192, std=0.05)
        floats[f"b{n}_wqkv"] = r(192, 576, std=0.04)
        floats[f"b{n}_bqkv"] = r(576, std=0.1)
        floats[f"b{n}_wproj"] = r(192, 192, std=0.04)
        floats[f"b{n}_bproj"] = r(192, std=0.1)
        floats[f"b{n}_gamma2"] = 0.8 + r(192, std=0.05)
        floats[f"b{n}_beta2"] = r(192, std=0.05)
        floats[f"b{n}_w1"] = r(192, 768, std=0.04)
        floats[f"b{n}_b1"] = r(768, std=0.1)
        floats[f"b{n}_w2"] = r(768, 192, std=0.04)
        floats[f"b{n}_b2"] = r(192, std=0.1)
    floats["final_gamma"] = 0.8 + r(192, std=0.05)
    floats["final_beta"] = r(192, std=0.05)
    floats["head_w"] = r(192, 1000, std=0.04)
    floats["head_b"] = r(1000, std=0.1)
    return floats


def make_table(exp=-7):
    return ScaleTable(weights={n: exp for n in WEIGHT_NAMES},
                      activations={n: exp for n in ACTIVATION_NAMES})


class FakeQuantTest(unittest.TestCase):
    def test_int8_grid_and_clip(self):
        x = torch.tensor([-3.0, -1.3, -0.25, 0.0, 0.25, 0.49, 0.51, 1.7, 5.0])
        out = fake_quant_int8(x, -2)          # grid 0.25, range +-32
        want = torch.tensor(
            [-3.0, -1.25, -0.25, 0.0, 0.25, 0.5, 0.5, 1.75, 5.0])
        self.assertTrue(torch.equal(out, want), f"got {out}")
        big = fake_quant_int8(torch.tensor([-40.0, 40.0]), -2)
        self.assertTrue(torch.equal(big, torch.tensor([-32.0, 31.75])))

    def test_q816_grid_and_clip(self):
        step = 2.0 ** -16
        x = torch.tensor([0.0, step, 3 * step, 130.0, -130.0])
        out = fake_quant_q816(x)
        want = torch.tensor(
            [0.0, step, 3 * step, 127.99998474121094, -128.0])
        self.assertTrue(torch.allclose(out, want, atol=1e-9))

    def test_uq08_grid_and_clip(self):
        x = torch.tensor([-0.1, 0.0, 0.001, 0.996, 1.5])
        out = fake_quant_uq08(x)
        want = torch.tensor([0.0, 0.0, 0.0, 0.99609375, 0.99609375])
        self.assertTrue(torch.allclose(out, want, atol=1e-9))

    def test_int32_range(self):
        out = fake_quant_int32(torch.tensor([-3.0e9, 3.0e9]), 0)
        self.assertTrue(torch.equal(out, torch.tensor(
            [float(-(1 << 31)), float((1 << 31) - 1)])))

    def test_ste_gradient(self):
        x = torch.randn(64, requires_grad=True)
        out = fake_quant_int8(x, -3).sum()
        out.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)),
                        "STE backward must be identity")

    def test_ste_gradient_q816_uq08(self):
        x = torch.randn(64, requires_grad=True)
        (fake_quant_q816(x).sum() + fake_quant_uq08(x).sum()).backward()
        self.assertTrue(torch.allclose(x.grad, torch.full_like(x, 2.0)))


class GeluMirrorTest(unittest.TestCase):
    def test_matches_contract_gelu(self):
        """The float shift-exp mirror tracks the integer ShiftGELU-ln2
        (differences are the dropped Q16 integer roundings only)."""
        from tools.p2.p2_sim import gelu_q16
        xs = (torch.linspace(-8.0, 8.0, 16385) * 65536).round().clamp(
            -(1 << 23), (1 << 23) - 1).to(torch.int64)
        ref = gelu_q16(xs).to(torch.float64) / 65536.0
        got = shiftgelu_float(xs.to(torch.float64) / 65536.0).double()
        err = (got - ref).abs()
        self.assertLess(err.mean().item(), 5e-4, "mean GELU mirror error")
        self.assertLess(err.max().item(), 1e-2, "max GELU mirror error")


class QatWiringTest(unittest.TestCase):
    """Small random model: exact-eval pipeline + train-path conformance."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20260815)
        cls.floats = make_floats()
        cls.table = make_table()
        cls.qat = QatDeiT(cls.floats, cls.table).to(DEVICE)
        cls.images = (torch.rand(2, 3, 224, 224) * 0.9 - 0.45).to(DEVICE)

    def test_param_names_complete(self):
        for name in PARAM_NAMES:
            self.assertIn(name, dict(self.qat.named_parameters()))
        self.assertEqual(len(list(self.qat.parameters())), len(PARAM_NAMES))

    def test_exact_forward_pipeline(self):
        """exact_forward == build_model + forward_batch_cfg, bit-exact."""
        from tools.p2.p2_quantize import build_model
        from tools.p2.p2_sim_ivit import NonlinConfig, forward_batch_cfg
        rec1, rec2 = {}, {}
        l1, e1 = exact_forward(self.qat, self.images, rec1)
        model = build_model(self.qat.tensors_dict(), self.qat.table, DEVICE)
        l2, e2 = forward_batch_cfg(model, self.images, NonlinConfig(), rec2)
        self.assertEqual(e1, e2)
        self.assertTrue(torch.equal(l1, l2), "exact pipeline bit mismatch")
        for name, t in rec1.items():
            self.assertTrue(torch.equal(t, rec2[name]), name)

    def _boundary_bins(self):
        with torch.no_grad():
            rec_t = {}
            logits_t = self.qat(self.images, rec_t)
            rec_e = {}
            logits_e, scale_e = exact_forward(self.qat, self.images, rec_e)
        return rec_t, rec_e, logits_t, logits_e, scale_e

    def test_train_path_shapes_and_grid(self):
        rec_t, _, logits_t, _, _ = self._boundary_bins()
        self.assertEqual(logits_t.shape, (2, 1000))
        s = self.table
        for name, t in rec_t.items():
            if name == "input" or name == "act_patch_matrix":
                exp = s.activation_exp("input")
            else:
                exp = s.activation_exp(name)
            step = 2.0 ** exp
            # every boundary value lies exactly on the int8 contract grid
            q = t / step
            self.assertTrue(torch.equal(q, torch.round(q)),
                            f"{name} not on grid {step}")
            self.assertTrue(q.min() >= -128 and q.max() <= 127,
                            f"{name} out of int8 range")

    def test_train_path_conformance_bins(self):
        rec_t, rec_e, logits_t, logits_e, scale_e = self._boundary_bins()
        s = self.table
        block_names = []
        for n in range(1, 13):
            for name in ("ln1_out", "qkv_out", "context_out", "msa_out",
                         "y", "ln2_out", "hidden", "ffn_out", "out"):
                block_names.append(f"b{n}_{name}")
        worst_mean, worst_max = 0.0, 0.0
        for name in block_names:
            self.assertIn(name, rec_e, f"exact rec missing {name}")
            step = 2.0 ** s.activation_exp(name)
            bins = ((rec_t[name] - rec_e[name].to(torch.float32) * step)
                    .abs() / step)
            worst_mean = max(worst_mean, float(bins.mean()))
            worst_max = max(worst_max, float(bins.max()))
        self.assertLess(worst_mean, MEAN_BINS_LIMIT,
                        f"mean bin distance {worst_mean:.3f}")
        self.assertLess(worst_max, MAX_BINS_LIMIT,
                        f"max bin distance {worst_max:.1f}")
        rel = ((logits_t.cpu() - logits_e.cpu().to(torch.float32)
                * (2.0 ** scale_e)).abs()
               / (logits_e.cpu().abs().to(torch.float32)
                  * (2.0 ** scale_e) + 1.0))
        self.assertLess(rel.mean().item(), LOGITS_MEAN_REL_LIMIT)
        self.assertLess(rel.max().item(), LOGITS_MAX_REL_LIMIT)

    def test_train_path_early_boundaries(self):
        """act_patch_embed / act_tokens / final_ln_out against the
        single-image integer path (forward_image_cfg records them)."""
        from tools.p2.p2_quantize import build_model
        from tools.p2.p2_sim_ivit import NonlinConfig, forward_image_cfg
        s = self.table
        with torch.no_grad():
            rec_t = {}
            self.qat(self.images[:1], rec_t)
            model = build_model(self.qat.tensors_dict(), self.qat.table,
                                DEVICE)
            rec_i = {}
            forward_image_cfg(model, self.images[0], NonlinConfig(),
                              rec_i, prune=False)
        for name in ("act_patch_embed", "act_tokens", "final_ln_out"):
            step = 2.0 ** s.activation_exp(name)
            bins = ((rec_t[name][0] - rec_i[name].to(torch.float32) * step)
                    .abs() / step)
            self.assertLess(bins.mean().item(), MEAN_BINS_LIMIT, name)
            self.assertLess(bins.max().item(), MAX_BINS_LIMIT, name)

    def test_gradients_flow(self):
        images = self.images.detach().clone()
        logits = self.qat(images)
        loss = torch.nn.functional.cross_entropy(
            logits, torch.tensor([17, 392], device=DEVICE))
        loss.backward()
        for name, p in self.qat.named_parameters():
            self.assertIsNotNone(p.grad, f"no grad for {name}")
            self.assertTrue(torch.isfinite(p.grad).all(), f"bad grad {name}")
            self.assertGreater(p.grad.abs().sum().item(), 0.0,
                               f"zero grad for {name}")


class TimmMappingTest(unittest.TestCase):
    """heatvit_to_timm_state must be the exact inverse of
    to_heatvit_tensors (needed by the recalib hooks)."""

    def test_heatvit_timm_roundtrip(self):
        from tools.p2.p2_quantize import to_heatvit_tensors
        from tools.p2.qat_data import heatvit_to_timm_state
        floats = make_floats()
        back = to_heatvit_tensors(heatvit_to_timm_state(floats))
        self.assertEqual(set(back), set(floats))
        for name, t in floats.items():
            self.assertTrue(torch.equal(back[name], t), name)

    def test_real_checkpoint_roundtrip(self):
        from tools.p2.p2_quantize import (
            CHECKPOINT, load_state_dict, to_heatvit_tensors)
        if not Path(CHECKPOINT).exists():
            self.skipTest("real checkpoint missing")
        from tools.p2.qat_data import heatvit_to_timm_state
        floats = to_heatvit_tensors(load_state_dict())
        back = to_heatvit_tensors(heatvit_to_timm_state(floats))
        for name, t in floats.items():
            self.assertTrue(torch.equal(back[name], t), name)


class RealCheckpointSmokeTest(unittest.TestCase):
    """Wiring against the real DeiT-T checkpoint + legacy scale table.

    Skipped when the checkpoint or the table is missing. Real weights
    carry larger dynamic range, so the fp32-vs-fp64 drift compounds more
    than in the random model; the assertions are prediction-level
    (argmax / top-5 agreement between the train path and the bit-exact
    integer path), which is the wiring signal that matters.
    """

    def test_real_checkpoint_conformance(self):
        from tools.p2.p2_quantize import (
            CHECKPOINT, load_state_dict, make_val_loader, to_heatvit_tensors)
        table_path = REPO_ROOT / "p2_out" / "scale_table.json"
        if not Path(CHECKPOINT).exists() or not table_path.exists():
            self.skipTest("real checkpoint or scale table missing")
        state = load_state_dict()
        floats = to_heatvit_tensors(state)
        table = ScaleTable.load(table_path)
        # LN input contract check for the exact path
        for name in ["act_tokens", *[f"b{n}_y" for n in range(1, 13)],
                     *[f"b{n}_out" for n in range(1, 13)]]:
            self.assertLessEqual(table.activations[name], 0, name)
        qat = QatDeiT(floats, table).to(DEVICE)
        loader = make_val_loader(8, batch_size=8, shuffle=False)
        images, _ = next(iter(loader))
        images = images.to(DEVICE)
        with torch.no_grad():
            logits_t = qat(images)
            logits_e, scale_e = exact_forward(qat, images)
        pred_t = logits_t.argmax(dim=1).cpu()
        pred_e = logits_e.argmax(dim=1).cpu()
        agree = (pred_t == pred_e).sum().item()
        self.assertGreaterEqual(agree, 6, f"argmax agreement {agree}/8")
        t5_t = logits_t.topk(5, dim=1).indices.cpu()
        t5_e = logits_e.topk(5, dim=1).indices.cpu()
        overlap = sum(len(set(a.tolist()) & set(b.tolist()))
                      for a, b in zip(t5_t, t5_e))
        self.assertGreaterEqual(overlap, 32,
                                f"top-5 overlap {overlap}/40")
        rel = ((logits_t.cpu() - logits_e.cpu().to(torch.float32)
                * (2.0 ** scale_e)).abs()
               / (logits_e.cpu().abs().to(torch.float32)
                  * (2.0 ** scale_e) + 1.0))
        self.assertLess(rel.mean().item(), 0.4,
                        f"real-checkpoint logits mean|rel|={rel.mean():.4f}")
        self.assertLess(rel.max().item(), 4.0,
                        f"real-checkpoint logits max|rel|={rel.max():.4f}")


class PrunedSelectorTest(unittest.TestCase):
    """P4 step 1: frozen Token Selector float mirror + pruned train forward.

    Uses the real P2-C selector checkpoint (p2_out/selectors_sup4.pt) and
    the legacy backbone table; skipped when either is missing.
    """

    SELECTOR_PATH = REPO_ROOT / "p2_out" / "selectors_sup4.pt"
    TABLE_PATH = REPO_ROOT / "p2_out" / "scale_table.json"

    @classmethod
    def setUpClass(cls):
        from tools.p2.qat_selector import merge_selector_scales
        if not cls.SELECTOR_PATH.exists() or not cls.TABLE_PATH.exists():
            raise unittest.SkipTest("selector checkpoint or table missing")
        cls.payload = torch.load(cls.SELECTOR_PATH, map_location="cpu",
                                 weights_only=False)
        cls.table = ScaleTable.load(cls.TABLE_PATH)
        merge_selector_scales(cls.table, cls.payload)

    def _tokens(self, seed=7, n=200):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(-127, 128, (1, n, 192), generator=g) \
            .to(torch.int8)

    def test_mirror_conformance(self):
        from tools.p2.p2_sim import SelectorP, token_selector
        from tools.p2.qat_selector import QatSelector
        for idx in (1, 2, 3):
            in_exp = self.table.activation_exp(f"b{3 * idx}_out")
            x_int = self._tokens(seed=7 + idx)
            sel = QatSelector(self.payload["selectors"][idx - 1],
                              self.table, idx).to(DEVICE)
            p = SelectorP(
                **{k: v.to(DEVICE)
                   for k, v in self.payload["selectors"][idx - 1].items()})
            with torch.no_grad():
                tok_int, pkg_int = token_selector(
                    x_int[0].to(DEVICE), False, p, self.table, idx, in_exp)
                x_f = x_int.to(DEVICE).float() * (2.0 ** in_exp)
                tok_f, pkg_f, _, _ = sel(x_f, self.table, in_exp, False)
            self.assertEqual(pkg_int, pkg_f, f"idx{idx} package flag")
            tok_fi = tok_f.squeeze(0) / (2.0 ** in_exp)
            # counts may differ by a few threshold-flip tokens
            self.assertLessEqual(abs(tok_int.shape[0] - tok_f.shape[1]),
                                 6, f"idx{idx} token count drift")
            if tok_int.shape[0] == tok_f.shape[1]:
                # kept-set overlap: tokens whose fused score sits within
                # ~1e-5 of 0.5 can flip between the two paths (Q16 tie
                # rounding); require >= 95% agreement both ways.
                n = tok_int.shape[0]
                d = (tok_int.float().unsqueeze(1)
                     - tok_fi.float().unsqueeze(0)).abs().max(
                    dim=-1).values
                ok_i = (d <= 1.5).sum(dim=1).gt(0).sum().item()
                ok_f = (d <= 1.5).sum(dim=0).gt(0).sum().item()
                self.assertGreaterEqual(ok_i, int(0.95 * n),
                                        f"idx{idx} int rows matched")
                self.assertGreaterEqual(ok_f, int(0.95 * n),
                                        f"idx{idx} float rows matched")

    def test_pruned_forward_gradient_flow(self):
        from tools.p2.qat_selector import attach_selectors
        qat = QatDeiT(make_floats(), self.table).to(DEVICE)
        attach_selectors(qat, self.payload, self.table)
        qat.train()
        img = torch.randn(2, 3, 224, 224, device=DEVICE)
        out = qat(img, prune=True)
        self.assertEqual(out.shape, (2, 1000))
        out.sum().backward()
        self.assertIsNotNone(qat.head_w.grad)
        self.assertIsNotNone(qat.b1_wqkv.grad)
        self.assertGreater(qat.head_w.grad.abs().sum().item(), 0.0)
        self.assertGreater(qat.b1_wqkv.grad.abs().sum().item(), 0.0)
        # frozen selectors: registered as buffers, not parameters
        self.assertEqual(sum(1 for _ in qat.selectors.parameters()), 0)

    def test_rate_loss_flow(self):
        """P4-2: return_rates gives 3 STE soft counts; the (rate-target)^2
        loss flows gradients back into the backbone feature path."""
        from tools.p2.qat_selector import attach_selectors
        qat = QatDeiT(make_floats(), self.table).to(DEVICE)
        attach_selectors(qat, self.payload, self.table)
        qat.train()
        img = torch.randn(2, 3, 224, 224, device=DEVICE)
        logits, rates = qat(img, prune=True, return_rates=True)
        self.assertEqual(logits.shape, (2, 1000))
        self.assertEqual(len(rates), 3)
        for r in rates:
            self.assertTrue(r.requires_grad)
        loss = sum(((r / 2.0 - t) / 197.0) ** 2
                   for r, t in zip(rates, (88.0, 45.0, 32.0)))
        loss.backward()
        self.assertIsNotNone(qat.b1_wqkv.grad)
        self.assertGreater(qat.b1_wqkv.grad.abs().sum().item(), 0.0)

    def test_pruned_eval_wiring(self):
        """The P4 bit-exact pruned eval helper runs on real QAT weights."""
        from tools.p2.p2_quantize import load_state_dict, to_heatvit_tensors
        from tools.p2 import p2_qat
        floats = to_heatvit_tensors(load_state_dict())
        qat = QatDeiT(floats, self.table).to(DEVICE)
        acc, correct, total = p2_qat.eval_pruned_exact(
            qat, self.SELECTOR_PATH, 32, DEVICE)
        self.assertEqual(total, 32)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
