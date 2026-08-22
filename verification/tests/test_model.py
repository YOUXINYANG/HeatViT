import unittest
from unittest import mock

from verification.heatvit_ref.model import (
    HeatViTModel,
    HeatViTParams,
    ModelResult,
)
from verification.heatvit_ref.weights import SEED, build_params

# The full deterministic image used by the end-to-end generator.
import random as _random


def det_image():
    rng = _random.Random(SEED)
    return [rng.randint(-128, 127) for _ in range(224 * 224 * 3)]


_CACHE = {}


def build_once():
    if "params" not in _CACHE:
        _CACHE["params"], _CACHE["summary"], _CACHE["counts"] = \
            build_params(det_image())
    return _CACHE["params"], _CACHE["summary"], _CACHE["counts"]


class ModelTest(unittest.TestCase):
    def test_call_order_with_reduced_config(self):
        # Reduced configuration: verify the pipeline order and checkpoint
        # keys with mocked integer layers.
        model = HeatViTModel()
        params = mock.Mock(spec=HeatViTParams)
        params.patch = mock.Mock()
        params.blocks = [mock.Mock() for _ in range(2)]
        params.selectors = [mock.Mock() for _ in range(1)]
        params.selector_blocks = (2,)
        params.final_gamma = [64] * 192
        params.final_beta = [0] * 192
        params.head_w = [[0] * 1000 for _ in range(192)]
        params.head_b = [0] * 1000
        params.final_gamma_scale_exp = -6
        params.final_beta_scale_exp = -7
        params.final_ln_out_scale_exp = -7
        params.blocks[-1].ffn.out_scale_exp = -7

        with mock.patch(
            "verification.heatvit_ref.model.patch_embedding",
            return_value=[[0] * 192 for _ in range(5)],
        ) as patch_fn, mock.patch(
            "verification.heatvit_ref.model.transformer_block",
            side_effect=lambda x, p: ([list(r) for r in x], {}),
        ) as block_fn, mock.patch(
            "verification.heatvit_ref.model.token_selector",
            side_effect=lambda x, pkg, p: mock.Mock(
                tokens=[list(r) for r in x],
                package_present=pkg,
                kept_normal_count=4,
                pruned_normal_count=0,
            ),
        ) as selector_fn:
            result = model.infer([0] * 150528, params)

        # Order: patch -> block1 -> block2 -> selector1 -> (blocks done).
        self.assertEqual(patch_fn.call_count, 1)
        self.assertEqual(block_fn.call_count, 2)
        self.assertEqual(selector_fn.call_count, 1)
        self.assertEqual(result.output_scale_exp, -14)
        self.assertIsInstance(result, ModelResult)
        for key in ("patch", "block_01", "block_02", "selector_01",
                    "final_ln", "logits"):
            self.assertIn(key, result.checkpoints)
        self.assertEqual(len(result.checkpoints["logits"]), 1000)

    def test_full_config_checkpoints_and_selectors(self):
        image = det_image()
        params, summary, token_counts = build_once()
        model = HeatViTModel()
        result = model.infer(image, params)

        mandatory = (["patch"] +
                     [f"block_{b:02d}" for b in range(1, 13)] +
                     [f"selector_{s:02d}" for s in range(1, 4)] +
                     ["final_ln", "logits"])
        self.assertEqual(sorted(result.checkpoints), sorted(mandatory))
        for s in range(1, 4):
            rows = result.checkpoints[f"selector_{s:02d}"]
            self.assertLessEqual(len(rows), 197)
            self.assertEqual(len(rows[0]), 192)
        self.assertEqual(len(result.checkpoints["logits"]), 1000)
        self.assertEqual(result.output_scale_exp, -14)
        self.assertEqual(len(result.selector_summary), 3)

        # The three selectors must each prune at least two normal tokens and
        # keep at least one, and token counts must strictly decrease.
        for entry in result.selector_summary:
            self.assertGreaterEqual(entry["kept_normal"], 1)
            self.assertGreaterEqual(entry["pruned_normal"], 2)
            self.assertEqual(entry["output_tokens"],
                             1 + entry["kept_normal"] + 1)
        self.assertEqual(token_counts[0], 197)
        for a, b in zip(token_counts, token_counts[1:]):
            self.assertLess(b, a)
        self.assertEqual(token_counts,
                         [e["input_tokens"] for e in result.selector_summary[0:1]] +
                         [e["output_tokens"] for e in result.selector_summary])

    def test_params_immutable_and_counts(self):
        params, summary, token_counts = build_once()
        self.assertEqual(len(params.blocks), 12)
        self.assertEqual(len(params.selectors), 3)
        self.assertEqual(len(params.final_gamma), 192)
        self.assertEqual(len(params.head_w), 192)
        self.assertEqual(len(params.head_w[0]), 1000)
        self.assertEqual(len(params.head_b), 1000)
        with self.assertRaises(Exception):
            params.blocks = ()

    def test_calibration_summary_fields(self):
        params, summary, token_counts = build_once()
        self.assertEqual(len(summary), 3)
        for idx, entry in enumerate(summary):
            self.assertEqual(entry["stage"], idx + 1)
            self.assertGreaterEqual(entry["attempt"], 0)
            self.assertLess(entry["attempt"], 256)
            self.assertGreaterEqual(entry["kept_normal"], 1)
            self.assertGreaterEqual(entry["pruned_normal"], 2)


if __name__ == "__main__":
    unittest.main()
