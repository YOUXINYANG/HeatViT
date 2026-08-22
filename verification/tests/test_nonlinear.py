import unittest

from verification.heatvit_ref.nonlinear import (
    PLAN_BP_1_Q16,
    PLAN_BP_2_Q16,
    PLAN_BP_3_Q16,
    gelu,
    layernorm,
    plan_sigmoid,
    softmax_attention,
    softmax_selector,
)


Q8_16_MIN = -(1 << 23)
Q8_16_MAX = (1 << 23) - 1
Q0_16_ONE = 1 << 16


class GeluTest(unittest.TestCase):
    def test_zero_maps_to_zero(self):
        self.assertEqual(gelu(0), 0)

    def test_known_quantized_values(self):
        self.assertEqual(gelu(0), 0)
        self.assertEqual(gelu(1), 1)
        self.assertEqual(gelu(-1), 0)
        self.assertEqual(gelu(32768), 19836)
        self.assertEqual(gelu(-32768), -12932)
        self.assertEqual(gelu(163965), 122974)
        self.assertEqual(gelu(-163965), -40991)
        self.assertEqual(gelu(8388607), 6291455)
        self.assertEqual(gelu(-8388608), -2097152)

    def test_min_max_stay_in_range(self):
        for x in (Q8_16_MIN, Q8_16_MAX):
            self.assertGreaterEqual(gelu(x), Q8_16_MIN)
            self.assertLessEqual(gelu(x), Q8_16_MAX)

    def test_monotonic_non_decreasing(self):
        previous = None
        for x in range(-40, 41):
            y = gelu(x * 4096)
            if previous is not None:
                self.assertGreaterEqual(y, previous)
            previous = y

    def test_rejects_float(self):
        with self.assertRaises(TypeError):
            gelu(0.5)


class PlanSigmoidTest(unittest.TestCase):
    def test_segment_breakpoints(self):
        self.assertEqual(plan_sigmoid(0), 32768)
        self.assertEqual(plan_sigmoid(PLAN_BP_1_Q16), 49152)
        self.assertEqual(plan_sigmoid(PLAN_BP_2_Q16), 60160)
        self.assertEqual(plan_sigmoid(PLAN_BP_3_Q16), 65536)
        self.assertEqual(plan_sigmoid(-PLAN_BP_1_Q16), 16384)
        self.assertEqual(plan_sigmoid(-PLAN_BP_2_Q16), 5376)
        self.assertEqual(plan_sigmoid(-PLAN_BP_3_Q16), 0)

    def test_threshold_neighborhood(self):
        self.assertEqual(plan_sigmoid(PLAN_BP_1_Q16 - 1), 49151)
        self.assertEqual(plan_sigmoid(PLAN_BP_1_Q16 + 1), 49152)
        self.assertEqual(plan_sigmoid(PLAN_BP_2_Q16 - 1), 60415)
        self.assertEqual(plan_sigmoid(PLAN_BP_2_Q16 + 1), 60160)
        self.assertEqual(plan_sigmoid(PLAN_BP_3_Q16 - 1), 65535)
        self.assertEqual(plan_sigmoid(PLAN_BP_3_Q16 + 1), 65536)

    def test_symmetry(self):
        for x in (0, 1, 65536, 155648, 327680, Q8_16_MIN, Q8_16_MAX):
            self.assertEqual(plan_sigmoid(-x), Q0_16_ONE - plan_sigmoid(x))

    def test_output_range(self):
        for x in (Q8_16_MIN, -327680, -1, 0, 1, 327680, Q8_16_MAX):
            self.assertIn(plan_sigmoid(x), range(Q0_16_ONE + 1))

    def test_rejects_float(self):
        with self.assertRaises(TypeError):
            plan_sigmoid(0.5)


class SoftmaxTest(unittest.TestCase):
    def test_attention_single_element_is_one(self):
        # P2 fix: attention delta2 = 1.0, single-element row saturates
        # UQ0.8 to 255 (was 128 under the old halving delta2 = 0.5).
        for row in ([0], [12345], [Q8_16_MIN], [Q8_16_MAX]):
            self.assertEqual(softmax_attention(row), [255])

    def test_selector_single_element_is_one(self):
        for row in ([0], [12345], [Q8_16_MIN], [Q8_16_MAX]):
            self.assertEqual(softmax_selector(row), [65536])

    def test_selector_equal_pair_halves(self):
        self.assertEqual(softmax_selector([0, 0]), [32768, 32768])
        self.assertEqual(softmax_selector([12345, 12345]), [32768, 32768])

    def test_row_sums_near_unit(self):
        for row in ([0] * 3, [0] * 197, [1, 2, 3, 4],
                    [Q8_16_MIN, Q8_16_MAX, 0], [-100000, 0, 100000]):
            selector = softmax_selector(row)
            attention = softmax_attention(row)
            self.assertTrue(abs(sum(selector) - Q0_16_ONE) <= 197)
            self.assertLessEqual(sum(attention), 256)

    def test_monotonic_in_input(self):
        row = list(range(-20, 21))
        self.assertEqual(softmax_selector(row), sorted(softmax_selector(row)))
        self.assertEqual(softmax_attention(row), sorted(softmax_attention(row)))

    def test_max_element_dominates(self):
        row = [-100000, 0, Q8_16_MAX, Q8_16_MIN, 500000]
        selector = softmax_selector(row)
        self.assertEqual(max(selector), selector[2])

    def test_empty_row_rejected(self):
        with self.assertRaises(ValueError):
            softmax_selector([])

    def test_rejects_float(self):
        with self.assertRaises(TypeError):
            softmax_selector([0.5, 0.0])


class LayerNormTest(unittest.TestCase):
    GAMMA_64 = [64] * 192
    BETA_0 = [0] * 192

    def test_zero_vector_is_zero(self):
        outputs, warn = layernorm(
            [0] * 192, self.GAMMA_64, self.BETA_0, -7, -6, -7, -7
        )
        self.assertEqual(outputs, [0] * 192)
        self.assertFalse(warn)

    def test_constant_vector_has_zero_normalized_term(self):
        outputs, warn = layernorm(
            [42] * 192, self.GAMMA_64, self.BETA_0, -7, -6, -7, -7
        )
        self.assertEqual(outputs, [0] * 192)
        self.assertFalse(warn)
        outputs, warn = layernorm(
            [42] * 192, self.GAMMA_64, [5] * 192, -7, -6, -7, -7
        )
        self.assertEqual(outputs, [5] * 192)
        self.assertFalse(warn)

    def test_negative_variance_clamps_and_warns(self):
        inputs = [127] * 95 + [90] * 97
        outputs, warn = layernorm(
            inputs, self.GAMMA_64, self.BETA_0, -23, -6, -7, -7
        )
        self.assertTrue(warn)
        self.assertEqual(outputs, [0] * 192)

    def test_monotone_rows(self):
        increasing = list(range(-128, 64))
        outputs, warn = layernorm(
            increasing, self.GAMMA_64, self.BETA_0, -7, -6, -7, -7
        )
        self.assertFalse(warn)
        self.assertEqual(outputs, sorted(outputs))
        self.assertTrue(all(-128 <= value <= 127 for value in outputs))

        decreasing = list(range(63, -129, -1))
        outputs, warn = layernorm(
            decreasing, self.GAMMA_64, self.BETA_0, -7, -6, -7, -7
        )
        self.assertFalse(warn)
        self.assertEqual(outputs, sorted(outputs, reverse=True))

    def test_mixed_row_stays_in_range(self):
        mixed = [(i % 37) * ((i % 2) * 2 - 1) for i in range(192)]
        outputs, warn = layernorm(
            mixed, self.GAMMA_64, self.BETA_0, -7, -6, -7, -7
        )
        self.assertFalse(warn)
        self.assertTrue(all(-128 <= value <= 127 for value in outputs))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            layernorm([0] * 192, self.GAMMA_64, self.BETA_0, 1, -6, -7, -7)
        with self.assertRaises(ValueError):
            layernorm([0] * 192, self.GAMMA_64, self.BETA_0, -33, -6, -7, -7)
        with self.assertRaises(ValueError):
            layernorm([0] * 191, self.GAMMA_64, self.BETA_0, -7, -6, -7, -7)
        with self.assertRaises(ValueError):
            layernorm([128] * 192, self.GAMMA_64, self.BETA_0, -7, -6, -7, -7)
        with self.assertRaises(TypeError):
            layernorm([0.5] * 192, self.GAMMA_64, self.BETA_0, -7, -6, -7, -7)


if __name__ == "__main__":
    unittest.main()
