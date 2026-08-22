import unittest

import numpy as np

from verification.heatvit_ref.fixed import requant
from verification.heatvit_ref.gemm import gemm, gemm_numpy, gemm_writeback


class GemmTest(unittest.TestCase):
    def test_row_major_and_transpose(self):
        a = [[1, 2, 3], [-1, 0, 1]]
        b = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(gemm(a, b, [7, -7], False), [[29, 21], [11, -3]])
        bt = [[1, 3, 5], [2, 4, 6]]
        self.assertEqual(gemm(a, bt, [7, -7], True), [[29, 21], [11, -3]])

    def test_no_bias(self):
        a = [[2, 0], [0, 3]]
        b = [[4], [5]]
        self.assertEqual(gemm(a, b, None, False), [[8], [15]])

    def test_scalar_matches_numpy(self):
        rng = np.random.default_rng(20260815)
        shapes = [(1, 1, 1), (7, 9, 5), (8, 24, 8), (9, 25, 17),
                  (8, 8, 64), (197, 192, 192)]
        for m, n, k in shapes:
            a = rng.integers(-128, 128, size=(m, k), dtype=np.int64)
            b = rng.integers(-128, 128, size=(k, n), dtype=np.int64)
            bias = rng.integers(-64, 64, size=(n,), dtype=np.int64)
            for transpose_b in (False, True):
                stored_b = b.T.copy() if transpose_b else b
                got = gemm(a.tolist(), stored_b.tolist(), bias.tolist(), transpose_b)
                want = gemm_numpy(a.tolist(), stored_b.tolist(), bias.tolist(), transpose_b)
                self.assertEqual(got, want, f"scalar/numpy mismatch for {(m, n, k)}")

    def test_unsigned_src0(self):
        self.assertEqual(
            gemm([[128]], [[-128]], None, False, a_unsigned=True),
            [[-16384]],
        )
        self.assertEqual(
            gemm([[255]], [[-1]], None, False, a_unsigned=True),
            [[-255]],
        )
        self.assertEqual(
            gemm_numpy([[128]], [[-128]], None, False, a_unsigned=True),
            [[-16384]],
        )

    def test_head_reference_isolates_heads(self):
        # Three distinct per-head 8x8x64 GEMMs with sentinel-free random data;
        # any cross-head mixing must make at least one output differ.
        rng = np.random.default_rng(20260815)
        outputs = []
        for _ in range(3):
            a = rng.integers(-128, 128, size=(8, 8), dtype=np.int64)
            b = rng.integers(-128, 128, size=(8, 64), dtype=np.int64)
            got = gemm(a.tolist(), b.tolist(), None, False)
            want = gemm_numpy(a.tolist(), b.tolist(), None, False)
            self.assertEqual(got, want)
            outputs.append(got)
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertNotEqual(outputs[1], outputs[2])
        self.assertNotEqual(outputs[0], outputs[2])

    def test_writeback_int8(self):
        accum = [[1, -1, 255, -255], [32768, -32769, 0, 100]]
        got = gemm_writeback(accum, -14, -7, 8)
        want = [[requant(v, -14, -7, 8) for v in row] for row in accum]
        self.assertEqual(got, want)
        for row in got:
            for value in row:
                self.assertGreaterEqual(value, -128)
                self.assertLessEqual(value, 127)

    def test_writeback_int32(self):
        got = gemm_writeback([[1 << 40, -(1 << 40), 5]], -14, -14, 32)
        self.assertEqual(got, [[(1 << 31) - 1, -(1 << 31), 5]])
        self.assertEqual(gemm_writeback([[7, -8]], -14, -14, 32), [[7, -8]])

    def test_writeback_rejects_unknown_bits(self):
        with self.assertRaises(ValueError):
            gemm_writeback([[1]], -14, -7, 16)

    def test_rejects_float(self):
        with self.assertRaises(TypeError):
            gemm([[1.0]], [[1]], None, False)
        with self.assertRaises(TypeError):
            gemm([[1]], [[1]], [1.0], False)
        with self.assertRaises(TypeError):
            gemm_writeback([[1.0]], -14, -7, 8)

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            gemm([[128]], [[1]], None, False)
        with self.assertRaises(ValueError):
            gemm([[1]], [[-129]], None, False)
        with self.assertRaises(ValueError):
            gemm([[256]], [[1]], None, False, a_unsigned=True)
        with self.assertRaises(ValueError):
            gemm([[1]], [[1]], [1 << 31], False)


if __name__ == "__main__":
    unittest.main()
