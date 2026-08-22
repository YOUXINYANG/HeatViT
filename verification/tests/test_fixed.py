import unittest

from verification.heatvit_ref.fixed import (
    isqrt,
    requant,
    round_shift_away,
    sat_signed,
    udiv,
)


class FixedTest(unittest.TestCase):
    def test_ties_away_from_zero(self):
        self.assertEqual(round_shift_away(1, 1), 1)
        self.assertEqual(round_shift_away(-1, 1), -1)
        self.assertEqual(round_shift_away(3, 1), 2)
        self.assertEqual(round_shift_away(-3, 1), -2)

    def test_saturation(self):
        self.assertEqual(sat_signed(128, 8), 127)
        self.assertEqual(sat_signed(-129, 8), -128)

    def test_scale_conversion(self):
        self.assertEqual(requant(255, -8, -7, 8), 127)
        self.assertEqual(requant(-255, -8, -7, 8), -128)

    def test_udiv(self):
        self.assertEqual(udiv(10, 3), (3, 1))
        self.assertEqual(udiv(0, 5), (0, 0))
        with self.assertRaises(ValueError):
            udiv(10, 0)

    def test_isqrt(self):
        self.assertEqual(isqrt(0), (0, 0))
        self.assertEqual(isqrt(15), (3, 6))
        self.assertEqual(isqrt(16), (4, 0))
        with self.assertRaises(ValueError):
            isqrt(-1)


if __name__ == "__main__":
    unittest.main()
