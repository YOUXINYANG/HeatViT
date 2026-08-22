"""Unit tests for tools/p2/scale_table.py (no torch dependency)."""

import shutil
import unittest
from pathlib import Path

from tools.p2.scale_table import (
    ACTIVATION_NAMES,
    WEIGHT_NAMES,
    ScaleTable,
    check_exp,
)

# Temporary files must live inside the workspace (sandbox-safe).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _full_table(weight_exp=-7, act_exp=-7):
    return ScaleTable(
        weights={name: weight_exp for name in WEIGHT_NAMES},
        activations={name: act_exp for name in ACTIVATION_NAMES},
    )


class TestCheckExp(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(check_exp("x", -32), -32)
        self.assertEqual(check_exp("x", 0), 0)
        self.assertEqual(check_exp("x", 31), 31)

    def test_out_of_range(self):
        for bad in (-33, 32):
            with self.assertRaises(ValueError):
                check_exp("x", bad)

    def test_non_integer(self):
        for bad in (1.5, "7", True, None):
            with self.assertRaises(ValueError):
                check_exp("x", bad)


class TestScaleTable(unittest.TestCase):
    def test_full_table_validates(self):
        _full_table().validate()

    def test_missing_weight_raises(self):
        table = _full_table()
        del table.weights["head_w"]
        with self.assertRaises(ValueError):
            table.validate()

    def test_missing_activation_raises(self):
        table = _full_table()
        del table.activations["final_ln_out"]
        with self.assertRaises(ValueError):
            table.validate()

    def test_unknown_name_raises(self):
        table = _full_table()
        table.weights["nonsense"] = -7
        with self.assertRaises(ValueError):
            table.validate()

    def test_bad_range_raises(self):
        table = _full_table()
        table.activations["input"] = 32
        with self.assertRaises(ValueError):
            table.validate()

    def test_lookup(self):
        table = _full_table(weight_exp=-6, act_exp=-8)
        self.assertEqual(table.weight_exp("b7_wqkv"), -6)
        self.assertEqual(table.activation_exp("b7_ln1_out"), -8)
        with self.assertRaises(KeyError):
            table.weight_exp("does_not_exist")

    def test_round_trip(self):
        table = _full_table(weight_exp=-6, act_exp=-8)
        # Sandbox notes: build/ and the redirected TEMP are write-denied in
        # some sessions, and tempfile.mkdtemp's 0o700 mode makes the created
        # directory inaccessible to later writes. Use a fixed workspace-root
        # scratch dir created with default permissions and clean up.
        tmp = REPO_ROOT / "p2_test_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            path = tmp / "scale_table.json"
            table.save(path)
            loaded = ScaleTable.load(path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(loaded.weights, table.weights)
        self.assertEqual(loaded.activations, table.activations)
        loaded.validate()

    def test_schema_size(self):
        # 3 patch-region weights + 12*8 block weights + 3*6 selector
        # weights + final gamma/beta + head = 3 + 96 + 18 + 3 = 120.
        self.assertEqual(len(WEIGHT_NAMES), 120)
        # input + 3 patch activations + 12*8 block + 3*7 selector + final ln.
        self.assertEqual(len(ACTIVATION_NAMES), 1 + 3 + 96 + 21 + 1)


if __name__ == "__main__":
    unittest.main()
