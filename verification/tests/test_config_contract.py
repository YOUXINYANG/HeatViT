import json
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "heatvit_t.json"
PACKAGE_PATH = REPO_ROOT / "rtl" / "include" / "heatvit_pkg.sv"
# The project keeps a single record document; the numeric contract lives in
# its Part 1 section 9.
CONTRACT_PATH = REPO_ROOT / "docs" / "heatvit.md"


class ConfigContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.package = PACKAGE_PATH.read_text(encoding="utf-8")

    def localparam_int(self, name):
        match = re.search(
            rf"localparam\s+int\s+{re.escape(name)}\s*=\s*(-?\d+);",
            self.package,
        )
        self.assertIsNotNone(match, f"missing package constant {name}")
        return int(match.group(1))

    def test_fixed_point_constants_match_config(self):
        expected = {
            "GELU_SLOPE_NUM_Q16": self.config["gelu_q16"]["slope_num"],
            "GELU_SLOPE_SHIFT": self.config["gelu_q16"]["slope_shift"],
            "GELU_SLOPE_ROUND_ADD": self.config["gelu_q16"]["slope_round"],
            "GELU_EXP_NEG_Q_MAX": self.config["gelu_q16"]["exp_neg_q_max"],
            "GELU_EXP_POS_Q_MAX": self.config["gelu_q16"]["exp_pos_q_max"],
            "EXP_LN2_Q16": self.config["exp_q16"]["ln2"],
            "EXP_QUAD_Q16": self.config["exp_q16"]["quad"],
            "EXP_OFFSET_Q16": self.config["exp_q16"]["offset"],
            "EXP_CONST_Q16": self.config["exp_q16"]["constant"],
            "SOFTMAX_DELTA_Q16_ATTENTION": self.config["softmax_delta_q16"]["attention"],
            "SOFTMAX_DELTA_Q16_SELECTOR": self.config["softmax_delta_q16"]["selector"],
        }
        for name, value in expected.items():
            self.assertEqual(self.localparam_int(name), value, name)

        eps = re.search(
            r"localparam\s+logic\s+\[47:0\]\s+LN_EPS_Q32\s*=\s*48'd(\d+);",
            self.package,
        )
        self.assertIsNotNone(eps, "missing LN_EPS_Q32")
        self.assertEqual(int(eps.group(1)), self.config["layernorm_epsilon_q32"])

    def test_descriptor_width_is_320(self):
        match = re.search(
            r"typedef\s+struct\s+packed\s*\{(.*?)\}\s*heatvit_desc_t;",
            self.package,
            re.S,
        )
        self.assertIsNotNone(match, "missing heatvit_desc_t struct")
        total = 0
        for line in match.group(1).splitlines():
            stripped = line.strip()
            field = re.match(r"logic\s*(?:signed\s*)?\[(\d+):0\]\s*\w+;", stripped)
            if field:
                total += int(field.group(1)) + 1
            elif re.match(r"heatvit_scale_t\s+\w+;", stripped):
                total += 6
        self.assertEqual(total, 320)

    def test_contract_document_covers_numeric_contract(self):
        self.assertTrue(
            CONTRACT_PATH.is_file(),
            "missing docs/heatvit.md",
        )
        text = CONTRACT_PATH.read_text(encoding="utf-8")
        for marker in (
            "Q8.16",
            "Q0.16",
            "UQ0.8",
            "Q0.32",
            "GELU_SLOPE_NUM_Q16",
            "LN_EPS_Q32",
            "4295",
            "94548",
            "320",
            "远离零",
            "饱和",
        ):
            self.assertIn(marker, text, f"contract document missing {marker}")


if __name__ == "__main__":
    unittest.main()
