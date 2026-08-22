import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_DIR = REPO_ROOT / "build" / "vectors" / "e2e"
SV_CONFIG = REPO_ROOT / "sim" / "generated" / "e2e_tb_config.sv"

MANDATORY = {
    "seed", "part", "descriptor_rom_sha256", "regions", "checkpoints",
    "selectors", "token_counts", "output_scale_exp", "watchdog_cycles",
    "files",
}
EXPECTED_CHECKPOINTS = [
    "patch",
    "block_01", "block_02", "block_03", "selector_01",
    "block_04", "block_05", "block_06", "selector_02",
    "block_07", "block_08", "block_09", "selector_03",
    "block_10", "block_11", "block_12",
    "final_ln", "logits",
]
EXPECTED_DESC_INDEX = [2, 15, 28, 41, 53, 66, 79, 92, 104, 117, 130, 143,
                       155, 168, 181, 194, 195, 196]


def _load_manifest():
    with open(E2E_DIR / "manifest.json", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_sv():
    text = SV_CONFIG.read_text(encoding="utf-8")

    def int_const(name):
        match = re.search(rf"localparam\s+int\s+{name}\s*=\s*(\S+);", text)
        if not match:
            raise KeyError(f"missing SV constant {name}")
        return int(match.group(1), 0)

    def hex_const(name):
        match = re.search(rf"localparam\s+logic\s+\[31:0\]\s+{name}\s*=\s*"
                          rf"32'h([0-9a-fA-F]+);", text)
        if not match:
            raise KeyError(f"missing SV constant {name}")
        return int(match.group(1), 16)

    def array_const(name, kind):
        match = re.search(rf"localparam\s+.*?\s+{name}\s*\[0:\d+\]\s*=\s*"
                          rf"'\{{(.*?)\}};", text, re.S)
        if not match:
            raise KeyError(f"missing SV array {name}")
        body = match.group(1).replace("\n", " ").strip()
        if kind == "hex":
            return [int(v, 16) for v in re.findall(r"32'h([0-9a-fA-F]+)",
                                                   body)]
        return [int(v, 0) for v in body.split(",") if v.strip()]

    return {
        "INPUT_BASE": hex_const("INPUT_BASE"),
        "INPUT_BYTES": int_const("INPUT_BYTES"),
        "WEIGHT_BASE": hex_const("WEIGHT_BASE"),
        "WEIGHT_BYTES": int_const("WEIGHT_BYTES"),
        "SCRATCH_BASE": hex_const("SCRATCH_BASE"),
        "SCRATCH_BYTES": int_const("SCRATCH_BYTES"),
        "OUTPUT_BASE": hex_const("OUTPUT_BASE"),
        "OUTPUT_BYTES": int_const("OUTPUT_BYTES"),
        "WATCHDOG_CYCLES": int_const("WATCHDOG_CYCLES"),
        "OUTPUT_SCALE_EXP": int_const("OUTPUT_SCALE_EXP"),
        "SELECTOR_IN_N": array_const("SELECTOR_IN_N", "int"),
        "SELECTOR_OUT_N": array_const("SELECTOR_OUT_N", "int"),
        "SELECTOR_PACKAGE": array_const("SELECTOR_PACKAGE", "int"),
        "CHECKPOINT_DESC_INDEX": array_const("CHECKPOINT_DESC_INDEX", "int"),
        "CHECKPOINT_OFFSET": array_const("CHECKPOINT_OFFSET", "hex"),
        "CHECKPOINT_BYTES": array_const("CHECKPOINT_BYTES", "int"),
    }


class E2eManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = _load_manifest()
        cls.sv = _parse_sv()

    def test_manifest_mandatory_keys(self):
        missing = MANDATORY - set(self.manifest)
        self.assertFalse(missing, f"missing manifest keys: {missing}")

    def test_seed_part_and_descriptor_hash(self):
        self.assertEqual(self.manifest["seed"], 20260815)
        self.assertEqual(self.manifest["part"], "xc7k325tfbg900-3")
        self.assertEqual(len(self.manifest["descriptor_rom_sha256"]), 64)
        rom_hash = self.manifest["descriptor_rom_sha256"]
        import hashlib
        with open(REPO_ROOT / "rtl/generated/heatvit_descriptors.mem",
                  "rb") as handle:
            self.assertEqual(rom_hash,
                             hashlib.sha256(handle.read()).hexdigest())

    def test_region_bases_and_bytes(self):
        reg = self.manifest["regions"]
        self.assertEqual(reg["input"]["base"], 0x00000000)
        self.assertEqual(reg["weight"]["base"], 0x01000000)
        self.assertEqual(reg["scratch"]["base"], 0x02000000)
        self.assertEqual(reg["output"]["base"], 0x03000000)
        for name, entry in reg.items():
            self.assertEqual(entry["bytes"] & 0x7, 0, f"{name} not 8-aligned")
        # No region may cross into the next base.
        self.assertLessEqual(reg["input"]["base"] + reg["input"]["bytes"],
                             0x01000000)
        self.assertLessEqual(reg["weight"]["base"] + reg["weight"]["bytes"],
                             0x02000000)
        self.assertLessEqual(reg["scratch"]["base"] + reg["scratch"]["bytes"],
                             0x03000000)
        self.assertLessEqual(reg["output"]["base"] + reg["output"]["bytes"],
                             0x03000000 + 0x100000)

    def test_checkpoint_table(self):
        cps = self.manifest["checkpoints"]
        self.assertEqual([c["name"] for c in cps], EXPECTED_CHECKPOINTS)
        self.assertEqual([c["desc_index"] for c in cps], EXPECTED_DESC_INDEX)
        for c in cps:
            self.assertEqual(c["offset"] & 0x7, 0, f"{c['name']} unaligned")
            self.assertGreater(c["bytes"], 0)
            self.assertEqual(len(c["sha256"]), 64)
            if c["name"] == "logits":
                self.assertEqual(c["scale_exp"], -14)
            else:
                self.assertEqual(c["scale_exp"], -7)

    def test_selector_summary(self):
        self.assertEqual(len(self.manifest["selectors"]), 3)
        counts = self.manifest["token_counts"]
        self.assertEqual(counts[0], 197)
        for idx, entry in enumerate(self.manifest["selectors"]):
            self.assertGreaterEqual(entry["kept_normal"], 1)
            self.assertGreaterEqual(entry["pruned_normal"], 2)
            self.assertEqual(entry["output_tokens"], counts[idx + 1])
            self.assertLess(counts[idx + 1], counts[idx])

    def test_output_scale_and_watchdog(self):
        self.assertEqual(self.manifest["output_scale_exp"], -14)
        self.assertGreater(self.manifest["watchdog_cycles"], 10_000_000)

    def test_file_hashes(self):
        import hashlib
        for name, digest in self.manifest["files"].items():
            path = E2E_DIR / name
            self.assertTrue(path.exists(), f"missing {name}")
            with open(path, "rb") as handle:
                self.assertEqual(digest,
                                 hashlib.sha256(handle.read()).hexdigest(),
                                 name)

    def test_sv_config_matches_json(self):
        reg = self.manifest["regions"]
        checks = {
            "INPUT_BASE": reg["input"]["base"],
            "INPUT_BYTES": reg["input"]["bytes"],
            "WEIGHT_BASE": reg["weight"]["base"],
            "WEIGHT_BYTES": reg["weight"]["bytes"],
            "SCRATCH_BASE": reg["scratch"]["base"],
            "SCRATCH_BYTES": reg["scratch"]["bytes"],
            "OUTPUT_BASE": reg["output"]["base"],
            "OUTPUT_BYTES": reg["output"]["bytes"],
            "WATCHDOG_CYCLES": self.manifest["watchdog_cycles"],
            "OUTPUT_SCALE_EXP": self.manifest["output_scale_exp"],
        }
        for name, expected in checks.items():
            self.assertEqual(self.sv[name], expected, name)

        counts = self.manifest["token_counts"]
        self.assertEqual(self.sv["SELECTOR_IN_N"],
                         [counts[0], counts[1], counts[2]])
        self.assertEqual(self.sv["SELECTOR_OUT_N"],
                         [counts[1], counts[2], counts[3]])
        pkgs = [e["package_present"] for e in self.manifest["selectors"]]
        self.assertEqual(self.sv["SELECTOR_PACKAGE"], pkgs)

        cps = self.manifest["checkpoints"]
        self.assertEqual(self.sv["CHECKPOINT_DESC_INDEX"],
                         [c["desc_index"] for c in cps])
        self.assertEqual(self.sv["CHECKPOINT_OFFSET"],
                         [reg[c["region"]]["base"] + c["offset"]
                          for c in cps])
        self.assertEqual(self.sv["CHECKPOINT_BYTES"],
                         [c["bytes"] for c in cps])


if __name__ == "__main__":
    unittest.main()
