import unittest

from verification.heatvit_ref.descriptor import (
    FLAG_DST_OUTPUT,
    FLAG_DYNAMIC_K,
    FLAG_DYNAMIC_M,
    FLAG_DYNAMIC_N,
    FLAG_HEAD_MODE,
    FLAG_OUTPUT_INT32,
    FLAG_RHS_TRANSPOSE,
    FLAG_SRC0_UNSIGNED,
    FLAG_SWAP_ACTIVATION,
    OP_ATTN_SOFTMAX,
    OP_FINISH,
    OP_GEMM,
    OP_LAYERNORM,
    OP_PATCHIFY,
    OP_SELECTOR_FINALIZE,
    Descriptor,
)
from tools.generate_descriptors import (
    ACT_SLOT,
    _scale_lookup,
    block_sequence,
    build_memory_map,
    build_schedule,
    classifier_sequence,
    final_layernorm_sequence,
    patch_sequence,
    selector_sequence,
)

F = lambda *bits: sum(1 << b for b in bits)  # noqa: E731


class ScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mm, cls.scratch_bytes, cls.weight_bytes = build_memory_map()
        cls.descs = build_schedule(cls.mm)

    def test_descriptor_count_and_finish(self):
        self.assertEqual(len(self.descs), 198)
        for idx, desc in enumerate(self.descs):
            self.assertEqual(desc.opcode == OP_FINISH, idx == 197,
                             f"OP_FINISH must be only at index 197, got {idx}")

    def test_fixed_index_landmarks(self):
        self.assertEqual(self.descs[42].opcode, OP_GEMM)
        self.assertEqual(self.descs[93].opcode, OP_GEMM)
        self.assertEqual(self.descs[144].opcode, OP_GEMM)
        self.assertEqual(self.descs[53].opcode, OP_SELECTOR_FINALIZE)
        self.assertEqual(self.descs[104].opcode, OP_SELECTOR_FINALIZE)
        self.assertEqual(self.descs[155].opcode, OP_SELECTOR_FINALIZE)
        self.assertEqual(self.descs[195].opcode, OP_LAYERNORM)
        self.assertEqual(self.descs[196].opcode, OP_GEMM)
        self.assertEqual(self.descs[197].opcode, OP_FINISH)

    def test_block_residual2_positions(self):
        # Residual2 (flag4 swap points) at block ends: 15, 28, 41, 66, 79,
        # 92, 117, 130, 143, 168, 181, 194.
        expected = [15, 28, 41, 66, 79, 92, 117, 130, 143, 168, 181, 194]
        for idx in expected:
            desc = self.descs[idx]
            self.assertTrue(desc.flags & F(FLAG_SWAP_ACTIVATION),
                            f"index {idx} must set flag 4")
            self.assertEqual(desc.dst_offset, 0, "Residual2 dst is the slot")
        # The three finalize swap points.
        for idx in (53, 104, 155):
            desc = self.descs[idx]
            self.assertTrue(desc.flags & F(FLAG_SWAP_ACTIVATION),
                            f"finalize {idx} must set flag 4")
        # Exactly 15 flag4 descriptors.
        flagged = [i for i, d in enumerate(self.descs)
                   if d.flags & F(FLAG_SWAP_ACTIVATION)]
        self.assertEqual(len(flagged), 15, f"flag4 count {len(flagged)}")

    def test_transformer_dynamic_flags(self):
        for idx, desc in enumerate(self.descs):
            if desc.opcode == OP_ATTN_SOFTMAX:
                self.assertTrue(desc.flags & F(FLAG_DYNAMIC_M))
                self.assertTrue(desc.flags & F(FLAG_DYNAMIC_N),
                                f"softmax {idx} must be dynamic N")
            if desc.opcode == OP_GEMM and desc.flags & F(FLAG_HEAD_MODE):
                if desc.flags & F(FLAG_RHS_TRANSPOSE):
                    # QK^T: transpose + head mode + dynamic N.
                    self.assertTrue(desc.flags & F(FLAG_DYNAMIC_N),
                                    f"QK {idx} must be dynamic N")
                if desc.flags & F(FLAG_SRC0_UNSIGNED):
                    # Attention*V: head mode + flag18 + dynamic K.
                    self.assertTrue(desc.flags & F(FLAG_DYNAMIC_K),
                                    f"AV {idx} must be dynamic K")
            # All dynamic transformer ops use param0[1:0] = 00.
            if desc.flags & F(FLAG_DYNAMIC_M) and not (
                    42 <= idx <= 53 or 93 <= idx <= 104 or
                    144 <= idx <= 155):
                self.assertEqual(desc.param0 & 0x3, 0,
                                 f"desc {idx} transformer dynamic param0")

    def test_selector_dynamic_flags(self):
        for idx in list(range(42, 54)) + list(range(93, 105)) + \
                list(range(144, 156)):
            desc = self.descs[idx]
            if desc.opcode == OP_SELECTOR_FINALIZE:
                self.assertEqual(desc.param0 & 0x3, 0,
                                 f"finalize {idx} param0[1:0] must be 00")
            else:
                self.assertTrue(desc.flags & F(FLAG_DYNAMIC_M),
                                f"selector desc {idx} must be dynamic M")
                self.assertEqual(desc.param0 & 0x3, 1,
                                 f"selector desc {idx} param0[1:0] must be 01")

    def test_classifier_head_flags(self):
        head = self.descs[196]
        self.assertEqual(head.opcode, OP_GEMM)
        self.assertEqual(head.m, 1)
        self.assertEqual(head.n, 1000)
        self.assertEqual(head.k, 192)
        self.assertTrue(head.flags & F(FLAG_OUTPUT_INT32),
                        "classifier must set flag 7")
        self.assertTrue(head.flags & F(FLAG_DST_OUTPUT),
                        "classifier must set flag 15")

    def test_final_layernorm_reads_activation_slot(self):
        ln = self.descs[195]
        self.assertEqual(ln.opcode, OP_LAYERNORM)
        self.assertEqual(ln.src0_offset, 0, "final LN reads the activation slot")

    def test_activation_offsets_are_slot_relative(self):
        for idx, desc in enumerate(self.descs):
            for name in ("src0_offset", "src1_offset", "aux_offset",
                         "dst_offset"):
                off = getattr(desc, name)
                # Activation references are slot-relative (< ACT_SLOT);
                # everything else lives beyond the two slots.
                self.assertTrue(off < ACT_SLOT or off >= 2 * ACT_SLOT,
                                f"desc {idx} {name}={off} inside slot gap")

    def test_all_offsets_aligned(self):
        for idx, desc in enumerate(self.descs):
            for name in ("src0_offset", "src1_offset", "bias_offset",
                         "aux_offset", "dst_offset"):
                self.assertEqual(getattr(desc, name) & 0x7, 0,
                                 f"desc {idx} {name} unaligned")

    def test_validate_accepts_schedule(self):
        for desc in self.descs:
            desc.validate()

    def test_validate_rejects_schema_violations(self):
        with self.assertRaises(ValueError):
            Descriptor(opcode=99).validate()
        with self.assertRaises(ValueError):
            Descriptor(opcode=OP_GEMM, reserved=1).validate()
        with self.assertRaises(ValueError):
            Descriptor(opcode=OP_GEMM, src0_scale_exp=32).validate()
        with self.assertRaises(ValueError):
            Descriptor(opcode=OP_GEMM, dst_scale_exp=-33).validate()
        with self.assertRaises(ValueError):
            Descriptor(opcode=OP_LAYERNORM,
                       flags=F(FLAG_SRC0_UNSIGNED)).validate()
        with self.assertRaises(ValueError):
            Descriptor(opcode=OP_PATCHIFY,
                       flags=F(FLAG_DYNAMIC_M), param0=0).validate()
        with self.assertRaises(ValueError):
            Descriptor(opcode=OP_GEMM, m=0, n=8, k=8).validate()
        with self.assertRaises(ValueError):
            Descriptor(opcode=OP_GEMM, m=8, n=8, k=8,
                       src0_offset=4).validate()

    def test_sequence_helper_sizes(self):
        sc = _scale_lookup(None)
        self.assertEqual(len(patch_sequence(self.mm, sc)), 3)
        self.assertEqual(len(block_sequence(1, self.mm, True, sc)), 13)
        self.assertEqual(len(selector_sequence(1, self.mm, True)), 12)
        self.assertEqual(len(final_layernorm_sequence(self.mm, sc)), 1)
        self.assertEqual(len(classifier_sequence(self.mm, sc)), 1)
        self.assertEqual(Descriptor.finish().opcode, OP_FINISH)

    def test_memory_map_regions(self):
        mm = self.mm
        scratch = mm["scratch"]
        self.assertEqual(scratch["buf0"], 0)
        self.assertEqual(scratch["buf1"], ACT_SLOT)
        self.assertTrue(self.scratch_bytes > 2 * ACT_SLOT)
        self.assertTrue(self.weight_bytes > 0)
        for name, off in scratch.items():
            if isinstance(off, int):
                self.assertEqual(off & 0x7, 0, f"scratch {name} unaligned")
        for name, off in mm["weight"].items():
            self.assertEqual(off & 0x7, 0, f"weight {name} unaligned")


if __name__ == "__main__":
    unittest.main()
