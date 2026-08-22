import tempfile
import unittest
from pathlib import Path

from verification.heatvit_ref.descriptor import (
    Descriptor,
    format_desc_hex,
)
from verification.heatvit_ref.layout import (
    add_pos,
    head_concat,
    patchify,
    qkv_unpack,
)
from verification.heatvit_ref.memory import (
    TensorArena,
    pack_int32_le,
    unpack_int32_le,
)
from verification.heatvit_ref.op_sequence import write_descriptors_mem


def qkv_value(token, kind, head, lane):
    """Deterministic int8 filler for the fused QKV matrix."""
    return ((token * 576 + kind * 192 + head * 64 + lane) * 37) % 200 - 100


class LayoutTest(unittest.TestCase):
    def test_patchify_flatten_order(self):
        # 4x4x3 NHWC image, patch=2: pixel (row, col) channel c is encoded
        # as ((row*4+col)*3+c), all values well inside int8.
        image = [
            ((row * 4 + col) * 3 + ch) & 0x7F
            for row in range(4)
            for col in range(4)
            for ch in range(3)
        ]
        patches = patchify(image, width=4, patch=2)
        self.assertEqual(len(patches), 4)
        self.assertTrue(all(len(p) == 12 for p in patches))

        first = [
            ((in_row * 4 + in_col) * 3 + ch) & 0x7F
            for in_row in (0, 1)
            for in_col in (0, 1)
            for ch in range(3)
        ]
        self.assertEqual(patches[0], first)

        last = [
            (((2 + in_row) * 4 + (2 + in_col)) * 3 + ch) & 0x7F
            for in_row in (0, 1)
            for in_col in (0, 1)
            for ch in range(3)
        ]
        self.assertEqual(patches[-1], last)

    def test_qkv_unpack_and_concat_inverse(self):
        tokens = 2
        fused = []
        for token in range(tokens):
            row = []
            for kind in range(3):
                for head in range(3):
                    for lane in range(64):
                        row.append(qkv_value(token, kind, head, lane))
            fused.append(row)

        unpacked = qkv_unpack(fused, tokens)
        for kind in range(3):
            for head in range(3):
                for token in range(tokens):
                    for lane in range(64):
                        self.assertEqual(
                            unpacked[kind][head][token][lane],
                            qkv_value(token, kind, head, lane),
                        )
        for kind in range(3):
            self.assertEqual(
                head_concat(unpacked[kind], tokens),
                [row[kind * 192:(kind + 1) * 192] for row in fused],
            )

    def test_qkv_full_size_round_trip(self):
        tokens = 197
        fused = [
            [
                qkv_value(token, kind, head, lane)
                for kind in range(3)
                for head in range(3)
                for lane in range(64)
            ]
            for token in range(tokens)
        ]
        unpacked = qkv_unpack(fused, tokens)
        self.assertEqual(
            [len(kind) for kind in unpacked], [3, 3, 3]
        )
        self.assertEqual(
            [len(head) for head in unpacked[0]], [tokens, tokens, tokens]
        )
        for kind in range(3):
            self.assertEqual(
                head_concat(unpacked[kind], tokens),
                [row[kind * 192:(kind + 1) * 192] for row in fused],
            )

    def test_add_pos_cls_and_position(self):
        patch_embed = [[1, 2], [3, 4]]
        cls = [10, 20]
        pos = [[1, 1], [2, 2], [5, 5]]
        got = add_pos(patch_embed, cls, pos)
        self.assertEqual(got, [[11, 21], [3, 4], [8, 9]])

        # int8 saturation still applies.
        got = add_pos([[127, -128]], [0, 0], [[127, -128], [1, 1]])
        self.assertEqual(got, [[127, -128], [127, -127]])

    def test_arena_alignment(self):
        arena = TensorArena()
        self.assertEqual(arena.allocate("a", 1), 0)
        self.assertEqual(arena.allocate("b", 8), 8)
        self.assertEqual(arena.allocate("c", 9), 16)
        self.assertEqual(arena.offset_of("a"), 0)
        self.assertEqual(arena.offset_of("b"), 8)
        self.assertEqual(arena.offset_of("c"), 16)
        for name in ("a", "b", "c"):
            self.assertEqual(arena.offset_of(name) % 8, 0)
        self.assertEqual(arena.total_bytes, 25)
        self.assertEqual(arena.padded_bytes, 32)

    def test_descriptor_round_trip_and_bit_order(self):
        desc = Descriptor(
            opcode=3,
            flags=0x008820,
            m=7,
            n=192,
            k=8,
            heads=3,
            src0_offset=0x00000010,
            src1_offset=0x01000000,
            bias_offset=0x01000100,
            aux_offset=0,
            dst_offset=0x02000000,
            src0_scale_exp=-7,
            src1_scale_exp=-7,
            aux_scale_exp=0,
            dst_scale_exp=-7,
            next_index=5,
            param0=1,
            param1=2,
            reserved=0,
        )
        word = desc.pack()
        self.assertGreaterEqual(word, 0)
        self.assertLess(word, 1 << 320)
        self.assertEqual(word & 0xF, 0)  # reserved occupies the low 4 bits
        self.assertEqual((word >> 312) & 0xFF, 3)  # opcode occupies the top 8 bits
        self.assertEqual(Descriptor.unpack(word), desc)

        hex_str = format_desc_hex(word)
        self.assertEqual(len(hex_str), 80)
        self.assertTrue(all(ch in "0123456789abcdef" for ch in hex_str))
        self.assertEqual(int(hex_str[:2], 16), desc.opcode)

    def test_descriptor_finish(self):
        finish = Descriptor.finish()
        self.assertEqual(finish.opcode, 14)  # OP_FINISH
        self.assertEqual(finish.reserved, 0)
        self.assertEqual(Descriptor.unpack(finish.pack()), finish)

    def test_descriptors_mem_file(self):
        descriptors = [
            Descriptor(opcode=3, flags=0x8800, m=1, n=1, k=1, heads=0,
                       src0_offset=0, src1_offset=8, bias_offset=16,
                       aux_offset=0, dst_offset=32,
                       src0_scale_exp=-7, src1_scale_exp=-7,
                       aux_scale_exp=0, dst_scale_exp=-7,
                       next_index=1, param0=0, param1=0, reserved=0),
            Descriptor.finish(),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "desc.mem"
            write_descriptors_mem(path, descriptors)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(all(len(line) == 80 for line in lines))
            self.assertEqual(Descriptor.unpack(int(lines[0], 16)), descriptors[0])

    def test_memory_little_endian_int32(self):
        values = [0, 1, -1, 0x7FFFFFFF, -0x80000000, 0x12345678]
        data = pack_int32_le(values)
        self.assertEqual(len(data), 24)
        self.assertEqual(unpack_int32_le(data, 6), values)
        self.assertEqual(data[20:24], bytes([0x78, 0x56, 0x34, 0x12]))


if __name__ == "__main__":
    unittest.main()
