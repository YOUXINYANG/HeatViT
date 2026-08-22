import unittest
from dataclasses import replace

import numpy as np

from verification.heatvit_ref.fixed import requant, sat_signed
from verification.heatvit_ref.gemm import gemm, gemm_writeback
from verification.heatvit_ref.layout import patchify
from verification.heatvit_ref.nonlinear import gelu
from verification.heatvit_ref.transformer import (
    BlockParams,
    FfnParams,
    MhsaParams,
    PatchParams,
    ffn,
    mhsa,
    patch_embedding,
    transformer_block,
)


def det_image(width, height):
    return [
        ((row * width + col) * 3 + ch) * 37 % 200 - 100
        for row in range(height)
        for col in range(width)
        for ch in range(3)
    ]


def random_block_params(rng):
    """Deterministic synthetic weights for one complete Transformer block."""
    mhsa_params = MhsaParams(
        ln_gamma=[int(v) for v in rng.integers(0, 128, size=192)],
        ln_beta=[int(v) for v in rng.integers(-32, 32, size=192)],
        wqkv=[
            [int(v) for v in row]
            for row in rng.integers(-8, 8, size=(192, 576), dtype=np.int64)
        ],
        bqkv=[int(v) for v in rng.integers(-64, 64, size=576)],
        wproj=[
            [int(v) for v in row]
            for row in rng.integers(-8, 8, size=(192, 192), dtype=np.int64)
        ],
        bproj=[int(v) for v in rng.integers(-64, 64, size=192)],
        wqkv_scale_exp=0, wproj_scale_exp=0,
    )
    ffn_params = FfnParams(
        ln_gamma=[int(v) for v in rng.integers(0, 128, size=192)],
        ln_beta=[int(v) for v in rng.integers(-32, 32, size=192)],
        w1=[
            [int(v) for v in row]
            for row in rng.integers(-128, 128, size=(192, 768), dtype=np.int64)
        ],
        b1=[int(v) for v in rng.integers(-64, 64, size=768)],
        w2=[
            [int(v) for v in row]
            for row in rng.integers(-128, 128, size=(768, 192), dtype=np.int64)
        ],
        b2=[int(v) for v in rng.integers(-64, 64, size=192)],
        w1_scale_exp=0, w2_scale_exp=0,
    )
    return BlockParams(mhsa=mhsa_params, ffn=ffn_params)


class TransformerTest(unittest.TestCase):
    def test_single_patch_patchify_and_gemm(self):
        # One 16x16 patch: first patch column of W is all ones, the rest zero.
        image = det_image(16, 16)
        patches = patchify(image, width=16, patch=16)
        self.assertEqual(len(patches), 1)

        weight = [[1 if c == 0 else 0 for c in range(192)] for _ in range(768)]
        bias = [0] * 192
        accum = gemm(patches, weight, bias, False)
        self.assertEqual(accum[0][0], sum(patches[0]))
        for c in range(1, 192):
            self.assertEqual(accum[0][c], 0)

        embed = gemm_writeback(accum, -14, -7, 8)
        self.assertEqual(embed[0][0], requant(sum(patches[0]), -14, -7, 8))

    def test_patch_embedding_full_size(self):
        rng = np.random.default_rng(20260815)
        width, height, patch = 224, 224, 16
        image = [
            int(v) for v in rng.integers(-128, 128, size=width * height * 3)
        ]
        weight = rng.integers(-128, 128, size=(768, 192), dtype=np.int64)
        bias = [0] * 192
        cls = [int(v) for v in rng.integers(-128, 128, size=192)]
        pos = rng.integers(-128, 128, size=(197, 192), dtype=np.int64)
        params = PatchParams(
            patch_weight=[list(row) for row in weight],
            patch_bias=bias,
            cls=cls,
            pos=[list(row) for row in pos],
            width=width,
            height=height,
            patch=patch,
            embed_dim=192,
            tokens=197,
        )

        patches = patchify(image, width, patch)
        self.assertEqual(len(patches), 196)
        # First, middle and last patch raster indices.
        for pr, pc in ((0, 0), (6, 13), (13, 13)):
            patch_idx = pr * 14 + pc
            expected = [
                int(image[((pr * 16 + in_row) * 224 + pc * 16 + in_col) * 3 + ch])
                for in_row in range(16)
                for in_col in range(16)
                for ch in range(3)
            ]
            self.assertEqual(patches[patch_idx], expected, f"patch {pr},{pc}")

        activation = patch_embedding(image, params)
        self.assertEqual(len(activation), 197)
        self.assertTrue(all(len(row) == 192 for row in activation))
        self.assertEqual(params.activation_scale_exp, -7)

        # CLS row uses only the CLS vector plus position row 0.
        cls_row = [
            sat_signed(cls[c] + int(pos[0][c]), 8) for c in range(192)
        ]
        self.assertEqual(activation[0], cls_row)

        # Patch rows recompute independently with an int64 matmul.
        w64 = np.asarray(weight, dtype=np.int64)
        for patch_idx, out_row in ((0, 1), (97, 98), (195, 196)):
            acc = (
                np.asarray([patches[patch_idx]], dtype=np.int64) @ w64
                + np.asarray(bias, dtype=np.int64)[None, :]
            )
            embed = [requant(int(v), -14, -7, 8) for v in acc[0]]
            expected = [
                sat_signed(embed[c] + int(pos[out_row][c]), 8) for c in range(192)
            ]
            self.assertEqual(activation[out_row], expected, f"patch {patch_idx}")

    def test_patch_params_immutable(self):
        params = PatchParams(
            patch_weight=[[0] * 192 for _ in range(768)],
            patch_bias=[0] * 192,
            cls=[0] * 192,
            pos=[[0] * 192 for _ in range(197)],
        )
        with self.assertRaises(Exception):
            params.embed_dim = 100

    def test_mhsa(self):
        n = 9
        # Token 0 is 64-hot in the head lanes; later tokens are hot outside
        # those lanes, so identity Q/K weights give one dominant score row.
        x = []
        for t in range(n):
            row = [0] * 192
            if t == 0:
                for c in range(64):
                    row[c] = 127
            else:
                row[63 + t] = 127
            x.append(row)
        ln_gamma = [64] * 192
        ln_beta = [0] * 192
        wqkv = [[0] * 576 for _ in range(192)]
        bqkv = [0] * 576
        # Q/V: identity for all heads; K is identity, negated or sign-
        # alternating per head so the three score matrices differ pairwise.
        for lane in range(64):
            wqkv[lane][lane] = 1                # Q h0
            wqkv[lane][64 + lane] = 1           # Q h1
            wqkv[lane][128 + lane] = 1          # Q h2
            wqkv[lane][192 + lane] = 1          # K h0
            wqkv[lane][256 + lane] = -1         # K h1 (negated)
            wqkv[lane][320 + lane] = (1 if lane % 2 == 0 else -1)  # K h2
            wqkv[lane][384 + lane] = 1          # V h0
            wqkv[lane][448 + lane] = 1          # V h1
            wqkv[lane][512 + lane] = 1          # V h2

        wproj = [[1 if r == c else 0 for c in range(192)] for r in range(192)]
        bproj = [0] * 192
        params = MhsaParams(
            ln_gamma=ln_gamma,
            ln_beta=ln_beta,
            wqkv=wqkv,
            bqkv=bqkv,
            wproj=wproj,
            bproj=bproj,
            wqkv_scale_exp=0, wproj_scale_exp=0,
        )

        output, ck = mhsa(x, params)
        self.assertEqual(len(output), n)
        self.assertTrue(all(len(row) == 192 for row in output))
        self.assertEqual([len(k) for k in ck["qkv"]], [3, 3, 3])
        self.assertEqual([len(h) for h in ck["qkv"][0]], [n, n, n])
        self.assertEqual(len(ck["score"]), 3)
        self.assertEqual([len(r) for r in ck["score"][0]], [n] * n)
        self.assertEqual(len(ck["prob"]), 3)
        self.assertEqual([len(r) for r in ck["prob"][0]], [n] * n)
        self.assertEqual(len(ck["context"]), 3)
        self.assertEqual([len(r) for r in ck["context"][0]], [64] * n)
        self.assertEqual(len(ck["concat"]), n)

        # Three heads must not share row maxima or denominators.
        self.assertNotEqual(ck["score"][0], ck["score"][1])
        self.assertNotEqual(ck["score"][1], ck["score"][2])
        self.assertNotEqual(ck["score"][0], ck["score"][2])
        self.assertNotEqual(ck["prob"][0], ck["prob"][1])
        self.assertNotEqual(ck["prob"][1], ck["prob"][2])

        # The dominant row catches a missing unsigned-src0 flag downstream.
        # (P2 fix: attention delta2 = 1.0, so the single dominant row
        # saturates UQ0.8 to 255 instead of the old 128.)
        self.assertEqual(ck["prob"][0][0][0], 255)
        # V is identity, so prob=255 times V[0][0]=127 requantizes to 127
        # under unsigned interpretation (signed interpretation would
        # give -127).
        self.assertEqual(ck["context"][0][0][0], 127)
        # weight_scale=0 plus identity projection makes output equal concat.
        self.assertEqual(output, ck["concat"])

        # Independently recompute head 0's score with an int64 matmul.
        q0 = np.asarray(ck["qkv"][0][0], dtype=np.int64)
        k0 = np.asarray(ck["qkv"][1][0], dtype=np.int64)
        acc = q0 @ k0.T
        expected_score = [
            [requant(int(v), -14, -17, 32) for v in row] for row in acc
        ]
        self.assertEqual(ck["score"][0], expected_score)

    def test_ffn(self):
        n = 13
        rng = np.random.default_rng(20260815)
        y = [
            [int(v) for v in row]
            for row in rng.integers(-128, 128, size=(n, 192), dtype=np.int64)
        ]
        ln_gamma = [int(v) for v in rng.integers(0, 128, size=192)]
        ln_beta = [int(v) for v in rng.integers(-32, 32, size=192)]
        w1 = [
            [int(v) for v in row]
            for row in rng.integers(-128, 128, size=(192, 768), dtype=np.int64)
        ]
        w1[0][0] = 127
        w1[0][1] = -128
        b1 = [int(v) for v in rng.integers(-64, 64, size=768)]
        w2 = [
            [int(v) for v in row]
            for row in rng.integers(-128, 128, size=(768, 192), dtype=np.int64)
        ]
        b2 = [int(v) for v in rng.integers(-64, 64, size=192)]
        params = FfnParams(
            ln_gamma=ln_gamma,
            ln_beta=ln_beta,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            w1_scale_exp=0, w2_scale_exp=0,
        )

        z, ck = ffn(y, params)
        self.assertEqual(len(z), n)
        self.assertTrue(all(len(row) == 192 for row in z))
        self.assertEqual(len(ck["ln2"]), n)
        self.assertEqual(len(ck["hidden"]), n)
        self.assertTrue(all(len(row) == 768 for row in ck["hidden"]))
        self.assertEqual(len(ck["ffn_out"]), n)

        # Independently recompute the GELU'd first-layer output.
        ln2 = np.asarray(ck["ln2"], dtype=np.int64)
        acc = ln2 @ np.asarray(w1, dtype=np.int64) + np.asarray(b1, dtype=np.int64)[None, :]
        hidden = []
        for row in acc:
            hidden.append([
                requant(gelu(requant(int(v), -7, -16, 24)), -16, -7, 8)
                for v in row
            ])
        self.assertEqual(ck["hidden"], hidden)

        # Residual with equal scales is a saturating add.
        for r in range(n):
            for c in range(192):
                self.assertEqual(
                    z[r][c],
                    sat_signed(y[r][c] + ck["ffn_out"][r][c], 8),
                )

    def test_block(self):
        n = 13
        rng = np.random.default_rng(20260815)
        x = [
            [int(v) for v in row]
            for row in rng.integers(-128, 128, size=(n, 192), dtype=np.int64)
        ]
        params0 = random_block_params(rng)
        params1 = random_block_params(rng)

        z0, ck0 = transformer_block(x, params0)
        self.assertEqual(len(z0), n)
        self.assertTrue(all(len(row) == 192 for row in z0))
        self.assertEqual(len(ck0["ln1"]), n)
        self.assertTrue(all(len(row) == 192 for row in ck0["ln1"]))
        self.assertEqual([len(h) for h in ck0["qkv"][0]], [n, n, n])
        self.assertEqual(len(ck0["prob"]), 3)
        self.assertEqual([len(r) for r in ck0["prob"][0]], [n] * n)
        self.assertEqual(len(ck0["context"]), 3)
        self.assertEqual([len(r) for r in ck0["context"][0]], [64] * n)
        self.assertEqual(len(ck0["concat"]), n)
        self.assertEqual(len(ck0["msa"]), n)
        self.assertEqual(len(ck0["y"]), n)
        self.assertEqual(len(ck0["ln2"]), n)
        self.assertTrue(all(len(row) == 768 for row in ck0["hidden"]))
        self.assertEqual(len(ck0["ffn_out"]), n)
        self.assertEqual(ck0["z"], z0)

        # Residual1 and Residual2 are saturating adds at the shared scale.
        for r in range(n):
            for c in range(192):
                self.assertEqual(
                    ck0["y"][r][c],
                    sat_signed(x[r][c] + ck0["msa"][r][c], 8),
                )
                self.assertEqual(
                    z0[r][c],
                    sat_signed(ck0["y"][r][c] + ck0["ffn_out"][r][c], 8),
                )

        # Two chained blocks: the second block input is the first block Z.
        z1, ck1 = transformer_block(z0, params1)
        self.assertEqual(len(z1), n)
        self.assertTrue(all(len(row) == 192 for row in z1))
        self.assertEqual(ck1["z"], z1)
        self.assertNotEqual(z0, z1)

    def test_block_order_sensitive(self):
        n = 13
        rng = np.random.default_rng(20260815)
        x = [
            [int(v) for v in row]
            for row in rng.integers(-128, 128, size=(n, 192), dtype=np.int64)
        ]
        params0 = random_block_params(rng)
        z_gold, _ = transformer_block(x, params0)

        # LN1/LN2 swap: MSA takes the FFN gamma/beta and vice versa.
        mhsa_alt = replace(
            params0.mhsa,
            ln_gamma=params0.ffn.ln_gamma,
            ln_beta=params0.ffn.ln_beta,
        )
        ffn_alt = replace(
            params0.ffn,
            ln_gamma=params0.mhsa.ln_gamma,
            ln_beta=params0.mhsa.ln_beta,
        )
        msa_alt, _ = mhsa(x, mhsa_alt)
        y_alt = [
            [sat_signed(x[r][c] + msa_alt[r][c], 8) for c in range(192)]
            for r in range(n)
        ]
        z_ln_swap, _ = ffn(y_alt, ffn_alt)
        self.assertNotEqual(z_ln_swap, z_gold)

        # Residual1/Residual2 swap means the FFN stage (with its residual)
        # runs first and the MSA stage (with its residual) runs second.
        ffn_first, _ = ffn(x, params0.ffn)
        msa_second, _ = mhsa(ffn_first, params0.mhsa)
        z_stage_swap = [
            [sat_signed(ffn_first[r][c] + msa_second[r][c], 8) for c in range(192)]
            for r in range(n)
        ]
        self.assertNotEqual(z_stage_swap, z_gold)


if __name__ == "__main__":
    unittest.main()
