import unittest

import numpy as np

from verification.heatvit_ref.fixed import round_div, sat_signed
from verification.heatvit_ref.selector import (
    KEEP_THRESHOLD,
    Q016_ONE,
    WARN_HEAD_DEN_ZERO,
    WARN_PACKAGE_DEN_ZERO,
    SelectorHeadWeightParams,
    SelectorLocalParams,
    SelectorParams,
    SelectorScoreParams,
    concat_local_global,
    finalize_tokens,
    fuse_head_scores,
    head_weight_mlp,
    mean_over_candidates,
    mean_over_head_lanes,
    per_head_local_mlp,
    per_head_score_mlp,
    token_selector,
)


def sentinel_tokens(n):
    """n tokens; token t fills all 192 channels with a distinct int8 pattern."""
    return [
        [((t * 53 + 7) % 251) - 125] * 192
        for t in range(n)
    ]


class SelectorTest(unittest.TestCase):
    def test_threshold_is_inclusive(self):
        result = finalize_tokens(
            cls=[1, 2], normal=[[3, 4], [5, 6]], incoming_package=None,
            scores=[32768, 32767])
        self.assertEqual([list(t) for t in result.tokens],
                         [[1, 2], [3, 4], [5, 6]])
        self.assertTrue(result.package_present)

    def test_incoming_package_never_becomes_normal(self):
        result = finalize_tokens(
            cls=[1], normal=[[2]], incoming_package=[9],
            scores=[65536, 65536])
        self.assertEqual([list(t) for t in result.tokens],
                         [[1], [2], [9]])
        self.assertTrue(result.package_present)

    def test_all_keep_no_package(self):
        n = 8
        tokens = sentinel_tokens(n)
        scores = [40000 + i * 10 for i in range(n - 1)]
        result = finalize_tokens(tokens[0], tokens[1:], None, scores)
        self.assertEqual([list(t) for t in result.tokens], tokens)
        self.assertFalse(result.package_present)
        self.assertEqual(result.kept_normal_count, n - 1)
        self.assertEqual(result.pruned_normal_count, 0)
        self.assertEqual(result.warnings, 0)
        self.assertEqual(len(result.tokens), n)

    def test_all_pruned_forms_single_package(self):
        n = 8
        tokens = sentinel_tokens(n)
        scores = [1000 + i * 100 for i in range(n - 1)]
        result = finalize_tokens(tokens[0], tokens[1:], None, scores)
        # CLS plus exactly one Package; weighted mean of all 7 candidates.
        self.assertEqual(len(result.tokens), 2)
        self.assertTrue(result.package_present)
        self.assertEqual(result.kept_normal_count, 0)
        self.assertEqual(result.pruned_normal_count, n - 1)
        den = sum(scores)
        expected = []
        for d in range(192):
            num = sum(scores[t] * tokens[1 + t][d] for t in range(n - 1))
            expected.append(sat_signed(round_div(num, den), 8))
        self.assertEqual(list(result.tokens[1]), expected)

    def test_mixed_pruning_stable_order(self):
        n = 12
        tokens = sentinel_tokens(n)
        # Alternating keep/prune scores.
        scores = [50000 if t % 2 == 0 else 1000 for t in range(n - 1)]
        result = finalize_tokens(tokens[0], tokens[1:], None, scores)
        kept = [tokens[1 + t] for t in range(n - 1) if t % 2 == 0]
        pruned = [(tokens[1 + t], scores[t]) for t in range(n - 1) if t % 2 == 1]
        self.assertEqual(
            [list(t) for t in result.tokens[1:1 + len(kept)]], kept)
        self.assertEqual(len(result.tokens), 1 + len(kept) + 1)
        self.assertTrue(result.package_present)
        den = sum(s for _, s in pruned)
        package = [
            sat_signed(round_div(sum(s * tok[d] for tok, s in pruned), den), 8)
            for d in range(192)
        ]
        self.assertEqual(list(result.tokens[-1]), package)

    def test_threshold_equal_keeps(self):
        n = 6
        tokens = sentinel_tokens(n)
        scores = [32768, 32767, 32768, 32767, 32768]
        result = finalize_tokens(tokens[0], tokens[1:], None, scores)
        # Exactly 32768 keeps, 32767 prunes.
        self.assertEqual(result.kept_normal_count, 3)
        self.assertEqual(result.pruned_normal_count, 2)
        self.assertEqual(len(result.tokens), 5)
        self.assertTrue(result.package_present)

    def test_incoming_package_with_mixed_pruning(self):
        n = 8
        tokens = sentinel_tokens(n)
        # Incoming package is the last candidate; its score is huge so it
        # would keep, but it must accumulate instead of becoming normal.
        scores = [50000, 50000, 1000, 1000, 1000, 1000, 65536]
        result = finalize_tokens(tokens[0], tokens[1:-1], tokens[-1], scores)
        self.assertEqual(result.kept_normal_count, 2)
        self.assertEqual(result.pruned_normal_count, 4)
        self.assertEqual(len(result.tokens), 4)  # CLS + 2 kept + 1 package
        self.assertEqual(list(result.tokens[1]), list(tokens[1]))
        self.assertEqual(list(result.tokens[2]), list(tokens[2]))
        self.assertEqual(result.tokens[-1] not in [tuple(t) for t in tokens[1:-1]],
                         True)
        # Package includes the incoming package token with its own score.
        participants = [(tokens[1 + t], scores[t]) for t in range(2, 6)]
        participants.append((tokens[-1], 65536))
        den = sum(s for _, s in participants)
        package = [
            sat_signed(round_div(sum(s * tok[d] for tok, s in participants),
                                 den), 8)
            for d in range(192)
        ]
        self.assertEqual(list(result.tokens[-1]), package)

    def test_package_den_zero_fallback(self):
        n = 6
        tokens = sentinel_tokens(n)
        scores = [0] * (n - 1)
        result = finalize_tokens(tokens[0], tokens[1:], None, scores)
        self.assertTrue(result.package_present)
        self.assertEqual(result.warnings, WARN_PACKAGE_DEN_ZERO)
        # Unweighted mean of the pruned candidates.
        expected = [
            sat_signed(round_div(sum(tokens[1 + t][d] for t in range(n - 1)),
                                 n - 1), 8)
            for d in range(192)
        ]
        self.assertEqual(list(result.tokens[-1]), expected)
        self.assertEqual(len(result.tokens), 2)

    def test_negative_package_numerator_rounds_away(self):
        # Channel 0 has a negative weighted numerator whose remainder is half
        # the denominator (ties round away); channel 1 divides exactly.
        token_a = [0, 100] + [100] * 190
        token_b = [-19, 0] + [-19] * 190
        result = finalize_tokens(
            cls=[0] * 192, normal=[token_a, token_b], incoming_package=None,
            scores=[3, 1])
        # Channel 0: 3*0 + 1*(-19) = -19; -19/4 = -4.75 -> -5.
        self.assertEqual(result.tokens[-1][0], -5)
        # Channel 1: 3*100 + 1*0 = 300; 300/4 = 75 exactly.
        self.assertEqual(result.tokens[-1][1], 75)

    def test_head_fuse_normal(self):
        scores = [[0], [32768], [65536]]       # 3 heads x 1 candidate
        weights = [[65536, 65536, 0]]          # 1 candidate x 3 heads
        fused, warn = fuse_head_scores(scores, weights)
        self.assertEqual(fused, (16384,))
        self.assertEqual(warn, 0)

    def test_head_fuse_zero_den(self):
        scores = [[1000], [2000], [3000]]
        weights = [[0, 0, 0]]
        fused, warn = fuse_head_scores(scores, weights)
        # Equal-weight mean: 6000 / 3 = 2000, warning bit set.
        self.assertEqual(fused, (2000,))
        self.assertEqual(warn, WARN_HEAD_DEN_ZERO)

    def test_head_fuse_saturates(self):
        scores = [[65536], [65536], [65536]]
        weights = [[65536, 65536, 65536]]
        fused, warn = fuse_head_scores(scores, weights)
        self.assertEqual(fused, (Q016_ONE,))
        self.assertEqual(warn, 0)

    def test_mean_over_candidates_negative_ties(self):
        local = [
            [[-2], [-1], [0], [1], [2]],
            [[0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0]],
        ]
        self.assertEqual(mean_over_candidates(local)[0], (0,))
        local = [
            [[-2], [-1]],
            [[0], [0]],
            [[0], [0]],
        ]
        # (-3)/2 -> -1.5 -> ties away from zero -> -2.
        self.assertEqual(mean_over_candidates(local)[0], (-2,))

    def test_concat_local_global_layout(self):
        local_fill = [0x11, 0x33, 0x55]
        local = [
            [[local_fill[h]] * 32, [(local_fill[h] + 0x10)] * 32]
            for h in range(3)
        ]
        glob = [
            [0xAA] * 32,
            [0xBB] * 32,
            [0xCC] * 32,
        ]
        out = concat_local_global(local, glob)
        self.assertEqual(len(out), 3)
        for h in range(3):
            self.assertEqual(len(out[h]), 2)
            self.assertEqual(len(out[h][0]), 64)
            self.assertEqual(list(out[h][0][:32]), [local_fill[h]] * 32)
            self.assertEqual(list(out[h][0][32:]), list(glob[h]))
            # Every candidate's last 32 lanes come from its own head's global.
            self.assertEqual(list(out[h][1][32:]), list(glob[h]))

    def test_mean_over_head_lanes(self):
        candidates = [
            (tuple(range(64)), tuple([1] * 64), tuple([-1] * 64)),
        ]
        stats = mean_over_head_lanes(candidates)
        # mean(range(64)) = 31.5 -> ties away -> 32.
        self.assertEqual(stats[0], (32, 1, -1))

    def test_token_selector_end_to_end(self):
        n = 6
        rng = np.random.default_rng(20260815)
        tokens = [
            [int(v) for v in row]
            for row in rng.integers(-128, 128, size=(n, 192), dtype=np.int64)
        ]
        params = SelectorParams(
            local=SelectorLocalParams(
                w=[[[int(v) for v in row] for row in
                    rng.integers(-8, 8, size=(64, 32), dtype=np.int64)]
                   for _ in range(3)],
                b=[[int(v) for v in rng.integers(-64, 64, size=32)]
                   for _ in range(3)],
            ),
            score=SelectorScoreParams(
                w1=[[[int(v) for v in row] for row in
                     rng.integers(-8, 8, size=(64, 32), dtype=np.int64)]
                    for _ in range(3)],
                b1=[[int(v) for v in rng.integers(-64, 64, size=32)]
                    for _ in range(3)],
                w2=[[[int(v) for v in row] for row in
                     rng.integers(-8, 8, size=(32, 16), dtype=np.int64)]
                    for _ in range(3)],
                b2=[[int(v) for v in rng.integers(-64, 64, size=16)]
                    for _ in range(3)],
                w3=[[[int(v) for v in row] for row in
                     rng.integers(-8, 8, size=(16, 2), dtype=np.int64)]
                    for _ in range(3)],
                b3=[[int(v) for v in rng.integers(-64, 64, size=2)]
                    for _ in range(3)],
            ),
            head_weight=SelectorHeadWeightParams(
                w1=[[int(v) for v in row] for row in
                    rng.integers(-8, 8, size=(3, 3), dtype=np.int64)],
                b1=[int(v) for v in rng.integers(-64, 64, size=3)],
                w2=[[int(v) for v in row] for row in
                    rng.integers(-8, 8, size=(3, 3), dtype=np.int64)],
                b2=[int(v) for v in rng.integers(-64, 64, size=3)],
            ),
        )

        result = token_selector(tokens, False, params)
        self.assertEqual(len(result.tokens), result.token_count)
        self.assertTrue(2 <= result.token_count <= n)
        self.assertEqual(len(result.local), 3)
        self.assertEqual(len(result.local[0]), n - 1)
        self.assertEqual(len(result.local[0][0]), 32)
        self.assertEqual(len(result.global_features), 3)
        self.assertEqual(len(result.global_features[0]), 32)
        self.assertEqual(len(result.head_scores), 3)
        self.assertEqual(len(result.head_scores[0]), n - 1)
        self.assertEqual(len(result.head_stats), n - 1)
        self.assertEqual(len(result.head_stats[0]), 3)
        self.assertEqual(len(result.head_weights), n - 1)
        self.assertEqual(len(result.head_weights[0]), 3)
        self.assertEqual(len(result.fused_scores), n - 1)
        for score in result.fused_scores:
            self.assertTrue(0 <= score <= Q016_ONE)

        # Output equals a manual finalize on the fused scores.
        manual = finalize_tokens(
            tokens[0], tokens[1:], None, list(result.fused_scores))
        self.assertEqual(list(result.tokens), list(manual.tokens))
        self.assertEqual(result.package_present, manual.package_present)
        self.assertEqual(result.kept_normal_count, manual.kept_normal_count)
        self.assertEqual(result.pruned_normal_count, manual.pruned_normal_count)

        # With an incoming package, the last candidate must accumulate.
        result_pkg = token_selector(tokens, True, params)
        manual_pkg = finalize_tokens(
            tokens[0], tokens[1:-1], tokens[-1], list(result_pkg.fused_scores))
        self.assertEqual(list(result_pkg.tokens), list(manual_pkg.tokens))
        self.assertEqual(result_pkg.token_count, len(manual_pkg.tokens))

    def test_keep_threshold_constant(self):
        self.assertEqual(KEEP_THRESHOLD, 32768)

    def test_three_consecutive_finalize(self):
        # Stage 1: no incoming Package, mixed pruning -> one trailing Package.
        tokens = sentinel_tokens(12)
        scores = [50000 if t % 2 == 0 else 1000 for t in range(11)]
        r1 = finalize_tokens(tokens[0], tokens[1:], None, scores)
        self.assertTrue(r1.package_present)
        self.assertEqual(len(r1.tokens), 8)

        # Stage 2: previous output as input; the trailing Package is the
        # last candidate and more normals are pruned.
        scores2 = [50000, 1000, 50000, 1000, 50000, 1000, 30000]
        r2 = finalize_tokens(r1.tokens[0], list(r1.tokens[1:-1]),
                             r1.tokens[-1], scores2)
        self.assertTrue(r2.package_present)
        self.assertLessEqual(len(r2.tokens), len(r1.tokens))
        self.assertEqual(len(r2.tokens), 5)

        # Stage 3: chain again with a fresh prune; token count must stay
        # non-increasing and the CLS + kept order must be preserved.
        scores3 = [40000, 1000, 40000, 65536]
        r3 = finalize_tokens(r2.tokens[0], list(r2.tokens[1:-1]),
                             r2.tokens[-1], scores3)
        self.assertTrue(r3.package_present)
        self.assertLessEqual(len(r3.tokens), len(r2.tokens))
        self.assertEqual(len(r3.tokens), 4)

        # Exactly one trailing Package per stage: the last token of each
        # output is the Package and all earlier outputs are CLS + kept.
        for result in (r1, r2, r3):
            self.assertTrue(result.package_present)
            self.assertEqual(len(result.tokens),
                             1 + result.kept_normal_count + 1)
            # CLS stays first and kept normals keep their relative order.
            self.assertEqual(list(result.tokens[0]), tokens[0])

        # Kept normals of stage 2 preserve the input order.
        kept2 = [list(t) for t in r2.tokens[1:-1]]
        self.assertEqual(kept2, [list(r1.tokens[1]), list(r1.tokens[3]),
                                 list(r1.tokens[5])])


if __name__ == "__main__":
    unittest.main()
