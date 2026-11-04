"""The 16 candidate conventions: eight distinct bijections of the array, point and array transforms agree, and a tie is not broken."""
from __future__ import annotations

import numpy as np
import pytest

from midas_sx.conventions import Convention, candidate_conventions, rank_conventions, transform_frame, transform_rc


def test_sixteen_distinct_candidates():
    cs = candidate_conventions()
    assert len(cs) == 16 and len({c.name for c in cs}) == 16


def test_point_and_array_transforms_agree_for_all_eight():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1000, size=(5, 6, 9))
    r, c = np.meshgrid(np.arange(6), np.arange(9), indexing="ij")
    for k in range(8):
        t = transform_frame(a, k)
        rr, cc = transform_rc(r, c, 6, 9, k)
        assert np.allclose(rr, np.round(rr)) and np.allclose(cc, np.round(cc))
        assert np.array_equal(t[:, rr.astype(int), cc.astype(int)], a[:, r, c]), k
        assert t.shape[-2:] == ((9, 6) if k >= 4 else (6, 9))


def test_the_eight_transforms_are_distinct_on_a_generic_array():
    a = np.arange(30).reshape(5, 6) * 1.0 + np.arange(30).reshape(5, 6)[::-1] ** 2
    imgs = [transform_frame(a, k) for k in range(8)]
    for i in range(8):
        for j in range(i + 1, 8):
            assert imgs[i].shape != imgs[j].shape or not np.array_equal(imgs[i], imgs[j]), (i, j)


def test_rank_refuses_a_tie_and_a_zero_score():
    cs = candidate_conventions()
    best, s1, ok = rank_conventions([(cs[0], 100.0), (cs[1], 95.0), (cs[2], 10.0)])
    assert best == cs[0] and s1 == 95.0 and not ok                       # 5 % margin < 20 %
    _, _, ok = rank_conventions([(cs[0], 100.0), (cs[1], 60.0)])
    assert ok
    _, _, ok = rank_conventions([(cs[0], 0.0), (cs[1], 0.0)])
    assert not ok
