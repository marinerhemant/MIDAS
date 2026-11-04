"""Merge statistics and the SHELX .hkl round trip."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from midas_hkls import SpaceGroup
from midas_sx.merge import merge_observations, r_int, read_hkl, write_hkl

SG = SpaceGroup.from_number(225)           # Fm-3m


def _obs(rows):
    return pd.DataFrame(rows, columns=["h", "k", "l", "F2", "sigF2"])


def test_equivalents_merge_to_one_row_with_a_weighted_mean():
    obs = _obs([(1, 1, 1, 100.0, 10.0), (-1, 1, 1, 120.0, 10.0), (1, -1, 1, 110.0, 5.0),
                (2, 0, 0, 50.0, 5.0)])
    m = merge_observations(obs, SG).set_index(["h", "k", "l"])
    assert len(m) == 2
    row = m.loc[SG.asu_representative(1, 1, 1)]
    w = np.array([1 / 100, 1 / 100, 1 / 25])
    assert row["F2"] == pytest.approx((w * np.array([100, 120, 110])).sum() / w.sum())
    assert row["n"] == 3
    assert row["sigF2"] >= row["sig_int"] - 1e-12 and row["sigF2"] >= row["sig_ext"] - 1e-12


def test_friedel_mates_are_merged():
    obs = _obs([(1, 2, 3, 10.0, 1.0), (-1, -2, -3, 12.0, 1.0)])
    assert len(merge_observations(obs, SpaceGroup.from_number(1))) == 1      # P1 still merges Friedel mates


def test_r_int_is_zero_for_identical_equivalents_and_nan_without_any():
    same = _obs([(1, 1, 1, 100.0, 5.0), (-1, 1, 1, 100.0, 5.0)])
    assert r_int(same, SG) == pytest.approx(0.0, abs=1e-15)
    single = _obs([(1, 1, 1, 100.0, 5.0), (2, 0, 0, 50.0, 5.0)])
    assert np.isnan(r_int(single, SG))
    off = _obs([(1, 1, 1, 90.0, 5.0), (-1, 1, 1, 110.0, 5.0)])
    assert r_int(off, SG) == pytest.approx(20.0 / 200.0)


def test_single_observation_keeps_its_own_sigma():
    m = merge_observations(_obs([(2, 0, 0, 50.0, 5.0)]), SG)
    assert m["sigF2"].iloc[0] == pytest.approx(5.0) and m["n"].iloc[0] == 1


def test_hkl_round_trip_and_format(tmp_path):
    m = pd.DataFrame(dict(h=[1, -2, 10], k=[1, 0, -3], l=[1, 4, 12], F2=[1.0e5, -3.5, 250.25], sigF2=[2.0e3, 1.2, 9.9]))
    p = tmp_path / "x.hkl"
    scale = write_hkl(p, m)
    lines = p.read_text().splitlines()
    assert lines[-1].split()[:3] == ["0", "0", "0"]
    assert all(len(line) == 32 for line in lines)                      # (3I4, 2F8.2, I4)
    back = read_hkl(p)
    assert list(zip(back.h, back.k, back.l)) == list(zip(m.h, m.k, m.l))
    assert np.allclose(back.F2.to_numpy() / scale, m.F2.to_numpy(), rtol=2e-4, atol=1e-4 / scale * 100)
    assert back.F2.abs().max() <= 9999.0 + 1e-6


def test_hkl_refuses_indices_that_do_not_fit_the_field():
    m = pd.DataFrame(dict(h=[1000], k=[0], l=[0], F2=[1.0], sigF2=[0.1]))
    with pytest.raises(ValueError):
        write_hkl("/dev/null", m)
