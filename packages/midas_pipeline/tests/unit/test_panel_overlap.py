"""Duplicate spots and relative alignment of overlapping panels (#75 follow-up).

Synthetic spot tables with a known injected misalignment: the module must find
the duplicates, set its own window from the data, and recover the relative
translation / scale / rotation with an error bar that covers the truth.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_pipeline.panel_overlap import (
    C_ETA, C_OME, C_RING, C_TTH, C_Y, C_Z, analyse_pair, analyse_panels,
    duplicate_masks,
)

LSD = 3.3e6                       # um
RADII = {1: 313_000.0, 2: 362_000.0, 3: 514_000.0, 4: 605_000.0}   # um
STEP = 0.5


def _table(y, z, ring, ome):
    """InputAll-order rows from lab positions (eta / 2theta derived from them)."""
    r = np.hypot(y, z)
    eta = np.degrees(np.arctan2(-y, z))
    tth = np.degrees(np.arctan(r / LSD))
    n = y.size
    out = np.zeros((n, 8))
    out[:, 0], out[:, 1], out[:, 2] = y, z, ome
    out[:, 3] = 50.0
    out[:, 4] = np.arange(1, n + 1)
    out[:, 5], out[:, 6], out[:, 7] = ring, eta, tth
    return out


def _spots(rng, n, rings=(1, 2, 3, 4)):
    ring = rng.choice(rings, n)
    rad = np.array([RADII[int(r)] for r in ring])
    eta = rng.uniform(-170, 170, n)
    ome = np.round(rng.uniform(-179, 179, n) / STEP) * STEP
    return _table(-rad * np.sin(np.radians(eta)), rad * np.cos(np.radians(eta)), ring, ome)


def _make_pair(rng, n_dup=300, n_only_a=150, n_only_b=150, *, t=(0.0, 0.0), s=0.0,
               rho=0.0, jitter_um=15.0):
    """Panel a, panel b = same reflections seen through a misaligned panel.

    b's positions are  p_b = p_a mapped by the INVERSE of the transform the
    analysis fits (p_a ~ t + (1+s) R(rho) p_b), plus jitter.
    """
    dup = _spots(rng, n_dup)
    a = np.vstack([dup, _spots(rng, n_only_a)])
    pa = dup[:, [C_Y, C_Z]]
    # p_a = t + (1+s) R p_b   =>   p_b = R^T (p_a - t) / (1+s)
    c, sn = np.cos(rho), np.sin(rho)
    Rt = np.array([[c, sn], [-sn, c]])
    pb = ((pa - np.asarray(t)) / (1.0 + s)) @ Rt.T
    pb = pb + rng.normal(0, jitter_um, pb.shape)
    b_dup = _table(pb[:, 0], pb[:, 1], dup[:, C_RING], dup[:, C_OME])
    b = np.vstack([b_dup, _spots(rng, n_only_b)])
    # fresh, distinct SpotIDs
    a[:, 4] = np.arange(1, a.shape[0] + 1)
    b[:, 4] = np.arange(1, b.shape[0] + 1)
    return a, b, n_dup


def test_finds_the_duplicates_and_nothing_else():
    rng = np.random.default_rng(1)
    a, b, n_dup = _make_pair(rng)
    st = analyse_pair(a, b, 1, 2, lsd_a=LSD, lsd_b=LSD, omega_step_deg=STEP)
    true = {(i, i) for i in range(n_dup)}
    got = set(zip(st.idx_a.tolist(), st.idx_b.tolist()))
    assert len(got & true) >= 0.97 * n_dup               # recall
    assert len(got - true) <= 0.02 * len(got)            # precision
    assert st.n_a == a.shape[0] and st.n_b == b.shape[0]


def test_recovers_translation_scale_rotation_within_its_error_bar():
    rng = np.random.default_rng(2)
    t, s, rho = (140.0, -60.0), 3e-4, 2e-4               # um, fraction, rad
    a, b, _ = _make_pair(rng, t=t, s=s, rho=rho)
    st = analyse_pair(a, b, 1, 2, lsd_a=LSD, lsd_b=LSD, omega_step_deg=STEP)
    assert st.n_pairs > 250
    # the model is linearised in (s, rho); the truth is within a few SE
    for got, se, want in ((st.translation_um[0], st.translation_se_um[0], t[0]),
                          (st.translation_um[1], st.translation_se_um[1], t[1]),
                          (st.scale, st.scale_se, s),
                          (st.rotation_mrad, st.rotation_se_mrad, 1e3 * rho)):
        assert abs(got - want) < 5 * se + 1e-9, (got, se, want)
    assert 5 < st.position_sigma_um < 40                  # ~ the injected jitter
    assert st.condition < 50                              # well-conditioned


def test_a_large_offset_does_not_hide_the_duplicates():
    """A 12-pixel translation moves eta and 2theta by amounts that depend on
    where on the ring the spot is, so no constant offset describes it; matching
    through the fitted transform must still find the pairs and narrow the window."""
    rng = np.random.default_rng(3)
    a, b, n_dup = _make_pair(rng, t=(2500.0, 0.0), jitter_um=10.0)   # 12.5 px at 200 um
    st = analyse_pair(a, b, 1, 2, lsd_a=LSD, lsd_b=LSD, omega_step_deg=STEP)
    assert st.n_pairs >= 0.9 * n_dup
    assert abs(st.translation_um[0] - 2500.0) < 5 * st.translation_se_um[0] + 1.0
    assert abs(st.d_eta[0]) > 0.02 or abs(st.d_eta[1]) > 0.05   # the offset shows in eta
    assert st.tol_pos_um < 500.0                          # window narrowed to the data


def test_no_overlap_gives_no_pairs():
    rng = np.random.default_rng(4)
    a, b = _spots(rng, 200), _spots(rng, 200)
    st = analyse_pair(a, b, 1, 2, lsd_a=LSD, lsd_b=LSD, omega_step_deg=STEP)
    # chance coincidences inside a window that is only a few sigma wide are rare
    assert st.n_pairs <= 3


def test_omega_wraps_at_180():
    rng = np.random.default_rng(5)
    a, b, n_dup = _make_pair(rng, n_dup=200)
    a[:50, C_OME] = 179.75
    b[:50, C_OME] = -179.75                               # same reflection across the seam
    st = analyse_pair(a, b, 1, 2, lsd_a=LSD, lsd_b=LSD, omega_step_deg=STEP)
    got = set(zip(st.idx_a.tolist(), st.idx_b.tolist()))
    assert sum((i, i) in got for i in range(50)) >= 45


def test_panels_with_different_lsd_are_compared_on_one_plane():
    rng = np.random.default_rng(6)
    a, b, n_dup = _make_pair(rng)
    lsd_b = 1.04 * LSD
    b[:, [C_Y, C_Z]] *= 1.04                              # the same rays on a farther plane
    b[:, C_TTH] = np.degrees(np.arctan(np.hypot(b[:, C_Y], b[:, C_Z]) / lsd_b))
    st = analyse_pair(a, b, 1, 2, lsd_a=LSD, lsd_b=lsd_b, omega_step_deg=STEP)
    assert abs(st.scale) < 5 * st.scale_se + 1e-6         # no spurious scale


def test_duplicate_masks_drop_exactly_one_record_per_pair():
    rng = np.random.default_rng(7)
    a, b, n_dup = _make_pair(rng)
    pairs = analyse_panels([a, b], [1, 2], [LSD, LSD], omega_step_deg=STEP)
    assert len(pairs) == 1
    drop = duplicate_masks([a, b], pairs, [1, 2])
    n = pairs[0].n_pairs
    assert drop[1].sum() == n and drop[0].sum() == 0      # earlier panel wins by default
    # a quality score overrides it: prefer b everywhere -> a's records are dropped
    drop2 = duplicate_masks([a, b], pairs, [1, 2],
                            prefer=[np.zeros(a.shape[0]), np.ones(b.shape[0])])
    assert drop2[0].sum() == n and drop2[1].sum() == 0
