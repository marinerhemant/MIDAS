"""Tests for the polytype-cell reflection indexer (9R mixed-hkl)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from midas_defect.polytype.cell_index import (
    NINE_R_SEQUENCE,
    PolytypeCell,
    close_packed_basis,
    index_reflections,
    nine_r_from_fcc,
    polytype_reflections,
    structure_factor_intensity,
)

A_FCC = 3.6356
B0 = 2.0 * math.pi / A_FCC
G = B0 * math.sqrt(3.0)  # |G_111|


def test_nine_r_cell_from_fcc():
    cell = nine_r_from_fcc(A_FCC)
    assert cell.a == pytest.approx(A_FCC / math.sqrt(2), rel=1e-6)
    assert cell.c == pytest.approx(9.0 * A_FCC / math.sqrt(3), rel=1e-6)
    assert cell.centering == "R-obverse"


def test_00l_rod_reproduces_n_over_3_ladder():
    """The 9R (0 0 l) rod with l=3,6,9,12,15,18 = the n*G/3 <111> ladder exactly."""
    cell = nine_r_from_fcc(A_FCC)
    U = np.eye(3)
    hkls, G_s = polytype_reflections(cell, U, (1, 1, 1), q_max_inv_A=6.2)
    qmag = np.linalg.norm(G_s, axis=1)
    for l, n in [(3, 1), (6, 2), (9, 3), (12, 4), (15, 5), (18, 6)]:
        # find the (0 0 +/-l) reflection
        m = (hkls[:, 0] == 0) & (hkls[:, 1] == 0) & (np.abs(hkls[:, 2]) == l)
        assert m.any(), f"(0 0 {l}) missing"
        assert qmag[m][0] == pytest.approx(n * G / 3.0, rel=2e-3)


def test_r_centering_obverse_rule():
    cell = nine_r_from_fcc(A_FCC)  # obverse: -h+k+l = 3n
    assert cell.allowed(0, 0, 3)       # 3 -> 3n
    assert not cell.allowed(0, 0, 1)   # 1
    assert not cell.allowed(1, 0, 2)   # -1+0+2 = 1
    assert cell.allowed(1, 0, 1)       # -1+0+1 = 0 -> 3n
    # reverse setting (the other stacking polarity) flips the rule
    rev = PolytypeCell(cell.a, cell.c, centering="R-reverse")
    assert rev.allowed(1, 0, 2)        # 1-0+2 = 3 -> 3n
    assert not rev.allowed(1, 0, 1)    # 1-0+1 = 2


def test_close_packed_basis_is_9r():
    b = close_packed_basis(NINE_R_SEQUENCE)
    assert b.shape == (9, 3)
    # equal d_111 spacing along c
    assert np.allclose(b[:, 2], np.arange(9) / 9.0)


def test_on_axis_satellites_are_structurally_extinct():
    """The crux of the volume-fraction question: an IDEAL 9R has NO on-axis n*G/3
    satellites. The R-centering-allowed (0 0 l) with l != 9n have |F|^2 = 0; only
    the FCC fundamentals (0 0 9)=111 and (0 0 18)=222 survive (|F|^2 = N^2 = 81)."""
    assert structure_factor_intensity((0, 0, 9)) == pytest.approx(81.0, abs=1e-6)
    assert structure_factor_intensity((0, 0, 18)) == pytest.approx(81.0, abs=1e-6)
    for l in (3, 6, 12, 15):  # G/3, 2G/3, 4G/3, 5G/3 -- forbidden gaps
        assert structure_factor_intensity((0, 0, l)) == pytest.approx(0.0, abs=1e-9)


def test_real_9r_superlattice_reflections_are_off_axis():
    """The genuine (nonzero-F) 9R superlattice reflections are mixed-hkl and lie far
    off the <111> axis (~83 deg), not on the G/3 ladder."""
    cell = nine_r_from_fcc(A_FCC)
    hkls, G_s, inten = polytype_reflections(
        cell, np.eye(3), (1, 1, 1), q_max_inv_A=3.1, with_intensity=True
    )
    qmag = np.linalg.norm(G_s, axis=1)
    cstar = G_s[(hkls[:, 0] == 0) & (hkls[:, 1] == 0) & (hkls[:, 2] == 9)][0]
    cstar = cstar / np.linalg.norm(cstar)
    # the strongest non-fundamental reflection below |q|<3.1
    nonfund = ~((hkls[:, 0] == 0) & (hkls[:, 1] == 0))
    j = np.where(nonfund)[0][np.argmax(inten[nonfund])]
    ang = np.degrees(np.arccos(min(1.0, abs(G_s[j] @ cstar) / qmag[j])))
    assert inten[j] > 1.0          # genuinely present
    assert ang > 60.0              # but far off the axis


def test_min_rel_intensity_drops_extinct_on_axis_rod():
    """Filtering by |F|^2 removes the extinct (0 0 l != 9n) rod points that
    R-centering alone would keep."""
    cell = nine_r_from_fcc(A_FCC)
    hkls, _ = polytype_reflections(cell, np.eye(3), (1, 1, 1), q_max_inv_A=6.2)
    on_axis_weak = ((hkls[:, 0] == 0) & (hkls[:, 1] == 0)
                    & (np.abs(hkls[:, 2]) % 9 != 0))
    assert on_axis_weak.any()      # geometric set keeps them
    hkls2, _ = polytype_reflections(
        cell, np.eye(3), (1, 1, 1), q_max_inv_A=6.2, min_rel_intensity=1e-3
    )
    kept_weak = ((hkls2[:, 0] == 0) & (hkls2[:, 1] == 0)
                 & (np.abs(hkls2[:, 2]) % 9 != 0))
    assert not kept_weak.any()     # intensity filter removes them


def test_period3_modulation_turns_on_axis_satellites():
    """The on-axis n*G/3 satellites are EXTINCT for an ideal 9R but turn on under a
    period-3 modulation, ~ amplitude^2 -- i.e. they measure modulation, not volume."""
    # ideal: extinct
    assert structure_factor_intensity((0, 0, 6)) == pytest.approx(0.0, abs=1e-9)
    # spacing modulation -> nonzero, and grows ~quadratically with amplitude
    i_small = structure_factor_intensity((0, 0, 6), spacing_modulation=0.02)
    i_big = structure_factor_intensity((0, 0, 6), spacing_modulation=0.04)
    assert i_small > 1e-6
    assert i_big / i_small == pytest.approx(4.0, rel=0.15)  # (0.04/0.02)^2 = 4
    # composition modulation also turns it on
    assert structure_factor_intensity((0, 0, 6), comp_modulation=0.05) > 1e-6
    # the fundamental stays strong; the off-axis 1st-order reflection is ~modulation-
    # independent (volume probe), unlike the on-axis satellite
    f0 = structure_factor_intensity((0, 0, 9))
    f1 = structure_factor_intensity((0, 0, 9), spacing_modulation=0.04)
    assert f1 == pytest.approx(f0, rel=0.05)


def test_sf_polarity_matches_centering_rule():
    """Nonzero |F|^2 reflections obey the centering rule of their polarity:
    obverse -> -h+k+l=3n, reverse -> h-k+l=3n. The obverse/reverse pair are the
    two stacking senses (the polytype's own twin / doublet origin)."""
    rng = np.random.default_rng(1)
    hkls = rng.integers(-3, 4, size=(400, 3))
    hkls = hkls[(hkls != 0).any(axis=1)]
    Iobv = structure_factor_intensity(hkls, reverse=False)
    Irev = structure_factor_intensity(hkls, reverse=True)
    obv_nonzero = hkls[Iobv > 1e-6]
    rev_nonzero = hkls[Irev > 1e-6]
    assert np.all((-obv_nonzero[:, 0] + obv_nonzero[:, 1] + obv_nonzero[:, 2]) % 3 == 0)
    assert np.all((rev_nonzero[:, 0] - rev_nonzero[:, 1] + rev_nonzero[:, 2]) % 3 == 0)


def test_index_synthetic_9r_reflection_is_significant():
    """A reflection placed exactly on a 9R lattice point indexes with ~0 residual
    and beats the random-orientation null; an off-lattice point does not."""
    cell = nine_r_from_fcc(A_FCC)
    U = np.eye(3)
    c_axes = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]
    hkls, G_s = polytype_reflections(cell, U, (1, 1, 1), q_max_inv_A=6.7)
    # pick a genuine mixed-index reflection (l not a multiple making it the rod)
    mixed = np.where((hkls[:, 0] != 0) & (np.linalg.norm(G_s, axis=1) > 5.5))[0]
    target = G_s[mixed[0]]

    rng = np.random.default_rng(0)
    res = index_reflections(
        np.vstack([target, target + np.array([0.6, -0.4, 0.5])]),
        cell, U, c_axes, null_trials=60, rng=rng,
    )
    # on-lattice point: tiny residual, significant
    assert res[0]["residual_inv_A"] < 0.02
    assert res[0]["significant"]
    # deliberately displaced point: not on the lattice, not significant
    assert res[1]["residual_inv_A"] > res[0]["residual_inv_A"]
    assert not res[1]["significant"]


def test_index_returns_expected_keys():
    cell = nine_r_from_fcc(A_FCC)
    U = np.eye(3)
    res = index_reflections(np.array([[3.0, 0.0, 0.0]]), cell, U,
                            [(1, 1, 1)], null_trials=20,
                            rng=np.random.default_rng(1))
    assert set(res[0]) >= {
        "q", "hkl", "grain", "c_axis", "residual_inv_A",
        "null_floor_inv_A", "indexed", "significant",
    }
