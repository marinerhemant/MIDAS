"""Harmonic collisions: a Pilatus cannot separate two reflections that
diffract at different energies onto the same pixel.

The clearest, most common source of collision is not a coincidence between
unrelated reflections -- it is EXACT: any integer multiple n*(h, k, L) of a
point already on the rod shares its direction, and therefore its pixel,
by construction (confirmed numerically below to machine precision). A
correct collision detector must catch this, not just approximate near-misses.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import itertools
import math
import numpy as np
import pytest

from midas_defect.rod_profile import (
    rod_path_laue, laue_predict_hkl, rod_collisions,
)

LSD, PX = 349_622.0, 172.0
NR, NC, BR, BC = 1679, 1475, 810.3, 737.2
A, C = 3.6116, 19.2516
B_MAT = np.diag([1 / A, 1 / A, 1 / C])
LAUE_KW = dict(lsd_um=LSD, pixel_um=PX, bc_row=BR, bc_col=BC,
               n_rows=NR, n_cols=NC, E_lo_keV=11.0, E_hi_keV=90.0)


def _rot(ax, deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    if ax == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


UROT = _rot("z", 20.0) @ _rot("x", -78.0)  # same as test_rod_profile_laue.py
H, K = 0, 1


def _candidate_box(hmax=6, kmax=6, lmax=15):
    hkl = [h for h in itertools.product(range(-hmax, hmax + 1), range(-kmax, kmax + 1),
                                        range(-lmax, lmax + 1)) if h != (0, 0, 0)]
    return np.array(hkl, float)


# ------------------------------------------------------------- laue_predict_hkl

def test_laue_predict_hkl_matches_rod_path_laue_for_a_single_hkl():
    for L in (0.0, 1.0):
        single = rod_path_laue(UROT, B_MAT, H, K, [L], **LAUE_KW)
        assert len(single) == 1
        pred = laue_predict_hkl(UROT, B_MAT, np.array([[H, K, L]]), **LAUE_KW)
        assert len(pred["row"]) == 1
        assert pred["row"][0] == pytest.approx(single.row[0], abs=1e-9)
        assert pred["col"][0] == pytest.approx(single.col[0], abs=1e-9)
        assert pred["energy_keV"][0] == pytest.approx(single.energy_keV[0], abs=1e-9)


def test_laue_predict_hkl_returns_only_the_accessible_subset():
    # (0, 0, 5) is never forward-scattering under U=I (see test_rod_profile_laue's
    # backscatter test); (H, K, 0) and (H, K, 1) are confirmed accessible.
    hkl = np.array([[0, 0, 5], [H, K, 0], [H, K, 1]], float)
    pred = laue_predict_hkl(UROT, B_MAT, hkl, **LAUE_KW)
    assert len(pred["row"]) == 2
    assert set(map(tuple, pred["hkl"])) == {(H, K, 0.0), (H, K, 1.0)}


def test_laue_predict_hkl_rejects_bad_shape():
    with pytest.raises(ValueError):
        laue_predict_hkl(UROT, B_MAT, np.array([1.0, 2.0, 3.0]), **LAUE_KW)


# ------------------------------------------------------------- rod_collisions

def test_rod_collisions_finds_an_exact_harmonic_alias():
    """(0, 2, -1) = 2 * (0, 1, -0.5): same direction as a point on our own
    rod, hence the exact same pixel -- a real, exact collision, not a
    near-miss, and from a genuinely different (h, k) row."""
    L_values = np.arange(-2, 3, 0.05)
    path = rod_path_laue(UROT, B_MAT, H, K, L_values, **LAUE_KW)
    cands = _candidate_box()
    res = rod_collisions(path, UROT, B_MAT, cands, **LAUE_KW)

    i = int(np.argmin(np.abs(path.L - (-0.5))))
    assert path.L[i] == pytest.approx(-0.5, abs=1e-9)
    assert res["collided"][i]
    assert tuple(res["nearest_hkl"][i]) == (0.0, 2.0, -1.0)
    assert res["nearest_dist_px"][i] < 1e-6


def test_rod_collisions_leaves_some_points_clean():
    """Not every point on this rod aliases with something in a modest box --
    the detector must distinguish contaminated from clean points, not flag
    everything or nothing."""
    L_values = np.arange(-2, 3, 0.05)
    path = rod_path_laue(UROT, B_MAT, H, K, L_values, **LAUE_KW)
    cands = _candidate_box()
    res = rod_collisions(path, UROT, B_MAT, cands, **LAUE_KW)
    assert 0 < res["collided"].sum() < len(path)
    clean = ~res["collided"]
    assert np.all(res["nearest_dist_px"][clean] > 5.0)


def test_rod_collisions_excludes_the_rods_own_integer_members():
    """The rod's own real Bragg point at L=0 or L=1 must not flag itself as
    contamination just because (H, K, 0)/(H, K, 1) is also in the candidate
    box passed in."""
    L_values = np.array([0.0, 1.0])
    path = rod_path_laue(UROT, B_MAT, H, K, L_values, **LAUE_KW)
    assert len(path) == 2
    cands = np.array([[H, K, 0.0], [H, K, 1.0]], float)  # only the rod's own members
    res = rod_collisions(path, UROT, B_MAT, cands, **LAUE_KW)
    assert not res["collided"].any()
    assert np.all(np.isnan(res["nearest_hkl"]))
    assert np.all(np.isinf(res["nearest_dist_px"]))


def test_rod_collisions_empty_path_returns_empty_arrays():
    path = rod_path_laue(np.eye(3), B_MAT, 0, 0, [], **LAUE_KW)
    res = rod_collisions(path, UROT, B_MAT, _candidate_box(2, 2, 5), **LAUE_KW)
    assert len(res["collided"]) == 0


def test_rod_collisions_no_accessible_candidates_means_no_collisions():
    L_values = np.arange(-2, 3, 0.05)
    path = rod_path_laue(UROT, B_MAT, H, K, L_values, **LAUE_KW)
    # a candidate list that is never forward-scattering under U = I-like UROT
    # at this energy band -- (0, 0, L) is backscattering for every L (see
    # test_rod_profile_laue's DROP_NO_ENERGY test for the same fact under U=I;
    # confirmed separately here that it also holds under UROT for this range).
    cands = np.array([[0.0, 0.0, l] for l in range(1, 5)])
    pred = laue_predict_hkl(UROT, B_MAT, cands, **LAUE_KW)
    assert len(pred["row"]) == 0, "test assumption: this candidate set must be inaccessible"
    res = rod_collisions(path, UROT, B_MAT, cands, **LAUE_KW)
    assert not res["collided"].any()
