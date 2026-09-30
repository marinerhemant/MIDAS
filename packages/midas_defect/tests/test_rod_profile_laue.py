"""The Laue (energy-sweep) rod walker -- cross-checked against rod_path.

rod_path_laue must be the SAME physics as rod_path with the roles of omega
and wavelength swapped: fix omega at 0 and solve for energy, instead of fixing
wavelength and solving for omega. The cross-check below constructs that
equivalence directly (solve for energy at omega=0, then verify the identical
point is recovered by feeding that energy back into rod_path as a fixed
wavelength with omega=0 in its window) rather than trusting the two
implementations to agree by construction.

(h, k) = (0, 1) under UROT is not an arbitrary choice: most (h, k) rows under
most orientations are either backscattering (no positive-energy solution) or
scatter off the edge of the detector -- confirmed by hand before writing
these tests, the same way a real analysis has to check before trusting a row.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import numpy as np
import pytest

from midas_defect.rod_profile import (
    rod_path, rod_path_laue, matched_control_path_laue, profile_along_laue,
    rod_significance, LauePath, RodProfile,
    DROP_NO_ENERGY, DROP_OFF_DETECTOR, DROP_MASKED,
)

LSD, PX = 349_622.0, 172.0
NR, NC, BR, BC = 1679, 1475, 810.3, 737.2
A, C = 3.6116, 19.2516
B_MAT = np.diag([1 / A, 1 / A, 1 / C])  # 1/d convention
LAUE_KW = dict(lsd_um=LSD, pixel_um=PX, bc_row=BR, bc_col=BC,
               n_rows=NR, n_cols=NC, E_lo_keV=11.0, E_hi_keV=90.0)


def _rot(ax, deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    if ax == "x": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Same rotation test_rod_profile.py already uses for a reachable (00L) rod
# in the omega-sweep case; (h, k) = (0, 1) is confirmed (below) to have a
# forward-scattering, on-detector energy solution across L in [-2, 3].
UROT = _rot("z", 20.0) @ _rot("x", -78.0)
H, K = 0, 1


def test_rod_path_laue_returns_points_and_counts_what_it_dropped():
    p = rod_path_laue(UROT, B_MAT, H, K, np.arange(-8, 8, 0.05), **LAUE_KW)
    assert len(p) > 20
    assert set(p.dropped) >= {DROP_NO_ENERGY, DROP_OFF_DETECTOR}
    assert sum(p.dropped.values()) + len(p) == len(np.arange(-8, 8, 0.05))
    assert "dropped" in str(p)
    assert isinstance(p, LauePath)


def test_backscatter_direction_is_dropped_as_no_energy_not_fabricated():
    # (0, 0, L) with U = I: g_s = (0, 0, L/C) has g_x = 0 for every L, never
    # < 0, so there is never a forward-scattering energy -- dropped, not
    # fabricated as some huge or zero "energy".
    p = rod_path_laue(np.eye(3), B_MAT, 0, 0, np.arange(1, 20, 1.0), **LAUE_KW)
    assert len(p) == 0
    assert p.dropped[DROP_NO_ENERGY] == 19


@pytest.mark.parametrize("L", [0.0, 1.0])
def test_energy_solve_matches_an_independent_omega_solve_at_zero(L):
    """Solve for energy at omega=0, then recover the SAME point via rod_path.

    If rod_path_laue's energy is right, feeding it back into rod_path (fixed
    wavelength, omega window containing 0) as the wavelength must find
    omega = 0 as a root and predict the identical row/col -- a real
    cross-check against an independently-tested function, not a tautology.
    """
    p = rod_path_laue(UROT, B_MAT, H, K, [L], **LAUE_KW)
    assert len(p) == 1, "L=0,1 are confirmed forward-scattering and on-detector for (H,K)"
    lam = 12.39842 / p.energy_keV[0]

    q = rod_path(UROT, B_MAT, H, K, [L], wavelength_A=lam,
                lsd_um=LSD, pixel_um=PX, bc_row=BR, bc_col=BC,
                n_rows=NR, n_cols=NC, omega_lo_deg=-0.5, omega_hi_deg=0.5)
    assert len(q) == 1, "the recovered energy must put omega=0 back on the Ewald sphere"
    assert q.omega_deg[0] == pytest.approx(0.0, abs=1e-6)
    assert q.row[0] == pytest.approx(p.row[0], abs=1e-6)
    assert q.col[0] == pytest.approx(p.col[0], abs=1e-6)


def test_matched_control_path_laue_offsets_hk():
    p = rod_path_laue(UROT, B_MAT, H, K, np.arange(-2, 3, 0.05), **LAUE_KW)
    c = matched_control_path_laue(UROT, B_MAT, H, K, np.arange(-2, 3, 0.05), **LAUE_KW)
    assert c.hk == (0.5, 1.5)
    assert p.hk == (H, K)


def _image_with_rod(path: LauePath, amp=500.0, bg=50.0, sigma_px=2.0, seed=0):
    rng = np.random.default_rng(seed)
    image = rng.normal(bg, 5.0, size=(NR, NC))
    yy, xx = np.mgrid[-8:9, -8:9]
    kernel = amp * np.exp(-(xx**2 + yy**2) / (2 * sigma_px**2))
    for r, c in zip(path.row, path.col):
        ri, ci = int(round(r)), int(round(c))
        r0, r1 = max(0, ri - 8), min(NR, ri + 9)
        c0, c1 = max(0, ci - 8), min(NC, ci + 9)
        image[r0:r1, c0:c1] += kernel[r0 - (ri - 8):r1 - (ri - 8), c0 - (ci - 8):c1 - (ci - 8)]
    return image, bg


def test_a_planted_laue_rod_is_recovered_and_a_blank_image_is_not():
    # (0, 4) rather than (H, K): confirmed by hand to have BOTH the rod and
    # its (0.5, 4.5) matched control land on-detector over an overlapping L
    # stretch -- unlike (H, K)=(0, 1), whose control's valid range does not
    # overlap its own (a real property of this geometry, not a bug: the
    # energy-sweep condition does not guarantee a rod and its half-integer
    # control share a reachable L window the way the omega-sweep case does).
    L_values = np.arange(4, 10, 0.02)
    path = rod_path_laue(UROT, B_MAT, 0, 4, L_values, **LAUE_KW)
    control = matched_control_path_laue(UROT, B_MAT, 0, 4, L_values, **LAUE_KW)
    assert len(path) > 10 and len(control) > 10

    image, bg = _image_with_rod(path, amp=800.0)
    mask = np.zeros((NR, NC), bool)

    rod_profile = profile_along_laue(image - bg, mask, path)
    control_profile = profile_along_laue(image - bg, mask, control)
    assert isinstance(rod_profile, RodProfile)
    assert np.all(np.isnan(rod_profile.omega_deg)), "Laue profiles carry no omega"

    sig = rod_significance(rod_profile, control_profile)
    assert sig["median_sigma"] > 5.0, "a planted rod must read as a large excess"

    blank = np.random.default_rng(1).normal(bg, 5.0, size=(NR, NC))
    blank_rod = profile_along_laue(blank - bg, mask, path)
    blank_control = profile_along_laue(blank - bg, mask, control)
    sig_blank = rod_significance(blank_rod, blank_control)
    assert abs(sig_blank["median_sigma"]) < 3.0, "no planted rod must not read as significant"


def test_profile_along_laue_masks_are_NaN_not_zero():
    L_values = np.arange(-1, 2, 0.05)
    path = rod_path_laue(UROT, B_MAT, H, K, L_values, **LAUE_KW)
    image = np.zeros((NR, NC))
    mask = np.ones((NR, NC), bool)  # everything masked
    profile = profile_along_laue(image, mask, path)
    assert np.all(np.isnan(profile.intensity))
    assert profile.dropped[DROP_MASKED] == len(path)


def test_profile_along_laue_rejects_bad_image():
    L_values = np.arange(-1, 2, 0.05)
    path = rod_path_laue(UROT, B_MAT, H, K, L_values, **LAUE_KW)
    with pytest.raises(ValueError):
        profile_along_laue(np.zeros((3, NR, NC)), np.zeros((NR, NC), bool), path)
    with pytest.raises(ValueError):
        profile_along_laue(np.zeros((NR, NC)), np.zeros((NR, NC), bool), path, reducer="bogus")


def test_rod_path_laue_rejects_bad_q_convention():
    with pytest.raises(ValueError):
        rod_path_laue(UROT, B_MAT, H, K, [1.0], q_convention="nope", **LAUE_KW)
