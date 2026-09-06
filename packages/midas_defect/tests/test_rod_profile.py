"""Rods measured on the frame stack.

Built on a synthetic rod planted in a synthetic stack, so every test can fail.
The properties pinned are the ones whose violation cost real results:

- the rod path must CURVE on the detector (a straight line drifts off it)
- not-observable must be NaN, never zero
- significance is sigma, never a ratio
- a control with zero scatter must be REFUSED, not used
- resolution-limited must return a lower bound, not a number
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import numpy as np
import pytest

from midas_defect.rod_profile import (
    rod_path, matched_control_path, profile_along, rod_significance,
    ring_L_marks, transverse_width, RodProfile,
    DROP_MASKED, DROP_OFF_DETECTOR, DROP_NO_OMEGA,
)

LAM, LSD, PX = 0.42459, 349_622.0, 172.0
NR, NC, BR, BC = 1679, 1475, 810.3, 737.2
A, C = 3.6116, 19.2516
B_MAT = np.diag([1 / A, 1 / A, 1 / C])           # 1/d convention
U = np.eye(3)
OM_LO, OM_HI, OM0, DOM = -18.0, 18.0, -17.5, 1.0
GEOKW = dict(wavelength_A=LAM, lsd_um=LSD, pixel_um=PX, bc_row=BR, bc_col=BC,
             n_rows=NR, n_cols=NC, omega_lo_deg=OM_LO, omega_hi_deg=OM_HI)


def _rot(ax, deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    if ax == "x": return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if ax == "y": return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


# A geometry where the (00L) rod actually reaches Bragg inside +/-18 deg:
# c* must lie near the horizontal plane perpendicular to the beam. With c*
# along the beam, or steeply out of plane, the rod never diffracts in a narrow
# rocking range at all -- which is a real property of narrow-wedge data, not a
# defect, and rod_path reports it as no_omega_solution rather than inventing
# points.
UROT = _rot("z", 20.0) @ _rot("x", -78.0)     # rod AND control both reachable
UCURVE = _rot("y", 4.0) @ _rot("x", -78.0)    # longer arc, visible curvature
USTRAIGHT = _rot("x", -90.0)                  # c* exactly along y: rod is STRAIGHT


# ------------------------------------------------------------------ the path

def test_rod_path_returns_points_and_counts_what_it_dropped():
    p = rod_path(UROT, B_MAT, 0, 0, np.arange(-16, 16, 0.05), **GEOKW)
    assert len(p) > 50
    assert set(p.dropped) >= {DROP_NO_OMEGA, DROP_OFF_DETECTOR}
    assert sum(p.dropped.values()) + len(p) == len(np.arange(-16, 16, 0.05))
    assert "dropped" in str(p)


def _chord_deviation(p):
    """Max distance of the path from the straight chord joining its ends."""
    r0, c0 = p.row[0], p.col[0]
    r1, c1 = p.row[-1], p.col[-1]
    v = np.array([r1 - r0, c1 - c0]); v = v / np.linalg.norm(v)
    return float(np.abs((p.row - r0) * (-v[1]) + (p.col - c0) * v[0]).max())


def test_the_rod_is_CURVED_on_the_detector():
    """The reason to walk in q-space. A straight detector line drifts off it.

    The magnitude is geometry-dependent -- see the companion test for the
    symmetric case where it is exactly zero -- so what is pinned here is that
    the curvature is real and larger than the transverse half-width matters at,
    not a particular number.
    """
    p = rod_path(UCURVE, B_MAT, 0, 0, np.arange(-16, 16, 0.05), **GEOKW)
    assert len(p) > 100
    assert _chord_deviation(p) > 0.5, (
        "the rod came out straight in a geometry chosen to curve it; the "
        "q-space walk would then buy nothing and something is wrong")


def test_the_rod_is_EXACTLY_straight_in_the_symmetric_geometry():
    """The code must not manufacture curvature where none exists.

    With c* exactly along lab +y the rod's detector image is a straight line by
    symmetry. If this ever shows curvature, the projection has a bug.
    """
    p = rod_path(USTRAIGHT, B_MAT, 0, 0, np.arange(-16, 16, 0.05), **GEOKW)
    assert len(p) > 100
    assert _chord_deviation(p) < 1e-6


def test_bragg_condition_holds_at_every_returned_point():
    p = rod_path(UROT, B_MAT, 0, 0, np.arange(-16, 16, 0.1), **GEOKW)
    assert len(p) > 50
    for i in range(0, len(p), 7):
        g_s = UROT @ (B_MAT @ np.array([0, 0, p.L[i]]))
        w = math.radians(p.omega_deg[i])
        gx = math.cos(w) * g_s[0] - math.sin(w) * g_s[1]
        assert gx == pytest.approx(-LAM * (g_s @ g_s) / 2.0, abs=1e-9)


def test_q_convention_is_not_silently_mixable():
    with pytest.raises(ValueError, match="q_convention"):
        rod_path(UROT, B_MAT, 0, 0, [1.0], q_convention="angstrom", **GEOKW)
    p1 = rod_path(UROT, B_MAT, 0, 0, np.arange(-10, 10, 0.1), **GEOKW)
    p2 = rod_path(UROT, B_MAT * 2 * math.pi, 0, 0, np.arange(-10, 10, 0.1),
                  q_convention="2pi/d", **GEOKW)
    assert len(p1) == len(p2)
    assert np.allclose(p1.row, p2.row, atol=1e-6)
    assert np.allclose(p1.col, p2.col, atol=1e-6)


def test_the_control_walks_the_same_geometry():
    L = np.arange(-16, 16, 0.05)
    rod = rod_path(UROT, B_MAT, 0, 0, L, **GEOKW)
    ctl = matched_control_path(UROT, B_MAT, 0, 0, L, **GEOKW)
    assert ctl.hk == (0.5, 0.5)
    assert len(rod) > 100 and len(ctl) > 100
    # same |q| ballpark and same detector region -- that is what "matched" means
    assert abs(np.median(ctl.q_mag) - np.median(rod.q_mag)) / np.median(rod.q_mag) < 1.0
    assert abs(np.median(ctl.row) - np.median(rod.row)) < 400


def test_an_UNREACHABLE_control_is_visible_not_silent():
    """In a narrow wedge the +/-1/2 offset can push the control out of range.

    That must show up as an empty path with a counted reason, so it can never
    be divided by as if it were a measurement.
    """
    L = np.arange(-16, 16, 0.05)
    ctl = matched_control_path(_rot("y", 4.0) @ _rot("x", -78.0), B_MAT, 0, 0,
                               L, **GEOKW)
    assert len(ctl) == 0
    assert ctl.dropped[DROP_NO_OMEGA] == len(L)


# --------------------------------------------------------------- the profile

def _stack_with_rod(path, amp=500.0, bg=0.0, sigma_px=2.0, seed=0):
    rng = np.random.default_rng(seed)
    st = rng.normal(bg, 5.0, size=(36, NR, NC)).astype(np.float32)
    for i in range(len(path)):
        kf = int(round((path.omega_deg[i] - OM0) / DOM))
        if not (0 <= kf < 36):
            continue
        r, c = int(round(path.row[i])), int(round(path.col[i]))
        rr = slice(max(r - 4, 0), r + 5); cc = slice(max(c - 4, 0), c + 5)
        yy, xx = np.mgrid[rr, cc]
        st[kf, rr, cc] += amp * np.exp(-(((yy - path.row[i]) ** 2 +
                                          (xx - path.col[i]) ** 2)
                                         / (2 * sigma_px ** 2)))
    return st


def test_a_planted_rod_is_recovered_and_a_blank_stack_is_not():
    L = np.arange(-12, 12, 0.1)
    p = rod_path(UROT, B_MAT, 0, 0, L, **GEOKW)
    mask = np.zeros((NR, NC), bool)
    hot = profile_along(_stack_with_rod(p), mask, p,
                        omega_first_deg=OM0, omega_step_deg=DOM)
    cold = profile_along(np.random.default_rng(1).normal(0, 5, (36, NR, NC)),
                         mask, p, omega_first_deg=OM0, omega_step_deg=DOM)
    assert np.nanmedian(hot.intensity) > 20 * np.nanstd(cold.intensity)
    assert abs(np.nanmedian(cold.intensity)) < 3.0


def test_masked_points_are_NaN_and_counted_never_zero():
    """A silent gap looks exactly like a real minimum in the rod."""
    L = np.arange(-12, 12, 0.1)
    p = rod_path(UROT, B_MAT, 0, 0, L, **GEOKW)
    mask = np.zeros((NR, NC), bool)
    lo = int(np.percentile(p.row, 40)); hi = int(np.percentile(p.row, 60))
    mask[lo:hi, :] = True
    prof = profile_along(_stack_with_rod(p), mask, p,
                         omega_first_deg=OM0, omega_step_deg=DOM)
    assert prof.dropped[DROP_MASKED] > 0
    assert np.isnan(prof.intensity).sum() >= prof.dropped[DROP_MASKED]
    assert not np.any(prof.intensity[np.isfinite(prof.intensity)] == 0.0)
    assert "observable" in str(prof)


def test_profile_rejects_a_bad_stack_or_reducer():
    p = rod_path(UROT, B_MAT, 0, 0, np.arange(-8, 8, 0.2), **GEOKW)
    with pytest.raises(ValueError, match="3-D|stack must"):
        profile_along(np.zeros((NR, NC)), np.zeros((NR, NC), bool), p,
                      omega_first_deg=OM0, omega_step_deg=DOM)
    with pytest.raises(ValueError, match="reducer"):
        profile_along(np.zeros((36, NR, NC)), np.zeros((NR, NC), bool), p,
                      omega_first_deg=OM0, omega_step_deg=DOM, reducer="median")


# ---------------------------------------------------------- significance

def _flat_profile(n, value, scatter, seed=0):
    rng = np.random.default_rng(seed)
    return RodProfile(L=np.arange(n),
                      intensity=rng.normal(value, scatter, n),
                      n_valid_px=np.full(n, 13), omega_deg=np.zeros(n),
                      dropped={})


def test_significance_is_sigma_and_detects_a_real_excess():
    rod = _flat_profile(300, 40.0, 5.0, 1)
    ctl = _flat_profile(300, 0.0, 5.0, 2)
    s = rod_significance(rod, ctl)
    assert s["median_sigma"] == pytest.approx(8.0, abs=1.5)
    assert s["n_rod"] == 300 and s["n_control"] == 300


def test_significance_is_near_zero_when_there_is_no_rod():
    s = rod_significance(_flat_profile(300, 0.0, 5.0, 3),
                         _flat_profile(300, 0.0, 5.0, 4))
    assert abs(s["median_sigma"]) < 1.0


def test_A_CONTROL_THAT_CANNOT_FAIL_IS_REFUSED():
    """The vacuous azimuthal-median control, caught rather than reported."""
    ctl = RodProfile(L=np.arange(50), intensity=np.zeros(50),
                     n_valid_px=np.full(50, 13), omega_deg=np.zeros(50),
                     dropped={})
    with pytest.raises(ValueError, match="cannot fail"):
        rod_significance(_flat_profile(50, 40.0, 5.0), ctl)


# ------------------------------------------------------------- ring marking

def test_ring_crossings_are_marked_not_deleted():
    L = np.arange(-14, 14, 0.05)
    p = rod_path(UROT, B_MAT, 0, 0, L, **GEOKW)
    rad = np.hypot(p.row - BR, p.col - BC)
    target = float(np.median(rad))
    marks = ring_L_marks(p, [target], bc_row=BR, bc_col=BC, tolerance_px=3.0)
    assert marks.size > 0
    assert marks.size < len(p)                    # marks, does not delete
    assert ring_L_marks(p, [], bc_row=BR, bc_col=BC).size == 0


# ------------------------------------------------------------------- width

def test_resolution_limited_returns_a_LOWER_BOUND_not_a_number():
    w = transverse_width(diffuse_fwhm_inv_A=0.0040, bragg_fwhm_inv_A=0.0042)
    assert w.resolution_limited
    assert w.coherence_length_A is None and w.excess_fwhm_inv_A is None
    assert w.lower_bound_A == pytest.approx(1 / 0.0042)
    assert "LOWER BOUND" in str(w)


def test_a_genuine_excess_deconvolves_the_instrument():
    w = transverse_width(diffuse_fwhm_inv_A=0.0100, bragg_fwhm_inv_A=0.0060)
    assert not w.resolution_limited
    assert w.excess_fwhm_inv_A == pytest.approx(math.sqrt(1e-4 - 3.6e-5))
    assert w.coherence_length_A == pytest.approx(1 / w.excess_fwhm_inv_A)
    # the excess is SMALLER than the raw width, i.e. the length is LONGER
    assert w.coherence_length_A > 1 / 0.0100


def test_width_rejects_nonsense():
    with pytest.raises(ValueError):
        transverse_width(0.0, 0.001)
    with pytest.raises(ValueError):
        transverse_width(0.001, -1.0)
