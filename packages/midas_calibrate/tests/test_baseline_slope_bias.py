"""A sloped background must not pull the ring centroid.

``extract_fitted_points`` takes an intensity-weighted centroid over a fixed
radial window. It used to remove the per-η window MINIMUM, a constant, which
leaves any background slope in the window. The centroid is then dragged toward
the high-background side: inward on a ceria frame whose background falls with
2θ, so the fit converged to a short Lsd, worse the weaker the ring.

Reported by a user of a CCD detector at a synchrotron beamline, 2026-10-11. Measured
at the TRUE geometry on a noise-free synthetic weak image: rings 1-3 sat
-0.37 / -0.65 / -0.23 px inward; flat background -> +0.003 px; reversing the
slope flips the sign. Four independent /verify lenses (claim a269ba0c1fe1).

The baseline is now a straight line through the window-end means. These tests
use a hand-built cake with zero tilt and zero distortion, so a returned point's
radius is simply ``hypot(Y - BC_y, Z - BC_z)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_calibrate.estep import CakeProfile, extract_fitted_points
from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import build_ring_table

A_CEO2 = 5.4116
WAVELENGTH = 0.172973
N_ETA = 8


def _params(Width=4040.0):
    # 4040 um at 200 um/px = a 20.2 px radial window, the window a 1600 um
    # default gives at 79.278 um pixels.
    N, px = 1024, 200.0
    return CalibrationParams(
        NrPixelsY=N, NrPixelsZ=N, pxY=px, pxZ=px, Lsd=400_000.0,
        BC_y=N / 2.0, BC_z=N / 2.0, Wavelength=WAVELENGTH,
        LatticeConstant=(A_CEO2,) * 3 + (90.0,) * 3, SpaceGroup=225,
        RhoD=float(N) * px, MaxRingRad=float(N) * 0.70, MinRingRad=20.0,
        Width=Width,
    )


def _cake_one_ring(p, rt, *, slope_per_px, peak=68.0, bg0=50.0, sigma=1.5,
                   phase=0.3):
    """A cake holding ONLY ring 0: Gaussian peak on a linear background.

    ``slope_per_px`` is the background change per radial pixel (negative = the
    background falls outward, as on the ceria frames). 0.18 counts/px over a
    20 px window is ring 1 of the weak CCD image (peak 68 on ~50).
    ``phase`` places the true ring off the 1-px bin grid.
    """
    r0 = float(rt.r_ideal_px[0]) + phase
    R = np.arange(np.floor(r0) - 16.5, np.floor(r0) + 17.5, 1.0)
    bg = bg0 + slope_per_px * (R - r0)
    prof = bg + peak * np.exp(-0.5 * ((R - r0) / sigma) ** 2)
    I = np.repeat(prof[:, None], N_ETA, axis=1)
    eta = np.linspace(-170.0, 170.0, N_ETA)
    cake = CakeProfile(R_centers=R, eta_centers=eta, intensity=I,
                       coverage=np.ones_like(I))
    return cake, r0


def _bias(p, rt, cake, r0):
    fits = [f for f in extract_fitted_points(cake, rt, p) if f.ring_idx == 0]
    assert len(fits) == N_ETA
    Y = np.array([f.Y_pix for f in fits])
    Z = np.array([f.Z_pix for f in fits])
    return float(np.mean(np.hypot(Y - p.BC_y, Z - p.BC_z)) - r0)


def _old_min_centroid_bias(p, rt, cake, r0):
    """The previous recipe, written out, to show this test would have caught it."""
    half_px = 0.5 * p.Width / p.pxY
    idx = np.where(np.abs(cake.R_centers - rt.r_ideal_px[0]) <= half_px)[0]
    B = cake.intensity[idx, 0]
    I = np.maximum(B - B.min(), 0.0)
    return float((I * cake.R_centers[idx]).sum() / I.sum() - r0)


@pytest.mark.parametrize("slope", [-0.18, +0.18])
def test_sloped_background_does_not_move_the_centroid(slope):
    p = _params()
    rt = build_ring_table(p)
    cake, r0 = _cake_one_ring(p, rt, slope_per_px=slope)
    # The old recipe really was biased here, in the direction of the slope.
    old = _old_min_centroid_bias(p, rt, cake, r0)
    assert abs(old) > 0.15 and np.sign(old) == np.sign(slope)
    # The line-through-the-ends baseline is not.
    assert abs(_bias(p, rt, cake, r0)) < 0.08


def test_flat_background_is_unbiased():
    p = _params()
    rt = build_ring_table(p)
    cake, r0 = _cake_one_ring(p, rt, slope_per_px=0.0)
    assert abs(_bias(p, rt, cake, r0)) < 0.02


def test_bias_does_not_grow_as_the_ring_weakens():
    """The old bias scaled as slope / peak; the new one must not."""
    p = _params()
    rt = build_ring_table(p)
    b = [abs(_bias(p, rt, *_cake_one_ring(p, rt, slope_per_px=-0.18, peak=pk)))
         for pk in (300.0, 68.0, 18.0)]
    assert max(b) < 0.1


def test_window_too_short_for_two_ends_falls_back_without_error():
    """Under 5 bins there are no two window ends; the constant baseline stays."""
    p = _params(Width=3.0 * 200.0)     # half_px = 1.5 -> 3 bins
    rt = build_ring_table(p)
    cake, r0 = _cake_one_ring(p, rt, slope_per_px=0.0)
    fits = [f for f in extract_fitted_points(cake, rt, p) if f.ring_idx == 0]
    assert len(fits) == N_ETA


@pytest.mark.parametrize("phase", [0.0, 0.3, 0.5])
def test_window_ends_on_the_ring_keep_the_constant_baseline(phase):
    """A 7.5 px window around a sigma 1.5 px ring has its ends on the tails.

    A line through the ends then cuts into one flank (measured +0.08 px on a
    flat-background ring, centred, noise-free). The guard must keep the old
    constant baseline there, which is unbiased on a flat background.
    """
    p = _params(Width=1500.0)          # 1500 um / 200 um = 7.5 px
    rt = build_ring_table(p)
    cake, r0 = _cake_one_ring(p, rt, slope_per_px=0.0, phase=phase)
    assert abs(_bias(p, rt, cake, r0)) < 0.03
