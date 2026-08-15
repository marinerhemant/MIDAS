"""MaxRingRad must actually bound the ring table.

Regression for a factor-of-2 bug: build_ring_table derived its 2theta cap as
``2*arctan(R_max/Lsd)``, but a ray scattered through 2theta lands at
``R = Lsd*tan(2theta)`` -- the factor 2 is already inside "2theta".  There was
also no MaxRingRad filter on r_ideal_px to mirror the MinRingRad one.

Net effect on a 2880^2 / 150 um detector at Lsd = 1 m with CeO2 at 107 keV:
asking for MaxRingRad = 1420 px admitted rings out to 2964 px (2.09x), i.e.
into the detector corners the caller asked to exclude.  Every
midas_calibrate_v2 pipeline inherits this, because v2 has no ring-table
builder of its own and imports this one.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import build_ring_table, max_resolvable_ring_radius_px


def _p(max_rr, min_rr=200.0, lsd=1_000_000.0, lam=0.11595):
    return CalibrationParams(
        NrPixelsY=2880, NrPixelsZ=2880, pxY=150.0, pxZ=150.0,
        Lsd=lsd, BC_y=1440.0, BC_z=1440.0, tx=0.0, ty=0.0, tz=0.0,
        Wavelength=lam, SpaceGroup=225,
        LatticeConstant=(5.4116,) * 3 + (90.0,) * 3,
        MaxRingRad=max_rr, MinRingRad=min_rr, RhoD=305657.0, nIterations=1,
    )


@pytest.mark.parametrize("max_rr", [400.0, 710.0, 1000.0, 1420.0])
def test_max_ring_rad_is_respected(max_rr):
    rt = build_ring_table(_p(max_rr))
    assert len(rt) > 0
    assert rt.r_ideal_px.max() <= max_rr + 1e-9, (
        f"MaxRingRad={max_rr} px admitted a ring at "
        f"{rt.r_ideal_px.max():.1f} px ({rt.r_ideal_px.max()/max_rr:.2f}x)")


def test_min_ring_rad_is_respected():
    rt = build_ring_table(_p(1420.0, min_rr=500.0))
    assert rt.r_ideal_px.min() >= 500.0 - 1e-9


def test_two_theta_cap_matches_the_radius_cap():
    """2theta_max and MaxRingRad must describe the same boundary."""
    max_rr, lsd, px = 1420.0, 1_000_000.0, 150.0
    rt = build_ring_table(_p(max_rr, lsd=lsd))
    tt_max_expected = np.degrees(np.arctan(max_rr * px / lsd))
    assert rt.two_theta_deg.max() <= tt_max_expected + 1e-9


def test_resolvability_cap_flags_dense_ring_tables():
    """CeO2 at 107 keV is unresolvable at short distance on a 150 um pixel."""
    # 330 mm: the pathological case that motivated the helper
    rt_short = build_ring_table(_p(1420.0, lsd=330_000.0))
    r_short, n_short = max_resolvable_ring_radius_px(rt_short,
                                                     min_separation_px=8.0,
                                                     r_min_px=200.0)
    # 1500 mm: comfortable
    rt_long = build_ring_table(_p(1420.0, lsd=1_500_000.0))
    r_long, n_long = max_resolvable_ring_radius_px(rt_long,
                                                   min_separation_px=8.0,
                                                   r_min_px=200.0)
    assert len(rt_short) > 50, "expected a dense table at short distance"
    assert n_short < n_long, (
        f"short distance should resolve FEWER rings, got {n_short} vs {n_long}")
    assert n_short <= 3, (
        f"CeO2 at 330 mm should resolve <=3 rings above 200 px, got {n_short}")


def test_resolvability_returns_none_when_too_few_rings():
    rt = build_ring_table(_p(1420.0, lsd=1_000_000.0))
    r, n = max_resolvable_ring_radius_px(rt, r_min_px=1e9)
    assert r is None and n == 0
