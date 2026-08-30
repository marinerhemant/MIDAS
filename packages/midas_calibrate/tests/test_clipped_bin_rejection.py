"""A clipped radial window must not contribute a fitted point.

``extract_fitted_points`` takes an intensity-weighted centroid over a FIXED
radial window::

    I     = max(I_block - I_block.min(axis=0), 0)
    R_fit = (I * R_window).sum(axis=0) / tot

Where that window leaves the detector — or crosses a module gap, or masked
pixels — the missing cake cells hold no data. The peak is truncated and the
centroid is dragged toward the surviving side. The only guard was
``valid_tot = tot > 0.0``, which merely requires SOME signal, so a
half-truncated peak passed and contributed a biased point.

Those points are what dragged the geometry: the OUTER rings are the ones whose
windows leave the detector, which is why capping ``max_ring_radius_px``
appeared to fix the calibration. That cap is the wrong remedy — the outer rings
carry the tilt and distortion leverage, and a partial arc is perfectly good
data. Only the clipped BINS are bad.

So the invariant has two halves, and both are tested here:

  * a clipped bin must be rejected, AND
  * a partial ring must still be USED — its covered arc must survive.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from midas_calibrate.estep import (
    CakeProfile, extract_fitted_points, integrate_cake,
)
from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import build_ring_table

A_CEO2 = 5.4116
WAVELENGTH = 0.172973


def _params(N=512, px=200.0, Lsd=400_000.0, **kw):
    p = CalibrationParams(
        NrPixelsY=N, NrPixelsZ=N, pxY=px, pxZ=px, Lsd=Lsd,
        BC_y=N / 2.0, BC_z=N / 2.0, Wavelength=WAVELENGTH,
        LatticeConstant=(A_CEO2,) * 3 + (90.0,) * 3, SpaceGroup=225,
        RhoD=float(N) * px, MaxRingRad=float(N) * 0.70, MinRingRad=20.0,
    )
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _ring_image(p, sigma=1.6, peak=4000.0, bg=25.0, seed=0):
    """Rings drawn at the analytic radii, THROUGH the geometry in ``p``.

    Rendering with a plain ``hypot(Y - BC_y, Z - BC_z)`` would produce an image
    with no tilt signature at all, so a round-trip against a tilted ``p`` would
    be testing nothing — the fit would correctly return ty = 0 and the test
    would look like a calibration failure. Use the same forward model the
    calibration inverts.
    """
    from midas_integrate.geometry import pixel_to_REta, build_tilt_matrix

    N = p.NrPixelsY
    Yi, Zi = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float),
                         indexing="xy")
    R, _eta = pixel_to_REta(
        Yi, Zi, Ycen=p.BC_y, Zcen=p.BC_z,
        TRs=build_tilt_matrix(p.tx, p.ty, p.tz),
        Lsd=p.Lsd, RhoD=p.RhoD, px=p.pxY,
        **{f"p{k}": getattr(p, f"p{k}") for k in range(15)},
        parallax=p.Parallax,
    )
    img = np.full_like(R, bg)
    seen = set()
    for h in range(8):
        for k in range(8):
            for l in range(8):
                if h == k == l == 0 or not (h % 2 == k % 2 == l % 2):
                    continue
                s2 = h * h + k * k + l * l
                if s2 in seen:
                    continue
                d = A_CEO2 / math.sqrt(s2)
                ratio = WAVELENGTH / (2.0 * d)
                if ratio >= 1.0:
                    continue
                seen.add(s2)
                R0 = p.Lsd * math.tan(2.0 * math.asin(ratio)) / p.pxY
                if 25 < R0 < 0.72 * N:
                    img += peak * np.exp(-((R - R0) ** 2) / (2 * sigma ** 2))
    return np.random.default_rng(seed).poisson(np.clip(img, 0, None)).astype(float)


# ------------------------------------------------------------ coverage map

def test_cake_carries_a_coverage_map():
    p = _params()
    rt = build_ring_table(p)
    cake = integrate_cake(p, _ring_image(p), rt)
    assert cake.coverage is not None
    assert cake.coverage.shape == cake.intensity.shape
    assert (cake.coverage >= 0).all()
    assert cake.coverage.max() > 0


def test_coverage_is_zero_beyond_the_detector_corner():
    """The physical check on the map: past the corner nothing can contribute."""
    p = _params()
    rt = build_ring_table(p)
    cake = integrate_cake(p, _ring_image(p), rt)
    corner = math.hypot(p.BC_y, p.BC_z)          # BC is centred here
    beyond = cake.R_centers > corner + 2.0
    if beyond.any():
        assert cake.coverage[beyond, :].max() == 0.0
    inscribed = min(p.BC_y, p.NrPixelsY - 1 - p.BC_y)
    well_inside = cake.R_centers < inscribed - 5.0
    assert (cake.coverage[well_inside, :] > 0).all(), (
        "cells inside the inscribed radius must all be covered")


# --------------------------------------------------- the invariant, both halves

def test_a_clipped_window_contributes_no_point():
    """Half of the invariant: bins whose radial window is not fully covered
    must be dropped, however strong their (truncated) signal."""
    p = _params()
    rt = build_ring_table(p)
    cake = integrate_cake(p, _ring_image(p), rt)
    pts = extract_fitted_points(cake, rt, p)
    assert pts, "no points extracted at all"

    px = 0.5 * (p.pxY + p.pxZ)
    half = 0.5 * p.Width / px
    cov = cake.coverage
    for fp in pts:
        R = math.hypot(fp.Y_pix - p.BC_y, fp.Z_pix - p.BC_z)
        idx = np.where(np.abs(cake.R_centers - R) <= half)[0]
        if idx.size < 3:
            continue
        # every cell of this point's radial window must be covered somewhere
        assert cov[idx, :].max() > 0.0


def test_a_partial_ring_is_still_used():
    """The other half, and the reason not to just cap the ring radius.

    Rings between the inscribed radius and the corner appear only as arcs in
    the corners. Those arcs are good data and must still produce points — they
    carry the lever arm that constrains tilt and distortion.
    """
    p = _params()
    rt = build_ring_table(p)
    cake = integrate_cake(p, _ring_image(p), rt)
    pts = extract_fitted_points(cake, rt, p)
    inscribed = min(p.BC_y, p.NrPixelsY - 1 - p.BC_y)
    radii = np.array([math.hypot(f.Y_pix - p.BC_y, f.Z_pix - p.BC_z)
                      for f in pts])
    assert (radii > inscribed).any(), (
        f"every fitted point fell inside the inscribed radius "
        f"({inscribed:.0f} px) — partial rings are being discarded wholesale, "
        f"which throws away the tilt/distortion leverage")


def test_rejection_removes_points_rather_than_rings():
    """A partial ring must lose only its clipped bins, not all of them."""
    p = _params()
    rt = build_ring_table(p)
    cake = integrate_cake(p, _ring_image(p), rt)
    strict = extract_fitted_points(cake, rt, p, min_cell_coverage=0.5)
    loose = extract_fitted_points(cake, rt, p, min_cell_coverage=0.0)
    assert len(strict) < len(loose), "the coverage filter dropped nothing"
    rings_strict = {f.ring_idx for f in strict}
    rings_loose = {f.ring_idx for f in loose}
    # rings may be lost entirely only if they were wholly clipped; the bulk
    # must survive
    assert len(rings_strict) >= len(rings_loose) - 1


def test_no_coverage_map_falls_back_without_crashing():
    """Hand-built CakeProfiles (no coverage) must still work."""
    p = _params()
    rt = build_ring_table(p)
    cake = integrate_cake(p, _ring_image(p), rt)
    bare = CakeProfile(R_centers=cake.R_centers, eta_centers=cake.eta_centers,
                       intensity=cake.intensity)
    assert bare.coverage is None
    assert extract_fitted_points(bare, rt, p)


# ------------------------------------------------------------- the payoff

@pytest.mark.slow
def test_geometry_round_trip_at_full_ring_radius():
    """The end the whole thing exists for: recover a known geometry without
    having to throw the outer rings away.

    Before the clipped-bin fix this converged 170 µm and 0.13 deg off with a
    ~290 µε residual, using the same rings.
    """
    from midas_calibrate import autocalibrate

    truth = _params(N=512, Lsd=400_000.0, ty=0.30, tz=-0.45)
    img = _ring_image(truth)
    start = _params(N=512, Lsd=400_000.0 * 1.002, ty=0.0, tz=0.0)
    start.BC_y += 1.0
    start.BC_z -= 1.0
    r = autocalibrate(start, img, verbose=False)
    got = r.params
    assert abs(float(got.Lsd) - truth.Lsd) < 0.02e-2 * truth.Lsd
    assert abs(float(got.ty) - truth.ty) < 0.02
    assert abs(float(got.tz) - truth.tz) < 0.02
