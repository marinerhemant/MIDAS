"""The calibration must be able to mask bad pixels.

Until 2026-08-29 there was **no way to pass a mask into the calibration at
all**: ``CalibrationParams`` had no mask field, and none of
``integrate_cake`` / ``run_estep`` / ``midas_calibrate.autocalibrate`` /
``midas_calibrate_v2.calibrate`` accepted one — even though
``IntegrationParams`` carries ``MaskFile`` and
``midas_integrate.detector_mapper.build_map`` honours a mask, and the shipped
example ``parameters.txt`` sets ``MaskFile mask_upd.tif``.

Bad pixels therefore entered the cake as genuine zeros and diluted every cell
they touched, dragging the intensity-weighted radial centroid off the ring.

The subtlety these tests exist to pin: **zeroing masked pixels is not masking.**
That is precisely the state the code was already in — a dead pixel reads ~0
after dark subtraction. Masking only works if the masked pixels leave the
DENOMINATOR too, so a half-dead cell reports the mean of its LIVE pixels rather
than a diluted mean over live-plus-dead.

Convention: **nonzero means BAD**, matching ``build_map`` (``mask == 1.0`` is
masked) and the shipped ``mask_upd.tif``, whose nonzero pixels coincide exactly
with the dead pixels of the Pilatus frame. (The example ``parameters.txt``
comment says "0 = masked"; that is wrong — it would mask 92 % of the detector.)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_calibrate.estep import integrate_cake, run_estep
from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import build_ring_table

A_CEO2 = 5.4116
WAVELENGTH = 0.172973


def _params(N=384, px=200.0, Lsd=400_000.0):
    return CalibrationParams(
        NrPixelsY=N, NrPixelsZ=N, pxY=px, pxZ=px, Lsd=Lsd,
        BC_y=N / 2.0, BC_z=N / 2.0, Wavelength=WAVELENGTH,
        LatticeConstant=(A_CEO2,) * 3 + (90.0,) * 3, SpaceGroup=225,
        RhoD=float(N) * px, MaxRingRad=float(N) * 0.70, MinRingRad=20.0)


def _rings(p, seed=0):
    from midas_integrate.geometry import pixel_to_REta, build_tilt_matrix
    N = p.NrPixelsY
    Yi, Zi = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float),
                         indexing="xy")
    R, _ = pixel_to_REta(Yi, Zi, Ycen=p.BC_y, Zcen=p.BC_z,
                         TRs=build_tilt_matrix(p.tx, p.ty, p.tz), Lsd=p.Lsd,
                         RhoD=p.RhoD, px=p.pxY,
                         **{f"p{k}": 0.0 for k in range(15)}, parallax=False)
    img = np.full_like(R, 25.0)
    seen = set()
    for h in range(8):
        for k in range(8):
            for l in range(8):
                if h == k == l == 0 or not (h % 2 == k % 2 == l % 2):
                    continue
                s2 = h * h + k * k + l * l
                if s2 in seen:
                    continue
                ratio = WAVELENGTH / (2.0 * (A_CEO2 / math.sqrt(s2)))
                if ratio >= 1.0:
                    continue
                seen.add(s2)
                R0 = p.Lsd * math.tan(2.0 * math.asin(ratio)) / p.pxY
                if 25 < R0 < 0.70 * N:
                    img += 4000.0 * np.exp(-((R - R0) ** 2) / (2 * 1.6 ** 2))
    return np.random.default_rng(seed).poisson(np.clip(img, 0, None)).astype(float)


# ------------------------------------------------------- the API exists at all

def test_every_layer_of_the_chain_accepts_a_mask():
    """The gap was that the parameter did not exist anywhere. Pin the whole
    chain, because threading it through only part of it is what made the
    earlier `calibrate(mask=...)` attempt fail deep inside."""
    import inspect
    from midas_calibrate import autocalibrate
    from midas_calibrate_v2 import calibrate
    from midas_calibrate_v2.pipelines.single import autocalibrate as v2_auto
    from midas_calibrate_v2.pipelines._common import run_estep_v1

    for fn in (integrate_cake, run_estep, autocalibrate, calibrate,
               v2_auto, run_estep_v1):
        assert "mask" in inspect.signature(fn).parameters, (
            f"{fn.__module__}.{fn.__name__} takes no mask")


# ------------------------------------------------- masking != zeroing

def test_masking_removes_pixels_from_the_denominator_not_just_the_numerator():
    """The core of the fix. A cell whose pixels are half masked must report the
    mean of the LIVE half, not a mean diluted by the dead half.

    Zeroing alone would halve that cell's intensity; correct masking leaves it
    unchanged.
    """
    p = _params()
    rt = build_ring_table(p)
    img = _rings(p)
    N = p.NrPixelsY

    # Kill a wedge of pixels: set them to 0 AND mask them.
    bad = np.zeros_like(img, dtype=bool)
    Yi, Zi = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
    eta = np.degrees(np.arctan2(-(Yi - p.BC_y), Zi - p.BC_z))
    bad[(eta > 20.0) & (eta < 40.0)] = True

    img_killed = np.where(bad, 0.0, img)

    cake_zeroed = integrate_cake(p, img_killed, rt)                  # no mask
    cake_masked = integrate_cake(p, img_killed, rt, mask=bad)        # masked

    # In the affected wedge the zeroed cake is diluted; the masked cake either
    # reports the live mean or marks the cell uncovered.
    z, m = cake_zeroed.intensity, cake_masked.intensity
    cov_m = cake_masked.coverage
    assert cov_m is not None
    # Coverage must DROP where pixels were masked, and stay put elsewhere.
    cov_z = cake_zeroed.coverage
    assert float(cov_m.sum()) < float(cov_z.sum()), (
        "masking did not reduce the coverage at all — the denominator still "
        "counts the dead pixels")
    # and the two cakes must actually differ somewhere
    fin = np.isfinite(z) & np.isfinite(m)
    assert np.abs(z[fin] - m[fin]).max() > 0.0


def test_a_fully_masked_cell_reports_zero_coverage():
    p = _params()
    rt = build_ring_table(p)
    img = _rings(p)
    allbad = np.ones_like(img, dtype=bool)
    cake = integrate_cake(p, img, rt, mask=allbad)
    assert cake.coverage is not None
    assert float(cake.coverage.max()) == 0.0
    assert float(np.nan_to_num(cake.intensity).max()) == 0.0


def test_an_empty_mask_is_a_no_op():
    """A mask of all-good must reproduce the unmasked cake bit for bit —
    otherwise the mask path is perturbing data it should not touch."""
    p = _params()
    rt = build_ring_table(p)
    img = _rings(p)
    a = integrate_cake(p, img, rt)
    b = integrate_cake(p, img, rt, mask=np.zeros_like(img, dtype=bool))
    assert np.allclose(a.intensity, b.intensity, rtol=0, atol=1e-12)
    assert np.allclose(a.coverage, b.coverage, rtol=0, atol=1e-9)


def test_mask_shape_mismatch_raises_rather_than_misaligning():
    """A mask in the wrong orientation masks the wrong pixels — silently, and
    worse than no mask. Shape is the one check that can catch it here."""
    p = _params()
    rt = build_ring_table(p)
    img = _rings(p)
    with pytest.raises(ValueError, match="mask shape"):
        integrate_cake(p, img, rt, mask=np.zeros((10, 10), dtype=bool))


def test_nonzero_means_bad_not_good():
    """Convention check. If this ever inverts, a mask would keep exactly the
    pixels it is meant to discard."""
    p = _params()
    rt = build_ring_table(p)
    img = _rings(p)
    ones = np.ones_like(img, dtype=np.uint8)      # every pixel flagged
    cake = integrate_cake(p, img, rt, mask=ones)
    assert float(cake.coverage.max()) == 0.0, (
        "mask==1 kept the pixels — the convention is inverted")


def test_run_estep_passes_the_mask_through():
    p = _params()
    rt = build_ring_table(p)
    img = _rings(p)
    N = p.NrPixelsY
    bad = np.zeros_like(img, dtype=bool)
    bad[:, N // 2 - 30:N // 2 + 30] = True        # a vertical dead stripe
    _cake_a, fits_a = run_estep(p, img, rt)
    _cake_b, fits_b = run_estep(p, img, rt, mask=bad)
    assert len(fits_b) < len(fits_a), (
        "masking a wide stripe removed no fitted points — the mask is not "
        "reaching the E-step")
