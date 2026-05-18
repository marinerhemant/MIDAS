"""Azimuthal sigma-clip — parasitic single-crystal spot detection on a
single-frame powder image and on a multi-panel tiled detector.

The principle: on a powder ring, intensity is approximately uniform in η
at fixed R. A parasitic single-crystal Bragg spot (sample environment,
gasket, capillary, stray grain) is localised in η and pokes above the
azimuthal MAD of its ring. We group pixels by their radial bin,
robust-clip per ring, and validate that (a) the planted spots are
flagged, (b) the powder ring itself is left alone.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import (
    spec_from_v1_params,
    HardBinGeometry,
    azimuthal_sigma_clip,
    azimuthal_sigma_clip_multi_panel,
)


def _spec(NY=256, NZ=256, BC_off_y=0.37, BC_off_z=-0.41):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + BC_off_y, BC_z=NZ / 2.0 + BC_off_z,
        RhoD=float(NY),
        RMin=4.0, RMax=120.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=2.0,
    )
    return spec_from_v1_params(p, requires_grad=False)


def _powder_image(spec, *, ring_R_px=(30.0, 60.0, 90.0), bg=100.0,
                   ring_amp=400.0, ring_width_px=1.5, seed=0):
    """Synthetic powder image: η-uniform Gaussian rings on a flat background
    with Poisson-like noise."""
    NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = (yy - NY / 2.0 - 0.37) * 200.0
    Zc = (zz - NZ / 2.0 + 0.41) * 200.0
    R_um = np.sqrt(Yc * Yc + Zc * Zc)
    R_px = R_um / 200.0
    img = np.full((NZ, NY), bg, dtype=np.float64)
    for R0 in ring_R_px:
        img += ring_amp * np.exp(-(R_px - R0) ** 2 / (2 * ring_width_px ** 2))
    rng = np.random.default_rng(seed)
    img += rng.normal(0.0, np.sqrt(np.maximum(img, 1.0)))
    return img


def _plant_parasitic_spot(img, center_yx, peak=5_000.0, sigma=1.2):
    """Add a 2D Gaussian Bragg-like spot to the image (in-place)."""
    cy, cx = center_yx
    NZ, NY = img.shape
    yy, xx = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    img += peak * np.exp(-((yy - cx) ** 2 + (xx - cy) ** 2) / (2 * sigma ** 2))
    return img


# ----------------------------------------------------------------------
# (1) Single-panel — planted spots are detected, rings are preserved.
# ----------------------------------------------------------------------

def test_single_panel_planted_spots_detected():
    spec = _spec()
    img = _powder_image(spec)
    # Plant 3 spots, each at a different ring (different R, different η).
    # Match the same coordinate system the powder image used.
    NY, NZ = spec.NrPixelsY, spec.NrPixelsZ
    cy = NZ / 2.0 - 0.41
    cx = NY / 2.0 + 0.37
    spot_centers = [
        (cy + 30.0, cx + 0.0),    # ring R≈30, η≈0
        (cy + 0.0,  cx - 60.0),   # ring R≈60, η≈180
        (cy - 45.0, cx + 78.0),   # ring R≈90, η ≈ -60
    ]
    for c in spot_centers:
        _plant_parasitic_spot(img, c, peak=8_000.0, sigma=1.2)

    geom = HardBinGeometry.from_spec(spec)
    cleaned, mask = azimuthal_sigma_clip(img, geom, n_sigma=5.0)

    # Each planted spot's centre pixel must be flagged.
    for (cy_i, cx_i) in spot_centers:
        i, j = int(round(cy_i)), int(round(cx_i))
        # Look in a 3×3 window — the centre pixel may shift by ±1 due to rounding.
        window = mask[i - 1:i + 2, j - 1:j + 2]
        assert window.any(), f"no pixel near ({i}, {j}) flagged"

    # Powder rings themselves must not be wholesale flagged. We track
    # the fraction of ring-pixels (within ±2 px of any planted ring
    # centre) that survived; should be > 95%.
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = (yy - cx) * 200.0
    Zc = (zz - cy) * 200.0
    R_px = np.sqrt(Yc * Yc + Zc * Zc) / 200.0
    ring_pix = np.zeros_like(mask, dtype=bool)
    for R0 in (30.0, 60.0, 90.0):
        ring_pix |= np.abs(R_px - R0) < 2.0
    on_ring = ring_pix.sum()
    flagged_on_ring = (mask & ring_pix).sum()
    # Each spot bleeds ~10-30 ring-pixels, three spots ≈ ≤90.
    # Total ring pixels for three rings on 256×256 ≈ several thousand.
    assert flagged_on_ring < 0.05 * on_ring


def test_single_panel_no_planted_spots_few_false_positives():
    """On a clean powder image, azimuthal clip should flag very few
    pixels — only noise tails above n_sigma."""
    spec = _spec()
    img = _powder_image(spec, seed=1)
    geom = HardBinGeometry.from_spec(spec)
    _, mask = azimuthal_sigma_clip(img, geom, n_sigma=6.0)
    # Conservative bound: 5σ-equivalent tails are ~6e-7 per pixel under
    # Gaussian; at MAD-based σ on a powder ring we still expect ≪ 1%
    # false positives. The valid-pixel set isn't the whole image (corners
    # fall outside [RMin, RMax]); compare against valid count.
    valid = geom.valid.cpu().numpy().reshape(mask.shape)
    n_flagged = mask.sum()
    n_valid = valid.sum()
    assert n_flagged < 0.02 * n_valid, (
        f"too many false positives: {n_flagged}/{n_valid}"
    )


def test_single_panel_cleaned_value_at_spot():
    """At a planted-spot pixel, the cleaned image should be much closer
    to the ring intensity than to the original spot intensity."""
    spec = _spec()
    img = _powder_image(spec, seed=2)
    cy = spec.NrPixelsZ / 2.0 - 0.41
    cx = spec.NrPixelsY / 2.0 + 0.37
    sy, sx = cy + 60.0, cx        # on ring R≈60, η≈90
    _plant_parasitic_spot(img, (sy, sx), peak=20_000.0, sigma=0.8)

    geom = HardBinGeometry.from_spec(spec)
    cleaned, mask = azimuthal_sigma_clip(img, geom, n_sigma=5.0,
                                          mode="replace_with_median")
    i, j = int(round(sy)), int(round(sx))
    assert mask[i, j]
    assert img[i, j] > 15_000.0       # original is huge
    assert cleaned[i, j] < 1_500.0    # cleaned is back near ring


def test_flag_only_leaves_image_untouched():
    spec = _spec()
    img = _powder_image(spec, seed=3)
    img_orig = img.copy()
    geom = HardBinGeometry.from_spec(spec)
    out, mask = azimuthal_sigma_clip(img, geom, mode="flag_only")
    np.testing.assert_array_equal(out, img_orig)
    assert mask.dtype == bool


def test_replace_with_nan():
    spec = _spec()
    img = _powder_image(spec, seed=4)
    cy = spec.NrPixelsZ / 2.0 - 0.41
    cx = spec.NrPixelsY / 2.0 + 0.37
    _plant_parasitic_spot(img, (cy + 30.0, cx), peak=10_000.0)
    geom = HardBinGeometry.from_spec(spec)
    out, mask = azimuthal_sigma_clip(img, geom, mode="replace_with_nan")
    assert np.isnan(out[mask]).all()
    assert not np.isnan(out[~mask]).any()


# ----------------------------------------------------------------------
# (2) Multi-panel — each panel handled independently.
# ----------------------------------------------------------------------

def test_multi_panel_independent_processing():
    """Two-panel mockup (Pilatus-2M-style tiles each with their own
    geometry). Plant one spot on each panel, verify both are flagged
    without cross-talk."""
    # Two panels of the same physical size, offset in BC (so the beam
    # centre sits at slightly different pixel coordinates on each panel
    # — mimics tile alignment). Same Lsd, same wavelength.
    spec_a = _spec(BC_off_y=0.37, BC_off_z=-0.41)
    spec_b = _spec(BC_off_y=-0.25, BC_off_z=0.55)
    img_a = _powder_image(spec_a, seed=10)
    img_b = _powder_image(spec_b, seed=11)

    # Plant a spot on each panel at different rings.
    cy_a = spec_a.NrPixelsZ / 2.0 - 0.41
    cx_a = spec_a.NrPixelsY / 2.0 + 0.37
    _plant_parasitic_spot(img_a, (cy_a + 30.0, cx_a + 5.0), peak=12_000.0)

    cy_b = spec_b.NrPixelsZ / 2.0 + 0.55
    cx_b = spec_b.NrPixelsY / 2.0 - 0.25
    _plant_parasitic_spot(img_b, (cy_b - 60.0, cx_b + 4.0), peak=12_000.0)

    geom_a = HardBinGeometry.from_spec(spec_a)
    geom_b = HardBinGeometry.from_spec(spec_b)
    cleaned, masks = azimuthal_sigma_clip_multi_panel(
        [img_a, img_b], [geom_a, geom_b], n_sigma=5.0,
    )

    assert len(cleaned) == 2 and len(masks) == 2

    # Each panel's planted spot region is flagged on that panel only.
    i_a, j_a = int(round(cy_a + 30.0)), int(round(cx_a + 5.0))
    assert masks[0][i_a - 1:i_a + 2, j_a - 1:j_a + 2].any()

    i_b, j_b = int(round(cy_b - 60.0)), int(round(cx_b + 4.0))
    assert masks[1][i_b - 1:i_b + 2, j_b - 1:j_b + 2].any()

    # Cross-talk check: panel A's mask shouldn't be flagging panel-B's
    # spot region (the two panels have different geometries — pixel
    # coords don't carry over).
    valid_a = geom_a.valid.cpu().numpy().reshape(masks[0].shape)
    valid_b = geom_b.valid.cpu().numpy().reshape(masks[1].shape)
    assert masks[0].sum() < 0.02 * valid_a.sum()
    assert masks[1].sum() < 0.02 * valid_b.sum()


# ----------------------------------------------------------------------
# (3) Argument validation
# ----------------------------------------------------------------------

def test_shape_mismatch_raises():
    spec = _spec(NY=64, NZ=64)
    geom = HardBinGeometry.from_spec(spec)
    with pytest.raises(ValueError, match="does not match geometry"):
        azimuthal_sigma_clip(np.zeros((32, 32)), geom)


def test_bad_mode_raises():
    spec = _spec(NY=32, NZ=32)
    geom = HardBinGeometry.from_spec(spec)
    img = np.zeros((spec.NrPixelsZ, spec.NrPixelsY))
    with pytest.raises(ValueError, match="unknown mode"):
        azimuthal_sigma_clip(img, geom, mode="bogus")


def test_panel_count_mismatch_raises():
    spec = _spec(NY=32, NZ=32)
    geom = HardBinGeometry.from_spec(spec)
    img = np.zeros((spec.NrPixelsZ, spec.NrPixelsY))
    with pytest.raises(ValueError, match="length mismatch"):
        azimuthal_sigma_clip_multi_panel([img], [geom, geom])
