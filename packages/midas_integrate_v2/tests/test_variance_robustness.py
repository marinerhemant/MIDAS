"""Tests for the three robustness improvements to per-bin σ:

1. ``error_model`` — 'poisson' | 'azimuthal' | 'hybrid'
2. ``correction`` — per-pixel intensity correction propagated into σ
3. ``empty_bin_value`` — NaN (default) for bins with zero accumulated weight
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import (
    spec_from_v1_params,
    HardBinGeometry, integrate_hard_with_variance,
    SubpixelBinGeometry, integrate_subpixel_with_variance,
    PolygonBinGeometry, integrate_polygon_with_variance,
)


def _spec(NY=24, NZ=24):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    return spec_from_v1_params(p, requires_grad=False)


# ----------------------------------------------------------------------
# (1) error_model
# ----------------------------------------------------------------------

def test_invalid_error_model_raises():
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 100.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="error_model"):
        integrate_hard_with_variance(img, geom, error_model="bogus")


def test_azimuthal_zero_on_constant_image_hard():
    """Constant image → zero pixel-to-pixel spread → σ_azim = 0 everywhere."""
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 100.0, dtype=torch.float64)
    _, sigma = integrate_hard_with_variance(img, geom,
                                              error_model="azimuthal")
    pop = torch.isfinite(sigma)
    assert torch.allclose(sigma[pop], torch.zeros_like(sigma[pop]),
                          atol=1e-12)


def test_hybrid_falls_back_to_poisson_on_constant_image():
    """Constant image: σ_azim=0 < σ_poisson, hybrid must pick poisson."""
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 100.0, dtype=torch.float64)
    _, sig_p = integrate_hard_with_variance(img, geom, error_model="poisson")
    _, sig_h = integrate_hard_with_variance(img, geom, error_model="hybrid")
    pop = torch.isfinite(sig_p) & torch.isfinite(sig_h)
    assert torch.allclose(sig_p[pop], sig_h[pop], atol=1e-12)


def test_azimuthal_captures_in_bin_spread_hard():
    """Build an image where in-bin pixels have wildly different values.
    σ_azim should dwarf σ_poisson — this is the regime Poisson under-
    estimates and the whole point of having an azimuthal mode."""
    s = _spec(NY=64, NZ=64)
    geom = HardBinGeometry.from_spec(s)
    # Pixel intensity = 100 + 1000 * (column index parity), so within
    # each polar bin we have alternating ~100 / ~1100 pixels: huge
    # azimuthal spread but moderate Poisson σ.
    yy, zz = np.meshgrid(np.arange(s.NrPixelsY), np.arange(s.NrPixelsZ),
                          indexing="xy")
    img_np = 100.0 + 1000.0 * (yy % 2).astype(np.float64)
    img = torch.from_numpy(img_np)
    _, sig_p = integrate_hard_with_variance(img, geom, error_model="poisson")
    _, sig_a = integrate_hard_with_variance(img, geom, error_model="azimuthal")
    _, sig_h = integrate_hard_with_variance(img, geom, error_model="hybrid")
    pop = torch.isfinite(sig_p) & torch.isfinite(sig_a) & (sig_p > 0)
    assert pop.any(), "need at least one populated bin"
    # On bins with the alternation, azimuthal σ must exceed Poisson σ.
    assert (sig_a[pop] > sig_p[pop]).any()
    # Hybrid >= both, by construction.
    assert (sig_h[pop] >= sig_p[pop] - 1e-12).all()
    assert (sig_h[pop] >= sig_a[pop] - 1e-12).all()


def test_azimuthal_polygon_runs_and_is_nonneg():
    s = _spec()
    geom = PolygonBinGeometry.from_spec(s)
    rng = np.random.default_rng(0)
    img = torch.from_numpy(rng.uniform(50.0, 150.0,
                                        size=(s.NrPixelsZ, s.NrPixelsY)))
    _, sig = integrate_polygon_with_variance(img, geom,
                                              error_model="azimuthal")
    finite = sig[torch.isfinite(sig)]
    assert (finite >= 0).all()


def test_azimuthal_subpixel_runs_and_is_nonneg():
    s = _spec()
    geom = SubpixelBinGeometry.from_spec(s, K=2)
    rng = np.random.default_rng(1)
    img = torch.from_numpy(rng.uniform(50.0, 150.0,
                                        size=(s.NrPixelsZ, s.NrPixelsY)))
    _, sig = integrate_subpixel_with_variance(img, geom,
                                                error_model="azimuthal")
    finite = sig[torch.isfinite(sig)]
    assert (finite >= 0).all()


# ----------------------------------------------------------------------
# (2) correction propagation
# ----------------------------------------------------------------------

def test_correction_scales_mean_and_sigma_consistently_hard():
    """If we divide every pixel by a constant c, the corrected mean
    should be (raw_mean / c) AND the corrected σ should be (raw_σ / c).
    This is the consistency the upstream pipeline was lacking."""
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 400.0, dtype=torch.float64)
    c_val = 4.0
    corr = torch.full_like(img, c_val)
    m_raw, s_raw = integrate_hard_with_variance(img, geom)
    m_corr, s_corr = integrate_hard_with_variance(img, geom, correction=corr)
    pop = torch.isfinite(m_raw) & torch.isfinite(m_corr)
    assert torch.allclose(m_corr[pop], m_raw[pop] / c_val, atol=1e-12)
    assert torch.allclose(s_corr[pop], s_raw[pop] / c_val, atol=1e-12)


def test_correction_matches_pre_division_on_user_variance():
    """When user supplies their own variance_image, dividing image by c
    and variance by c² before calling should match passing `correction=c`."""
    s = _spec()
    geom = PolygonBinGeometry.from_spec(s)
    rng = np.random.default_rng(2)
    img_np = rng.uniform(80.0, 120.0, size=(s.NrPixelsZ, s.NrPixelsY))
    var_np = rng.uniform(5.0, 25.0, size=(s.NrPixelsZ, s.NrPixelsY))
    corr_np = rng.uniform(0.8, 1.2, size=(s.NrPixelsZ, s.NrPixelsY))
    img = torch.from_numpy(img_np)
    var_img = torch.from_numpy(var_np)
    corr = torch.from_numpy(corr_np)

    m_a, s_a = integrate_polygon_with_variance(
        img, geom, variance_image=var_img, correction=corr,
    )
    m_b, s_b = integrate_polygon_with_variance(
        img / corr, geom, variance_image=var_img / (corr * corr),
    )
    pop = torch.isfinite(m_a) & torch.isfinite(m_b)
    assert torch.allclose(m_a[pop], m_b[pop], atol=1e-10)
    assert torch.allclose(s_a[pop], s_b[pop], atol=1e-10)


def test_correction_shape_mismatch_raises():
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 100.0, dtype=torch.float64)
    bad = torch.full((4, 4), 1.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="correction shape"):
        integrate_hard_with_variance(img, geom, correction=bad)


# ----------------------------------------------------------------------
# (3) empty-bin handling
# ----------------------------------------------------------------------

def test_empty_bins_default_to_nan_polygon():
    """With a tiny eta-band (so most eta slices are empty for the
    detector geometry), the default behaviour should mark unpopulated
    bins as NaN rather than silent 0."""
    p = IntegrationParams(
        NrPixelsY=24, NrPixelsZ=24,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=12.37, BC_z=11.59, RhoD=24.0,
        # Push RMin far past the detector edge so every bin is empty.
        RMin=500.0, RMax=520.0, RBinSize=2.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    s = spec_from_v1_params(p, requires_grad=False)
    geom = PolygonBinGeometry.from_spec(s)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 100.0, dtype=torch.float64)
    mean, sigma = integrate_polygon_with_variance(img, geom)
    # Every bin should be NaN (no pixel intersects this far-out ring).
    assert torch.isnan(mean).all()
    assert torch.isnan(sigma).all()


def test_empty_bins_zero_legacy_optin():
    """Passing empty_bin_value=0.0 restores the prior silent-zero
    behaviour for back-compat."""
    p = IntegrationParams(
        NrPixelsY=24, NrPixelsZ=24,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=12.37, BC_z=11.59, RhoD=24.0,
        RMin=500.0, RMax=520.0, RBinSize=2.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    s = spec_from_v1_params(p, requires_grad=False)
    geom = PolygonBinGeometry.from_spec(s)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 100.0, dtype=torch.float64)
    mean, sigma = integrate_polygon_with_variance(img, geom,
                                                    empty_bin_value=0.0)
    assert torch.all(mean == 0.0)
    assert torch.all(sigma == 0.0)


def test_partial_empty_bins_nan_does_not_poison_finite_bins():
    """Mixed populated / empty bins: NaN must only land on the empty
    ones, finite bins must hold their proper (mean, σ). Force a mix by
    masking out a quadrant of the detector."""
    s = _spec()
    NY, NZ = s.NrPixelsY, s.NrPixelsZ
    # Mask the upper-right quadrant: removes whole bins on a coarse grid.
    mask = np.zeros((NZ, NY), dtype=np.uint8)
    mask[: NZ // 2, NY // 2:] = 1
    geom = HardBinGeometry.from_spec(s, mask=mask)
    img = torch.full((NZ, NY), 100.0, dtype=torch.float64)
    mean, sigma = integrate_hard_with_variance(img, geom)
    finite = torch.isfinite(mean)
    nan = torch.isnan(mean)
    assert finite.any() and nan.any()
    # populated bins recover the constant-image mean.
    assert torch.allclose(mean[finite], torch.full_like(mean[finite], 100.0),
                          atol=1e-12)
    # sigma also NaN at the same locations.
    assert torch.all(torch.isnan(sigma) == nan)
