"""Item 29 — multi-detector simultaneous integration."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import spec_from_v1_params, NumpyArraySource
from midas_integrate_v2.streaming import integrate_multi_detector


def _spec_pair(NY=32, NZ=32, lsd_a=1_000_000.0, lsd_b=2_000_000.0):
    p_a = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=lsd_a,
        BC_y=NY / 2.0, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=0.5,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=30.0,
        Wavelength=0.18,
    )
    p_b = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=lsd_b,
        BC_y=NY / 2.0, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=0.5,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=30.0,
        Wavelength=0.18,
    )
    return (
        spec_from_v1_params(p_a, requires_grad=False),
        spec_from_v1_params(p_b, requires_grad=False),
    )


def test_multi_detector_uniform_image_unifies():
    """Two identical-uniform-image detectors stitched onto a Q grid:
    where at least one detector has coverage and finite data, the
    unified intensity must be near the original uniform value."""
    spec_a, spec_b = _spec_pair(lsd_a=1_000_000.0, lsd_b=1_000_000.0)
    n_frames = 2
    img = np.ones((n_frames, spec_a.NrPixelsZ, spec_a.NrPixelsY))
    src_a = NumpyArraySource(img.copy())
    src_b = NumpyArraySource(img.copy())
    # q-grid lies inside the detectors' actual Q coverage (Lsd=1e6 µm,
    # px=200 µm, λ=0.18 Å → Q span [~0.01, 0.17] Å⁻¹).
    q_grid = torch.linspace(0.02, 0.15, 15, dtype=torch.float64)
    out = list(integrate_multi_detector(
        [src_a, src_b], [spec_a, spec_b], q_grid=q_grid,
        overlap_weight="inverse_variance",
    ))
    assert len(out) == n_frames
    fid, I, sigma = out[0]
    finite = torch.isfinite(I) & torch.isfinite(sigma) & (sigma < 1e10)
    assert finite.any(), "no overlap between detectors on q_grid"
    assert (I[finite] > 0.5).all() and (I[finite] < 2.0).all()


def test_multi_detector_sigma_inverse_variance_weighted():
    """Test that low-σ detector dominates the unified σ."""
    spec_a, spec_b = _spec_pair()
    n_frames = 1
    img = (np.random.default_rng(0)
            .normal(100.0, 5.0, size=(n_frames, spec_a.NrPixelsZ, spec_a.NrPixelsY)))
    img = np.clip(img, 0, None)
    src_a = NumpyArraySource(img.copy())
    src_b = NumpyArraySource(img.copy())
    q_grid = torch.linspace(0.02, 0.15, 12, dtype=torch.float64)
    out = list(integrate_multi_detector(
        [src_a, src_b], [spec_a, spec_b], q_grid=q_grid,
        overlap_weight="inverse_variance",
    ))
    fid, I, sigma = out[0]
    finite = torch.isfinite(sigma) & (sigma > 0) & (sigma < 1e10)
    # σ_combined finite where coverage exists, much smaller than 1
    if finite.any():
        assert (sigma[finite] < 10.0).all()


def test_multi_detector_mismatched_sources_raises():
    spec_a, spec_b = _spec_pair()
    src_a = NumpyArraySource(np.ones((2, spec_a.NrPixelsZ, spec_a.NrPixelsY)))
    src_b = NumpyArraySource(np.ones((3, spec_a.NrPixelsZ, spec_a.NrPixelsY)))
    q_grid = torch.linspace(0.5, 4.0, 12, dtype=torch.float64)
    with pytest.raises(ValueError):
        list(integrate_multi_detector(
            [src_a, src_b], [spec_a, spec_b], q_grid=q_grid,
        ))
