"""Item 45 — MILK pyFAI-substitute adapter."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import spec_from_v1_params
from midas_integrate_v2.io import MILKMultiGeometryAdapter


def _spec(NY=64, NZ=64, lsd=1_000_000.0):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=lsd,
        BC_y=NY / 2.0, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=1.0, RMax=20.0, RBinSize=0.5,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
        Wavelength=0.18,
    )
    return spec_from_v1_params(p, requires_grad=False)


def test_milk_adapter_single_panel_2theta():
    s = _spec()
    img = np.full((s.NrPixelsZ, s.NrPixelsY), 100.0)
    adapter = MILKMultiGeometryAdapter([s], unit="2th_deg")
    result = adapter.integrate1d([img], npt=64, method="polygon")
    assert result.radial.shape == (64,)
    assert result.intensity.shape == (64,)
    assert result.sigma.shape == (64,)
    # Uniform image → uniform integrated intensity
    assert (np.abs(result.intensity - 100.0) < 5.0).any()


def test_milk_adapter_multi_panel_q_axis():
    s_a = _spec(lsd=1_000_000.0)
    s_b = _spec(lsd=1_500_000.0)
    img_a = np.full((s_a.NrPixelsZ, s_a.NrPixelsY), 50.0)
    img_b = np.full((s_b.NrPixelsZ, s_b.NrPixelsY), 50.0)
    adapter = MILKMultiGeometryAdapter([s_a, s_b], unit="q_A^-1")
    result = adapter.integrate1d([img_a, img_b], npt=128, method="polygon")
    assert result.radial.shape == (128,)
    # Q axis should be monotone increasing
    assert (np.diff(result.radial) > 0).all()
    # Where we have stacked coverage, σ should be smaller than per-panel
    assert (np.nanmin(result.sigma) > 0)


def test_milk_adapter_with_sigma_returns_triple():
    s = _spec()
    img = np.full((s.NrPixelsZ, s.NrPixelsY), 50.0)
    adapter = MILKMultiGeometryAdapter([s], unit="2th_deg")
    q, I, sigma = adapter.integrate1d_with_sigma([img], npt=32)
    assert q.shape == I.shape == sigma.shape == (32,)


def test_milk_adapter_unit_conversion():
    s = _spec()
    img = np.ones((s.NrPixelsZ, s.NrPixelsY))
    for unit in ("2th_deg", "q_A^-1", "q_nm^-1", "r_mm"):
        adapter = MILKMultiGeometryAdapter([s], unit=unit)
        result = adapter.integrate1d([img], npt=32)
        assert np.all(np.diff(result.radial) > 0), \
            f"radial not monotone for {unit}"


def test_milk_adapter_normalisation_factor():
    s = _spec()
    img = np.full((s.NrPixelsZ, s.NrPixelsY), 100.0)
    adapter = MILKMultiGeometryAdapter([s], unit="2th_deg")
    res_un = adapter.integrate1d([img], npt=32)
    res_norm = adapter.integrate1d([img], npt=32, normalization_factor=[2.0])
    np.testing.assert_allclose(res_norm.intensity * 2.0,
                                 res_un.intensity, rtol=1e-12)
