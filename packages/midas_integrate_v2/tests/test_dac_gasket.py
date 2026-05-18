"""Item 6 — DAC gasket mask + η-coverage diagnostic."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import build_provenance, spec_from_v1_params
from midas_integrate_v2.dac import build_gasket_mask, eta_coverage_per_ring


def _spec(NY=64, NZ=64):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=1.0, RMax=20.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
    )
    return spec_from_v1_params(p, requires_grad=False)


def test_full_open_mask_full_coverage():
    s = _spec()
    mask = build_gasket_mask(
        NrPixelsY=s.NrPixelsY, NrPixelsZ=s.NrPixelsZ,
        BC=(float(s.BC_y), float(s.BC_z)),
        eta_open_deg=(-180.0, 180.0),
        symmetry="single",
    )
    assert mask.dtype == bool
    assert mask.all()
    rings = torch.tensor([10.0, 15.0])
    cov = eta_coverage_per_ring(s, mask, rings)
    np.testing.assert_allclose(cov.numpy(), [1.0, 1.0], atol=1e-6)


def test_two_fold_dac_mask_half_coverage():
    s = _spec()
    # ±10° wedge twice (top and bottom): roughly 20°/180° = ~0.11 of η
    mask = build_gasket_mask(
        NrPixelsY=s.NrPixelsY, NrPixelsZ=s.NrPixelsZ,
        BC=(float(s.BC_y), float(s.BC_z)),
        eta_open_deg=(-10.0, 10.0),
        symmetry="two_fold",
    )
    rings = torch.tensor([10.0])
    cov = eta_coverage_per_ring(s, mask, rings).numpy()
    # Two 20°-wide wedges → ~40°/360° = ~0.11 in η
    assert cov[0] < 0.20
    assert cov[0] > 0.05


def test_four_fold_dac_mask_more_coverage():
    s = _spec()
    mask = build_gasket_mask(
        NrPixelsY=s.NrPixelsY, NrPixelsZ=s.NrPixelsZ,
        BC=(float(s.BC_y), float(s.BC_z)),
        eta_open_deg=(-10.0, 10.0),
        symmetry="four_fold",
    )
    rings = torch.tensor([10.0])
    cov = eta_coverage_per_ring(s, mask, rings).numpy()
    # Four 20° wedges → ~80°/360° = ~0.22
    assert cov[0] > 0.15
    assert cov[0] < 0.30


def test_provenance_carries_eta_coverage():
    s = _spec()
    mask = build_gasket_mask(
        NrPixelsY=s.NrPixelsY, NrPixelsZ=s.NrPixelsZ,
        BC=(float(s.BC_y), float(s.BC_z)),
        eta_open_deg=(-30.0, 30.0),
        symmetry="two_fold",
    )
    rings = torch.tensor([8.0, 14.0])
    cov = eta_coverage_per_ring(s, mask, rings)
    meta = build_provenance(s, eta_coverage_per_ring=cov)
    assert "eta_coverage_per_ring" in meta.extra
    vals = meta.extra["eta_coverage_per_ring"]
    assert len(vals) == 2
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_unknown_symmetry_raises():
    with pytest.raises(ValueError):
        build_gasket_mask(
            NrPixelsY=16, NrPixelsZ=16, BC=(8.0, 8.0),
            eta_open_deg=(-10.0, 10.0),
            symmetry="seven_fold",
        )
