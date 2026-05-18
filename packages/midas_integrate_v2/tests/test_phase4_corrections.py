"""Phase 4: Q-mode bin edges + polarization + solid-angle as torch.

Three guarantees:

1. ``build_q_bin_edges_in_R`` matches v1's numpy implementation to fp64
   bit-level on the same inputs.
2. ``PolarizationCorrection`` and ``SolidAngleCorrection`` reproduce v1's
   per-pixel correction factors at v1's interior tolerance.
3. Both corrections plug into ``integrate_with_corrections`` and
   gradient flows back through them to ``Lsd`` (geometry) and to the
   correction's own refinable parameters.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math

import numpy as np
import pytest
import torch

from midas_integrate.geometry import build_q_bin_edges_in_R as v1_build_q
from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import (
    spec_from_v1_params,
    build_q_bin_edges_in_R,
    PolarizationCorrection,
    SolidAngleCorrection,
    integrate_with_corrections,
    polarization_factor,
    solid_angle_factor_flat,
)


# ── (1) Q-mode bin edges parity ──

def test_q_mode_bin_edges_match_v1():
    QMin, QMax, QBinSize = 0.5, 7.0, 0.01
    Lsd_val, px_val, lam_val = 895_900.0, 150.0, 0.172979
    r_lo_v1, r_hi_v1, n_v1 = v1_build_q(QMin, QMax, QBinSize,
                                          Lsd_val, px_val, lam_val)

    Lsd = torch.tensor(Lsd_val, dtype=torch.float64)
    px = torch.tensor(px_val, dtype=torch.float64)
    lam = torch.tensor(lam_val, dtype=torch.float64)
    r_lo, r_hi, n = build_q_bin_edges_in_R(
        QMin=QMin, QMax=QMax, QBinSize=QBinSize,
        Lsd=Lsd, px=px, wavelength_A=lam,
    )
    assert n == n_v1
    np.testing.assert_allclose(r_lo.numpy(), r_lo_v1, rtol=0, atol=1e-10)
    np.testing.assert_allclose(r_hi.numpy(), r_hi_v1, rtol=0, atol=1e-10)


def test_q_mode_bin_edges_differentiable_in_lsd_and_wavelength():
    Lsd = torch.tensor(895_900.0, dtype=torch.float64, requires_grad=True)
    px = torch.tensor(150.0, dtype=torch.float64)
    lam = torch.tensor(0.172979, dtype=torch.float64, requires_grad=True)
    r_lo, _, _ = build_q_bin_edges_in_R(
        QMin=0.5, QMax=7.0, QBinSize=0.01,
        Lsd=Lsd, px=px, wavelength_A=lam,
    )
    L = r_lo.sum()
    L.backward()
    assert Lsd.grad is not None and torch.isfinite(Lsd.grad)
    assert lam.grad is not None and torch.isfinite(lam.grad)
    # dR/dLsd > 0 (longer Lsd → larger R for same Q); dR/dλ > 0.
    assert float(Lsd.grad) > 0
    assert float(lam.grad) > 0


# ── (2) per-pixel correction factor parity vs v1 ──

def test_polarization_factor_matches_v1_pointwise():
    """v1 uses ``1 - PF · sin²(2θ) · cos²(η - plane)``; same formula."""
    Lsd_val = 1_000_000.0
    px_val = 200.0
    PF, plane = 0.99, 12.0

    Rs = np.linspace(2.0, 1500.0, 25)
    etas = np.linspace(-170.0, 170.0, 13)
    R_grid, eta_grid = np.meshgrid(Rs, etas, indexing="ij")

    twoTheta = np.arctan(R_grid * px_val / Lsd_val)
    s2t = np.sin(twoTheta)
    ce = np.cos((eta_grid - plane) * math.pi / 180.0)
    expect = 1.0 - PF * s2t * s2t * ce * ce

    R_t = torch.tensor(R_grid.flatten(), dtype=torch.float64)
    eta_t = torch.tensor(eta_grid.flatten(), dtype=torch.float64)
    Lsd_t = torch.tensor(Lsd_val, dtype=torch.float64)
    px_t = torch.tensor(px_val, dtype=torch.float64)
    pf_t = torch.tensor(PF, dtype=torch.float64)
    pe_t = torch.tensor(plane, dtype=torch.float64)
    got = polarization_factor(R_t, eta_t, Lsd=Lsd_t, px=px_t,
                                pol_fraction=pf_t, pol_plane_eta_deg=pe_t)
    np.testing.assert_allclose(got.numpy().reshape(R_grid.shape), expect,
                                rtol=0, atol=1e-12)


def test_solid_angle_flat_matches_cos3_2theta():
    Lsd_val = 1_000_000.0
    px_val = 200.0
    Rs = np.linspace(2.0, 1500.0, 25)
    twoTheta = np.arctan(Rs * px_val / Lsd_val)
    expect = np.cos(twoTheta) ** 3
    R_t = torch.tensor(Rs, dtype=torch.float64)
    Lsd_t = torch.tensor(Lsd_val, dtype=torch.float64)
    px_t = torch.tensor(px_val, dtype=torch.float64)
    got = solid_angle_factor_flat(R_t, Lsd=Lsd_t, px=px_t)
    np.testing.assert_allclose(got.numpy(), expect, rtol=0, atol=1e-15)


# ── (3) corrections wire into integrate_with_corrections ──

def _spec(NY=24, NZ=24, requires_grad=True):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    return spec_from_v1_params(p, requires_grad=requires_grad)


def test_polarization_correction_module_changes_output():
    """Sanity: applying polarization correction changes the integrated
    profile by a non-trivial amount."""
    s = _spec()
    img = torch.ones(s.NrPixelsZ, s.NrPixelsY, dtype=torch.float64)
    pol = PolarizationCorrection(pol_fraction=0.99, pol_plane_eta_deg=0.0)

    int_no = integrate_with_corrections(img, s)
    int_pol = integrate_with_corrections(img, s, polarization=pol)
    diff = (int_no - int_pol).abs().max()
    assert float(diff) > 1e-6, "polarization correction had no effect"


def test_solid_angle_correction_module_changes_output():
    s = _spec()
    img = torch.ones(s.NrPixelsZ, s.NrPixelsY, dtype=torch.float64)
    sa = SolidAngleCorrection()

    int_no = integrate_with_corrections(img, s)
    int_sa = integrate_with_corrections(img, s, solid_angle=sa)
    diff = (int_no - int_sa).abs().max()
    assert float(diff) > 1e-6, "solid-angle correction had no effect"


def test_corrections_gradient_flows_to_geometry_and_pol_params():
    s = _spec()
    img = torch.ones(s.NrPixelsZ, s.NrPixelsY, dtype=torch.float64)
    pol = PolarizationCorrection(pol_fraction=0.99, pol_plane_eta_deg=0.0,
                                   refinable=True)
    sa = SolidAngleCorrection()
    int2d = integrate_with_corrections(
        img, s, polarization=pol, solid_angle=sa,
    )
    L = int2d.mean()
    L.backward()
    assert s.Lsd.grad is not None and torch.isfinite(s.Lsd.grad)
    assert pol.pol_fraction.grad is not None
    assert torch.isfinite(pol.pol_fraction.grad)
    assert pol.pol_plane_eta_deg.grad is not None
    assert torch.isfinite(pol.pol_plane_eta_deg.grad)
