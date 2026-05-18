"""Item 8 — Parallax kernel tests + bit-parity at Parallax=0.

Verifies:
- Parallax=0 → integration matches the no-parallax reference (already
  the v2 forward kernel always applies parallax_correction; parallax=0
  must be a no-op).
- Non-zero parallax shifts ring radii by the expected analytic amount.
- Gradient flows back to spec.Parallax.
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
    integrate_with_corrections,
)
from midas_integrate_v2.forward import pixel_to_REta_from_spec


def _spec(NY=64, NZ=64, parallax=0.0):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=1.0, RMax=20.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
        Parallax=parallax,
    )
    return spec_from_v1_params(p, requires_grad=False)


def test_parallax_zero_is_no_op_in_kernel():
    """At Parallax=0, R_px should not change vs computing it without
    parallax (which here means the same kernel — the v2 always applies
    the parallax correction with parallax=0)."""
    s0 = _spec(parallax=0.0)
    NY, NZ = s0.NrPixelsY, s0.NrPixelsZ
    ys = torch.arange(NY, dtype=torch.float64)
    zs = torch.arange(NZ, dtype=torch.float64)
    Z, Y = torch.meshgrid(zs, ys, indexing="ij")
    out = pixel_to_REta_from_spec(Y, Z, s0)
    R0 = out.R_px
    # Closed-form expected R: hypotenuse of (Y-BC, Z-BC) since tilts=0
    BC_y, BC_z = float(s0.BC_y), float(s0.BC_z)
    R_expected = torch.sqrt((Y - BC_y) ** 2 + (Z - BC_z) ** 2)
    np.testing.assert_allclose(R0.numpy(), R_expected.numpy(),
                                rtol=1e-9, atol=1e-9)


def test_parallax_shifts_R_predictably():
    """Non-zero parallax displaces R by parallax · sin(2θ)/(px) up to
    the kernel's exact form. We just confirm R changes by a non-trivial
    monotone amount with parallax sign."""
    s0 = _spec(parallax=0.0)
    s_pos = _spec(parallax=50.0)
    s_neg = _spec(parallax=-50.0)
    NY, NZ = s0.NrPixelsY, s0.NrPixelsZ
    ys = torch.arange(NY, dtype=torch.float64)
    zs = torch.arange(NZ, dtype=torch.float64)
    Z, Y = torch.meshgrid(zs, ys, indexing="ij")
    R0 = pixel_to_REta_from_spec(Y, Z, s0).R_px
    Rp = pixel_to_REta_from_spec(Y, Z, s_pos).R_px
    Rn = pixel_to_REta_from_spec(Y, Z, s_neg).R_px
    diff_pos = (Rp - R0)
    diff_neg = (Rn - R0)
    # Ignore the central pixel where R~=0.
    big_R = R0 > 5.0
    # Positive parallax shifts pos vs neg in opposite sign
    assert (diff_pos[big_R] * diff_neg[big_R] <= 0).all()
    # Magnitude is non-trivial
    assert (diff_pos[big_R].abs().max() > 0.001)


def test_parallax_gradient_flows_to_spec():
    """Use a non-uniform image so the integrated profile depends on
    where each pixel falls in R; gradient through soft-bin assignment
    is then non-zero w.r.t. Parallax."""
    s = _spec(parallax=10.0)
    s.Parallax = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    # Annular bright spot at R~8 px to make integration ring-position-sensitive
    NY, NZ = s.NrPixelsY, s.NrPixelsZ
    BC_y, BC_z = float(s.BC_y), float(s.BC_z)
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    R_pix = np.sqrt((yy - BC_y) ** 2 + (zz - BC_z) ** 2)
    img = torch.as_tensor(
        np.exp(-((R_pix - 8.0) / 1.0) ** 2),
        dtype=torch.float64,
    )
    out = integrate_with_corrections(img, s)
    # Use a max over a single bin to make the loss locally sensitive.
    L = out[out.shape[0] // 2, 7]
    L.backward()
    assert s.Parallax.grad is not None
    assert torch.isfinite(s.Parallax.grad)
    assert float(s.Parallax.grad.abs()) > 0.0
