"""Smoke tests — exercise the M0/M1 building blocks without real images.

These tests are intentionally light: they verify that the forward primitives
compile, autograd flows, and the parameter framework round-trips.  Full
parity tests vs v1 require calibrant datasets and live in a separate suite.
"""
from __future__ import annotations

import math

import pytest
import torch


def test_parameters_pack_unpack():
    from midas_calibrate_v2.parameters import (
        Parameter, CalibrationSpec,
    )
    from midas_calibrate_v2.parameters.pack import (
        pack_spec, unpack_spec, refined_indices, refined_subset,
    )

    s = CalibrationSpec()
    s.add(Parameter("a", 1.0, refined=True))
    s.add(Parameter("b", 2.0, refined=False))
    s.add(Parameter("c", torch.tensor([3.0, 4.0]), refined=True))

    x, info = pack_spec(s)
    assert x.shape == (4,)
    assert info.refined == [True, False, True]
    sub = refined_subset(x, info)
    # 'a' (1) and 'c' (2) → 3 entries
    assert sub.shape == (3,)
    assert sub.tolist() == [1.0, 3.0, 4.0]
    out = unpack_spec(x, info, s)
    assert float(out["a"]) == 1.0
    assert float(out["b"]) == 2.0
    assert out["c"].tolist() == [3.0, 4.0]


def test_geometry_autograd():
    """Verify gradients flow to every refined geometry parameter."""
    from midas_calibrate_v2.forward.geometry import pixel_to_REta

    Y = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float64)
    Z = torch.tensor([100.0, 250.0, 400.0], dtype=torch.float64)
    Lsd = torch.tensor(1_000_000.0, dtype=torch.float64, requires_grad=True)
    BC_y = torch.tensor(1024.0, dtype=torch.float64, requires_grad=True)
    BC_z = torch.tensor(1024.0, dtype=torch.float64, requires_grad=True)
    tx = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    tz = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    p = torch.zeros(15, dtype=torch.float64, requires_grad=True)
    parallax = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    pxY = torch.tensor(200.0, dtype=torch.float64, requires_grad=True)
    pxZ = torch.tensor(200.0, dtype=torch.float64, requires_grad=True)
    rho_d = torch.tensor(1500.0, dtype=torch.float64)

    out = pixel_to_REta(
        Y, Z, Lsd=Lsd, BC_y=BC_y, BC_z=BC_z,
        tx=tx, ty=ty, tz=tz, p_coeffs=p, parallax=parallax,
        pxY=pxY, pxZ=pxZ, rho_d=rho_d,
    )
    loss = out.R_px.sum() + out.eta_deg.sum()
    loss.backward()
    for v in (Lsd, BC_y, BC_z, tx, ty, tz, p, parallax, pxY, pxZ):
        assert v.grad is not None, f"no gradient for {v}"
        assert torch.isfinite(v.grad).all(), f"non-finite gradient for {v}"


def test_parallax_zero_grad_clean():
    """Parallax = 0 should still carry gradient (v1 bug, fixed in v2)."""
    from midas_calibrate_v2.forward.parallax import parallax_correction

    R_px = torch.tensor([1000.0, 1500.0], dtype=torch.float64)
    rad_um = torch.tensor([2e5, 3e5], dtype=torch.float64)
    Lsd = torch.tensor(1e6, dtype=torch.float64)
    parallax = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    px = torch.tensor(200.0, dtype=torch.float64)

    out = parallax_correction(R_px, rad_um, Lsd, parallax, px)
    out.sum().backward()
    assert parallax.grad is not None
    # Gradient should equal sin(2θ)/px summed, NOT zero from a graph break.
    assert parallax.grad.abs() > 0


def test_distortion_basis_v1_layout():
    from midas_calibrate_v2.forward.distortion import (
        distortion_factor, v1_term_layout,
    )
    R_norm = torch.tensor(0.5, dtype=torch.float64)
    eta = torch.tensor(45.0, dtype=torch.float64)
    p = torch.linspace(-0.01, 0.01, 15, dtype=torch.float64)
    D = distortion_factor(R_norm, eta, p, terms=v1_term_layout())
    assert torch.isfinite(D)


def test_panel_layout_regular():
    from midas_calibrate_v2.forward.panels import PanelLayout, apply_panel_shifts

    layout = PanelLayout.regular(n_y=2, n_z=3, sy=10, sz=20, gap_y=2, gap_z=3)
    assert layout.n_panels() == 6
    assert layout.panel_index_mask is not None
    assert layout.panel_index_mask.shape == (22, 66)

    Y = torch.tensor([5.0, 15.0], dtype=torch.float64)
    Z = torch.tensor([10.0, 30.0], dtype=torch.float64)
    panel_idx = torch.tensor([0, 4], dtype=torch.long)
    delta_yz = torch.zeros(6, 2, dtype=torch.float64, requires_grad=True)
    delta_theta = torch.zeros(6, dtype=torch.float64, requires_grad=True)

    Yn, Zn = apply_panel_shifts(Y, Z, panel_idx, layout, delta_yz, delta_theta)
    Yn.sum().backward()
    assert delta_yz.grad is not None


def test_pseudo_strain_residual_smoke():
    from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual

    Y = torch.tensor([100.0, 200.0], dtype=torch.float64)
    Z = torch.tensor([100.0, 250.0], dtype=torch.float64)
    rtt = torch.tensor([5.0, 7.0], dtype=torch.float64)
    p = {
        "Lsd": torch.tensor(1e6, requires_grad=True),
        "BC_y": torch.tensor(1024.0, requires_grad=True),
        "BC_z": torch.tensor(1024.0, requires_grad=True),
        "tx": torch.tensor(0.0),
        "ty": torch.tensor(0.0, requires_grad=True),
        "tz": torch.tensor(0.0, requires_grad=True),
        "Parallax": torch.tensor(0.0),
        "pxY": torch.tensor(200.0, requires_grad=True),
        "pxZ": torch.tensor(200.0),
    }
    for i in range(15):
        p[f"p{i}"] = torch.tensor(0.0, requires_grad=True)

    r = pseudo_strain_residual(
        Y, Z, rtt, p, rho_d=torch.tensor(1500.0),
    )
    r.sum().backward()
    assert p["Lsd"].grad is not None
    assert p["pxY"].grad is not None


def test_bragg_round_trip():
    from midas_calibrate_v2.forward.bragg import two_theta_from_d, d_from_two_theta

    d = torch.tensor([3.123, 1.951, 1.249], dtype=torch.float64)
    lam = torch.tensor(1.5406, dtype=torch.float64)
    tt = two_theta_from_d(d, lam)
    d_back = d_from_two_theta(tt, lam)
    assert torch.allclose(d_back, d, atol=1e-9)


def test_v1_compat_spec():
    """v1 → v2 spec conversion preserves geometry values."""
    from midas_calibrate.params import CalibrationParams
    from midas_calibrate_v2.compat.from_v1 import spec_from_v1_params

    v1 = CalibrationParams()
    v1.Lsd = 900_000.0
    v1.BC_y = 1024.0
    v1.BC_z = 1024.0
    v1.ty = 0.1
    v1.tz = 0.2
    v1.pxY = 200.0
    v1.pxZ = 200.0
    v1.NrPixelsY = 2048
    v1.NrPixelsZ = 2048
    v1.MaxRingRad = 1500.0
    v1.SpaceGroup = 225
    v1.LatticeConstant = (5.41, 5.41, 5.41, 90.0, 90.0, 90.0)
    v1.Wavelength = 0.2
    for i in range(15):
        setattr(v1, f"p{i}", 0.0)

    s = spec_from_v1_params(v1)
    assert "Lsd" in s.parameters
    assert "pxY" in s.parameters
    assert "pxZ" in s.parameters
    assert "tx" in s.parameters
    assert s.parameters["tx"].refined is False  # v1 keeps tx fixed
    # By default v2 inherits v1's refinement choices: pxY, pxZ start fixed.
    assert s.parameters["pxY"].refined is False
    assert s.parameters["pxZ"].refined is False
