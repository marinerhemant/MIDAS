"""A pixel exactly on the beam centre must not kill the detector's gradient.

``R = sqrt(y^2 + z^2)`` at the origin is a cone point (the derivative depends
on the approach direction) and ``eta = atan2(0, 0)`` is undefined, not merely
singular -- a pixel at the beam centre has no azimuth. So there is no correct
gradient to restore; the decision (spec_autograd_classB_classC.md, B1) is that
such a pixel is not an observation: R = 0, eta = 0, and EXACTLY zero gradient.

What makes it matter is that torch sums per-pixel gradients, so one NaN pixel
made ``dR/dBC`` NaN for the whole detector -- and ``BC 1022 1022`` (integer)
ships in the midas_ff_pipeline templates while midas_integrate_v2 evaluates
this on the full integer pixel grid.
"""

from __future__ import annotations

import pytest
import torch

from midas_calibrate_v2.forward.geometry import pixel_to_REta


def _call(Y, Z, bc_y, bc_z):
    return pixel_to_REta(
        Y, Z,
        Lsd=torch.tensor(1_000_000.0, dtype=torch.float64),
        BC_y=bc_y, BC_z=bc_z,
        tx=torch.tensor(0.0, dtype=torch.float64),
        ty=torch.tensor(0.0, dtype=torch.float64),
        tz=torch.tensor(0.0, dtype=torch.float64),
        p_coeffs=torch.zeros(15, dtype=torch.float64),
        parallax=torch.tensor(0.0, dtype=torch.float64),
        pxY=torch.tensor(200.0, dtype=torch.float64),
        pxZ=torch.tensor(200.0, dtype=torch.float64),
        rho_d=torch.tensor(1000.0, dtype=torch.float64),
    )


def _grid(lo, hi, bc_y, bc_z):
    ys = torch.arange(float(lo), float(hi), dtype=torch.float64)
    zs = torch.arange(float(lo), float(hi), dtype=torch.float64)
    Z, Y = torch.meshgrid(zs, ys, indexing="ij")
    return _call(Y, Z, bc_y, bc_z)


def test_single_pixel_on_the_beam_centre():
    BC = torch.tensor([1022.0, 1022.0], dtype=torch.float64, requires_grad=True)
    out = _call(torch.tensor([1022.0], dtype=torch.float64),
                torch.tensor([1022.0], dtype=torch.float64), BC[0], BC[1])
    assert float(out.R_px) == 0.0
    g = torch.autograd.grad(out.R_px.sum(), BC)[0]
    assert torch.isfinite(g).all()
    assert float(g.abs().max()) == 0.0, "the BC pixel must contribute nothing"


def test_whole_grid_gradient_survives_the_singular_pixel():
    """The headline failure: one bad pixel, no gradient anywhere."""
    BC = torch.tensor([1022.0, 1022.0], dtype=torch.float64, requires_grad=True)
    out = _grid(1020, 1025, BC[0], BC[1])
    g = torch.autograd.grad(out.R_px.sum(), BC)[0]
    assert torch.isfinite(g).all(), "one BC pixel poisoned the whole detector"


def test_grid_gradient_equals_the_same_grid_with_the_centre_pixel_dropped():
    """The zeroed pixel must be equivalent to excluding it, not to biasing it."""
    BC = torch.tensor([1022.0, 1022.0], dtype=torch.float64, requires_grad=True)
    out = _grid(1020, 1025, BC[0], BC[1])
    g_all = torch.autograd.grad(out.R_px.sum(), BC, retain_graph=True)[0]

    BC2 = torch.tensor([1022.0, 1022.0], dtype=torch.float64, requires_grad=True)
    ys = torch.arange(1020.0, 1025.0, dtype=torch.float64)
    zs = torch.arange(1020.0, 1025.0, dtype=torch.float64)
    Z, Y = torch.meshgrid(zs, ys, indexing="ij")
    keep = ~((Y == 1022.0) & (Z == 1022.0))
    out2 = _call(Y[keep], Z[keep], BC2[0], BC2[1])
    g_kept = torch.autograd.grad(out2.R_px.sum(), BC2)[0]

    assert torch.allclose(g_all, g_kept, atol=1e-12)


@pytest.mark.parametrize("bcy", [1021.5, 1021.9, 1022.0, 1022.1, 1022.5])
def test_gradient_finite_across_an_integer_crossing(bcy):
    """BC is refined, so it can land on an integer mid-optimisation."""
    BC = torch.tensor([bcy, 1022.0], dtype=torch.float64, requires_grad=True)
    out = _grid(1020, 1025, BC[0], BC[1])
    g = torch.autograd.grad(out.R_px.sum(), BC)[0]
    assert torch.isfinite(g).all()


def test_off_centre_values_and_gradients_unchanged():
    """A gradient fix must not move the forward model anywhere else."""
    BC = torch.tensor([1022.0, 1022.0], dtype=torch.float64, requires_grad=True)
    out = _call(torch.tensor([1023.0], dtype=torch.float64),
                torch.tensor([1022.0], dtype=torch.float64), BC[0], BC[1])
    assert abs(float(out.R_px) - 1.0) < 1e-12
    g = torch.autograd.grad(out.R_px.sum(), BC)[0]
    assert abs(float(g[0]) + 1.0) < 1e-12
    assert abs(float(g[1])) < 1e-12


@pytest.mark.parametrize("rho_d", [0.0, -1.0, None])
def test_non_positive_rho_d_does_not_nan_the_whole_detector(rho_d):
    """``RhoD = 0`` is an ordinary spec — it is what you get when no
    distortion was ever calibrated — and it must not poison the geometry.

    ``rho = R / rho_d`` was formed unguarded, so a zero normalisation radius
    gave every pixel inf and then, multiplied by the (zero) distortion
    coefficients, NaN. Measured before the guard: 262144 of 262144 pixels NaN
    from ``eval_pixel_REta`` on a 512x512 spec. It surfaced only when the
    polygon integration kernel started routing through this function; the
    hard and subpixel kernels had been returning NaN for such specs already.
    """
    BC = torch.tensor([100.0, 100.0], dtype=torch.float64)
    ys = torch.arange(0.0, 40.0, dtype=torch.float64)
    zs = torch.arange(0.0, 40.0, dtype=torch.float64)
    Z, Y = torch.meshgrid(zs, ys, indexing="ij")
    kw = dict(
        Lsd=torch.tensor(1_000_000.0, dtype=torch.float64),
        BC_y=BC[0], BC_z=BC[1],
        tx=torch.tensor(0.0, dtype=torch.float64),
        ty=torch.tensor(0.0, dtype=torch.float64),
        tz=torch.tensor(0.0, dtype=torch.float64),
        p_coeffs=torch.zeros(15, dtype=torch.float64),
        parallax=torch.tensor(0.0, dtype=torch.float64),
        pxY=torch.tensor(200.0, dtype=torch.float64),
        pxZ=torch.tensor(200.0, dtype=torch.float64),
    )
    if rho_d is not None:
        kw["rho_d"] = torch.tensor(rho_d, dtype=torch.float64)
    out = pixel_to_REta(Y, Z, **kw)
    assert torch.isfinite(out.R_px).all(), "non-positive RhoD NaN-ed R"
    assert torch.isfinite(out.eta_deg).all(), "non-positive RhoD NaN-ed eta"
    # And the values must be the plain undistorted geometry.
    expect = torch.sqrt((Y - BC[0]) ** 2 + (Z - BC[1]) ** 2)
    assert torch.allclose(out.R_px, expect, atol=1e-9)


def test_positive_rho_d_still_normalises_distortion():
    """The guard must not disable distortion when RhoD is legitimately set."""
    common = dict(
        Lsd=torch.tensor(1_000_000.0, dtype=torch.float64),
        BC_y=torch.tensor(100.0, dtype=torch.float64),
        BC_z=torch.tensor(100.0, dtype=torch.float64),
        tx=torch.tensor(0.0, dtype=torch.float64),
        ty=torch.tensor(0.0, dtype=torch.float64),
        tz=torch.tensor(0.0, dtype=torch.float64),
        parallax=torch.tensor(0.0, dtype=torch.float64),
        pxY=torch.tensor(200.0, dtype=torch.float64),
        pxZ=torch.tensor(200.0, dtype=torch.float64),
    )
    p = torch.zeros(15, dtype=torch.float64)
    p[0] = 0.05                                   # iso_R2
    Y = torch.tensor([[150.0]], dtype=torch.float64)
    Z = torch.tensor([[100.0]], dtype=torch.float64)
    r_off = pixel_to_REta(Y, Z, p_coeffs=torch.zeros(15, dtype=torch.float64),
                          rho_d=torch.tensor(20000.0, dtype=torch.float64),
                          **common).R_px
    r_on = pixel_to_REta(Y, Z, p_coeffs=p,
                         rho_d=torch.tensor(20000.0, dtype=torch.float64),
                         **common).R_px
    assert torch.isfinite(r_on).all()
    assert float((r_on - r_off).abs()) > 1e-6, "distortion was silently disabled"


def test_half_pixel_beam_centre_is_untouched():
    BC = torch.tensor([1022.5, 1022.0], dtype=torch.float64, requires_grad=True)
    out = _call(torch.tensor([1022.0], dtype=torch.float64),
                torch.tensor([1022.0], dtype=torch.float64), BC[0], BC[1])
    assert abs(float(out.R_px) - 0.5) < 1e-12
    g = torch.autograd.grad(out.R_px.sum(), BC)[0]
    assert abs(float(g[0]) - 1.0) < 1e-12
