"""The omega solver: accuracy and gradients where the old form failed.

The solver satisfies  -Gx cos(w) + Gy sin(w) = v,  so the residual of THAT
identity is the arbiter -- not agreement with the previous implementation,
which was itself biased. The closed form

    rho = |(Gx, Gy)|,  phi = atan2(Gy, -Gx),  w = phi +- acos(v/rho)

replaced a Gy^2-divided quadratic whose ``+ 1e-7`` denominators perturbed every
spot, and whose ``safe_arccos`` froze both value and gradient in a +-0.0256 deg
band around omega = 0.

C parity is covered separately by tests/test_c_comparison.py (FF and NF); note
its tolerance is 0.5 deg, which confirms no regression but cannot resolve the
1e-2 deg effects tested here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry, solve_omega


@pytest.fixture(autouse=True)
def _float64_default():
    """float64 for this module only.

    Setting the default dtype at import time leaks into every test module
    imported afterwards -- it broke 49 tests in test_forward.py, which expects
    the float32 default.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _solve(Gx, Gy, v):
    """The SHIPPED solver -- not a copy of it.

    An earlier draft of this file reimplemented the closed form here, which
    tested the copy rather than the code and silently dropped the tangency
    guard. ``solve_omega`` is module-level precisely so the test can call it.
    """
    wp, wn, _ = solve_omega(Gx, Gy, v)
    return wp, wn


def _residual(w, Gx, Gy, v):
    return torch.abs(-Gx * torch.cos(w) + Gy * torch.sin(w) - v)


def _coeffs(n, gy_scale, seed):
    rng = np.random.default_rng(seed)
    Gx = torch.tensor(rng.normal(0, 0.35, n))
    Gy = torch.tensor(rng.uniform(-gy_scale, gy_scale, n)) if gy_scale < 0.1 \
        else torch.tensor(rng.normal(0, gy_scale, n))
    v = torch.tensor(rng.normal(0, 0.20, n))
    keep = (v * v) <= (Gx * Gx + Gy * Gy)
    return Gx[keep], Gy[keep], v[keep]


@pytest.mark.parametrize("gy_scale,label", [
    (0.35, "ordinary"), (3e-4, "small |Gy|"), (1e-8, "tiny |Gy|"),
])
def test_constraint_residual_is_at_machine_precision(gy_scale, label):
    """The old form gave median 1.2e-07 (ordinary) to 3.0e-04 (tiny |Gy|)."""
    Gx, Gy, v = _coeffs(50_000, gy_scale, seed=0)
    wp, wn = _solve(Gx, Gy, v)
    r = torch.minimum(_residual(wp, Gx, Gy, v), _residual(wn, Gx, Gy, v))
    assert float(r.median()) < 1e-15, f"{label}: median residual {float(r.median()):.3e}"
    assert float(r.max()) < 1e-12, f"{label}: max residual {float(r.max()):.3e}"


def test_gy_exactly_zero_needs_no_special_branch():
    """The old code carried a separate |Gy| < 1e-12 branch. Gy = 0 is now
    just phi = atan2(0, -Gx), which is 0 or pi. Nothing special about it."""
    rng = np.random.default_rng(1)
    Gx = torch.tensor(rng.normal(0, 0.35, 20_000))
    Gy = torch.zeros_like(Gx)
    v = torch.tensor(rng.normal(0, 0.20, 20_000))
    keep = (v * v) <= (Gx * Gx)
    Gx, Gy, v = Gx[keep], Gy[keep], v[keep]
    wp, wn = _solve(Gx, Gy, v)
    r = torch.minimum(_residual(wp, Gx, Gy, v), _residual(wn, Gx, Gy, v))
    assert float(r.max()) < 1e-12


def test_no_frozen_band_near_omega_zero():
    """Recover a KNOWN omega inside the band the old clamp pinned.

    Old: median error 1.56e-02 deg, i.e. the answer was the band edge rather
    than the spot's actual omega.
    """
    rng = np.random.default_rng(2)
    n = 50_000
    Gx = torch.tensor(rng.normal(0, 0.35, n))
    Gy = torch.tensor(rng.normal(0, 0.35, n))
    w_true = torch.tensor(rng.uniform(-0.02, 0.02, n)) * math.pi / 180.0
    v = -Gx * torch.cos(w_true) + Gy * torch.sin(w_true)   # exact by construction
    wp, wn = _solve(Gx, Gy, v)
    # Either branch may carry the true root; take the nearer.
    err = torch.minimum((wp - w_true).abs(), (wn - w_true).abs())
    err_deg = err * 180.0 / math.pi
    assert float(err_deg.median()) < 1e-10, (
        f"median error {float(err_deg.median()):.3e} deg "
        f"(old form: 1.56e-02 deg -- the frozen band)")
    assert float(np.percentile(err_deg.numpy(), 99.9)) < 1e-6


def test_no_epsilon_bias_at_moderate_gy():
    """``y2 + 1e-7`` perturbed EVERY coefficient, not only small-|Gy| spots.

    At |Gy| ~ 0.35 the old relative perturbation was ~1e-6, which is what put
    a ~1e-7 residual on ordinary spots. Nothing here may depend on a tolerance
    that loose.
    """
    Gx, Gy, v = _coeffs(20_000, 0.35, seed=3)
    wp, wn = _solve(Gx, Gy, v)
    r = torch.minimum(_residual(wp, Gx, Gy, v), _residual(wn, Gx, Gy, v))
    assert float((r > 1e-14).double().mean()) < 1e-3


# ------------------------------------------------------------- gradients

def test_gradient_finite_and_fd_matching_at_omega_zero():
    """omega = 0 is an ORDINARY spot. Its gradient must be right, not merely
    finite -- the old clamp made it exactly 0."""
    GX, GY = 0.3, 0.4
    # w = 0  =>  -Gx cos 0 + Gy sin 0 = -Gx, so v = -Gx puts a root at omega=0.
    # It lands on the MINUS branch here (phi + acos(v/rho) is the other root).
    v = torch.tensor(-GX)

    def f(gx, gy):
        return _solve(gx, gy, v)[1]

    Gx = torch.tensor(GX, requires_grad=True)
    Gy = torch.tensor(GY, requires_grad=True)
    out = f(Gx, Gy)
    assert abs(float(out.detach())) < 1e-12, "setup should put omega at 0"
    g = torch.autograd.grad(out, [Gx, Gy])
    assert all(torch.isfinite(x).all() for x in g)
    h = 1e-7
    for i, (val, gi) in enumerate(zip((GX, GY), g)):
        a = [torch.tensor(GX), torch.tensor(GY)]
        a[i] = torch.tensor(val + h); up = float(f(*a))
        a[i] = torch.tensor(val - h); dn = float(f(*a))
        fd = (up - dn) / (2 * h)
        assert abs(float(gi) - fd) < 1e-4, f"param {i}: {float(gi)} vs FD {fd}"
    # The old clamp made this exactly 0. At least one component must be alive.
    assert max(abs(float(x)) for x in g) > 1e-6, "gradient is dead at omega = 0"


def test_one_tangential_spot_does_not_poison_the_batch():
    """acos'(+-1) is infinite at exact tangency; torch sums gradients, so an
    unguarded singular spot would NaN every other spot's gradient."""
    Gx = torch.tensor([0.3, 0.3, 0.3], requires_grad=True)
    Gy = torch.tensor([0.4, 0.4, 0.4], requires_grad=True)
    v = torch.tensor([0.5, 0.1, -0.2])           # v[0] = rho exactly: tangency
    wp, _ = _solve(Gx, Gy, v)
    g = torch.autograd.grad(wp.sum(), [Gx, Gy])
    assert all(torch.isfinite(x).all() for x in g), "tangency poisoned the batch"


# ------------------------------------------------------- end to end

def _ff_model():
    geom = HEDMGeometry(
        Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=0.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0, wavelength=0.295,
    )
    a, wl = 2.87, 0.295
    hkls_int = torch.tensor([[1, 1, 0], [2, 0, 0], [2, 1, 1], [2, 2, 0]])
    hkls_cart = (torch.eye(3) / a @ hkls_int.double().T).T
    thetas = torch.asin(wl / (2.0 * (1.0 / torch.norm(hkls_cart, dim=-1))))
    return HEDMForwardModel(hkls=hkls_cart, thetas=thetas, geometry=geom,
                            hkls_int=hkls_int.double(),
                            device=torch.device("cpu"))


def test_model_forward_is_finite_and_differentiable():
    model = _ff_model()
    eul = torch.tensor([[0.4, 0.6, 0.8]], requires_grad=True)
    pos = torch.zeros(1, 3)
    out = model(eul.unsqueeze(0), pos.unsqueeze(0))
    assert torch.isfinite(out.omega).all()
    assert torch.isfinite(out.eta).all()
    g = torch.autograd.grad(
        (out.omega * out.valid).sum(), eul, allow_unused=True)[0]
    assert g is not None and torch.isfinite(g).all()


def test_model_omega_within_180():
    model = _ff_model()
    rng = np.random.default_rng(0)
    eul = torch.tensor(rng.uniform(0, 2 * np.pi, size=(64, 3)))
    out = model(eul.unsqueeze(0), torch.zeros(64, 3).unsqueeze(0))
    om = out.omega[out.valid > 0.5]
    assert float(om.abs().max()) <= 180.0 + 1e-9
