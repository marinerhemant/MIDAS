"""Per-reflection corrections: each checked against a reference that does not share its derivation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_hkls import corrections as C


# ----------------------------------------------------------------- geometry

def test_scattered_direction_inverts_the_midas_eta_definition():
    tt = np.array([5.0, 20.0, 60.0, 120.0])
    eta = np.array([-170.0, -30.0, 0.0, 90.0])
    s = C.scattered_direction(tt, eta)
    assert np.allclose(np.linalg.norm(s, axis=-1), 1.0, atol=1e-14)
    assert np.allclose(np.degrees(np.arccos(s[..., 0])), tt, atol=1e-10)
    eta_back = np.degrees(np.arctan2(-s[..., 1], s[..., 2]))
    assert np.allclose(eta_back, eta, atol=1e-10)


def test_rotate_about_axis_is_right_handed():
    # +90 deg about z takes x to y, y to -x
    assert np.allclose(C.rotate_about_axis([1, 0, 0], [0, 0, 1], 90.0), [0, 1, 0], atol=1e-15)
    assert np.allclose(C.rotate_about_axis([0, 1, 0], [0, 0, 1], 90.0), [-1, 0, 0], atol=1e-15)
    v = np.array([0.3, -0.2, 0.9])
    ax = np.array([0.2, 0.1, 1.0])
    out = C.rotate_about_axis(v, ax, 37.0)
    assert np.isclose(np.linalg.norm(out), np.linalg.norm(v), atol=1e-14)
    k = ax / np.linalg.norm(ax)
    assert np.isclose(out @ k, v @ k, atol=1e-14)            # axis component preserved


# ------------------------------------------------------------- polarization

@pytest.mark.parametrize("f", [0.0, 0.3, 0.99, 1.0])
def test_polarization_matches_the_thomson_vector_mixture(f):
    """P = sum over the two transverse states of weight * (1 - (eps . s)^2)."""
    tt = np.linspace(2.0, 120.0, 40)[:, None]
    eta = np.linspace(-180.0, 180.0, 73)[None, :]
    s = C.scattered_direction(tt, eta)
    eps_h = np.array([0.0, 1.0, 0.0])                         # horizontal, perpendicular to the beam
    eps_v = np.array([0.0, 0.0, 1.0])
    ref = (0.5 * (1 + f) * (1 - (s @ eps_h) ** 2)
           + 0.5 * (1 - f) * (1 - (s @ eps_v) ** 2))
    got = C.polarization_factor(tt, eta, pol_fraction=f, pol_plane_eta_deg=90.0)
    assert np.allclose(got, ref, atol=1e-14)


def test_polarization_limits():
    tt = 30.0
    # fully polarized: horizontal scattering plane (eta=90) gives cos^2 2theta, vertical (eta=0) gives 1
    assert C.polarization_factor(tt, 90.0, pol_fraction=1.0) == pytest.approx(math.cos(math.radians(tt)) ** 2)
    assert C.polarization_factor(tt, 0.0, pol_fraction=1.0) == pytest.approx(1.0)
    # unpolarized: no preferred azimuth
    etas = np.linspace(-180, 180, 37)
    p = C.polarization_factor(tt, etas, pol_fraction=0.0)
    assert np.allclose(p, p[0], atol=1e-15)


@pytest.mark.parametrize("f", [0.0, 0.5, 0.99, 1.0])
def test_polarization_ring_average_is_independent_of_f(f):
    tt = 40.0
    etas = np.linspace(0, 360, 20000, endpoint=False)
    avg = C.polarization_factor(tt, etas, pol_fraction=f).mean()
    assert avg == pytest.approx(0.5 * (1 + math.cos(math.radians(tt)) ** 2), rel=1e-12)


@pytest.mark.parametrize("plane", [90.0, 0.0, 33.0])
@pytest.mark.parametrize("f", [0.99, 0.5, 1.0])
def test_polarization_equals_integrate_v2_mixture(f, plane):
    torch = pytest.importorskip("torch")
    iv2 = pytest.importorskip("midas_integrate_v2.corrections.intensity")
    Lsd, px = 1_000_000.0, 200.0
    tt = np.linspace(3.0, 50.0, 25)
    eta = np.linspace(-175.0, 175.0, 31)
    TT, ETA = np.meshgrid(tt, eta, indexing="ij")
    R_px = np.tan(np.radians(TT)) * Lsd / px
    ref = iv2.polarization_factor(
        torch.tensor(R_px), torch.tensor(ETA), Lsd=torch.tensor(Lsd, dtype=torch.float64),
        px=torch.tensor(px, dtype=torch.float64), pol_fraction=torch.tensor(f, dtype=torch.float64),
        pol_plane_eta_deg=torch.tensor(plane, dtype=torch.float64), model="mixture").numpy()
    got = C.polarization_factor(TT, ETA, pol_fraction=f, pol_plane_eta_deg=plane)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-14)


# ------------------------------------------------------------------ Lorentz

def test_rotation_lorentz_vertical_axis_closed_form():
    tt = np.array([8.0, 20.0, 45.0])[:, None]
    eta = np.array([-120.0, -45.0, 20.0, 80.0, 150.0])[None, :]
    s = C.scattered_direction(tt, eta)
    ref = 1.0 / (np.sin(np.radians(tt)) * np.abs(np.sin(np.radians(eta))))
    assert np.allclose(C.rotation_lorentz(s), ref, rtol=1e-12)
    # the rotation sense does not matter
    assert np.allclose(C.rotation_lorentz(s, axis=[0, 0, -1]), ref, rtol=1e-12)


def _ewald_roots(q0, axis, n=3600):
    """omega roots of f(omega) = |s0 + q(omega)| - 1 (k = 1), by scan + bisection."""
    s0 = C.BEAM_DIRECTION

    def f(om):
        return np.linalg.norm(s0 + C.rotate_about_axis(q0, axis, om)) - 1.0

    grid = np.linspace(0.0, 360.0, n + 1)
    vals = np.array([f(g) for g in grid])
    roots = []
    for i in range(n):
        if vals[i] == 0.0 or vals[i] * vals[i + 1] < 0.0:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(lo) * f(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            roots.append(0.5 * (lo + hi))
    return roots, f


@pytest.mark.parametrize("axis", [[0, 0, 1], [0.12, -0.07, 1.0], [0.0, 0.3, 1.0]])
def test_rotation_lorentz_matches_finite_difference_of_the_ewald_function(axis):
    """L * |d f / d omega| must be 1 (k = 1) at every Ewald crossing: tilted axes included."""
    rng = np.random.default_rng(7)
    n_checked = 0
    for _ in range(12):
        v = rng.normal(size=3)
        q0 = 0.9 * v / np.linalg.norm(v)
        roots, f = _ewald_roots(q0, axis)
        for om in roots:
            s = C.BEAM_DIRECTION + C.rotate_about_axis(q0, axis, om)
            L = float(C.rotation_lorentz(s, axis=axis, min_abs=0.0))
            h = 1e-5
            dfdw = (f(om + h) - f(om - h)) / (2 * math.radians(h))      # per radian
            if abs(dfdw) < 1e-2:                                        # skip near-tangent crossings
                continue
            assert L * abs(dfdw) == pytest.approx(1.0, rel=1e-5)
            n_checked += 1
    assert n_checked >= 20


def test_rotation_lorentz_returns_nan_near_the_axis_not_a_huge_number():
    s = C.scattered_direction(np.array([20.0, 20.0]), np.array([0.0, 90.0]))   # eta=0: ray in the plane containing the axis
    L = C.rotation_lorentz(s, min_abs=1e-3)
    assert np.isnan(L[0]) and np.isfinite(L[1])


# --------------------------------------------------------------- absorption

def test_diamond_mu_is_positive_and_falls_with_energy():
    lo, hi = C.diamond_mu_per_cm(1.0), C.diamond_mu_per_cm(0.4)
    assert lo > hi > 0.0


def test_diamond_transmission_normal_incidence_and_oblique_exit():
    mu, t = 1.0, 2.5                                  # cm^-1, mm
    c = np.array([1.0, 0.0, 0.0])                     # DAC axis along the beam
    s0 = C.BEAM_DIRECTION
    # exit 30 deg off the axis, same hemisphere: downstream anvil
    s = C.scattered_direction(30.0, 90.0)
    got = C.diamond_anvil_transmission(s, c, mu_per_cm=mu, t_up_mm=t, t_down_mm=2.0)
    expect = math.exp(-(mu / 10.0) * (t / 1.0 + 2.0 / math.cos(math.radians(30.0))))
    assert float(got) == pytest.approx(expect, rel=1e-13)
    # back-reflection (2theta = 150 deg) leaves through the UPSTREAM anvil
    sb = C.scattered_direction(150.0, 10.0)
    gb = C.diamond_anvil_transmission(sb, c, mu_per_cm=mu, t_up_mm=t, t_down_mm=2.0)
    eb = math.exp(-(mu / 10.0) * (t / 1.0 + t / abs(math.cos(math.radians(150.0)))))
    assert float(gb) == pytest.approx(eb, rel=1e-13)


def test_diamond_transmission_blocks_rays_outside_the_opening():
    c = np.array([1.0, 0.0, 0.0])
    inside = C.scattered_direction(34.0, 0.0)
    outside = C.scattered_direction(36.0, 0.0)
    kw = dict(mu_per_cm=1.0, t_up_mm=2.5, half_opening_deg=35.0)
    assert float(C.diamond_anvil_transmission(inside, c, **kw)) > 0.0
    assert float(C.diamond_anvil_transmission(outside, c, **kw)) == 0.0
    # tilting the cell by omega moves the incident ray off the axis too
    c_tilt = C.rotate_about_axis(c, [0, 0, 1], 36.0)
    assert float(C.diamond_anvil_transmission(C.scattered_direction(1.0, 0.0), c_tilt, **kw)) == 0.0


@pytest.mark.parametrize("laue", [True, False])
@pytest.mark.parametrize("ci,co", [(1.0, 1.0), (0.9, 0.5), (0.4, 0.8)])
@pytest.mark.parametrize("mu_h", [0.0, 0.05, 1.5])
def test_slab_self_absorption_matches_depth_quadrature(laue, ci, co, mu_h):
    h_um = 10.0
    mu = mu_h / (h_um * 1e-4)                         # cm^-1 such that mu*h = mu_h
    z = (np.arange(200000) + 0.5) / 200000
    path = z / ci + ((1 - z) / co if laue else z / co)
    ref = np.exp(-mu_h * path).mean()
    got = C.slab_self_absorption(ci, co, mu_per_cm=mu, thickness_um=h_um, laue=laue)
    assert float(got) == pytest.approx(ref, rel=1e-6, abs=1e-12)


def test_polarization_factor_accepts_torch_tensors_and_is_differentiable():
    torch = pytest.importorskip("torch")
    tt = np.array([5.0, 20.0, 60.0, 120.0]); eta = np.array([0.0, 37.0, 90.0, 215.0])
    for f in (0.0, 0.5, 0.99, 1.0):
        want = C.polarization_factor(tt, eta, pol_fraction=f, pol_plane_eta_deg=90.0)
        got = C.polarization_factor(torch.tensor(tt), torch.tensor(eta), pol_fraction=f, pol_plane_eta_deg=90.0)
        assert isinstance(got, torch.Tensor) and np.allclose(got.numpy(), want, rtol=1e-13)
    x = torch.tensor(tt, dtype=torch.float64, requires_grad=True)
    C.polarization_factor(x, torch.tensor(eta), pol_fraction=0.9).sum().backward()
    assert torch.isfinite(x.grad).all() and float(x.grad.abs().sum()) > 0
    assert isinstance(C.polarization_factor(30.0, 45.0), (float, np.floating, np.ndarray))          # the numpy path is unchanged
