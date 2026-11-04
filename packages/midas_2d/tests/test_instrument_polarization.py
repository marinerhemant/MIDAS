"""solid_angle_polarization: the polarisation factor is evaluated per pixel at the azimuth of the scattered ray (it used to be the single-direction value everywhere)."""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from midas_2d.instrument import solid_angle_polarization          # noqa: E402

LAM = 0.5
K0 = 2.0 * math.pi / LAM


def _q(two_theta_deg, phi_deg):
    """The q vector (1/A) that scatters by 2theta at azimuth phi about the beam z, phi measured from the detector x axis."""
    tt, ph = math.radians(two_theta_deg), math.radians(phi_deg)
    kf = K0 * np.array([math.sin(tt) * math.cos(ph), math.sin(tt) * math.sin(ph), math.cos(tt)])
    return kf - np.array([0.0, 0.0, K0])


def _p(two_theta_deg, phi_deg, K, **kw):
    """Polarisation part only: divide out the cos^3 solid-angle factor."""
    q = torch.tensor(_q(two_theta_deg, phi_deg), dtype=torch.float64)
    return float(solid_angle_polarization(q, wavelength_A=LAM, polarization=K, **kw)) / math.cos(math.radians(two_theta_deg)) ** 3


def test_default_is_unpolarised_and_azimuth_independent():
    for tt in (10.0, 60.0, 80.0):
        want = 0.5 * (1.0 + math.cos(math.radians(tt)) ** 2)
        for phi in (0.0, 37.0, 90.0, 215.0):
            assert _p(tt, phi, 0.5) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("tt", [20.0, 60.0, 80.0])
def test_fully_polarised_beam_is_cos2_in_plane_and_one_perpendicular(tt):
    c2 = math.cos(math.radians(tt)) ** 2
    assert _p(tt, 0.0, 1.0) == pytest.approx(c2, abs=1e-12)          # scattering plane contains the polarisation vector (the old value everywhere)
    assert _p(tt, 180.0, 1.0) == pytest.approx(c2, abs=1e-12)
    assert _p(tt, 90.0, 1.0) == pytest.approx(1.0, abs=1e-12)        # perpendicular to it: sigma, no suppression
    assert _p(tt, 270.0, 1.0) == pytest.approx(1.0, abs=1e-12)


def test_the_azimuthal_average_does_not_depend_on_the_polarisation():
    tt = 60.0
    phis = np.linspace(0.0, 360.0, 3600, endpoint=False)
    want = 0.5 * (1.0 + math.cos(math.radians(tt)) ** 2)             # 0.625
    for K in (0.5, 0.7, 1.0):
        avg = np.mean([_p(tt, ph, K) for ph in phis[::10]])
        assert avg == pytest.approx(want, abs=1e-9)


def test_polarisation_axis_rotates_the_pattern():
    assert _p(60.0, 90.0, 1.0, polarization_axis=(0.0, 1.0, 0.0)) == pytest.approx(0.25, abs=1e-12)    # plane now contains y
    assert _p(60.0, 0.0, 1.0, polarization_axis=(0.0, 1.0, 0.0)) == pytest.approx(1.0, abs=1e-12)
    assert _p(60.0, 45.0, 1.0, polarization_axis=(1.0, 1.0, 0.0)) == pytest.approx(0.25, abs=1e-12)    # a diagonal axis, ray along it


def test_partial_polarisation_interpolates_linearly_in_k():
    # K = 0.75 is f = 0.5: P = 0.5 (1 + cos^2 - 0.5 cos(2 phi) sin^2); at 2theta = 60 deg, phi = 0: 0.5 (1 + 0.25 - 0.5 * 0.75) = 0.4375
    assert _p(60.0, 0.0, 0.75) == pytest.approx(0.4375, abs=1e-12)


def _old(q, K):
    """The function as it was before 2026-10-21 (single-direction polarisation), for the gradient comparison at K = 0.5."""
    k = 2.0 * math.pi / LAM
    qmag = torch.linalg.vector_norm(q, dim=-1)
    two_th = 2.0 * torch.asin(torch.clamp(qmag / (2.0 * k), max=1.0))
    c2 = torch.cos(two_th)
    return ((1.0 - K) + K * c2 * c2) * torch.cos(two_th).clamp(min=1e-6) ** 3


def test_the_value_is_the_old_value_at_the_default_and_the_gradient_too():
    """K = 0.5 is azimuth independent, so value AND dQ-gradient must equal the old function's (a detached polarisation part changed the gradient by 23 % on a BCDI grid)."""
    rng = np.random.default_rng(1)
    q = torch.tensor(np.array([_q(tt, ph) for tt, ph in zip(rng.uniform(5, 70, 40), rng.uniform(0, 360, 40))]), dtype=torch.float64, requires_grad=True)
    new = solid_angle_polarization(q, wavelength_A=LAM)
    old = _old(q, 0.5)
    assert torch.allclose(new, old, rtol=1e-13, atol=1e-15)
    gn, = torch.autograd.grad(new.sum(), q)
    go, = torch.autograd.grad(old.sum(), q)
    assert torch.allclose(gn, go, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("K", [0.5, 0.8, 1.0])
def test_autograd_matches_finite_differences_for_every_k_and_axis(K):
    rng = np.random.default_rng(2)
    q0 = np.array([_q(tt, ph) for tt, ph in zip(rng.uniform(10, 60, 6), rng.uniform(0, 360, 6))])
    for axis in ((1.0, 0.0, 0.0), (0.3, -2.0, 5.0)):
        q = torch.tensor(q0, dtype=torch.float64, requires_grad=True)
        out = solid_angle_polarization(q, wavelength_A=LAM, polarization=K, polarization_axis=axis)
        g, = torch.autograd.grad(out.sum(), q)
        fd = np.zeros_like(q0)
        h = 1e-6
        for i in range(q0.shape[0]):
            for c in range(3):
                qp, qm = q0.copy(), q0.copy(); qp[i, c] += h; qm[i, c] -= h
                f = lambda a: float(solid_angle_polarization(torch.tensor(a, dtype=torch.float64), wavelength_A=LAM, polarization=K, polarization_axis=axis)[i])
                fd[i, c] = (f(qp) - f(qm)) / (2 * h)
        assert np.allclose(g.numpy(), fd, rtol=1e-6, atol=1e-9), (K, axis, np.abs(g.numpy() - fd).max())


def test_shapes_dtype_zero_q_and_the_clamp():
    q = torch.tensor(np.array([[_q(30.0, 10.0), _q(40.0, 200.0)]]), dtype=torch.float32)             # (1, 2, 3)
    out = solid_angle_polarization(q, wavelength_A=LAM, polarization=0.9)
    assert out.shape == (1, 2) and out.dtype == torch.float32
    assert float(solid_angle_polarization(torch.zeros(3, dtype=torch.float64), wavelength_A=LAM, polarization=1.0)) == pytest.approx(1.0, abs=1e-15)
    z = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    solid_angle_polarization(z, wavelength_A=LAM, polarization=1.0).backward()
    assert torch.isfinite(z.grad).all()                                                              # q on the beam axis: the azimuth is undefined, the gradient must not be NaN
    back = solid_angle_polarization(torch.tensor(_q(100.0, 0.0), dtype=torch.float64), wavelength_A=LAM, polarization=1.0)
    assert torch.isfinite(back) and float(back) < 1e-12                                              # a flat detector cannot see 2theta >= 90 deg: the cos^3 factor is clamped


def test_bad_arguments_are_refused():
    q = torch.tensor(_q(30.0, 0.0), dtype=torch.float64)
    with pytest.raises(ValueError):
        solid_angle_polarization(q, wavelength_A=LAM, polarization=1.2)
    with pytest.raises(ValueError):
        solid_angle_polarization(q, wavelength_A=LAM, polarization_axis=(0.0, 0.0, 1.0))
