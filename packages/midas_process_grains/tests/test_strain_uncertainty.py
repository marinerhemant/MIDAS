"""``compute.strain_uncertainty`` — errors on eFab and eKen.

The physics check is :func:`test_kenesei_exposes_the_weak_xx_direction`: in FF
geometry the beam runs along x, so ``g_x`` is ~0.07 against ~0.7 for y and z
and eps_xx is barely constrained. A UQ that reported eps_xx as well-determined
would be wrong, and that is exactly the failure this catches.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_process_grains.compute.strain_uncertainty import (
    VOIGT_ORDER,
    fable_strain_covariance,
    kenesei_strain_covariance,
)

CUBIC = np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0])


def _ff_like_g(n=200, seed=0):
    """g-vectors with an FF-shaped distribution: tiny g_x, large g_y/g_z."""
    rng = np.random.default_rng(seed)
    eta = rng.uniform(-np.pi, np.pi, n)
    gx = rng.normal(0.0, 0.07, n)
    gy, gz = 0.7 * np.cos(eta), 0.7 * np.sin(eta)
    g = np.column_stack([gx, gy, gz])
    return g / np.linalg.norm(g, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
#  eKen — from the least-squares normal equations
# ---------------------------------------------------------------------------

def test_kenesei_exposes_the_weak_xx_direction():
    """sigma(eps_xx) must dominate: the beam is along x, so g_x is small."""
    g = _ff_like_g()
    d0 = np.full(len(g), 2.0)
    rng = np.random.default_rng(1)
    ds = d0 * (1.0 + rng.normal(0.0, 1e-4, len(g)))
    res = kenesei_strain_covariance(g, ds, d0)
    s = dict(zip(VOIGT_ORDER, res.sigma_voigt))
    assert s["xx"] > 5 * s["yy"], (
        f"eps_xx should be far worse determined than eps_yy; got {s}")
    assert s["xx"] > 5 * s["zz"]


def test_kenesei_sigma_shrinks_with_more_spots():
    """More observations, tighter posterior — roughly as 1/sqrt(n)."""
    d0v = 2.0
    out = {}
    for n in (40, 640):
        g = _ff_like_g(n, seed=3)
        d0 = np.full(n, d0v)
        rng = np.random.default_rng(4)
        ds = d0 * (1.0 + rng.normal(0.0, 1e-4, n))
        out[n] = kenesei_strain_covariance(g, ds, d0).sigma_voigt[1]
    ratio = out[40] / out[640]
    assert 2.0 < ratio < 8.0, f"expected ~4x (sqrt(16)); got {ratio:.2f}"


def test_kenesei_sigma_scales_with_the_noise_level():
    """s^2 is estimated from the residual, so 10x noise -> ~10x sigma."""
    g = _ff_like_g(300, seed=5)
    d0 = np.full(len(g), 2.0)
    sig = {}
    for noise in (1e-5, 1e-4):
        rng = np.random.default_rng(6)
        ds = d0 * (1.0 + rng.normal(0.0, noise, len(g)))
        sig[noise] = kenesei_strain_covariance(g, ds, d0).sigma_voigt[1]
    assert 5.0 < sig[1e-4] / sig[1e-5] < 20.0


def test_kenesei_needs_more_spots_than_parameters():
    g = _ff_like_g(6)
    d0 = np.full(6, 2.0)
    with pytest.raises(ValueError, match="need >= 7 spots"):
        kenesei_strain_covariance(g, d0, d0)


def test_ridge_uses_the_sandwich_not_the_bare_inverse():
    """The regularised estimator is biased; (G'G+aI)^-1 alone understates it.

    The sandwich (G'G+aI)^-1 G'G (G'G+aI)^-1 is strictly smaller than the bare
    inverse, so the two must not coincide once alpha bites.
    """
    g = _ff_like_g(200, seed=7)
    d0 = np.full(len(g), 2.0)
    rng = np.random.default_rng(8)
    ds = d0 * (1.0 + rng.normal(0.0, 1e-4, len(g)))
    plain = kenesei_strain_covariance(g, ds, d0, regularization=0.0)
    ridge = kenesei_strain_covariance(g, ds, d0, regularization=1e-2)
    assert ridge.sigma_voigt[0] < plain.sigma_voigt[0], (
        "ridge should tighten the weak xx direction")
    assert "sandwich" in ridge.method


def test_kenesei_hydrostatic_uses_the_covariance():
    g = _ff_like_g(200, seed=9)
    d0 = np.full(len(g), 2.0)
    rng = np.random.default_rng(10)
    ds = d0 * (1.0 + rng.normal(0.0, 1e-4, len(g)))
    r = kenesei_strain_covariance(g, ds, d0)
    J = np.zeros(6); J[:3] = 1/3
    expect = float(np.sqrt(J @ r.cov_voigt @ J))
    independent = float(np.sqrt(((J**2) * np.diag(r.cov_voigt)).sum()))
    assert r.sigma_hydrostatic == pytest.approx(expect, rel=1e-10)
    assert r.sigma_hydrostatic != pytest.approx(independent, rel=1e-6), (
        "the normal components are correlated; the independent sum must differ")


def test_weights_change_the_answer():
    g = _ff_like_g(120, seed=11)
    d0 = np.full(len(g), 2.0)
    rng = np.random.default_rng(12)
    ds = d0 * (1.0 + rng.normal(0.0, 1e-4, len(g)))
    a = kenesei_strain_covariance(g, ds, d0).sigma_voigt
    w = np.ones(len(g)); w[: len(g)//2] = 0.1
    b = kenesei_strain_covariance(g, ds, d0, weights=w).sigma_voigt
    assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
#  eFab — propagated from the lattice covariance
# ---------------------------------------------------------------------------

def test_fable_zero_lattice_covariance_gives_zero_strain_sigma():
    r = fable_strain_covariance(CUBIC, CUBIC, np.zeros((6, 6)))
    assert np.allclose(r.sigma_voigt, 0.0)
    assert r.sigma_hydrostatic == pytest.approx(0.0)


def test_fable_sigma_scales_linearly_with_lattice_sigma():
    """eps is locally linear in latc, so 2x the lattice sigma is 2x the strain
    sigma (covariance scales by 4)."""
    C = np.diag([1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10])
    r1 = fable_strain_covariance(CUBIC, CUBIC, C)
    r2 = fable_strain_covariance(CUBIC, CUBIC, 4.0 * C)
    np.testing.assert_allclose(r2.sigma_voigt, 2.0 * r1.sigma_voigt, rtol=1e-8)


def test_fable_isotropic_lattice_error_maps_to_hydrostatic_strain():
    """A common scale error on a, b, c is a pure hydrostatic strain.

    da/a = 1e-4 on each, fully correlated, must give sigma(eps_hydro) ~ 1e-4
    and leave the shear components untouched.
    """
    a = 3.6
    s = 1e-4 * a                       # absolute sigma on each length
    C = np.zeros((6, 6))
    C[:3, :3] = s ** 2                 # perfectly correlated a, b, c
    r = fable_strain_covariance([a, a, a, 90, 90, 90], [a, a, a, 90, 90, 90], C)
    assert r.sigma_hydrostatic == pytest.approx(1e-4, rel=0.02)
    for k, nm in enumerate(VOIGT_ORDER):
        if nm in ("xy", "xz", "yz"):
            assert r.sigma_voigt[k] < 1e-9, f"{nm} should be untouched"


def test_fable_rejects_bad_covariance_shape():
    with pytest.raises(ValueError, match=r"cov_latc must be"):
        fable_strain_covariance(CUBIC, CUBIC, np.zeros((3, 3)))


def test_as_dict_labels_all_six_components():
    r = fable_strain_covariance(CUBIC, CUBIC, np.eye(6) * 1e-10)
    d = r.as_dict("eFab")
    for nm in VOIGT_ORDER:
        assert f"sigma_eFab_{nm}" in d
    assert "sigma_eFab_hydro" in d
