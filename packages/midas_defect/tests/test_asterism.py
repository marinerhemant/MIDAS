import numpy as np
import pytest

from midas_defect.asterism import (
    asterism_anisotropy_per_grain,
    edge_fraction_per_grain,
    per_grain_asterism_tensor,
)


def _planted_grain_voxels(direction: np.ndarray, sigma_along: float, sigma_perp: float, n: int, rng):
    """Voxels around the origin with anisotropic Gaussian cloud along ``direction``."""
    direction = direction / np.linalg.norm(direction)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = tmp - np.dot(tmp, direction) * direction
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    a = rng.normal(0, sigma_along, size=n)
    b = rng.normal(0, sigma_perp, size=n)
    c = rng.normal(0, sigma_perp, size=n)
    return a[:, None] * direction + b[:, None] * e1 + c[:, None] * e2


# -- second_moment ----------------------------------------------------------

def test_per_grain_asterism_tensor_recovers_planted_anisotropy():
    rng = np.random.default_rng(0)
    direction = np.array([1.0, 0.0, 0.0])
    n_vox = 200
    dq = _planted_grain_voxels(direction, sigma_along=0.05, sigma_perp=0.01, n=n_vox, rng=rng)
    Pn = np.zeros((n_vox, 3))
    qs = dq + Pn  # so qs - Pn = dq
    vals = np.ones(n_vox)
    g_of_v = np.zeros(n_vox, dtype=int)
    mask = np.ones(n_vox, dtype=bool)

    M = per_grain_asterism_tensor(qs, vals, g_of_v, Pn, mask, n_grains=1)
    eigvals = np.linalg.eigvalsh(M[0])
    # Largest eigenvalue should be along the planted direction; lambda_max / lambda_min > 5.
    assert eigvals[-1] / eigvals[0] > 5.0
    # Largest eigenvalue ~ sigma_along^2; smallest ~ sigma_perp^2.
    assert 0.5 * 0.05**2 < eigvals[-1] < 1.5 * 0.05**2


def test_per_grain_asterism_tensor_nan_for_grain_below_min_voxels():
    rng = np.random.default_rng(1)
    dq = rng.normal(scale=0.01, size=(5, 3))
    qs = dq.copy()
    Pn = np.zeros((5, 3))
    vals = np.ones(5)
    g_of_v = np.zeros(5, dtype=int)
    mask = np.ones(5, dtype=bool)
    M = per_grain_asterism_tensor(
        qs, vals, g_of_v, Pn, mask, n_grains=1, min_voxels_per_grain=10
    )
    assert np.isnan(M[0]).all()


# -- direction / edge fraction ----------------------------------------------

def test_edge_fraction_pure_edge_is_one():
    # Planted: asterism direction parallel to mean q -> f_edge = 1.
    rng = np.random.default_rng(0)
    direction = np.array([0.0, 0.0, 1.0])
    n_vox = 500
    dq = _planted_grain_voxels(direction, sigma_along=0.05, sigma_perp=0.005, n=n_vox, rng=rng)
    Pn = np.tile(np.array([0.0, 0.0, 5.0]), (n_vox, 1))  # mean q along z
    qs = dq + Pn
    vals = np.ones(n_vox)
    g_of_v = np.zeros(n_vox, dtype=int)
    mask = np.ones(n_vox, dtype=bool)

    M = per_grain_asterism_tensor(qs, vals, g_of_v, Pn, mask, n_grains=1)
    f = edge_fraction_per_grain(M, qs, vals, g_of_v, mask, n_grains=1)
    assert f[0] > 0.9


def test_edge_fraction_pure_screw_is_zero():
    # Planted: asterism direction perpendicular to mean q -> f_edge ~ 0.
    rng = np.random.default_rng(0)
    direction = np.array([1.0, 0.0, 0.0])
    n_vox = 500
    dq = _planted_grain_voxels(direction, sigma_along=0.05, sigma_perp=0.005, n=n_vox, rng=rng)
    Pn = np.tile(np.array([0.0, 0.0, 5.0]), (n_vox, 1))  # mean q along z
    qs = dq + Pn
    vals = np.ones(n_vox)
    g_of_v = np.zeros(n_vox, dtype=int)
    mask = np.ones(n_vox, dtype=bool)
    M = per_grain_asterism_tensor(qs, vals, g_of_v, Pn, mask, n_grains=1)
    f = edge_fraction_per_grain(M, qs, vals, g_of_v, mask, n_grains=1)
    assert f[0] < 0.1


# -- eigenvalue spectrum ----------------------------------------------------

def test_asterism_anisotropy_recovers_eigenvalue_order_and_ratios():
    rng = np.random.default_rng(0)
    direction = np.array([0.0, 0.0, 1.0])
    n_vox = 500
    dq = _planted_grain_voxels(direction, sigma_along=0.05, sigma_perp=0.01, n=n_vox, rng=rng)
    Pn = np.zeros((n_vox, 3))
    qs = dq.copy()
    vals = np.ones(n_vox)
    g_of_v = np.zeros(n_vox, dtype=int)
    mask = np.ones(n_vox, dtype=bool)
    M = per_grain_asterism_tensor(qs, vals, g_of_v, Pn, mask, n_grains=1)
    out = asterism_anisotropy_per_grain(M)
    assert (out["eigvals_sorted"][0, :-1] <= out["eigvals_sorted"][0, 1:]).all()
    # ratio ~ (0.05/0.01)^2 = 25
    assert out["anisotropy_max_min"][0] > 10.0
    # mean_q_per_grain not provided -> radial/azimuthal NaN
    assert np.isnan(out["radial_eigenvalue"][0])


def test_asterism_anisotropy_radial_vs_azimuthal_with_mean_q():
    rng = np.random.default_rng(0)
    direction = np.array([0.0, 0.0, 1.0])
    n_vox = 500
    dq = _planted_grain_voxels(direction, sigma_along=0.05, sigma_perp=0.005, n=n_vox, rng=rng)
    Pn = np.tile(np.array([0.0, 0.0, 5.0]), (n_vox, 1))
    qs = dq + Pn
    vals = np.ones(n_vox)
    g_of_v = np.zeros(n_vox, dtype=int)
    mask = np.ones(n_vox, dtype=bool)
    M = per_grain_asterism_tensor(qs, vals, g_of_v, Pn, mask, n_grains=1)
    mean_q = np.array([[0.0, 0.0, 5.0]])
    out = asterism_anisotropy_per_grain(M, mean_q_per_grain=mean_q)
    # Edge-aligned asterism: radial eigenvalue should dominate the azimuthal.
    assert out["radial_eigenvalue"][0] > out["azimuthal_eigenvalue"][0] * 5
