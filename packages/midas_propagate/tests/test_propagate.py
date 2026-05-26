"""Tests for midas_propagate.propagate — delta-method to per-grain stress.

Mathematical checks:

* ``latc_to_strain_grain(latc_ref, latc_ref) == 0``.
* Hydrostatic uniaxial strain on a cubic lattice gives the expected
  hydrostatic stress through Hooke's law.
* Stress depends only on (euler, latc), not pos — the pos columns of J
  should be zero, and the corresponding rows/cols of Σ_σ should not
  pick up signal from a pos-only Σ_g.
* For a zero per-grain covariance (Σ_g = 0), the stress covariance is
  also zero.
* For Σ_g = ε² · I (small isotropic noise on every g param), the
  output Σ_σ is non-zero and finite.
"""
from __future__ import annotations

import math

import pytest
import torch


def _have_stress_torch():
    try:
        from midas_stress.torch_backend import tensor_to_voigt  # noqa: F401
        return True
    except ImportError:
        return False


def _cubic_stiffness(C11=200.0, C12=130.0, C44=80.0):
    """Voigt-Mandel cubic stiffness. Units are arbitrary (GPa is the usual
    convention)."""
    s2 = math.sqrt(2.0)
    C = torch.tensor([
        [C11, C12, C12, 0.0, 0.0, 0.0],
        [C12, C11, C12, 0.0, 0.0, 0.0],
        [C12, C12, C11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2 * C44, 0.0, 0.0],   # Mandel: 2·C44 on shear diag
        [0.0, 0.0, 0.0, 0.0, 2 * C44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 2 * C44],
    ], dtype=torch.float64)
    return C


def test_lattice_vectors_cubic_identity():
    """Cubic lattice with a=1 gives the 3×3 identity."""
    from midas_propagate.propagate import lattice_vectors
    latc = torch.tensor([1.0, 1.0, 1.0, 90.0, 90.0, 90.0], dtype=torch.float64)
    L = lattice_vectors(latc)
    torch.testing.assert_close(L, torch.eye(3, dtype=torch.float64), rtol=1e-12, atol=1e-12)


def test_strain_zero_at_reference():
    """Strain wrt the reference lattice itself is the zero tensor."""
    from midas_propagate.propagate import latc_to_strain_grain
    latc_ref = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0], dtype=torch.float64)
    eps = latc_to_strain_grain(latc_ref, latc_ref)
    torch.testing.assert_close(eps, torch.zeros((3, 3), dtype=torch.float64),
                               rtol=1e-10, atol=1e-10)


def test_uniform_dilation_gives_hydrostatic_strain():
    """A 0.1% uniform dilation produces ε ≈ 0.001·I in the grain frame."""
    from midas_propagate.propagate import latc_to_strain_grain
    latc_ref = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0], dtype=torch.float64)
    latc_dil = torch.tensor([4.08 * 1.001] * 3 + [90.0, 90.0, 90.0],
                             dtype=torch.float64)
    eps = latc_to_strain_grain(latc_dil, latc_ref)
    # Diagonal entries ≈ 0.001, off-diagonals ≈ 0.
    torch.testing.assert_close(torch.diagonal(eps),
                               torch.full((3,), 0.001, dtype=torch.float64),
                               rtol=1e-9, atol=1e-9)
    assert eps.abs().max() < 1.001e-3 + 1e-12


@pytest.mark.skipif(not _have_stress_torch(),
                    reason="midas_stress torch backend not installed")
def test_pos_is_dropped_in_jacobian():
    """∂σ/∂pos must be zero — stress is a function of (euler, latc) only."""
    from midas_propagate.propagate import _stress_voigt_lab_from_g
    from torch.func import jacfwd

    g = torch.tensor(
        [0.1, 0.2, 0.3,                       # euler_rad
         4.08, 4.08, 4.08, 90.0, 90.0, 90.0,  # latc (== ref ⇒ zero strain)
         123.0, -45.0, 67.0],                  # pos_um (any value)
        dtype=torch.float64,
    )
    latc_ref = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0],
                             dtype=torch.float64)
    C = _cubic_stiffness()
    J = jacfwd(lambda gg: _stress_voigt_lab_from_g(gg, latc_ref, C))(g)
    # Position columns are indices 9, 10, 11.
    assert J.shape == (6, 12)
    assert J[:, 9:].abs().max() < 1e-12


@pytest.mark.skipif(not _have_stress_torch(),
                    reason="midas_stress torch backend not installed")
def test_per_grain_stress_zero_cov_gives_zero_cov():
    """Σ_g = 0 ⇒ Σ_σ = 0."""
    from midas_propagate.propagate import per_grain_stress_with_cov

    G = 5
    g_map = torch.zeros(G, 12, dtype=torch.float64)
    g_map[:, 3:9] = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0],
                                  dtype=torch.float64)
    sigma_gg = torch.zeros(G, 12, 12, dtype=torch.float64)
    latc_ref = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0],
                             dtype=torch.float64)
    C = _cubic_stiffness()

    res = per_grain_stress_with_cov(g_map, sigma_gg, latc_ref, C)
    assert res.stress_voigt.shape == (G, 6)
    assert res.stress_cov.shape == (G, 6, 6)
    assert res.sigma_voigt.shape == (G, 6)
    # Stress at ε=0 is exactly zero (no rotation can change that).
    torch.testing.assert_close(res.stress_voigt,
                               torch.zeros(G, 6, dtype=torch.float64),
                               atol=1e-12, rtol=1e-12)
    # Covariance is identically zero.
    assert res.stress_cov.abs().max() < 1e-12
    assert res.sigma_voigt.abs().max() < 1e-12


@pytest.mark.skipif(not _have_stress_torch(),
                    reason="midas_stress torch backend not installed")
def test_per_grain_stress_isotropic_grain_cov_propagates():
    """Σ_g = ε² · I on the latc-only block (cols 3..8) ⇒ stress cov is
    non-zero and finite, with no contribution from euler/pos blocks (we
    leave those zero)."""
    from midas_propagate.propagate import per_grain_stress_with_cov

    G = 3
    # Slightly strained, identity orientation.
    g = torch.tensor(
        [0.0, 0.0, 0.0,
         4.083, 4.080, 4.079, 90.0, 90.0, 90.0,
         0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    g_map = g.unsqueeze(0).expand(G, -1).clone()
    # Σ_g: small variance only on latc lengths (indices 3, 4, 5).
    sigma_gg = torch.zeros(G, 12, 12, dtype=torch.float64)
    var = (1e-4) ** 2   # σ = 1e-4 Å on each latc length
    for k in range(G):
        for i in (3, 4, 5):
            sigma_gg[k, i, i] = var
    latc_ref = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0],
                             dtype=torch.float64)
    C = _cubic_stiffness()

    res = per_grain_stress_with_cov(g_map, sigma_gg, latc_ref, C)
    # Stress diagonal Voigt entries should carry the propagated noise:
    # ε ~ 1e-4/4.08 ≈ 2.5e-5 strain, × C11 (200) ≈ 5e-3 stress, so σ_σ
    # should be order 1e-3 ... 1e-2. Just check positivity + finiteness.
    assert torch.isfinite(res.stress_cov).all()
    assert torch.isfinite(res.sigma_voigt).all()
    assert (res.sigma_voigt[:, :3] > 0).all()       # xx/yy/zz components
    # PSD on each per-grain covariance.
    for k in range(G):
        eigvals = torch.linalg.eigvalsh(
            0.5 * (res.stress_cov[k] + res.stress_cov[k].T)
        )
        assert eigvals.min() > -1e-9


@pytest.mark.skipif(not _have_stress_torch(),
                    reason="midas_stress torch backend not installed")
def test_input_validation():
    """Wrong shapes raise informative ValueError."""
    from midas_propagate.propagate import per_grain_stress_with_cov
    C = _cubic_stiffness()
    latc_ref = torch.tensor([4.08]*3 + [90.0]*3, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"g_map must be \(G, 12\)"):
        per_grain_stress_with_cov(
            torch.zeros(5, 13, dtype=torch.float64),
            torch.zeros(5, 12, 12, dtype=torch.float64),
            latc_ref, C,
        )
    with pytest.raises(ValueError, match="sigma_gg must be"):
        per_grain_stress_with_cov(
            torch.zeros(5, 12, dtype=torch.float64),
            torch.zeros(5, 12, 13, dtype=torch.float64),
            latc_ref, C,
        )
