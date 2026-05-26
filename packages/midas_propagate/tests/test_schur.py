"""Tests for midas_propagate.schur — per-grain Schur-marginal covariance.

Strategy: build small synthetic Hessian blocks where the joint inverse can
be computed in closed form, then check that ``per_grain_schur_marginal``
recovers the corresponding marginal block exactly.
"""
from __future__ import annotations

import math

import pytest
import torch


def _build_joint_inverse_ground_truth(
    H_gg: torch.Tensor,         # (n_g, n_g)
    H_gc: torch.Tensor,         # (n_g, n_c)
    H_cc: torch.Tensor,         # (n_c, n_c)
) -> torch.Tensor:
    """Build the full joint Hessian and return its (g, g) inverse block.

    Used as a reference: ``per_grain_schur_marginal`` should match this for
    a single grain when the calibration prior precision equals ``H_cc``.
    """
    n_g, n_c = H_gg.shape[0], H_cc.shape[0]
    H_joint = torch.zeros((n_g + n_c, n_g + n_c), dtype=H_gg.dtype)
    H_joint[:n_g, :n_g] = H_gg
    H_joint[:n_g, n_g:] = H_gc
    H_joint[n_g:, :n_g] = H_gc.T
    H_joint[n_g:, n_g:] = H_cc
    sigma_joint = torch.linalg.inv(H_joint)
    return sigma_joint[:n_g, :n_g]


def test_schur_matches_full_inversion_single_grain():
    """One grain, well-conditioned blocks. Schur path must equal the
    direct (g, g) block of the full joint inverse."""
    from midas_propagate.schur import per_grain_schur_marginal

    torch.manual_seed(0)
    n_g, n_c = 4, 6
    # SPD H_gg.
    A = torch.randn(n_g, n_g, dtype=torch.float64)
    H_gg_single = A @ A.T + torch.eye(n_g, dtype=torch.float64)
    # SPD H_cc with mild conditioning.
    B = torch.randn(n_c, n_c, dtype=torch.float64)
    H_cc = B @ B.T + 0.5 * torch.eye(n_c, dtype=torch.float64)
    # Cross block.
    H_gc_single = torch.randn(n_g, n_c, dtype=torch.float64) * 0.3
    sigma_cc = torch.linalg.inv(H_cc)

    # Reference: invert the 10x10 joint Hessian and take the (g, g) block.
    sigma_gg_ref = _build_joint_inverse_ground_truth(
        H_gg_single, H_gc_single, H_cc,
    )

    result = per_grain_schur_marginal(
        H_gg=H_gg_single.unsqueeze(0),
        H_gc=H_gc_single.unsqueeze(0),
        sigma_cc=sigma_cc,
        ridge_g=1e-12,
    )
    sigma_gg_schur = result.sigma_gg_calmarg.squeeze(0)
    torch.testing.assert_close(sigma_gg_schur, sigma_gg_ref, rtol=1e-9, atol=1e-12)


def test_schur_marg_widens_relative_to_frozen():
    """Marginalisation must inflate (PSD-greater-than-or-equal) the
    per-grain covariance, *provided each grain's implied joint Hessian
    is PSD*.

    The API takes one shared Sigma_cc (the paper-1 use case: calibration
    posterior from the calibrant data alone). For the joint per-grain
    Hessian ``[[H_gg, H_gc], [H_cg, inv(Sigma_cc)]]`` to stay PSD we
    keep H_gc small relative to H_gg's spectral floor — same regime that
    holds in the real HEDM setup, where calibration-grain coupling is
    bounded by the spot-position derivative scale."""
    from midas_propagate.schur import (
        per_grain_schur_marginal, sigma_inflation_ratio,
    )

    torch.manual_seed(1)
    G, n_g, n_c = 50, 3, 5
    # Shared calibration covariance.
    B = torch.randn(n_c, n_c, dtype=torch.float64)
    sigma_cc = B @ B.T + 0.5 * torch.eye(n_c, dtype=torch.float64)
    # Per-grain SPD H_gg with strong diagonal dominance.
    A = torch.randn(G, n_g, n_g, dtype=torch.float64)
    H_gg = A @ A.transpose(-1, -2) + 5.0 * torch.eye(n_g, dtype=torch.float64)
    # Small H_gc so the Schur correction stays inside H_gg's PSD cone.
    H_gc = torch.randn(G, n_g, n_c, dtype=torch.float64) * 0.05

    res = per_grain_schur_marginal(H_gg, H_gc, sigma_cc, ridge_g=1e-12)
    ratio = sigma_inflation_ratio(res.sigma_gg_frozen, res.sigma_gg_calmarg)
    assert torch.all(ratio >= 1.0 - 1e-9), (
        f"calibration-marginalised sigma must be >= frozen sigma; got ratio "
        f"min {ratio.min().item()}"
    )


def test_schur_zero_coupling_no_inflation():
    """If H_gc = 0 (calibration doesn't couple to this grain), Schur
    correction is zero and Sigma_marg == Sigma_frozen exactly."""
    from midas_propagate.schur import (
        per_grain_schur_marginal, sigma_inflation_ratio,
    )

    torch.manual_seed(2)
    G, n_g, n_c = 10, 4, 6
    A = torch.randn(G, n_g, n_g, dtype=torch.float64)
    H_gg = A @ A.transpose(-1, -2) + torch.eye(n_g, dtype=torch.float64)
    H_gc = torch.zeros(G, n_g, n_c, dtype=torch.float64)
    B = torch.randn(n_c, n_c, dtype=torch.float64)
    sigma_cc = B @ B.T + 0.5 * torch.eye(n_c, dtype=torch.float64)

    res = per_grain_schur_marginal(H_gg, H_gc, sigma_cc, ridge_g=1e-12)
    torch.testing.assert_close(
        res.sigma_gg_calmarg, res.sigma_gg_frozen, rtol=1e-9, atol=1e-12,
    )
    ratio = sigma_inflation_ratio(res.sigma_gg_frozen, res.sigma_gg_calmarg)
    torch.testing.assert_close(ratio, torch.ones_like(ratio),
                               rtol=1e-7, atol=1e-9)


def test_schur_handles_rank_deficient_sigma_cc():
    """Calibration covariance with structural rank-deficiency (a few
    eigendirections at zero) is the production case — must not NaN or blow
    up. The clipped eigenmodes get pseudo-inverted to zero (Moore-Penrose)
    so they contribute nothing to the Schur correction."""
    from midas_propagate.schur import per_grain_schur_marginal

    torch.manual_seed(3)
    G, n_g, n_c = 5, 3, 8

    A = torch.randn(G, n_g, n_g, dtype=torch.float64)
    H_gg = A @ A.transpose(-1, -2) + torch.eye(n_g, dtype=torch.float64)
    H_gc = torch.randn(G, n_g, n_c, dtype=torch.float64) * 0.2

    # Build a rank-3 sigma_cc (5 eigenvalues are zero).
    rank = 3
    U = torch.linalg.qr(torch.randn(n_c, n_c, dtype=torch.float64))[0]
    eigvals = torch.zeros(n_c, dtype=torch.float64)
    eigvals[:rank] = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
    sigma_cc = (U * eigvals.unsqueeze(0)) @ U.T

    res = per_grain_schur_marginal(
        H_gg, H_gc, sigma_cc, ridge_g=1e-12, cc_eig_tol=1e-12,
    )
    assert res.cc_rank_used == rank
    assert torch.isfinite(res.sigma_gg_calmarg).all()
    assert torch.isfinite(res.sigma_gg_frozen).all()

    # Sanity: marginal still PSD (positive diagonal).
    diag = torch.diagonal(res.sigma_gg_calmarg, dim1=-2, dim2=-1)
    assert torch.all(diag >= 0.0)


def test_schur_real_world_sizes_runs():
    """Smoke test at paper-1 scale: 200 grains × 12 grain params × 21
    calibration params. Should complete in << 1 s on CPU and produce
    finite outputs (no NaN/inf). Built from a per-grain SPD joint so the
    PSD inflation property holds."""
    from midas_propagate.schur import per_grain_schur_marginal

    torch.manual_seed(4)
    G, n_g, n_c = 200, 12, 21
    B = torch.randn(n_c, n_c, dtype=torch.float64)
    sigma_cc = B @ B.T + 0.5 * torch.eye(n_c, dtype=torch.float64)
    A = torch.randn(G, n_g, n_g, dtype=torch.float64)
    H_gg = A @ A.transpose(-1, -2) + 5.0 * torch.eye(n_g, dtype=torch.float64)
    H_gc = torch.randn(G, n_g, n_c, dtype=torch.float64) * 0.05

    res = per_grain_schur_marginal(
        H_gg, H_gc, sigma_cc, with_diagnostics=True,
    )
    assert res.sigma_gg_frozen.shape == (G, n_g, n_g)
    assert res.sigma_gg_calmarg.shape == (G, n_g, n_g)
    assert res.info_inflation_eigvals.shape == (G, n_g)
    assert torch.isfinite(res.sigma_gg_calmarg).all()
    assert torch.isfinite(res.info_inflation_eigvals).all()
    # All inflation eigvals must be >= 1 (within a tiny tolerance for the
    # ridge perturbation).
    assert (res.info_inflation_eigvals.min() > 1.0 - 1e-6)


def test_diagonal_sigma_shape_and_values():
    """``per_grain_diagonal_sigma`` returns ``sqrt(diag(...))`` per grain
    and clamps small negative entries (from numerical noise) to zero."""
    from midas_propagate.schur import per_grain_diagonal_sigma

    G, n_g = 4, 3
    cov = torch.zeros(G, n_g, n_g, dtype=torch.float64)
    diag_vals = torch.tensor([4.0, 9.0, 1e-20], dtype=torch.float64)
    for k in range(G):
        cov[k] = torch.diag(diag_vals)
    sigma = per_grain_diagonal_sigma(cov)
    assert sigma.shape == (G, n_g)
    torch.testing.assert_close(sigma[0], torch.tensor([2.0, 3.0, 1e-10],
                                                        dtype=torch.float64))


def test_schur_device_and_dtype_preserved():
    """Output dtype/device matches input — sanity that the per-grain
    blocks don't get silently downcast."""
    from midas_propagate.schur import per_grain_schur_marginal

    torch.manual_seed(5)
    H_gg = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    H_gc = torch.zeros((1, 3, 4), dtype=torch.float64)
    sigma_cc = torch.eye(4, dtype=torch.float64)

    res = per_grain_schur_marginal(H_gg, H_gc, sigma_cc)
    assert res.sigma_gg_frozen.dtype == torch.float64
    assert res.sigma_gg_calmarg.dtype == torch.float64
    assert res.sigma_gg_frozen.device == H_gg.device


def test_profile_unidentifiable_caps_huge_variances():
    """``profile_unidentifiable`` zeroes eigendirections whose variance
    sits orders of magnitude above the median — the production case is
    Σ_cc with a few distortion-phi directions at σ² ~ 10¹² alongside
    geometric directions at σ² ~ 10⁻⁴."""
    from midas_propagate.schur import profile_unidentifiable

    torch.manual_seed(7)
    # Build a Σ_cc with 4 well-conditioned eigenvalues + 2 huge ones.
    U = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))[0]
    eigvals = torch.tensor([1e-4, 1e-3, 1e-2, 1.0, 1e10, 1e12],
                            dtype=torch.float64)
    sigma_cc = (U * eigvals.unsqueeze(0)) @ U.T

    capped = profile_unidentifiable(sigma_cc, relative_cap=1e6)
    # The 1e10 and 1e12 eigenvalues (>> 1e6 * median(~1e-3) = 1e3) get zeroed.
    capped_eigvals = torch.linalg.eigvalsh(capped)
    assert capped_eigvals.max() < 1e6, (
        f"max kept eigenvalue {capped_eigvals.max().item():.3e} should be < cap"
    )
    # The well-conditioned eigenvalues are preserved (within sym-tolerance).
    assert (capped_eigvals > 1e-5).sum() >= 4


def test_profile_unidentifiable_var_cap_explicit():
    """Explicit ``var_cap`` overrides ``relative_cap`` and applies to
    eigenvalues directly."""
    from midas_propagate.schur import profile_unidentifiable

    U = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))[0]
    eigvals = torch.tensor([0.5, 1.0, 100.0, 1000.0], dtype=torch.float64)
    sigma_cc = (U * eigvals.unsqueeze(0)) @ U.T

    capped = profile_unidentifiable(sigma_cc, var_cap=50.0)
    kept = torch.linalg.eigvalsh(capped)
    assert kept.max() <= 50.0 + 1e-10
    assert (kept > 0.4).sum() == 2   # only 0.5 and 1.0 stay nonzero


def test_schur_input_validation():
    """Shape mismatches raise ValueError with useful message."""
    from midas_propagate.schur import per_grain_schur_marginal

    H_gg = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    H_gc = torch.zeros((1, 3, 4), dtype=torch.float64)
    sigma_cc = torch.eye(4, dtype=torch.float64)

    # n_g mismatch between H_gg and H_gc.
    bad_H_gc = torch.zeros((1, 4, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match="H_gg and H_gc disagree"):
        per_grain_schur_marginal(H_gg, bad_H_gc, sigma_cc)

    # n_c mismatch between H_gc and sigma_cc.
    bad_sigma_cc = torch.eye(5, dtype=torch.float64)
    with pytest.raises(ValueError, match="sigma_cc dimension"):
        per_grain_schur_marginal(H_gg, H_gc, bad_sigma_cc)

    # Non-square H_gg.
    bad_H_gg = torch.zeros((1, 3, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match="H_gg must be"):
        per_grain_schur_marginal(bad_H_gg, H_gc, sigma_cc)
