"""The scoring protocol the 12-D result rests on.

`decompose_affine` / `weighted_corr` were load-bearing for a published claim before
they had a single direct test. A bug in either would not crash anything -- it would
quietly change the headline number -- so they are pinned here against closed forms.
"""
import math

import pytest
import torch

from midas_dct_tt import (affine_basis, decompose_affine, intensity_scales,
                          profiled_intensity_residual, weighted_corr)

DT = torch.float64


@pytest.fixture
def pos():
    torch.manual_seed(0)
    return torch.randn(300, 3, dtype=DT)


# --- affine decomposition --------------------------------------------------
def test_affine_basis_is_one_x_y_z(pos):
    A = affine_basis(pos)
    assert A.shape == (pos.shape[0], 4)
    assert torch.allclose(A[:, 0], torch.ones_like(A[:, 0]))
    assert torch.allclose(A[:, 1:], pos)


def test_pure_affine_field_leaves_no_residual(pos):
    """A uniform F plus a linear gradient is exactly 36 dof -- all of it affine."""
    torch.manual_seed(1)
    H0 = torch.randn(3, 3, dtype=DT)
    G = torch.randn(3, 3, 3, dtype=DT)
    H = H0.expand(pos.shape[0], 3, 3) + torch.einsum("nk,kij->nij", pos, G)
    fit, res = decompose_affine(H, pos)
    assert float(res.abs().max()) < 1e-12
    assert torch.allclose(fit, H, atol=1e-12)


def test_decomposition_sums_back_to_the_original(pos):
    torch.manual_seed(2)
    H = torch.randn(pos.shape[0], 3, 3, dtype=DT)
    fit, res = decompose_affine(H, pos)
    assert torch.allclose(fit + res, H, atol=1e-12)


def test_residual_is_orthogonal_to_the_affine_basis(pos):
    """The defining property of a least-squares projection."""
    torch.manual_seed(3)
    H = torch.randn(pos.shape[0], 3, 3, dtype=DT)
    _, res = decompose_affine(H, pos)
    A = affine_basis(pos)
    assert float((A.T @ res.reshape(-1, 9)).abs().max()) < 1e-10


def test_quadratic_field_is_mostly_not_affine(pos):
    """x*y cannot be represented by [1,x,y,z], so most of it must survive."""
    torch.manual_seed(4)
    M = torch.randn(3, 3, dtype=DT)
    H = (pos[:, 0] * pos[:, 1]).reshape(-1, 1, 1) * M
    fit, res = decompose_affine(H, pos)
    assert float(res.norm() / H.norm()) > 0.8


def test_weights_restrict_the_fit_to_where_they_are_nonzero(pos):
    """Occupancy weighting must make vacuum voxels unable to steer the fit."""
    torch.manual_seed(5)
    H = torch.randn(pos.shape[0], 3, 3, dtype=DT)
    w = torch.zeros(pos.shape[0], dtype=DT)
    w[:50] = 1.0
    fit_w, _ = decompose_affine(H, pos, w)
    fit_sub, _ = decompose_affine(H[:50], pos[:50])
    assert torch.allclose(fit_w[:50], fit_sub, atol=1e-9)


# --- weighted correlation --------------------------------------------------
def test_weighted_corr_self_and_negation(pos):
    torch.manual_seed(6)
    H = torch.randn(pos.shape[0], 3, 3, dtype=DT)
    assert weighted_corr(H, H) == pytest.approx(1.0, abs=1e-12)
    assert weighted_corr(H, -H) == pytest.approx(-1.0, abs=1e-12)


def test_weighted_corr_is_scale_and_offset_invariant(pos):
    torch.manual_seed(7)
    H = torch.randn(pos.shape[0], 3, 3, dtype=DT)
    assert weighted_corr(H, 3.5 * H + 2.0) == pytest.approx(1.0, abs=1e-12)


def test_weighted_corr_matches_unweighted_pearson_when_weights_are_flat(pos):
    torch.manual_seed(8)
    a = torch.randn(pos.shape[0], 3, 3, dtype=DT)
    b = torch.randn(pos.shape[0], 3, 3, dtype=DT)
    x, y = a.reshape(-1), b.reshape(-1)
    x, y = x - x.mean(), y - y.mean()
    ref = float(x @ y / (x.norm() * y.norm()))
    assert weighted_corr(a, b) == pytest.approx(ref, abs=1e-12)
    assert weighted_corr(a, b, torch.ones(pos.shape[0], dtype=DT)) == pytest.approx(ref, abs=1e-12)


def test_zero_weight_voxels_cannot_contribute(pos):
    """The fix for '62% of the score came from vacuum'."""
    torch.manual_seed(9)
    a = torch.randn(pos.shape[0], 3, 3, dtype=DT)
    b = a.clone()
    w = torch.ones(pos.shape[0], dtype=DT)
    w[100:] = 0.0
    b[100:] = torch.randn(pos.shape[0] - 100, 3, 3, dtype=DT) * 50.0
    assert weighted_corr(a, b, w) == pytest.approx(1.0, abs=1e-12)


# --- intensity scales ------------------------------------------------------
def test_intensity_scales_recovers_exact_per_reflection_factors():
    torch.manual_seed(10)
    pred = torch.rand(3, 48, dtype=DT) + 1.0
    truth = torch.tensor([[2.0], [0.5], [7.0]], dtype=DT)
    got = intensity_scales(pred, truth * pred)
    assert torch.allclose(got, truth, atol=1e-12)


def test_intensity_scales_is_the_least_squares_minimiser():
    """a_r must minimise sum_s (a*pred - obs)^2 -- check the derivative vanishes."""
    torch.manual_seed(11)
    pred = torch.rand(2, 20, dtype=DT) + 1.0
    obs = torch.rand(2, 20, dtype=DT)
    a = intensity_scales(pred, obs)
    grad = ((a * pred - obs) * pred).sum(dim=-1)
    assert float(grad.abs().max()) < 1e-12


def test_profiled_residual_is_zero_at_the_truth_for_any_flux_error():
    torch.manual_seed(12)
    pred = torch.rand(3, 30, dtype=DT) + 1.0
    for c in ([1.0, 1.0, 1.0], [1.0472, 1.0472, 1.0472], [0.5, 1.0, 2.0]):
        obs = pred * torch.tensor(c, dtype=DT).unsqueeze(-1)
        assert float(profiled_intensity_residual(pred, obs)) < 1e-25


def test_profiled_residual_value_is_invariant_only_for_EQUAL_factors():
    """The precise statement. A uniform rescale leaves the value untouched; unequal
    per-reflection factors do not, because the normalisation is a global sum."""
    torch.manual_seed(13)
    pred = torch.rand(3, 30, dtype=DT) + 1.0
    obs = torch.rand(3, 30, dtype=DT) + 1.0
    base = float(profiled_intensity_residual(pred, obs))
    uniform = float(profiled_intensity_residual(pred, obs * 1.0472))
    assert uniform == pytest.approx(base, rel=1e-12)
    uneven = float(profiled_intensity_residual(
        pred, obs * torch.tensor([0.5, 1.0, 2.0], dtype=DT).unsqueeze(-1)))
    assert abs(uneven - base) / base > 1e-3


def test_profiled_residual_is_differentiable():
    torch.manual_seed(14)
    pred = (torch.rand(2, 12, dtype=DT) + 1.0).requires_grad_(True)
    obs = torch.rand(2, 12, dtype=DT) + 1.0
    profiled_intensity_residual(pred, obs).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
