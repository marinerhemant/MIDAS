"""Strain-solver tests — synthetic ground-truth recovery, autograd."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_process_grains.compute.strain import (
    PerSpotStrainResult,
    build_design_matrix,
    solve_strain_kenesei_bounded,
    solve_strain_kenesei_unbounded,
    solve_strain_fable_beaudoin,
    solve_strain_lstsq,            # backwards-compat alias of unbounded
    solve_strain_lattice,          # backwards-compat alias of fable_beaudoin
    voigt6_to_tensor,
)


def _random_unit_vectors(n, rng):
    g = rng.standard_normal((n, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g


def test_design_matrix_columns_for_unit_x_axis():
    """A pure x-direction g-vector has design row [1, 0, 0, 0, 0, 0]."""
    g = torch.tensor([[1.0, 0.0, 0.0]])
    G = build_design_matrix(g)
    np.testing.assert_allclose(G.numpy()[0], [1, 0, 0, 0, 0, 0])


def test_voigt6_to_tensor_round_trip():
    eps = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    T = voigt6_to_tensor(eps)
    expected = torch.tensor([[1.0, 4.0, 5.0], [4.0, 2.0, 6.0], [5.0, 6.0, 3.0]])
    np.testing.assert_allclose(T.numpy(), expected.numpy())


def test_recover_known_strain_from_synthetic_spots():
    """Generate spots from a known strain tensor; lstsq recovers it.

    For each unit g_hat, the relative d-spacing change is
    ``Δd/d = g_hat^T ε g_hat``.
    """
    rng = np.random.default_rng(42)
    n = 60
    g = _random_unit_vectors(n, rng)
    eps_true = np.array([
        [1e-4, 5e-5, -2e-5],
        [5e-5, -3e-4, 1e-5],
        [-2e-5, 1e-5, 2e-4],
    ])

    bb = np.einsum("ni,ij,nj->n", g, eps_true, g)
    ds_0 = np.full(n, 2.0784)             # arbitrary reference
    ds_obs = ds_0 * (1.0 + bb)

    g_t = torch.from_numpy(g)
    ds_obs_t = torch.from_numpy(ds_obs)
    ds_0_t = torch.from_numpy(ds_0)

    res = solve_strain_lstsq(g_t, ds_obs_t, ds_0_t)
    assert isinstance(res, PerSpotStrainResult)
    assert res.n_spots == n
    np.testing.assert_allclose(
        res.epsilon_tensor.numpy(), eps_true, atol=1e-12
    )
    assert float(res.residual_norm) < 1e-12


def test_sample_frame_g_required_lab_frame_g_fabricates_shear():
    """Regression: the Kenesei design matrix must use the SAMPLE-frame ĝ
    (ω-rotated), not a lab-frame ĝ recomputed from the detector (y, z).

    A grain's strain tensor is fixed in the sample frame. Each reflection's
    Δd/d depends on ``ĝ_sample^T ε ĝ_sample``. Spots are recorded at many ω, so
    their lab-frame ĝ = Rz(ω)·ĝ_sample differ. Fitting one tensor against the
    lab-frame ĝ (the FF c-omp strain bug) mixes ω frames and dumps the
    variation into spurious off-diagonal shear. This test pins both: the
    sample-frame fit recovers truth exactly; the lab-frame fit does not and
    fabricates large shear.
    """
    rng = np.random.default_rng(7)
    n = 80
    g_sample = _random_unit_vectors(n, rng)
    omega = rng.uniform(-180.0, 180.0, size=n)            # spread over ω

    eps_true = np.array([
        [1.5e-3, 1.0e-5, -2.0e-5],
        [1.0e-5, 1.4e-3, 3.0e-5],
        [-2.0e-5, 3.0e-5, 1.6e-3],
    ])                                                     # ~isotropic, tiny shear
    bb = np.einsum("ni,ij,nj->n", g_sample, eps_true, g_sample)
    ds_0 = np.full(n, 2.0784)
    ds_obs = ds_0 * (1.0 + bb)

    # Lab-frame ĝ = Rz(ω) · ĝ_sample (ω about the vertical lab axis).
    om = np.deg2rad(omega)
    c, s = np.cos(om), np.sin(om)
    g_lab = np.empty_like(g_sample)
    g_lab[:, 0] = c * g_sample[:, 0] - s * g_sample[:, 1]
    g_lab[:, 1] = s * g_sample[:, 0] + c * g_sample[:, 1]
    g_lab[:, 2] = g_sample[:, 2]

    ds_obs_t = torch.from_numpy(ds_obs)
    ds_0_t = torch.from_numpy(ds_0)

    res_sample = solve_strain_lstsq(torch.from_numpy(g_sample), ds_obs_t, ds_0_t)
    res_lab = solve_strain_lstsq(torch.from_numpy(g_lab), ds_obs_t, ds_0_t)

    # Sample-frame ĝ recovers truth to machine precision with ~zero residual.
    np.testing.assert_allclose(
        res_sample.epsilon_tensor.numpy(), eps_true, atol=1e-12
    )
    assert float(res_sample.residual_norm) < 1e-12

    # Lab-frame ĝ cannot fit ONE tensor (the data live in many ω frames), so
    # the residual is orders of magnitude larger and the recovered tensor is
    # provably wrong — the diagnostic signature of the FF c-omp strain bug.
    assert float(res_lab.residual_norm) > 1e3 * float(res_sample.residual_norm)
    assert float(res_lab.residual_norm) > 1e-4
    err_lab = np.linalg.norm(res_lab.epsilon_tensor.numpy() - eps_true)
    err_sample = np.linalg.norm(res_sample.epsilon_tensor.numpy() - eps_true)
    assert err_lab > 1e3 * max(err_sample, 1e-15)


def test_recover_zero_strain_at_no_change():
    rng = np.random.default_rng(0)
    n = 12
    g = torch.from_numpy(_random_unit_vectors(n, rng))
    ds_0 = torch.full((n,), 1.5)
    ds_obs = ds_0.clone()  # no change → strain = 0
    res = solve_strain_lstsq(g, ds_obs, ds_0)
    np.testing.assert_allclose(res.epsilon_voigt.numpy(), np.zeros(6), atol=1e-13)


def test_lstsq_is_differentiable():
    """Autograd through the backslash works."""
    rng = np.random.default_rng(7)
    n = 30
    g = torch.from_numpy(_random_unit_vectors(n, rng)).requires_grad_(False)
    eps_true = np.array([1e-4, -2e-4, 3e-4, 1e-5, -2e-5, 5e-6])
    bb = (g ** 2 @ eps_true[:3]) + 2 * (
        g[:, 0] * g[:, 1] * eps_true[3]
        + g[:, 0] * g[:, 2] * eps_true[4]
        + g[:, 1] * g[:, 2] * eps_true[5]
    )
    ds_0 = torch.full((n,), 2.0)
    # ds_obs depends on eps_true through bb -- but for the autograd test we
    # treat ds_obs as a leaf with grad enabled.
    ds_obs = (ds_0 * (1.0 + bb)).detach().clone().requires_grad_(True)

    res = solve_strain_lstsq(g, ds_obs, ds_0)
    loss = (res.epsilon_voigt ** 2).sum()
    loss.backward()
    assert ds_obs.grad is not None
    assert torch.isfinite(ds_obs.grad).all()
    # Gradient must be non-trivial — strain is a non-zero linear function of ds_obs.
    assert float(ds_obs.grad.abs().max()) > 0


def test_kenesei_bounded_recovers_known_strain_within_bounds():
    """Bounded Kenesei (paper Eq. 8-11 + ±0.01 box) recovers a strain that
    lies inside the box."""
    rng = np.random.default_rng(31)
    n = 60
    g = rng.standard_normal((n, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    eps_true = np.array([1e-3, -2e-4, 3e-4, 1e-5, -2e-5, 5e-6])
    bb = (g ** 2 @ eps_true[:3]) + 2 * (
        g[:, 0] * g[:, 1] * eps_true[3]
        + g[:, 0] * g[:, 2] * eps_true[4]
        + g[:, 1] * g[:, 2] * eps_true[5]
    )
    ds_0 = np.full(n, 1.27)
    ds_obs = ds_0 * (1.0 + bb)
    res = solve_strain_kenesei_bounded(
        torch.from_numpy(g),
        torch.from_numpy(ds_obs),
        torch.from_numpy(ds_0),
    )
    np.testing.assert_allclose(
        res.epsilon_voigt.numpy(), eps_true, atol=2e-6,
    )


def test_kenesei_bounded_clamps_blow_up_at_box_edge():
    """When the unbounded lstsq blows up out of [-0.01, 0.01], the bounded
    version saturates at the boundary."""
    rng = np.random.default_rng(33)
    n = 240
    # Construct an FF-HEDM-like geometry: g_x small and nearly constant.
    two_theta = math.radians(7.8)
    g_x_const = -math.sin(two_theta / 2)
    eta = rng.uniform(0, 2 * math.pi, size=n)
    g = np.stack([
        np.full(n, g_x_const),
        math.cos(two_theta / 2) * np.cos(eta),
        math.cos(two_theta / 2) * np.sin(eta),
    ], axis=1)
    g /= np.linalg.norm(g, axis=1, keepdims=True)

    # Use a strain that lstsq would interpret as huge ε_xx via the noise.
    bb = rng.normal(scale=2e-3, size=n)        # white noise; truth is 0
    ds_0 = np.full(n, 1.27)
    ds_obs = ds_0 * (1.0 + bb)

    res_bounded = solve_strain_kenesei_bounded(
        torch.from_numpy(g),
        torch.from_numpy(ds_obs),
        torch.from_numpy(ds_0),
    )
    # The bounded result must lie in the box on every component.
    assert (res_bounded.epsilon_voigt.abs() <= 0.0100001).all().item()


def test_kenesei_bounded_no_autograd_required():
    """Bounded path runs under torch.no_grad without errors."""
    rng = np.random.default_rng(7)
    n = 30
    g = rng.standard_normal((n, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    ds_0 = np.full(n, 1.5)
    ds_obs = ds_0 * 1.0001
    with torch.no_grad():
        res = solve_strain_kenesei_bounded(
            torch.from_numpy(g),
            torch.from_numpy(ds_obs),
            torch.from_numpy(ds_0),
        )
    assert res.n_spots == n


def test_solve_strain_lstsq_alias_still_works():
    """Backwards-compat alias resolves to unbounded Kenesei."""
    assert solve_strain_lstsq is solve_strain_kenesei_unbounded
    assert solve_strain_lattice is solve_strain_fable_beaudoin


def test_too_few_spots_raises():
    g = torch.from_numpy(np.eye(3))      # only 3 spots
    ds = torch.tensor([2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match=r"≥\s*6"):
        solve_strain_lstsq(g, ds, ds.clone())


def test_tikhonov_recovers_strain_on_well_conditioned_geometry():
    """With α=1e-6, regularised lstsq should agree with pure lstsq on a
    well-conditioned system to within a few × α."""
    rng = np.random.default_rng(99)
    n = 50
    g = rng.standard_normal((n, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    eps_true = np.array([1e-3, -2e-4, 3e-4, 1e-5, -2e-5, 0.5e-5])
    bb = (g ** 2 @ eps_true[:3]) + 2 * (
        g[:, 0] * g[:, 1] * eps_true[3]
        + g[:, 0] * g[:, 2] * eps_true[4]
        + g[:, 1] * g[:, 2] * eps_true[5]
    )
    ds_0 = np.full(n, 2.0)
    ds_obs = ds_0 * (1.0 + bb)

    g_t = torch.from_numpy(g)
    d_obs_t = torch.from_numpy(ds_obs)
    d_0_t = torch.from_numpy(ds_0)

    res_unreg = solve_strain_lstsq(g_t, d_obs_t, d_0_t)
    res_reg = solve_strain_lstsq(g_t, d_obs_t, d_0_t, regularization=1e-6)

    np.testing.assert_allclose(
        res_unreg.epsilon_voigt.numpy(), eps_true, atol=1e-12,
    )
    # Regularised result is biased by α toward zero; for well-conditioned
    # data the bias is α × ‖ε‖ which is ≪ 1e-3 here.
    np.testing.assert_allclose(
        res_reg.epsilon_voigt.numpy(), eps_true, atol=1e-5,
    )


def test_tikhonov_well_conditioned_round_trip_keeps_truth():
    """At α=1e-3 in column-normalised form, well-conditioned synthetic data
    are recovered to within a few × α × ‖ε‖ — i.e. essentially unbiased.

    The full FF-HEDM ε_xx blow-up pathology is exercised at the dataset
    level by the smoke run on alleppey; constructing a synthetic that
    exhibits the same numerical conditioning as a real 240-spot FCC grain
    while remaining unit-testable is brittle, so we settle here for a
    well-conditioned no-regression test.
    """
    rng = np.random.default_rng(2)
    n = 60
    g = rng.standard_normal((n, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    eps_true = np.array([1e-3, -2e-4, 3e-4, 1e-5, -2e-5, 5e-6])
    bb = (g ** 2 @ eps_true[:3]) + 2 * (
        g[:, 0] * g[:, 1] * eps_true[3]
        + g[:, 0] * g[:, 2] * eps_true[4]
        + g[:, 1] * g[:, 2] * eps_true[5]
    )
    ds_0 = np.full(n, 1.27)
    ds_obs = ds_0 * (1.0 + bb)
    res = solve_strain_lstsq(
        torch.from_numpy(g),
        torch.from_numpy(ds_obs),
        torch.from_numpy(ds_0),
        regularization=1e-3,
    )
    np.testing.assert_allclose(
        res.epsilon_voigt.numpy(), eps_true, atol=5e-6,
    )


def test_weighted_solve_accepts_weights():
    """Weighting all-ones reproduces the unweighted answer."""
    rng = np.random.default_rng(123)
    n = 20
    g = torch.from_numpy(_random_unit_vectors(n, rng))
    eps_true = np.array([1e-4, -1e-4, 2e-4, 0, 0, 0])
    bb = (g ** 2 @ eps_true[:3])
    ds_0 = torch.full((n,), 1.5)
    ds_obs = ds_0 * (1.0 + bb)

    a = solve_strain_lstsq(g, ds_obs, ds_0)
    b = solve_strain_lstsq(g, ds_obs, ds_0, weights=torch.ones(n))
    np.testing.assert_allclose(
        a.epsilon_voigt.numpy(), b.epsilon_voigt.numpy(), atol=1e-12
    )
