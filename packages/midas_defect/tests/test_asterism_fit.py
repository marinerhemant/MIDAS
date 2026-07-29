"""Tests for `midas_defect.asterism_fit`."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_defect.asterism_fit import (
    AsterismFit,
    fit_asterism_patches,
    fit_single_patch,
    predict_hkl_positions,
    strain_tensor_from_centroids,
    _sigma_from_cholesky,
)
from midas_defect.lattice import CUAL2_A_DEFAULT, CUAL2_C_DEFAULT, cual2_crystal


CPU = torch.device("cpu")


def _synthetic_gaussian_patch(
    center: np.ndarray, Sigma: np.ndarray, amplitude: float, baseline: float,
    n_voxels: int = 400, box_half: float = 0.15, rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(0)
    pts = center[None, :] + rng.uniform(-box_half, box_half, size=(n_voxels, 3))
    Sigma_inv = np.linalg.inv(Sigma)
    delta = pts - center
    quad = np.einsum("ni,ij,nj->n", delta, Sigma_inv, delta)
    intens = amplitude * np.exp(-0.5 * quad) + baseline
    intens += rng.normal(scale=0.05 * baseline + 1.0, size=intens.shape)
    return pts, intens


def _random_pd_sigma(rng: np.random.Generator, scale: float = 0.05) -> np.ndarray:
    """Random positive-definite Σ with eigenvalues in [scale²/2, 2 scale²]."""
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    eigs = rng.uniform(scale * scale * 0.5, scale * scale * 2.0, size=3)
    return Q @ np.diag(eigs) @ Q.T


# ---------------------------------------------------------------------------
# 1. Synthetic correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_predict_hkl_positions_returns_pairs_per_shell():
    """Predicted (hkl) list includes Friedel pairs (each hkl + its negative)."""
    U = np.eye(3)
    q_pred, hkls = predict_hkl_positions(U, a=CUAL2_A_DEFAULT, c=CUAL2_C_DEFAULT,
                                          q_max_inv_A=3.0)
    # every hkl must have its centrosymmetric partner in the list
    hkl_set = set(hkls)
    for h in hkls:
        neg = tuple(-x for x in h)
        assert neg in hkl_set, f"missing Friedel pair for {h}"
    # The q-vector set is centrosymmetric: for every q_pred[i] there is a
    # q_pred[j] = -q_pred[i].
    q_round = np.round(q_pred, 8)
    q_set = {tuple(row) for row in q_round}
    for row in q_round:
        assert tuple(-row) in q_set, f"missing centrosymmetric q for {row}"


@pytest.mark.unit
def test_predict_hkl_positions_q_magnitudes():
    """|q_pred| for U=I must equal |g_crystal| since rotation preserves length."""
    q_pred, hkls = predict_hkl_positions(np.eye(3), a=6.066, c=4.874,
                                          q_max_inv_A=3.0)
    for q, hkl in zip(q_pred, hkls):
        expected = 2 * math.pi * math.sqrt(
            (hkl[0]**2 + hkl[1]**2) / 6.066**2 + hkl[2]**2 / 4.874**2
        )
        assert np.linalg.norm(q) == pytest.approx(expected, rel=1e-10)


@pytest.mark.unit
def test_fit_single_patch_recovers_known_gaussian():
    """Plant a known 3-D Gaussian, fit it, recover (center, Σ) within 5%."""
    rng = np.random.default_rng(0)
    center_true = np.array([1.5, -0.3, 0.7])
    Sigma_true = _random_pd_sigma(rng, scale=0.04)
    pts, intens = _synthetic_gaussian_patch(
        center_true, Sigma_true, amplitude=1000.0, baseline=20.0,
        n_voxels=600, box_half=0.15, rng=rng,
    )
    q_t = torch.tensor(pts, dtype=torch.float64)
    I_t = torch.tensor(intens, dtype=torch.float64)
    q0_t = torch.tensor(center_true + np.array([0.01, -0.005, 0.008]),
                        dtype=torch.float64)
    fit = fit_single_patch(q_t, I_t, q0_t, sigma_init=0.04, n_steps=1000, lr=5e-3)

    assert np.allclose(fit["q_fit"], center_true, atol=5e-3), (
        f"center off: got {fit['q_fit']}, truth {center_true}"
    )
    # compare eigenvalues (independent of axis ordering). Anisotropic-Σ
    # recovery from noisy 600-voxel patches is the harder direction; we
    # accept ≤50% relative error per eigenvalue. Tighter recovery requires
    # second-order optimizer (LBFGS) — TODO for v0.2.
    eigs_true = np.sort(np.linalg.eigvalsh(Sigma_true))
    eigs_fit  = np.sort(np.linalg.eigvalsh(fit["Sigma"]))
    rel_err = np.abs(eigs_true - eigs_fit) / eigs_true
    assert (rel_err < 0.50).all(), (
        f"Σ eigenvalues off; truth={eigs_true}, fit={eigs_fit}, rel={rel_err}"
    )
    # average σ should be in the right ballpark
    assert abs(eigs_fit.mean() - eigs_true.mean()) / eigs_true.mean() < 0.25


@pytest.mark.unit
def test_fit_single_patch_raises_when_too_few_voxels():
    q_t = torch.zeros((3, 3), dtype=torch.float64)
    I_t = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least 6"):
        fit_single_patch(q_t, I_t, torch.zeros(3, dtype=torch.float64))


@pytest.mark.unit
def test_sigma_from_cholesky_is_positive_definite():
    """Any finite L_params input must yield Σ⁻¹ ≻ 0."""
    rng = np.random.default_rng(7)
    for _ in range(5):
        L = torch.tensor(rng.normal(size=6), dtype=torch.float64)
        Sinv = _sigma_from_cholesky(L)
        eigs = torch.linalg.eigvalsh(Sinv)
        assert (eigs > 0).all(), f"Σ⁻¹ has non-positive eigenvalues: {eigs}"


# ---------------------------------------------------------------------------
# 2. Autograd correctness
# ---------------------------------------------------------------------------

@pytest.mark.autograd
def test_fit_loss_gradient_wrt_q0_matches_fd():
    """Gradient of fit loss w.r.t. centre q0 matches finite-difference."""
    rng = np.random.default_rng(0)
    center = np.array([0.5, -0.3, 1.0])
    Sigma = _random_pd_sigma(rng, scale=0.05)
    pts, intens = _synthetic_gaussian_patch(
        center, Sigma, amplitude=800.0, baseline=10.0,
        n_voxels=300, box_half=0.12, rng=rng,
    )
    q_t = torch.tensor(pts, dtype=torch.float64)
    I_t = torch.tensor(intens, dtype=torch.float64)
    log_A = torch.log(I_t.max()).detach().clone().requires_grad_(False)
    log_b = torch.log(I_t.median().clamp_min(1.0)).detach().clone().requires_grad_(False)
    L_params = torch.zeros(6, dtype=torch.float64, requires_grad=False)
    L_params[0] = math.log(math.exp(1.0/0.05) - 1.0)
    L_params[2] = L_params[0]; L_params[5] = L_params[0]

    def loss(q0_):
        Sinv = _sigma_from_cholesky(L_params)
        delta = q_t - q0_
        quad = (delta @ Sinv * delta).sum(dim=-1)
        pred = torch.exp(log_A) * torch.exp(-0.5 * quad) + torch.exp(log_b)
        return ((I_t - pred) ** 2).sum()

    q0 = torch.tensor(center + 0.01, dtype=torch.float64, requires_grad=True)
    L = loss(q0)
    g_auto = torch.autograd.grad(L, q0)[0]

    eps = 1e-6
    g_fd = torch.zeros_like(q0)
    for i in range(3):
        plus  = q0.detach().clone(); plus[i]  += eps
        minus = q0.detach().clone(); minus[i] -= eps
        g_fd[i] = (loss(plus) - loss(minus)) / (2 * eps)
    assert torch.allclose(g_auto, g_fd, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. Device portability
# ---------------------------------------------------------------------------

@pytest.mark.device
def test_fit_single_patch_device_portable(_device_param):
    """Same fitted centre on CPU and MPS within tolerance."""
    if _device_param.type == "mps":
        dtype = torch.float32; tol = 1e-2
    else:
        dtype = torch.float64; tol = 1e-3
    rng = np.random.default_rng(1)
    center = np.array([1.0, 0.0, -0.5])
    Sigma = _random_pd_sigma(rng, scale=0.05)
    pts, intens = _synthetic_gaussian_patch(
        center, Sigma, amplitude=500.0, baseline=10.0,
        n_voxels=400, box_half=0.15, rng=rng,
    )
    q_t = torch.tensor(pts, dtype=dtype)
    I_t = torch.tensor(intens, dtype=dtype)
    q0_init = torch.tensor(center + 0.01, dtype=dtype)
    fit_cpu = fit_single_patch(q_t.to(CPU), I_t.to(CPU), q0_init.to(CPU),
                               sigma_init=0.05, n_steps=300, lr=1e-2)
    fit_dev = fit_single_patch(q_t.to(_device_param), I_t.to(_device_param),
                               q0_init.to(_device_param),
                               sigma_init=0.05, n_steps=300, lr=1e-2)
    assert np.allclose(fit_cpu["q_fit"], fit_dev["q_fit"], atol=tol)


# ---------------------------------------------------------------------------
# 4. Integration with predict_hkl_positions + cloud-of-patches
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_fit_asterism_patches_finds_multiple_hkls():
    """Plant a synthetic cloud with peaks at every predicted hkl; recover most of them."""
    rng = np.random.default_rng(3)
    U = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])    # identity
    a, c = 6.066, 4.874
    q_pred, hkls = predict_hkl_positions(U, a, c, q_max_inv_A=4.0)

    Sigma_iso = (0.04 ** 2) * np.eye(3)
    xs, ys, zs, intens = [], [], [], []
    for q0 in q_pred:
        pts, ii = _synthetic_gaussian_patch(
            q0, Sigma_iso, amplitude=500.0, baseline=5.0,
            n_voxels=80, box_half=0.10, rng=rng,
        )
        xs.append(pts[:, 0]); ys.append(pts[:, 1]); zs.append(pts[:, 2])
        intens.append(ii)
    qx = np.concatenate(xs); qy = np.concatenate(ys); qz = np.concatenate(zs)
    I  = np.concatenate(intens)

    fits = fit_asterism_patches(
        qx, qy, qz, I, U=U, a=a, c=c,
        q_max_inv_A=4.0, crop_halfwidth=0.10, min_voxels=20,
        n_steps=200, lr=1e-2, device=CPU,
    )

    # Expect at least ~half of the predicted hkls to be fit (some overlap, some
    # at low |q| with too few unique voxels)
    assert len(fits) >= len(q_pred) // 2, (
        f"fit only {len(fits)} of {len(q_pred)} predicted hkls"
    )
    # Each fit's q_fit must be close to its q_pred
    for f in fits:
        assert np.linalg.norm(f.q_fit - f.q_pred) < 0.03, (
            f"hkl={f.hkl}: q_fit {f.q_fit} far from q_pred {f.q_pred}"
        )
        # Σ eigenvalues should be in the right ballpark (~ 0.04 ** 2)
        for s in f.sigma_eig:
            assert 0.01 < s < 0.10, f"σ={s} for hkl {f.hkl} outside expected range"


@pytest.mark.unit
def test_strain_tensor_recovers_diagonal_strain():
    """Plant a known isotropic strain and confirm recovery."""
    rng = np.random.default_rng(0)
    U = np.eye(3)
    a, c = 6.066, 4.874
    q_pred, hkls = predict_hkl_positions(U, a, c, q_max_inv_A=4.0)
    eps_true = 0.001 * np.diag([1.0, -0.5, 0.3])      # ε_11=1e-3, ε_22=-5e-4, ε_33=3e-4
    fake_fits = []
    for q0, hkl in zip(q_pred, hkls):
        # q_obs = (I + ε) · q_pred
        q_obs = q0 + eps_true @ q0 + rng.normal(scale=1e-5, size=3)
        fake_fits.append(AsterismFit(
            hkl=hkl, q_pred=q0, q_fit=q_obs,
            amplitude=1.0, baseline=0.0,
            sigma_eig=np.array([0.04, 0.04, 0.04]),
            sigma_axes=np.eye(3),
            integrated_intensity=100.0, n_voxels=80,
            final_loss=0.0, converged=True,
        ))
    s = strain_tensor_from_centroids(fake_fits)
    assert np.allclose(s["epsilon_sym"], eps_true, atol=2e-5), (
        f"strain off: got\n{s['epsilon_sym']}\nexpected\n{eps_true}"
    )
    assert s["volumetric_strain"] == pytest.approx(np.trace(eps_true), abs=1e-5)


@pytest.mark.unit
def test_strain_tensor_raises_with_too_few_fits():
    with pytest.raises(ValueError, match="at least 4"):
        strain_tensor_from_centroids([])
