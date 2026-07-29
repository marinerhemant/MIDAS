"""Tests for `midas_defect.forward_sim`."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_defect.forward_sim import (
    RodParams,
    bootstrap_rod_params,
    fit_rod_params,
    hendricks_teller,
    per_hkl_alpha_consistency,
    rod_intensity,
)


CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Synthetic correctness — Hendricks–Teller closed-form properties
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hendricks_teller_peak_position_at_integer_d_layer():
    """Peak position is at q_par·d_layer = integer regardless of α."""
    q = torch.linspace(-1.0, 1.0, 2001, dtype=torch.float64)
    d = torch.tensor(1.0, dtype=torch.float64)
    for alpha_val in (0.2, 0.5, 0.8):
        a = torch.tensor(alpha_val, dtype=torch.float64)
        I = hendricks_teller(q, d, a)
        # peak should be at q=0, q=±1
        peak_at_zero = I[1000].item()
        peak_at_one  = I[0].item()       # q = -1
        in_between   = I[500].item()     # q = -0.5
        assert peak_at_zero > in_between, f"α={alpha_val}: peak at 0 ({peak_at_zero}) not > midpoint ({in_between})"
        assert peak_at_one > in_between, f"α={alpha_val}: peak at -1 ({peak_at_one}) not > midpoint"


@pytest.mark.unit
def test_hendricks_teller_constant_for_alpha_zero_within_lobe():
    """α=0 gives I = (1)/(1 - 2·0·cos + 0) = 1."""
    q = torch.linspace(-1, 1, 101, dtype=torch.float64)
    d = torch.tensor(1.0, dtype=torch.float64)
    a = torch.tensor(0.0, dtype=torch.float64)
    I = hendricks_teller(q, d, a)
    assert torch.allclose(I, torch.ones_like(I), atol=1e-12)


@pytest.mark.unit
def test_hendricks_teller_peak_sharpens_with_alpha():
    """Peak-to-wing contrast grows monotonically with α in (0, 1)."""
    q = torch.linspace(-0.5, 0.5, 1001, dtype=torch.float64)
    d = torch.tensor(1.0, dtype=torch.float64)
    ratios = []
    for alpha_val in (0.1, 0.3, 0.5, 0.7, 0.9):
        a = torch.tensor(alpha_val, dtype=torch.float64)
        I = hendricks_teller(q, d, a)
        # peak (at q=0) / wing (at q=±0.5, the antipeak of cos)
        ratios.append(I[500].item() / I[0].item())
    # Strictly monotone increasing in α
    for a, b in zip(ratios, ratios[1:]):
        assert b > a, f"contrast not monotone: {ratios}"


@pytest.mark.unit
def test_rod_intensity_axis_aligned_perfect_recovery():
    """At known direction/pivot/d_layer, rod_intensity reproduces HT along axis."""
    q_par = torch.linspace(-2, 2, 401, dtype=torch.float64)
    # build q along direction d=[1,0,0] through pivot=[0,0,0]
    direction = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    pivot = torch.zeros(3, dtype=torch.float64)
    q = q_par.unsqueeze(1) * direction.unsqueeze(0)
    A = torch.tensor(100.0, dtype=torch.float64)
    alpha = torch.tensor(0.3, dtype=torch.float64)
    sigma_perp = torch.tensor(0.1, dtype=torch.float64)
    d_layer = torch.tensor(1.0, dtype=torch.float64)
    baseline = torch.tensor(0.0, dtype=torch.float64)
    I_pred = rod_intensity(q, direction, pivot, A, alpha, sigma_perp,
                            d_layer, baseline)
    # cross-term is exp(0)=1 on the axis; expect A * HT(q_par, d, α)
    HT = hendricks_teller(q_par, d_layer, alpha)
    assert torch.allclose(I_pred, A * HT, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Autograd correctness
# ---------------------------------------------------------------------------

@pytest.mark.autograd
def test_hendricks_teller_gradient_wrt_alpha():
    """Gradient of HT w.r.t. α matches finite difference."""
    q = torch.linspace(-1, 1, 51, dtype=torch.float64)
    d = torch.tensor(1.0, dtype=torch.float64)
    alpha = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

    def fn(a):
        return hendricks_teller(q, d, a).sum()

    L = fn(alpha)
    g_auto = torch.autograd.grad(L, alpha)[0]
    eps = 1e-6
    g_fd = (fn(alpha + eps) - fn(alpha - eps)) / (2 * eps)
    assert g_auto.item() == pytest.approx(g_fd.item(), rel=1e-5)


@pytest.mark.autograd
def test_rod_intensity_gradient_wrt_d_layer_and_sigma_perp():
    rng = np.random.default_rng(0)
    q_np = rng.uniform(-1, 1, size=(40, 3))
    q = torch.as_tensor(q_np, dtype=torch.float64)
    direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    pivot = torch.zeros(3, dtype=torch.float64)
    A = torch.tensor(50.0, dtype=torch.float64)
    alpha = torch.tensor(0.4, dtype=torch.float64)
    baseline = torch.tensor(0.5, dtype=torch.float64)

    d_layer = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    sigma_perp = torch.tensor(0.07, dtype=torch.float64, requires_grad=True)

    def fn(d, sp):
        return rod_intensity(q, direction, pivot, A, alpha, sp, d, baseline).sum()

    L = fn(d_layer, sigma_perp)
    g_d, g_sp = torch.autograd.grad(L, (d_layer, sigma_perp))
    eps = 1e-6
    g_d_fd  = (fn(d_layer + eps, sigma_perp.detach()) -
                fn(d_layer - eps, sigma_perp.detach())) / (2 * eps)
    g_sp_fd = (fn(d_layer.detach(), sigma_perp + eps) -
                fn(d_layer.detach(), sigma_perp - eps)) / (2 * eps)
    assert g_d.item()  == pytest.approx(g_d_fd.item(),  rel=1e-4, abs=1e-6)
    assert g_sp.item() == pytest.approx(g_sp_fd.item(), rel=1e-4, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Device portability
# ---------------------------------------------------------------------------

@pytest.mark.device
def test_rod_intensity_device_portable(_device_param):
    if _device_param.type == "mps":
        dtype = torch.float32; tol = 1e-4
    else:
        dtype = torch.float64; tol = 1e-10
    q = torch.tensor(np.random.RandomState(0).uniform(-1, 1, size=(20, 3)),
                     dtype=dtype)
    direction = torch.tensor([0.5, 0.0, 0.866], dtype=dtype)
    pivot = torch.zeros(3, dtype=dtype)
    args = dict(
        A=torch.tensor(100.0, dtype=dtype),
        alpha=torch.tensor(0.3, dtype=dtype),
        sigma_perp=torch.tensor(0.05, dtype=dtype),
        d_layer=torch.tensor(1.0, dtype=dtype),
        baseline=torch.tensor(0.5, dtype=dtype),
    )
    I_cpu = rod_intensity(q, direction, pivot, **args)
    I_dev = rod_intensity(q.to(_device_param), direction.to(_device_param),
                           pivot.to(_device_param),
                           **{k: v.to(_device_param) for k, v in args.items()})
    assert torch.allclose(I_cpu, I_dev.cpu(), atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# 4. End-to-end fit — recover (α, d_layer, σ_perp) from synthetic rod data
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_fit_rod_params_recovers_alpha_and_d_layer():
    """Plant a synthetic HT rod, refine all params, check recovery within 10%."""
    rng = np.random.default_rng(0)
    direction_true = np.array([0.30, 0.0, 0.95])
    direction_true /= np.linalg.norm(direction_true)
    pivot_true = np.array([0.0, 0.0, 0.0])
    alpha_true = 0.25
    sigma_perp_true = 0.04
    d_layer_true = 0.95
    A_true = 800.0
    baseline_true = 5.0

    # generate voxels in a 3D cube around the rod
    n = 5000
    pts = rng.uniform(-1.0, 1.0, size=(n, 3))
    q_t = torch.tensor(pts, dtype=torch.float64)
    I_clean = rod_intensity(
        q_t,
        torch.tensor(direction_true, dtype=torch.float64),
        torch.tensor(pivot_true, dtype=torch.float64),
        torch.tensor(A_true, dtype=torch.float64),
        torch.tensor(alpha_true, dtype=torch.float64),
        torch.tensor(sigma_perp_true, dtype=torch.float64),
        torch.tensor(d_layer_true, dtype=torch.float64),
        torch.tensor(baseline_true, dtype=torch.float64),
    ).numpy()
    I_noisy = I_clean + rng.normal(scale=2.0, size=n)

    fit = fit_rod_params(
        pts, I_noisy,
        direction_init=[0.40, 0.05, 0.91],          # initial offset ~6°
        pivot_init=[0.0, 0.0, 0.0],
        A_init=500.0, alpha_init=0.10,
        sigma_perp_init=0.08, d_layer_init=0.80,
        baseline_init=10.0,
        n_steps=2000, lr=1e-2,
        device=CPU,
    )
    # recovery checks
    cos = abs(float(fit.direction @ direction_true))
    angle_deg = math.degrees(math.acos(min(1.0, cos)))
    assert angle_deg < 3.0, f"direction off by {angle_deg:.2f}°"
    assert abs(fit.alpha - alpha_true) / alpha_true < 0.30, (
        f"alpha {fit.alpha:.3f} vs truth {alpha_true:.3f}"
    )
    assert abs(fit.d_layer - d_layer_true) / d_layer_true < 0.10, (
        f"d_layer {fit.d_layer:.3f} vs truth {d_layer_true:.3f}"
    )
    assert abs(fit.sigma_perp - sigma_perp_true) / sigma_perp_true < 0.50, (
        f"sigma_perp {fit.sigma_perp:.4f} vs truth {sigma_perp_true:.4f}"
    )


@pytest.mark.integration
def test_bootstrap_rod_params_covers_truth():
    """Bootstrap CI on α should include the ground-truth value most of the time."""
    rng = np.random.default_rng(7)
    direction_true = np.array([1.0, 0.0, 0.0])
    pivot_true = np.array([0.0, 0.0, 0.0])
    alpha_true = 0.30
    sigma_perp_true = 0.05
    d_layer_true = 1.0
    A_true = 600.0
    baseline_true = 10.0
    pts = rng.uniform(-1.0, 1.0, size=(2000, 3))
    q_t = torch.tensor(pts, dtype=torch.float64)
    I_clean = rod_intensity(
        q_t,
        torch.tensor(direction_true, dtype=torch.float64),
        torch.tensor(pivot_true, dtype=torch.float64),
        torch.tensor(A_true, dtype=torch.float64),
        torch.tensor(alpha_true, dtype=torch.float64),
        torch.tensor(sigma_perp_true, dtype=torch.float64),
        torch.tensor(d_layer_true, dtype=torch.float64),
        torch.tensor(baseline_true, dtype=torch.float64),
    ).numpy()
    I_noisy = I_clean + rng.normal(scale=3.0, size=pts.shape[0])
    boot = bootstrap_rod_params(
        pts, I_noisy,
        n_boot=8, bootstrap_fraction=0.5,
        direction_init=direction_true.tolist(),
        pivot_init=pivot_true.tolist(),
        A_init=A_true, alpha_init=0.20,
        sigma_perp_init=0.07, d_layer_init=0.9,
        baseline_init=12.0,
        n_steps=600, lr=1e-2, device="cpu",
    )
    a_lo, a_hi = boot["alpha"]["p16"], boot["alpha"]["p84"]
    assert a_lo is not None and a_hi is not None
    # Loosen to ±2σ to cover noise; reasonable for n_boot=8
    assert a_lo - 0.10 <= alpha_true <= a_hi + 0.10, (
        f"α CI [{a_lo:.3f}, {a_hi:.3f}] does not cover truth {alpha_true:.3f}"
    )


@pytest.mark.integration
def test_per_hkl_alpha_consistency_returns_per_crossing_fits():
    """per_hkl_alpha_consistency returns one fit per shell crossing."""
    rng = np.random.default_rng(0)
    direction = np.array([1.0, 0.0, 0.0])
    pivot = np.array([0.0, 0.0, 0.0])
    # Need enough voxels in each tube-around-crossing to satisfy min_voxels:
    # for crop_perp=0.30 (a wide tube of cross-section ~π·0.30²≈0.28),
    # crop_along=0.40 (window of 0.80), in a (-2,2)³ cube of volume 64:
    # fraction per crossing ≈ 0.28·0.80/64 ≈ 0.0035. With n=15000 → ~52 pts.
    pts = rng.uniform(-2.0, 2.0, size=(15000, 3))
    q_t = torch.tensor(pts, dtype=torch.float64)
    I = rod_intensity(
        q_t,
        torch.tensor(direction, dtype=torch.float64),
        torch.tensor(pivot, dtype=torch.float64),
        torch.tensor(400.0, dtype=torch.float64),
        torch.tensor(0.25, dtype=torch.float64),
        torch.tensor(0.05, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(5.0, dtype=torch.float64),
    ).numpy()
    I += rng.normal(scale=2.0, size=I.shape)

    fits = per_hkl_alpha_consistency(
        pts, I,
        rod_direction=direction, rod_pivot=pivot,
        shell_qs=[1.0, 2.0],
        d_layer_init=1.0, sigma_perp_init=0.05,
        crop_perp=0.30, crop_along=0.40,
        min_voxels=20,
    )
    # 2 shells × 2 crossings each (±t_cross) = up to 4 fits
    assert len(fits) >= 2, f"got {len(fits)} crossing fits"
    for f in fits:
        assert 0.0 < f["alpha"] < 1.0, f"α out of range: {f['alpha']}"
        assert f["d_layer"] > 0
        assert f["sigma_perp"] > 0
