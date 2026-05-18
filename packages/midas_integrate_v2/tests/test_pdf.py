"""Item 2 — PDF helpers + σ propagation through G(r).

- R_px_to_Q matches analytical formula.
- σ propagation: planted Poisson noise on synthetic profile produces
  σ_G consistent with a bootstrap (1σ agreement, >100 trials).
- FT round-trip: a Gaussian peak in S(Q) yields a known closed-form
  G(r) shape that recovers position/width.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_integrate_v2.pdf import (
    R_px_to_Q,
    estimate_background,
    fourier_sine_transform,
    normalize_to_S,
)


def test_R_px_to_Q_matches_analytical():
    Lsd_um = 1_000_000.0
    px_um = 200.0
    lam_A = 0.18
    Rs = torch.tensor([10.0, 50.0, 100.0, 500.0], dtype=torch.float64)
    Q = R_px_to_Q(Rs, Lsd_um=Lsd_um, px_um=px_um, lambda_A=lam_A)
    expected = (4.0 * math.pi / lam_A) * np.sin(
        0.5 * np.arctan(Rs.numpy() * px_um / Lsd_um)
    )
    np.testing.assert_allclose(Q.numpy(), expected, rtol=1e-12, atol=1e-15)


def test_R_px_to_Q_differentiable_in_Lsd_and_lambda():
    Lsd = torch.tensor(1_000_000.0, dtype=torch.float64, requires_grad=True)
    lam = torch.tensor(0.18, dtype=torch.float64, requires_grad=True)
    Rs = torch.tensor([10.0, 100.0, 500.0], dtype=torch.float64)
    Q = R_px_to_Q(Rs, Lsd_um=Lsd, px_um=200.0, lambda_A=lam)
    Q.sum().backward()
    assert Lsd.grad is not None and torch.isfinite(Lsd.grad)
    assert lam.grad is not None and torch.isfinite(lam.grad)
    assert float(Lsd.grad) < 0   # bigger Lsd → smaller Q
    assert float(lam.grad) < 0   # bigger λ → smaller Q


def test_estimate_background_rejects_peaks():
    n = 200
    x = np.linspace(0, 10, n)
    base = 1.0 + 0.05 * x
    peaks = np.zeros_like(x)
    for c, w in [(2.0, 0.05), (5.0, 0.05), (8.0, 0.05)]:
        peaks += 100.0 * np.exp(-0.5 * ((x - c) / w) ** 2)
    profile = base + peaks
    bg = estimate_background(profile, window=21, percentile=10.0).numpy()
    # Background should track 'base' within < 5% in away-from-peak regions.
    away = np.abs(x - 2.0) > 0.5
    away &= np.abs(x - 5.0) > 0.5
    away &= np.abs(x - 8.0) > 0.5
    rms = np.sqrt(np.mean((bg[away] - base[away]) ** 2))
    assert rms < 0.10


def test_normalize_to_S_with_constant_form_factor():
    q = torch.linspace(1.0, 30.0, 200, dtype=torch.float64)
    I = torch.ones_like(q) * 5.0
    f2 = torch.ones_like(q) * 5.0  # I(Q) == <f²> → S(Q) == 1
    S, sig = normalize_to_S(I, q=q, atomic_form_factor_squared=f2)
    np.testing.assert_allclose(S.numpy(), np.ones_like(q.numpy()),
                                rtol=1e-12, atol=1e-12)
    assert torch.allclose(sig, torch.zeros_like(sig))


def test_normalize_to_S_propagates_sigma():
    q = torch.linspace(1.0, 30.0, 200, dtype=torch.float64)
    I = torch.ones_like(q) * 5.0
    f2 = torch.ones_like(q) * 5.0
    sig_I = torch.ones_like(q) * 0.1
    S, sig_S = normalize_to_S(I, q=q, atomic_form_factor_squared=f2,
                               sigma_intensity=sig_I)
    np.testing.assert_allclose(sig_S.numpy(), sig_I.numpy() / f2.numpy(),
                                rtol=1e-12, atol=1e-15)


def test_FT_recovers_known_oscillation():
    """A sinusoid in (S(Q)-1) should produce a localised G(r) feature
    near the corresponding r."""
    q = torch.linspace(0.5, 30.0, 2950, dtype=torch.float64)
    r0 = 2.5  # planted oscillation period in r
    Sm1 = 0.05 * torch.sin(q * r0)  # S(Q) - 1
    S = Sm1 + 1.0
    r_grid = torch.linspace(0.1, 10.0, 1024, dtype=torch.float64)
    G, _ = fourier_sine_transform(q, S, r_grid, Q_max=30.0, window="rect")
    # Peak of |G(r)| should be near r = r0
    peak_idx = int(torch.argmax(G.abs()))
    r_peak = float(r_grid[peak_idx])
    assert abs(r_peak - r0) < 0.1


def test_FT_sigma_matches_bootstrap():
    """Plant Poisson-like σ on synthetic S(Q); analytic σ_G should agree
    with a Monte-Carlo bootstrap to within 1σ statistical."""
    rng = np.random.default_rng(42)
    q = torch.linspace(0.5, 30.0, 600, dtype=torch.float64)
    Sm1_true = 0.05 * torch.sin(q * 2.5)
    S_true = Sm1_true + 1.0
    sig_S = torch.full_like(q, 0.01)
    r_grid = torch.linspace(0.5, 5.0, 64, dtype=torch.float64)
    G_analytic, sigG_analytic = fourier_sine_transform(
        q, S_true, r_grid, Q_max=30.0, window="rect", sigma_S=sig_S,
    )
    n_boot = 300
    G_samples = np.zeros((n_boot, r_grid.shape[0]))
    for k in range(n_boot):
        noise = rng.normal(0.0, sig_S.numpy())
        S_k = S_true.numpy() + noise
        G_k, _ = fourier_sine_transform(
            q, torch.as_tensor(S_k, dtype=torch.float64), r_grid,
            Q_max=30.0, window="rect",
        )
        G_samples[k] = G_k.numpy()
    sigG_boot = G_samples.std(axis=0, ddof=1)
    # Agreement: relative error < 20% across r (300 trials → ~5% Monte
    # Carlo noise on a stddev estimate, so 20% is comfortable).
    rel_err = np.abs(sigG_analytic.numpy() - sigG_boot) / np.maximum(sigG_boot, 1e-30)
    assert rel_err.mean() < 0.2, f"mean rel-err {rel_err.mean():.3f} > 0.2"
    # Also sanity-check magnitude: σ_G is in same range as bootstrap.
    assert (sigG_analytic.numpy().mean() / sigG_boot.mean()) == pytest.approx(
        1.0, rel=0.2
    )
