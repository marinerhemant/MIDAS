"""Rev-10 tests: wide-band SAXS analysis primitives."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_pdf.saxs import (
    sphere_form_factor_squared,
    guinier_fit, GuinierFit,
    porod_fit, PorodFit,
    porod_invariant,
    kratky_plot,
    worm_like_chain_form_factor_squared,
)


# ---------------------------------------------------------------------------
# Guinier
# ---------------------------------------------------------------------------

def test_guinier_recovers_sphere_Rg():
    """For a uniform sphere of radius R, R_g = R · √(3/5) exactly."""
    R = 50.0
    Rg_theory = R * np.sqrt(3 / 5)
    q = torch.linspace(0.001, 0.5, 1000, dtype=torch.float64)
    I = sphere_form_factor_squared(q, R)
    g = guinier_fit(q, I, QRg_max=1.0, Rg_initial_A=30.0)
    assert abs(g.Rg_A - Rg_theory) / Rg_theory < 0.02
    assert g.QRg_max <= 1.01


def test_guinier_recovers_I0():
    """I(Q=0) = V² for a sphere."""
    R = 40.0
    V = 4 / 3 * np.pi * R ** 3
    q = torch.linspace(0.001, 0.5, 1000, dtype=torch.float64)
    I = sphere_form_factor_squared(q, R)
    g = guinier_fit(q, I, QRg_max=1.0, Rg_initial_A=30.0)
    assert abs(g.I0 / V ** 2 - 1) < 0.05


def test_guinier_returns_finite_uncertainties():
    R = 30.0
    q = torch.linspace(0.001, 0.5, 500, dtype=torch.float64)
    I = sphere_form_factor_squared(q, R)
    g = guinier_fit(q, I)
    assert np.isfinite(g.Rg_sigma)
    assert np.isfinite(g.I0_sigma)
    assert g.Rg_sigma >= 0
    assert g.I0_sigma >= 0


def test_guinier_with_noise_recovers_within_uncertainty():
    """Noisy data still recovers R_g within a few σ."""
    R = 60.0
    Rg_theory = R * np.sqrt(3 / 5)
    q = torch.linspace(0.001, 0.05, 200, dtype=torch.float64)
    I_clean = sphere_form_factor_squared(q, R)
    rng = torch.Generator().manual_seed(0)
    noise = 0.03 * I_clean * torch.randn(I_clean.shape, generator=rng,
                                            dtype=torch.float64)
    I_noisy = (I_clean + noise).clamp(min=1e-3)
    sig = 0.03 * I_clean.abs()
    g = guinier_fit(q, I_noisy, sigma_I=sig, QRg_max=1.0, Rg_initial_A=30.0)
    # Within 3σ (very loose given the QRg convergence)
    assert abs(g.Rg_A - Rg_theory) < max(3 * g.Rg_sigma, 3.0)


# ---------------------------------------------------------------------------
# Porod
# ---------------------------------------------------------------------------

def test_porod_fit_returns_positive_K_and_uncertainty():
    R = 40.0
    q = torch.linspace(0.001, 1.5, 2000, dtype=torch.float64)
    I = sphere_form_factor_squared(q, R)
    p = porod_fit(q, I, Q_min=0.4)
    assert p.K_porod > 0
    assert p.K_sigma >= 0


def test_porod_fit_rejects_too_few_points():
    q = torch.linspace(0.5, 0.6, 2, dtype=torch.float64)
    I = torch.ones_like(q)
    with pytest.raises(ValueError):
        porod_fit(q, I, Q_min=0.55)


# ---------------------------------------------------------------------------
# Porod invariant
# ---------------------------------------------------------------------------

def test_porod_invariant_positive_on_sphere():
    R = 40.0
    q = torch.linspace(0.001, 1.0, 2000, dtype=torch.float64)
    I = sphere_form_factor_squared(q, R)
    Q_inv = porod_invariant(q, I)
    assert Q_inv > 0


def test_porod_invariant_range_gate():
    """Restricting the integration range gives a smaller invariant."""
    R = 40.0
    q = torch.linspace(0.001, 1.0, 2000, dtype=torch.float64)
    I = sphere_form_factor_squared(q, R)
    full = porod_invariant(q, I)
    partial = porod_invariant(q, I, Q_min=0.1, Q_max=0.5)
    assert 0 < partial < full


# ---------------------------------------------------------------------------
# Kratky plot
# ---------------------------------------------------------------------------

def test_kratky_returns_QsquaredI():
    q = torch.linspace(0.01, 0.5, 20, dtype=torch.float64)
    I = torch.exp(-q ** 2 * 10.0)
    q_out, IQ2 = kratky_plot(q, I)
    assert torch.allclose(q_out, q)
    assert torch.allclose(IQ2, q ** 2 * I)


# ---------------------------------------------------------------------------
# Worm-like chain
# ---------------------------------------------------------------------------

def test_wlc_zero_Q_limit_is_one():
    """|F(Q=0)|² = 1 for the normalised WLC form factor."""
    q = torch.tensor([1e-8, 1e-6, 1e-4], dtype=torch.float64)
    F2 = worm_like_chain_form_factor_squared(q, contour_length_A=500.0,
                                                persistence_length_A=15.0)
    assert torch.all(torch.abs(F2 - 1.0) < 1e-3)


def test_wlc_monotonic_decreasing():
    """|F|² decreases with Q for typical WLC parameters."""
    q = torch.linspace(0.001, 0.5, 200, dtype=torch.float64)
    F2 = worm_like_chain_form_factor_squared(q, contour_length_A=500.0,
                                                persistence_length_A=15.0)
    assert torch.all(torch.diff(F2) < 1e-9)


def test_wlc_differentiable_in_L_and_b():
    q = torch.linspace(0.01, 0.2, 20, dtype=torch.float64)
    L = torch.tensor(500.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
    F2 = worm_like_chain_form_factor_squared(q, L, b)
    F2.sum().backward()
    assert torch.isfinite(L.grad)
    assert torch.isfinite(b.grad)


def test_wlc_stays_positive():
    q = torch.linspace(0.001, 1.0, 200, dtype=torch.float64)
    F2 = worm_like_chain_form_factor_squared(q, contour_length_A=500.0,
                                                persistence_length_A=15.0)
    assert torch.all(F2 >= 0.0)
