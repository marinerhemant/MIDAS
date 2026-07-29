"""Rev-15 Bayesian model comparison (WAIC / LOO) tests."""
from __future__ import annotations

import math

import pytest
import torch

from midas_pdf.model_comparison import (
    compare_models,
    gaussian_pointwise_loglik,
    loo,
    waic,
)


# ---------------------------------------------------------------------------
# Pointwise Gaussian log-likelihood
# ---------------------------------------------------------------------------

def test_gaussian_loglik_shape():
    obs = torch.zeros(10, dtype=torch.float64)
    sigma = torch.full((10,), 0.5, dtype=torch.float64)
    mu = torch.randn(50, 10, dtype=torch.float64)
    ll = gaussian_pointwise_loglik(obs, sigma, mu)
    assert ll.shape == (50, 10)


def test_gaussian_loglik_matches_analytic():
    """Single-point single-sample: -0.5*log(2πσ²) - 0.5*(y-μ)²/σ²."""
    obs = torch.tensor([1.0], dtype=torch.float64)
    sigma = torch.tensor([0.5], dtype=torch.float64)
    mu = torch.tensor([[0.7]], dtype=torch.float64)                # (1,1)
    ll = gaussian_pointwise_loglik(obs, sigma, mu)
    expected = (-0.5 * math.log(2 * math.pi * 0.25)
                - 0.5 * (0.3 / 0.5) ** 2)
    assert abs(float(ll[0, 0]) - expected) < 1e-10


def test_gaussian_loglik_shape_mismatch_raises():
    obs = torch.zeros(10, dtype=torch.float64)
    sigma = torch.full((10,), 0.5, dtype=torch.float64)
    bad = torch.zeros(50, 5, dtype=torch.float64)                   # wrong N
    with pytest.raises(ValueError, match="mu_samples"):
        gaussian_pointwise_loglik(obs, sigma, bad)


# ---------------------------------------------------------------------------
# WAIC
# ---------------------------------------------------------------------------

def test_waic_perfect_fit_gives_finite_positive_value():
    """No noise, zero-variance sample stack: p_waic → 0, lppd finite."""
    obs = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    sigma = torch.full((3,), 0.5, dtype=torch.float64)
    mu = obs.unsqueeze(0).repeat(50, 1)                            # perfect
    res = waic(obs, sigma, mu)
    assert torch.isfinite(torch.tensor([res.value, res.se, res.lppd,
                                        res.p_eff])).all()
    assert res.p_eff == pytest.approx(0.0, abs=1e-9)
    # -2 * lppd for a perfect fit: -2 * 3 * (-0.5 log(2π·0.25)) = ...
    expected_lppd = 3 * (-0.5 * math.log(2 * math.pi * 0.25))
    assert abs(res.value - (-2.0 * expected_lppd)) < 1e-8


def test_waic_better_model_has_lower_value():
    """Model A fits, model B misses badly.  A's WAIC < B's WAIC."""
    torch.manual_seed(42)
    obs = torch.linspace(0.0, 1.0, 30, dtype=torch.float64)
    sigma = torch.full_like(obs, 0.05)
    mu_good = obs.unsqueeze(0) + 0.02 * torch.randn(80, 30, dtype=torch.float64)
    mu_bad  = torch.randn(80, 30, dtype=torch.float64) * 2.0        # random noise
    a = waic(obs, sigma, mu_good, name="good")
    b = waic(obs, sigma, mu_bad,  name="bad")
    assert a.value < b.value


def test_waic_accepts_precomputed_log_lik():
    torch.manual_seed(0)
    log_lik = torch.randn(50, 20, dtype=torch.float64) * 0.1 - 1.0
    res = waic(log_lik=log_lik)
    assert torch.isfinite(torch.tensor([res.value, res.se])).all()


def test_waic_missing_inputs_raises():
    with pytest.raises(ValueError, match="log_lik"):
        waic()


# ---------------------------------------------------------------------------
# LOO
# ---------------------------------------------------------------------------

def test_loo_shape_and_finite():
    torch.manual_seed(0)
    obs = torch.linspace(0.0, 1.0, 20, dtype=torch.float64)
    sigma = torch.full_like(obs, 0.05)
    mu = obs.unsqueeze(0) + 0.01 * torch.randn(60, 20, dtype=torch.float64)
    res = loo(obs, sigma, mu)
    assert res.pointwise.shape == (20,)
    assert torch.isfinite(torch.tensor([res.value, res.se, res.p_eff])).all()


def test_loo_close_to_waic_for_well_behaved():
    """WAIC and LOO are asymptotically equal; for a good, well-mixed sample
    stack they should agree to within a few percent."""
    torch.manual_seed(1)
    obs = torch.linspace(0.0, 1.0, 40, dtype=torch.float64)
    sigma = torch.full_like(obs, 0.05)
    mu = obs.unsqueeze(0) + 0.02 * torch.randn(200, 40, dtype=torch.float64)
    w = waic(obs, sigma, mu)
    l = loo(obs, sigma, mu)
    rel = abs(w.value - l.value) / max(abs(w.value), 1.0)
    assert rel < 0.2, f"WAIC {w.value} vs LOO {l.value} diverged"


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

def test_compare_two_models_picks_lower_waic():
    torch.manual_seed(3)
    obs = torch.linspace(0.0, 1.0, 30, dtype=torch.float64)
    sigma = torch.full_like(obs, 0.05)
    mu_good = obs.unsqueeze(0) + 0.02 * torch.randn(80, 30, dtype=torch.float64)
    mu_bad  = obs.unsqueeze(0) + 0.20 * torch.randn(80, 30, dtype=torch.float64)
    good = waic(obs, sigma, mu_good, name="two_phase")
    bad  = waic(obs, sigma, mu_bad,  name="one_phase")
    comp = compare_models({"one_phase": bad, "two_phase": good})
    assert comp.winner == "two_phase"
    assert comp.delta >= 0.0
    assert comp.se_delta > 0.0


def test_compare_models_requires_two():
    torch.manual_seed(0)
    res = waic(log_lik=torch.randn(5, 5, dtype=torch.float64))
    with pytest.raises(ValueError, match="at least two"):
        compare_models({"only": res})


def test_compare_models_shape_mismatch_raises():
    a = waic(log_lik=torch.randn(5, 10, dtype=torch.float64))
    b = waic(log_lik=torch.randn(5, 8, dtype=torch.float64))
    with pytest.raises(ValueError, match="pointwise shapes"):
        compare_models({"a": a, "b": b})
