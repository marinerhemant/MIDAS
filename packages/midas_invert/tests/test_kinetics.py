"""WS-7: JMAK fitting + rate-law discovery from in-situ HEDM series."""
import pytest
import torch

from midas_invert.kinetics import discover_rate_law, fit_jmak, jmak_fraction, rate_library

DT = torch.float64


@pytest.mark.unit
def test_jmak_fraction_monotonic_0_to_1():
    t = torch.linspace(0, 20, 100, dtype=DT)
    X = jmak_fraction(t, k=0.3, n=2.0)
    assert float(X[0]) == 0.0
    assert float(X[-1]) > 0.99
    assert torch.all(X[1:] >= X[:-1] - 1e-9)


@pytest.mark.autograd
@pytest.mark.parametrize("k_true,n_true", [(0.3, 1.0), (0.5, 2.5), (0.2, 4.0)])
def test_fit_jmak_recovers_k_and_n(k_true, n_true):
    t = torch.linspace(0.05, 25, 120, dtype=DT)
    X = jmak_fraction(t, k=k_true, n=n_true)
    out = fit_jmak(t, X, init_k=0.1, init_n=1.0, steps=3000, lr=0.02)
    assert abs(out["k"] - k_true) / k_true < 0.05
    assert abs(out["n"] - n_true) / n_true < 0.05


@pytest.mark.autograd
def test_fit_jmak_robust_to_noise():
    torch.manual_seed(0)
    t = torch.linspace(0.05, 25, 150, dtype=DT)
    X = jmak_fraction(t, k=0.4, n=2.0)
    X = torch.clamp(X + 0.01 * torch.randn_like(X), 0.0, 1.0)
    out = fit_jmak(t, X, steps=3000, lr=0.02)
    assert abs(out["k"] - 0.4) / 0.4 < 0.12
    assert abs(out["n"] - 2.0) / 2.0 < 0.12


@pytest.mark.unit
def test_discover_first_order_rate_law():
    """First-order X = 1 - exp(-k t): dX/dt = k(1-X) = k - kX -> monomial
    signature {"1": +k, "X": -k}, others ~0."""
    k = 0.5
    t = torch.linspace(0, 12, 400, dtype=DT)
    X = 1.0 - torch.exp(-k * t)
    coeffs = discover_rate_law(X, t)
    assert abs(coeffs["1"] - k) / k < 0.05
    assert abs(coeffs["X"] + k) / k < 0.05      # coefficient is -k
    assert abs(coeffs["X2"]) < 0.02 and abs(coeffs["X3"]) < 0.02


@pytest.mark.unit
def test_discover_autocatalytic_rate_law():
    """Logistic dX/dt = k X(1-X) = kX - kX^2 -> signature {"X": +k, "X2": -k}."""
    k = 0.8
    t = torch.linspace(0, 20, 600, dtype=DT)
    X0 = 0.02
    X = X0 * torch.exp(k * t) / (1 - X0 + X0 * torch.exp(k * t))
    coeffs = discover_rate_law(X, t)
    assert abs(coeffs["X"] - k) / k < 0.08
    assert abs(coeffs["X2"] + k) / k < 0.08     # coefficient is -k
    assert abs(coeffs["1"]) < 0.05 and abs(coeffs["X3"]) < 0.1
