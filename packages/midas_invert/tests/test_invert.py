"""Tests for the shared inversion primitives."""
import math

import pytest
import torch

from midas_invert import (
    ParameterMLP,
    cosine_loss,
    fisher_information,
    fit,
    laplace_uncertainty,
    mixture_deconvolution,
    next_best_measurement,
    rank_measurements,
    relative_l2_loss,
    train_surrogate,
)

DT = torch.float64


@pytest.mark.unit
def test_cosine_loss_scale_invariant_and_zero_at_match():
    a = torch.tensor([1., 2., 4., 7., 11.], dtype=DT)   # asymmetric
    assert float(cosine_loss(a, a)) < 1e-12
    assert float(cosine_loss(5 * a, a)) < 1e-12          # scale-invariant
    assert float(cosine_loss(a, a.flip(0))) > 0.05       # different shape


@pytest.mark.autograd
def test_fit_recovers_scalar_adam_and_lbfgs():
    target = torch.linspace(0, 1, 30, dtype=DT)
    for opt in ("adam", "lbfgs"):
        a = torch.tensor(0.0, dtype=DT, requires_grad=True)
        out = fit([a], lambda: relative_l2_loss(a * target, 3.0 * target),
                  steps=200 if opt == "adam" else 50, lr=0.1 if opt == "adam" else 1.0,
                  optimizer=opt)
        assert abs(float(a) - 3.0) < 1e-2, (opt, float(a))


@pytest.mark.autograd
def test_laplace_uncertainty_quadratic():
    # loss = 0.5 * sum k_i (theta_i - mu_i)^2 -> cov = diag(1/k_i) (noise_var=1, NLL form)
    k = torch.tensor([4.0, 1.0], dtype=DT)
    mu = torch.tensor([1.0, -2.0], dtype=DT)

    def nll(t):
        return 0.5 * (k * (t - mu) ** 2).sum()

    res = laplace_uncertainty(nll, mu)
    assert torch.allclose(res["std"], torch.sqrt(1.0 / k), atol=1e-6)


@pytest.mark.unit
def test_mixture_deconvolution_recovers_weights():
    # basis: 4 distinct bell components; observed = known mixture
    x = torch.linspace(-3, 3, 80, dtype=DT)
    centers = [-2.0, -0.5, 1.0, 2.5]
    basis = torch.stack([torch.exp(-0.5 * ((x - c) / 0.4) ** 2) for c in centers])
    w_true = torch.tensor([0.1, 0.5, 0.3, 0.1], dtype=DT)
    obs = w_true @ basis
    w = mixture_deconvolution(obs, basis, loss="rel_l2", steps=1500, lr=0.1)
    assert int(torch.argmax(w)) == int(torch.argmax(w_true))
    assert torch.allclose(w, w_true, atol=0.06)


@pytest.mark.unit
def test_fisher_information_ranks_and_picks_best():
    t = torch.linspace(0, 4, 40, dtype=DT)
    forward = lambda w: torch.sin(w * t)
    fi = fisher_information(forward, theta=2.0)
    assert fi.shape == t.shape and float(fi[0]) < float(fi[20])
    ranked = rank_measurements(forward, theta=2.0)
    assert [r[1] for r in ranked] == sorted([r[1] for r in ranked], reverse=True)
    assert next_best_measurement(forward, theta=2.0)[1] >= float(fi.max()) - 1e-9


@pytest.mark.unit
def test_surrogate_learns_linear_map():
    torch.manual_seed(0)
    X = torch.randn(400, 6, dtype=DT)
    W = torch.randn(6, 2, dtype=DT)
    Y = X @ W
    model, info = train_surrogate(X, Y, epochs=400, lr=5e-3, seed=0)
    assert float(info["val_mae"].mean()) < 0.1


@pytest.mark.unit
def test_information_matrix_and_greedy_design():
    from midas_invert import greedy_optimal_design, information_matrix
    # two parameters; candidate measurements with different sensitivity directions
    t = torch.linspace(0.2, 4, 12, dtype=DT)

    def forward(theta):   # theta = [a, b]; pred_m = a*sin(t_m) + b*cos(t_m)
        return theta[0] * torch.sin(t) + theta[1] * torch.cos(t)

    F = information_matrix(forward, torch.tensor([1.0, 1.0], dtype=DT))
    assert F.shape == (2, 2)
    assert torch.linalg.slogdet(F)[0] > 0          # positive-definite
    chosen = greedy_optimal_design(forward, torch.tensor([1.0, 1.0], dtype=DT), k=4)
    assert len(chosen) == 4 and len(set(i for i, _ in chosen)) == 4   # distinct


@pytest.mark.unit
def test_discover_dynamics_damped_oscillator():
    from midas_invert import discover_dynamics, integrate_latent_ode
    omega, gamma = 2.0, 0.4
    t = torch.linspace(0, 6, 600, dtype=DT)
    x = integrate_latent_ode(torch.tensor([-omega ** 2, -gamma, 0.0], dtype=DT), 1.0, 0.0, t)
    out = discover_dynamics(x, t, threshold=0.05)
    assert abs(out["omega"] - omega) / omega < 0.05
    assert abs(out["gamma"] - gamma) / gamma < 0.1
    assert out["x3"] == 0.0


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_device_fit(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if device == "mps" and not (hasattr(torch.backends, "mps")
                                and torch.backends.mps.is_available()):
        pytest.skip("no MPS")
    dt = torch.float32 if device == "mps" else DT
    a = torch.tensor(0.0, dtype=dt, device=device, requires_grad=True)
    tgt = torch.ones(10, dtype=dt, device=device)
    out = fit([a], lambda: ((a - 2.0) ** 2).mean() + 0 * tgt.sum(), steps=200, lr=0.1)
    assert abs(float(a) - 2.0) < 1e-2
