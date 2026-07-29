"""Frontier tier: multimodal fusion, EOM discovery, ensemble, active learning."""
import math

import pytest
import torch

from midas_2d import (
    build_crystal_tensor,
    carrier_density,
    discover_eom,
    fisher_information,
    fit_multimodal,
    integrate_latent_ode,
    lattice_temperature_from_carriers,
    next_best_measurement,
    polydisperse_rod,
    rank_measurements,
    recover_thickness_distribution,
    strain_two_channel,
    xray_only_degeneracy,
)

DT = torch.float64


# --------------------------------------------------------- multimodal fusion

def _make_multimodal_truth(t, Xi, alpha, *, amp=0.6, tr=0.1, td=1.5, tep=1.0):
    n = carrier_density(t, amp, tr, td)
    T = lattice_temperature_from_carriers(t, n, tep)
    eps = strain_two_channel(n, T, Xi, alpha)
    return n, eps


@pytest.mark.unit
def test_xray_residual_localizes_Xi_when_timescales_separate():
    """When the electronic (fast spike) and thermal (slow plateau) strain
    contributions are well separated in time, the X-ray strain shape alone
    already localizes the deformation potential Xi (sharp residual minimum at
    the true value).  The optical channel then confirms it and independently
    pins the carrier dynamics."""
    t = torch.linspace(0.01, 5, 200, dtype=DT)
    Xi_true, alpha_true = 1.0, 0.5
    n, eps = _make_multimodal_truth(t, Xi_true, alpha_true)
    fixed = dict(amp=0.6, tau_rise=0.1, tau_decay=1.5, tau_ep=1.0)
    Xi_grid = torch.linspace(0.2, 1.8, 12, dtype=DT)
    Xg, resid = xray_only_degeneracy(t, eps, Xi_grid, fixed_taus=fixed, steps=250)
    Xi_best = float(Xg[int(torch.argmin(resid))])
    assert abs(Xi_best - Xi_true) < 0.2


@pytest.mark.autograd
@pytest.mark.slow
def test_multimodal_recovers_deformation_potential():
    """Joint optical + X-ray inversion pins Xi and alpha."""
    t = torch.linspace(0.01, 5, 200, dtype=DT)
    Xi_true, alpha_true = 1.0, 0.5
    n, eps = _make_multimodal_truth(t, Xi_true, alpha_true)
    from midas_2d import optical_signal
    O = optical_signal(n, 1.0)

    rec = fit_multimodal(O, eps, t, use_optical=True, steps=2500, lr=0.03)
    assert abs(rec["Xi"] - Xi_true) < 0.15
    assert abs(rec["alpha"] - alpha_true) < 0.15


# ---------------------------------------------------------- EOM discovery

@pytest.mark.unit
def test_discover_damped_oscillator():
    """Recover a damped-oscillator EOM v_dot = -omega^2 x - gamma v from a
    trajectory by sparse regression (SINDy), without assuming it -- and the
    spurious cubic term is thresholded to exactly zero."""
    omega_true, gamma_true = 2.0, 0.4
    t = torch.linspace(0, 6, 600, dtype=DT)
    coeffs_true = torch.tensor([-omega_true ** 2, -gamma_true, 0.0], dtype=DT)
    x_obs = integrate_latent_ode(coeffs_true, x0=1.0, v0=0.0, t=t)

    out = discover_eom(x_obs, t, threshold=0.05)
    assert abs(out["omega"] - omega_true) / omega_true < 0.05
    assert abs(out["gamma"] - gamma_true) / gamma_true < 0.1
    assert out["x3"] == 0.0          # thresholded out -> sparse, correct equation


@pytest.mark.unit
def test_latent_ode_energy_decay_under_damping():
    t = torch.linspace(0, 10, 400, dtype=DT)
    coeffs = torch.tensor([-4.0, -0.5, 0.0], dtype=DT)   # omega^2=4, gamma=0.5
    x = integrate_latent_ode(coeffs, x0=1.0, v0=0.0, t=t)
    # amplitude decays
    assert x[:50].abs().max() > x[-50:].abs().max()


# ------------------------------------------------------- ensemble heterogeneity

@pytest.mark.unit
def test_polydisperse_smears_fringes():
    """A spread of thicknesses washes out the fringe minima vs a single N."""
    ct = build_crystal_tensor()
    l = torch.linspace(0.6, 1.4, 400, dtype=DT)
    n_grid = [3, 4, 5, 6, 7]
    mono = polydisperse_rod(ct, (1., 1.), [5], torch.tensor([1.0], dtype=DT), l)
    poly = polydisperse_rod(ct, (1., 1.), n_grid,
                            torch.tensor([1., 1., 1., 1., 1.], dtype=DT), l)
    # fringe contrast (min/max) is higher (deeper minima) for the mono sample
    def contrast(I):
        return float(I.min() / I.max())
    assert contrast(poly) > contrast(mono)


@pytest.mark.autograd
@pytest.mark.slow
def test_recover_thickness_distribution():
    """Recover a peaked thickness distribution from the smeared rod."""
    ct = build_crystal_tensor()
    l = torch.linspace(0.55, 1.45, 500, dtype=DT)
    n_grid = [3, 4, 5, 6, 7, 8]
    w_true = torch.tensor([0.05, 0.15, 0.5, 0.2, 0.07, 0.03], dtype=DT)
    obs = polydisperse_rod(ct, (1., 1.), n_grid, w_true, l)

    out = recover_thickness_distribution(obs, ct, (1., 1.), n_grid, l, steps=800)
    w_rec = out["weights"]
    # the recovered distribution peaks at the same thickness (N=5, index 2)
    assert int(torch.argmax(w_rec)) == int(torch.argmax(w_true))
    # and the mean thickness is close
    grid = torch.tensor(n_grid, dtype=DT)
    mean_true = float((w_true / w_true.sum() * grid).sum())
    mean_rec = float((w_rec * grid).sum())
    assert abs(mean_rec - mean_true) < 0.4


# ------------------------------------------------------------- active learning

@pytest.mark.unit
def test_fisher_information_ranks_informative_measurements():
    """For recovering an oscillation frequency, the most informative delays are
    where d(signal)/d(omega) is largest (late times, where phase error
    accumulates) -- not t=0."""
    t = torch.linspace(0, 4, 50, dtype=DT)

    def forward(omega):
        return torch.sin(omega * t)        # signal vs candidate delays

    fi = fisher_information(forward, theta=2.0)
    assert fi.shape == t.shape and torch.isfinite(fi).all()
    # t=0 carries no frequency information; a later delay carries more
    assert float(fi[0]) < float(fi[25])
    best = next_best_measurement(forward, theta=2.0)
    assert best[1] >= float(fi.max()) - 1e-9


@pytest.mark.unit
def test_rank_measurements_orders_by_information():
    t = torch.linspace(0.1, 4, 20, dtype=DT)
    forward = lambda w: torch.sin(w * t)
    ranked = rank_measurements(forward, theta=1.5)
    fis = [r[1] for r in ranked]
    assert fis == sorted(fis, reverse=True)
