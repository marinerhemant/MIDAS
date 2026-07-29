"""Two-temperature (electron-phonon g), heat diffusion (kappa), differentiable MD."""
import math

import pytest
import torch

from midas_2d import (
    bragg_from_trajectory,
    cdse_supercell,
    fit_electron_phonon_coupling,
    fit_thermal_diffusivity,
    harmonic_force,
    heat_diffusion_1d,
    lattice_T_to_intensity_ratio,
    recover_potential_from_movie,
    two_temperature_model,
    velocity_verlet,
)

DT = torch.float64
A = 6.077


def _q(hkl):
    return (2 * math.pi / A) * torch.tensor(hkl, dtype=DT)


# ----------------------------------------------------- two-temperature model

@pytest.mark.unit
def test_ttm_energy_flows_electrons_to_lattice():
    t = torch.linspace(0, 4, 400, dtype=DT)
    Te, Tl = two_temperature_model(t, g=2.0, C_e=1.0, C_l=3.0, pump_amp=1.0)
    # electrons heat first and peak higher; lattice rises later and they
    # equilibrate (Te approaches Tl by the end)
    assert int(torch.argmax(Te)) < int(torch.argmax(Tl))
    assert Te.max() > Tl.max()
    assert abs(float(Te[-1] - Tl[-1])) < 0.1 * float(Tl[-1] + 1e-9) + 1e-3


@pytest.mark.unit
def test_stronger_coupling_heats_lattice_sooner():
    t = torch.linspace(0, 4, 400, dtype=DT)
    _, Tl_weak = two_temperature_model(t, g=0.5, pump_amp=1.0)
    _, Tl_strong = two_temperature_model(t, g=4.0, pump_amp=1.0)
    # at an early-mid delay, stronger coupling has dumped more energy into the
    # lattice (robust to the exact final values)
    i_mid = 150  # t ~ 1.5
    assert float(Tl_strong[i_mid]) > float(Tl_weak[i_mid])


@pytest.mark.autograd
@pytest.mark.slow
def test_recover_electron_phonon_coupling():
    t = torch.linspace(0, 4, 300, dtype=DT)
    g_true = 2.5
    _, Tl = two_temperature_model(t, g=g_true, C_e=1.0, C_l=3.0, pump_amp=1.0)
    obs = lattice_T_to_intensity_ratio(Tl, q_perp=2.0, k_spring=30.0)

    out = fit_electron_phonon_coupling(obs, t, q_perp=2.0, k_spring=30.0,
                                       C_e=1.0, C_l=3.0, init_g=0.8,
                                       steps=1200, lr=0.05)
    assert abs(out["g"] - g_true) / g_true < 0.15


# --------------------------------------------------------------- heat diffusion

@pytest.mark.unit
def test_heat_diffusion_conserves_and_spreads():
    Nz = 40
    T0 = torch.zeros(Nz, dtype=DT)
    T0[-5:] = 1.0                         # hot surface
    T = heat_diffusion_1d(T0, kappa=0.3, dz=1.0, dt=0.2, n_steps=200)
    # insulated boundaries conserve total heat; the front spreads inward
    assert abs(float(T[-1].sum() - T0.sum())) < 1e-6
    assert float(T[-1, 0]) > float(T0[0])    # heat reached the substrate side
    assert float(T[-1].max()) < float(T0.max())  # peak diffused down


@pytest.mark.autograd
def test_recover_thermal_diffusivity():
    Nz = 40
    z = torch.arange(Nz, dtype=DT)
    T0 = torch.zeros(Nz, dtype=DT); T0[-5:] = 1.0
    kappa_true = 0.35
    T = heat_diffusion_1d(T0, kappa=kappa_true, dz=1.0, dt=0.2, n_steps=120)
    obs_strain = 1e-3 * T                   # eps = alpha * T

    out = fit_thermal_diffusivity(obs_strain, z, T0, alpha=1e-3, dt=0.2,
                                  n_steps=120, init_kappa=0.1, steps=300, lr=0.1)
    assert abs(out["kappa"] - kappa_true) / kappa_true < 0.1


# ----------------------------------------------------------- differentiable MD

@pytest.mark.unit
def test_velocity_verlet_conserves_energy_harmonic():
    torch.manual_seed(0)
    r_eq = torch.zeros(8, 3, dtype=DT)
    r0 = r_eq.clone(); r0[:, 2] += 0.1
    v0 = torch.zeros_like(r0)
    k = torch.tensor([0.0, 0.0, 4.0], dtype=DT)
    force = lambda r: harmonic_force(r, r_eq, k)
    traj = velocity_verlet(r0, v0, force, dt=0.02, n_steps=500)
    # energy = 1/2 m v^2 + 1/2 k x^2 conserved; check oscillation amplitude stays bounded
    zmax = traj[:, :, 2].abs().max()
    assert abs(float(zmax) - 0.1) < 0.02            # no blow-up / strong drift


@pytest.mark.unit
def test_md_oscillation_frequency_matches_sqrt_k_over_m():
    r_eq = torch.zeros(4, 3, dtype=DT)
    r0 = r_eq.clone(); r0[:, 2] += 0.05
    v0 = torch.zeros_like(r0)
    k_perp = 9.0
    force = lambda r: harmonic_force(r, r_eq, torch.tensor([0., 0., k_perp], dtype=DT))
    dt, n = 0.01, 2000
    traj = velocity_verlet(r0, v0, force, dt=dt, n_steps=n, mass=1.0)
    z = traj[:, 0, 2]
    # period from zero-crossings; omega = sqrt(k/m)
    sign = torch.sign(z - z.mean())
    crossings = torch.where(sign[1:] != sign[:-1])[0]
    # two crossings per period
    period = 2 * float(crossings.diff().float().mean()) * dt
    omega_meas = 2 * math.pi / period
    assert abs(omega_meas - math.sqrt(k_perp)) / math.sqrt(k_perp) < 0.05


@pytest.mark.autograd
@pytest.mark.slow
def test_learn_potential_from_diffraction_movie():
    """Recover the spring constant by differentiating an MD trajectory to match
    the Bragg-intensity oscillation."""
    from midas_2d.md_integrator import coherent_mode_kick
    coords, elements, _ = cdse_supercell((3, 3, 4), dtype=DT)
    k_true = 8.0
    r0 = coherent_mode_kick(coords, 0.04)   # non-uniform -> visible in |A|^2
    v0 = torch.zeros_like(r0)
    force = lambda r: harmonic_force(r, coords, torch.tensor([0., 0., k_true], dtype=DT))
    dt, n = 0.02, 300
    traj = velocity_verlet(r0, v0, force, dt=dt, n_steps=n)
    obs = bragg_from_trajectory(traj, elements, _q([0., 0., 2.]))

    out = recover_potential_from_movie(obs, coords, elements, _q([0., 0., 2.]),
                                       amp0=0.04, dt=dt, n_steps=n, init_k=3.0,
                                       steps=250, lr=0.1)
    assert abs(out["k_perp"] - k_true) / k_true < 0.15
