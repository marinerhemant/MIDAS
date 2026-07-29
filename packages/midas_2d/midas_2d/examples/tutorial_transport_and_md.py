"""Showcase: diffraction -> transport/coupling coefficients, and learning the
potential by differentiating molecular dynamics.

Three physical *numbers* pulled out of diffraction signals:

Left   -- ELECTRON-PHONON COUPLING g: a two-temperature model (carriers heat,
          then transfer to the lattice at rate g) drives the Bragg amplitude;
          we recover g from the intensity transient.
Middle -- THERMAL DIFFUSIVITY kappa: a surface-deposited heat pulse spreads with
          diffusivity kappa, seen as a depth-resolved strain front; recovered
          from the strain(z,t) movie.
Right  -- INTERATOMIC STIFFNESS k: a coherent phonon rings the lattice; we
          differentiate the MD trajectory to recover the spring constant from
          the Bragg-intensity oscillation.

Saves a three-panel figure to dev/paper/figures/.
"""
from __future__ import annotations

import math
import os

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


def main(out_dir=None, seed=0):
    torch.manual_seed(seed)

    # ---- (1) electron-phonon coupling g --------------------------------------
    t = torch.linspace(0, 4, 300, dtype=DT)
    g_true = 2.5
    Te, Tl = two_temperature_model(t, g=g_true, C_e=1.0, C_l=3.0, pump_amp=1.0)
    obs_ratio = lattice_T_to_intensity_ratio(Tl, q_perp=2.0, k_spring=30.0)
    rec_g = fit_electron_phonon_coupling(obs_ratio, t, q_perp=2.0, k_spring=30.0,
                                         init_g=0.8, steps=1200, lr=0.05)
    _, Tl_fit = two_temperature_model(t, g=rec_g["g"], C_e=1.0, C_l=3.0, pump_amp=rec_g["pump_amp"])
    fit_ratio = lattice_T_to_intensity_ratio(Tl_fit, q_perp=2.0, k_spring=30.0)

    # ---- (2) thermal diffusivity kappa ---------------------------------------
    Nz = 40
    z = torch.arange(Nz, dtype=DT)
    T0 = torch.zeros(Nz, dtype=DT); T0[-5:] = 1.0
    kappa_true = 0.35
    Tzt = heat_diffusion_1d(T0, kappa=kappa_true, dz=1.0, dt=0.2, n_steps=120)
    obs_strain = 1e-3 * Tzt
    rec_k = fit_thermal_diffusivity(obs_strain, z, T0, alpha=1e-3, dt=0.2,
                                    n_steps=120, init_kappa=0.1, steps=300, lr=0.1)

    # ---- (3) interatomic stiffness from a diffraction movie ------------------
    from midas_2d.md_integrator import coherent_mode_kick
    coords, elements, _ = cdse_supercell((3, 3, 4), dtype=DT)
    k_true = 8.0
    r0 = coherent_mode_kick(coords, 0.04)   # non-uniform standing-wave mode
    v0 = torch.zeros_like(r0)
    force = lambda r: harmonic_force(r, coords, torch.tensor([0., 0., k_true], dtype=DT))
    dt_md, n_md = 0.02, 300
    traj = velocity_verlet(r0, v0, force, dt=dt_md, n_steps=n_md)
    I_md = bragg_from_trajectory(traj, elements, _q([0., 0., 2.]))
    rec_pot = recover_potential_from_movie(I_md, coords, elements, _q([0., 0., 2.]),
                                           amp0=0.04, dt=dt_md, n_steps=n_md,
                                           init_k=3.0, steps=250, lr=0.1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.4))

    ax1.plot(t.numpy(), Te.numpy(), label="electron T_e(t)", color="tab:orange")
    ax1.plot(t.numpy(), Tl.numpy(), label="lattice T_l(t)", color="tab:red")
    ax1b = ax1.twinx()
    ax1b.plot(t.numpy(), obs_ratio.numpy(), "k.", ms=3, label="measured I/I0")
    ax1b.plot(t.numpy(), fit_ratio.detach().numpy(), "b-", lw=1,
              label=f"fit: g={rec_g['g']:.2f} (true {g_true})")
    ax1.set_xlabel("delay"); ax1.set_ylabel("temperature")
    ax1b.set_ylabel("Bragg I/I0")
    ax1.set_title("Electron-phonon coupling g\n(recovered from amplitude transient)")
    ax1.legend(loc="upper left", fontsize=8); ax1b.legend(loc="right", fontsize=8)

    im = ax2.imshow(obs_strain.detach().numpy(), aspect="auto", origin="lower",
                    cmap="inferno", extent=[0, Nz, 0, 120 * 0.2])
    ax2.set_xlabel("depth z"); ax2.set_ylabel("delay")
    ax2.set_title(f"Heat front -> diffusivity kappa\nrecovered {rec_k['kappa']:.3f} (true {kappa_true})")
    fig.colorbar(im, ax=ax2, label="strain eps(z,t)")

    tt = (torch.arange(n_md + 1) * dt_md).numpy()
    ax3.plot(tt, (I_md / I_md[0]).numpy(), "o", ms=3, label="Bragg I(t) movie")
    rec_force = lambda r: harmonic_force(r, coords, torch.tensor([0., 0., rec_pot["k_perp"]], dtype=DT))
    traj_fit = velocity_verlet(r0, v0, rec_force, dt=dt_md, n_steps=n_md)
    I_fit = bragg_from_trajectory(traj_fit, elements, _q([0., 0., 2.])).detach()
    ax3.plot(tt, (I_fit / I_fit[0]).numpy(), "r-", lw=1,
             label=f"MD fit: k={rec_pot['k_perp']:.1f} (true {k_true})")
    ax3.set_xlabel("delay"); ax3.set_ylabel("Bragg I/I0 at (0 0 2)")
    ax3.set_title("Interatomic stiffness from diffraction\n(by differentiating the MD)")
    ax3.legend(fontsize=8)

    fig.tight_layout()
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dev", "paper", "figures")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "transport_coefficients_and_md.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")
    print(f"g={rec_g['g']:.3f} kappa={rec_k['kappa']:.3f} k_md={rec_pot['k_perp']:.3f}")
    return out_path


if __name__ == "__main__":
    main()
