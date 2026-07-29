"""Showcase: the frontier tier.

Left   -- MULTI-MODAL FUSION: X-ray strain alone cannot separate the electronic
          (deformation-potential) and thermal contributions (flat residual
          valley); adding the optical carrier trace pins the deformation
          potential Xi.
Middle -- EQUATION-OF-MOTION DISCOVERY: from a structural trajectory we recover
          v_dot = -omega^2 x - gamma v (a damped phonon) without assuming it.
Right  -- ENSEMBLE HETEROGENEITY: recover the thickness distribution of a
          polydisperse sample from the smeared fringes.

Saves a three-panel figure to dev/paper/figures/.
"""
from __future__ import annotations

import math
import os

import torch

from midas_2d import (
    build_crystal_tensor,
    carrier_density,
    discover_eom,
    fit_multimodal,
    integrate_latent_ode,
    lattice_temperature_from_carriers,
    optical_signal,
    polydisperse_rod,
    recover_thickness_distribution,
    strain_two_channel,
    xray_only_degeneracy,
)

DT = torch.float64


def main(out_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.4))

    # ---- (1) multimodal fusion ----
    t = torch.linspace(0.01, 5, 200, dtype=DT)
    Xi_true, alpha_true = 1.0, 0.5
    n = carrier_density(t, 0.6, 0.1, 1.5)
    T = lattice_temperature_from_carriers(t, n, 1.0)
    eps = strain_two_channel(n, T, Xi_true, alpha_true)
    O = optical_signal(n, 1.0)
    fixed = dict(amp=0.6, tau_rise=0.1, tau_decay=1.5, tau_ep=1.0)
    Xi_grid = torch.linspace(0.2, 1.8, 16, dtype=DT)
    Xg, resid = xray_only_degeneracy(t, eps, Xi_grid, fixed_taus=fixed, steps=300)
    rec = fit_multimodal(O, eps, t, use_optical=True, steps=2000, lr=0.03)

    ax1.plot(Xg.numpy(), resid.numpy(), "k.-", label="X-ray-only residual")
    ax1.axvline(Xi_true, color="0.5", ls=":", label=f"true Xi={Xi_true}")
    ax1.axvline(rec["Xi"], color="r", lw=2, label=f"fusion Xi={rec['Xi']:.2f}")
    ax1.set_xlabel("deformation potential Xi")
    ax1.set_ylabel("X-ray-only fit residual")
    ax1.set_title("X-ray strain localizes Xi\n(optical confirms + pins carrier dynamics)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # ---- (2) EOM discovery ----
    omega_true, gamma_true = 2.0, 0.4
    te = torch.linspace(0, 6, 300, dtype=DT)
    x_obs = integrate_latent_ode(torch.tensor([-omega_true**2, -gamma_true, 0.0], dtype=DT),
                                 1.0, 0.0, te)
    out = discover_eom(x_obs, te, l1=1e-4, steps=2000, lr=0.02)
    x_fit = integrate_latent_ode(torch.tensor([out["x"], out["v"], out["x3"]], dtype=DT),
                                 1.0, 0.0, te)
    ax2.plot(te.numpy(), x_obs.numpy(), "o", ms=3, label="structural trajectory")
    ax2.plot(te.numpy(), x_fit.detach().numpy(), "r-",
             label=f"discovered: w={out['omega']:.2f}, g={out['gamma']:.2f}")
    ax2.set_xlabel("delay"); ax2.set_ylabel("order parameter x(t)")
    ax2.set_title("Equation-of-motion discovery\n(damped phonon, not assumed)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ---- (3) ensemble heterogeneity ----
    ct = build_crystal_tensor()
    l = torch.linspace(0.55, 1.45, 500, dtype=DT)
    n_grid = [3, 4, 5, 6, 7, 8]
    w_true = torch.tensor([0.05, 0.15, 0.5, 0.2, 0.07, 0.03], dtype=DT)
    obs = polydisperse_rod(ct, (1., 1.), n_grid, w_true, l)
    rec_d = recover_thickness_distribution(obs, ct, (1., 1.), n_grid, l, steps=800)
    x = torch.arange(len(n_grid)).numpy()
    ax3.bar(x - 0.18, (w_true / w_true.sum()).numpy(), width=0.35, label="true", color="0.6")
    ax3.bar(x + 0.18, rec_d["weights"].numpy(), width=0.35, label="recovered", color="r", alpha=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels(n_grid)
    ax3.set_xlabel("thickness N3 (cells)"); ax3.set_ylabel("population fraction")
    ax3.set_title("Ensemble thickness distribution\n(from smeared fringes)")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dev", "paper", "figures")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "frontier_multimodal_eom_ensemble.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")
    print(f"fusion Xi={rec['Xi']:.3f} alpha={rec['alpha']:.3f}; "
          f"EOM omega={out['omega']:.3f} gamma={out['gamma']:.3f}")
    return out_path


if __name__ == "__main__":
    main()
