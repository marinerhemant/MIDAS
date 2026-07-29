"""Showcase: transient lattice softening + coherent phonon, from diffraction.

Two ultrafast signatures, both inverted through the differentiable MD-coupled
forward:

Left  -- TRANSIENT SOFTENING: a pump-probe series where the out-of-plane spring
constant k_perp drops (lattice softens) then recovers.  We recover k_perp(t)
straight from the diffraction intensity ratios -- diffraction as a loss on the
*potential*, not just the structure.

Right -- COHERENT PHONON: a damped out-of-plane breathing mode modulates a
Bragg reflection; we recover its frequency and damping from the time series.

Saves a two-panel figure to dev/paper/figures/.
"""
from __future__ import annotations

import math
import os

import torch

from midas_2d import (
    bragg_timeseries,
    cdse_supercell,
    coherent_intensity,
    ensemble_intensity,
    fit_coherent_phonon,
    recover_stiffness,
    strain_wave,
    thermal_ensemble,
)

DT = torch.float64
A = 6.077


def _q(hkl):
    return (2 * math.pi / A) * torch.tensor(hkl, dtype=DT)


def main(out_dir=None, seed=0):
    torch.manual_seed(seed)
    coords, elements, _ = cdse_supercell((4, 4, 4), dtype=DT)
    q_panel = torch.stack([_q([2., 0., 0.]), _q([0., 0., 2.]), _q([0., 0., 4.])])
    I_ref = coherent_intensity(coords, elements, q_panel)

    # ---- transient softening: k_perp dips then recovers -----------------------
    # Keep the whole range in the *identifiable* (soft enough to move
    # measurably) regime: baseline k=25, softening dip to ~8.
    delays = torch.linspace(0.0, 1.0, 5, dtype=DT)
    k_par_true = 25.0
    k_perp_true = 25.0 - 17.0 * torch.exp(-((delays - 0.4) / 0.25) ** 2)

    eps = torch.randn(48, *coords.shape, dtype=DT)
    k_perp_rec = []
    for ti in range(len(delays)):
        frames = thermal_ensemble(coords, torch.tensor(k_par_true, dtype=DT),
                                  k_perp_true[ti], eps=eps)
        obs_ratio = ensemble_intensity(frames, elements, q_panel, coherent=True) / I_ref
        out = recover_stiffness(obs_ratio, coords, elements, q_panel,
                                n_frames=48, steps=500, lr=0.1, seed=1, init_k=20.0)
        k_perp_rec.append(out["k_perp"])
    k_perp_rec = torch.tensor(k_perp_rec)

    # ---- coherent phonon ------------------------------------------------------
    t = torch.linspace(0.0, 3.0, 64, dtype=DT)
    f_true, tau_true, amp_true = 2.0, 1.2, 0.03
    I_phonon = bragg_timeseries(coords, elements, _q([0., 0., 2.]), t,
                                amp_true, f_true, tau_true)
    rec = fit_coherent_phonon(I_phonon, coords, elements, _q([0., 0., 2.]), t,
                              init={"amp": 0.01, "freq": 1.4, "tau": 0.8},
                              steps=1800, lr=0.02)
    I_fit = bragg_timeseries(coords, elements, _q([0., 0., 2.]), t,
                             rec["amp"], rec["freq"], rec["tau"]).detach()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    d = delays.numpy()
    axL.plot(d, k_perp_true.numpy(), "k--", label="planted k_perp(t)")
    axL.plot(d, k_perp_rec.numpy(), "r.-", ms=10, label="recovered k_perp(t)")
    axL.axhline(k_par_true, color="b", ls=":", label="k_par (in-plane)")
    axL.set_xlabel("pump-probe delay (arb.)")
    axL.set_ylabel("out-of-plane spring constant k_perp")
    axL.set_title("Transient lattice softening\n(recovered from diffraction)")
    axL.legend()
    axL.grid(alpha=0.3)

    axR.plot(t.numpy(), (I_phonon / I_phonon[0]).numpy(), "o", ms=4,
             label="measured Bragg I(t)")
    axR.plot(t.numpy(), (I_fit / I_fit[0]).numpy(), "r-",
             label=f"fit: f={rec['freq']:.2f}, tau={rec['tau']:.2f}")
    axR.set_xlabel("pump-probe delay (arb.)")
    axR.set_ylabel("I(t) / I(0)  at (0 0 2)")
    axR.set_title("Coherent phonon\n(frequency + damping recovered)")
    axR.legend()
    axR.grid(alpha=0.3)

    fig.tight_layout()
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dev", "paper", "figures")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stiffness_softening_and_phonon.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")
    print(f"phonon recovered: {rec}")
    return out_path


if __name__ == "__main__":
    main()
