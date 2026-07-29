"""Showcase: MD-coupled anisotropic transient disordering of a CdSe nanoplatelet.

This is the novel end-to-end story for the Schaller/Flanders line:

1. Build an explicit few-layer CdSe platelet (atoms, not a unit cell).
2. Simulate a pump-probe delay series as MD-like ensembles whose *out-of-plane*
   displacement grows with delay while the in-plane stays fixed -- exactly the
   anisotropic disordering they measure.
3. Compute diffraction DIRECTLY FROM THE ATOMS with the differentiable engine
   (no Debye-Waller assumed): out-of-plane reflections decay faster than
   in-plane ones -- the anisotropic signature emerges from the coordinates.
4. Invert: recover u_perp(t) and u_par(t) by gradient descent and show they
   match both the planted values and the MSD tensor read straight off the MD
   frames. The whole chain MD -> diffraction -> parameters is differentiable.

Saves a two-panel figure to dev/paper/figures/.
"""
from __future__ import annotations

import math
import os

import torch

from midas_2d import (
    TransientMSD,
    cdse_supercell,
    cosine_loss,
    dwf_amplitude,
    ensemble_intensity,
    fit,
    msd_tensor_from_frames,
)

DT = torch.float64
A = 6.077


def _q(hkl):
    return (2.0 * math.pi / A) * torch.tensor(hkl, dtype=DT)


def main(out_dir=None, seed=0, n_frames=48):
    torch.manual_seed(seed)
    coords, elements, _ = cdse_supercell((6, 6, 4), dtype=DT)

    # Pump-probe delays: out-of-plane sigma_z rises, in-plane sigma_xy fixed.
    delays = torch.linspace(0.0, 1.0, 6, dtype=DT)          # arbitrary delay units
    sigma_xy = 0.06                                          # Angstrom
    sigma_z = 0.05 + 0.22 * delays                           # grows with delay

    # A panel of reflections: in-plane (h00) vs out-of-plane (00l) character.
    # Use ALLOWED zinc-blende reflections only (all-same-parity); (0 0 3) is
    # mixed-parity / forbidden and shows pure thermal-diffuse scattering, which
    # would mislead the inversion.
    refl = {
        "in-plane (2 0 0)": _q([2.0, 0.0, 0.0]),
        "out-of-plane (0 0 2)": _q([0.0, 0.0, 2.0]),
        "out-of-plane (0 0 4)": _q([0.0, 0.0, 4.0]),
    }
    q_all = torch.stack(list(refl.values()))                # (R, 3)

    # Forward: frame-averaged coherent intensity per delay, per reflection.
    I_meas = torch.zeros(len(delays), q_all.shape[0], dtype=DT)
    U_md = torch.zeros(len(delays), 3, dtype=DT)            # MD-derived MSD diag
    for ti, _d in enumerate(delays):
        sig = torch.tensor([sigma_xy, sigma_xy, float(sigma_z[ti])], dtype=DT)
        frames = coords[None] + sig * torch.randn(n_frames, *coords.shape, dtype=DT)
        I_meas[ti] = ensemble_intensity(frames, elements, q_all, coherent=True)
        U_md[ti] = torch.diag(msd_tensor_from_frames(frames))

    I_norm = I_meas / I_meas[0]                             # relative to t=0

    # ---- Inversion -----------------------------------------------------------
    # The intensity ratio R_r(t) = I_r(t)/I_r(0) = exp(-q . dU(t) . q) depends
    # ONLY on the change dU(t) = U(t) - U(0).  So we fit dU(t) (>= 0, softplus),
    # which is gauge-free, then add the known t=0 baseline for absolute MSDs.
    u_par0 = sigma_xy ** 2
    u_perp0 = 0.05 ** 2
    dmsd = TransientMSD(len(delays), u_par0=1e-5, u_perp0=1e-5)

    def loss_fn():
        # dwf_amplitude(q, du)^2 == exp(-q . du . q) == the intensity ratio.
        R = torch.stack([dmsd.amplitude(q_all, t) ** 2 for t in range(len(delays))])
        return ((R - I_norm) ** 2).sum() / (I_norm.pow(2).sum())

    fit(dmsd.parameters(), loss_fn, steps=2000, lr=0.02)
    u_perp_rec = (u_perp0 + dmsd.u_perp).detach()
    u_par_rec = (u_par0 + dmsd.u_par).detach()

    # ---------------------------------------------------------------- plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = delays.numpy()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    for j, name in enumerate(refl):
        axL.plot(d, I_norm[:, j].numpy(), "o-", label=name)
    axL.set_xlabel("pump-probe delay (arb.)")
    axL.set_ylabel("I(t) / I(0)")
    axL.set_title("Anisotropic transient disordering\n(emerges from the atoms, no DWF assumed)")
    axL.legend()
    axL.grid(alpha=0.3)

    axR.plot(d, (sigma_z.numpy() ** 2), "k--", label="planted u_perp(t)")
    axR.plot(d, U_md[:, 2].numpy(), "s", ms=7, mfc="none", label="MD-derived u_zz(t)")
    axR.plot(d, u_perp_rec.numpy(), "r.-", label="recovered u_perp(t)")
    axR.plot(d, u_par_rec.numpy(), "b.-", label="recovered u_par(t)")
    axR.axhline(sigma_xy ** 2, color="b", ls=":", alpha=0.6, label="planted u_par")
    axR.set_xlabel("pump-probe delay (arb.)")
    axR.set_ylabel("mean-square displacement (A^2)")
    axR.set_title("Differentiable inversion recovers the MSD\n(planted = MD-derived = recovered)")
    axR.legend(fontsize=8)
    axR.grid(alpha=0.3)

    fig.tight_layout()
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dev", "paper", "figures")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "md_transient_anisotropic_disorder.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")
    print(f"recovered u_perp(t): {u_perp_rec.tolist()}")
    print(f"recovered u_par(t):  {u_par_rec.tolist()}")
    return out_path


if __name__ == "__main__":
    main()
