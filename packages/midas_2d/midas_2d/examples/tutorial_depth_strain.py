"""Showcase: depth-resolved strain reconstruction (the d-spacing-vs-depth signal).

A laser-excited film carries a strain gradient through its thickness: the
out-of-plane d-spacing varies with depth, making the Bragg peak ASYMMETRIC.
Because we scatter from explicit atoms, that profile is a per-atom displacement
u_z(z) -- and it is differentiable, so we invert the asymmetric peak back to the
depth profile.

Left  -- the planted vs recovered depth strain profile d(z)/d0 - 1.
Right -- the (1 1 1) Bragg peak: symmetric (cold) vs asymmetric (strained), with
         the reconstruction overlaid.

Saves a two-panel figure to the figure directory (dev/paper/figures/ in a clone, ./figures otherwise).
"""
from __future__ import annotations

import math
import os

import torch

from midas_2d import (
    cdse_supercell,
    coherent_intensity,
    depth_resolved_intensity,
    linear_strain,
    recover_depth_strain,
    strain_to_displacement,
)
from midas_2d.examples._figures import figure_dir

DT = torch.float64
A = 6.077


def _rod(l, hk=(1.0, 1.0)):
    h = torch.full_like(l, hk[0]); k = torch.full_like(l, hk[1])
    return (2 * math.pi / A) * torch.stack([h, k, l], dim=-1)


def main(out_dir=None):
    coords, elements, _ = cdse_supercell((7, 7, 8), dtype=DT)   # thicker -> clearer
    z = coords[:, 2]

    # planted: surface (large z) expanded, substrate side relaxed
    eps = linear_strain(z, eps_surface=0.02, eps_substrate=0.0)
    order = torch.argsort(z)
    u_sorted = strain_to_displacement(z[order], eps[order])
    u_true = torch.empty_like(u_sorted); u_true[order] = u_sorted

    l = torch.linspace(0.5, 1.5, 700, dtype=DT)
    q = _rod(l)
    I_cold = coherent_intensity(coords, elements, q)
    I_strained = depth_resolved_intensity(coords, elements, q, u_z=u_true)

    z_ctrl = torch.linspace(float(z.min()), float(z.max()), 7, dtype=DT)
    out = recover_depth_strain(I_strained, coords, elements, q, z_ctrl,
                               steps=1500, lr=0.01, smooth_weight=1e2)
    I_fit = depth_resolved_intensity(coords, elements, q, u_z=out["u_atom"]).detach()

    # The cleanly recoverable quantity is the cumulative DISPLACEMENT u(z)
    # (|A|^2 is invariant to a rigid z-shift, so compare mean-removed).
    zc = z_ctrl.numpy()
    u_true_ctrl = torch.stack([u_true[(z - zz).abs().argmin()] for zz in z_ctrl]).numpy()
    u_rec_ctrl = out["u_ctrl"].numpy()
    u_true_ctrl = u_true_ctrl - u_true_ctrl.mean()
    u_rec_ctrl = u_rec_ctrl - u_rec_ctrl.mean()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    axL.plot(zc, u_true_ctrl, "k--o", label="planted displacement u(z)")
    axL.plot(zc, u_rec_ctrl, "r.-", label="recovered (from the peak)")
    axL.set_xlabel("depth z (A)")
    axL.set_ylabel("out-of-plane displacement u(z)  (A, mean-removed)")
    axL.set_title("Depth-resolved lattice-displacement reconstruction")
    axL.legend(); axL.grid(alpha=0.3)

    axR.plot(l.numpy(), (I_cold / I_cold.max()).numpy(), color="0.6",
             label="cold (symmetric)")
    axR.plot(l.numpy(), (I_strained / I_strained.max()).numpy(), "b",
             label="strained (asymmetric)")
    axR.plot(l.numpy(), (I_fit / I_fit.max()).numpy(), "r--", label="reconstruction")
    axR.set_xlabel("continuous Miller index  l")
    axR.set_ylabel("intensity (norm.)")
    axR.set_title("(1 1 1) Bragg peak: strain gradient -> asymmetry")
    axR.legend(); axR.grid(alpha=0.3)

    fig.tight_layout()
    out_dir = figure_dir(out_dir)
    out_path = os.path.join(out_dir, "depth_resolved_strain.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
