"""Showcase: coherent reciprocal-space map + differentiable phase retrieval.

* Left  -- a coherent 2-D reciprocal-space map |A(Q)|^2 around the (1 1 1) node
  of a few-layer CdSe platelet, showing the finite-size streak along the
  out-of-plane (l) direction (the coherent face of the thickness fringes).
* Right -- autograd phase retrieval: |FFT(psi)|^2 speckle inverted back to the
  complex object's modulus inside a support, the differentiable alternative to
  iterative ER/HIO and the natural slot for a learned prior.

Saves a two-panel figure to the figure directory (dev/paper/figures/ in a clone, ./figures otherwise).
"""
from __future__ import annotations

import os

import torch

from midas_2d import (
    bcdi_forward,
    cdse_supercell,
    phase_retrieval,
    reciprocal_space_map,
)
from midas_2d.examples._figures import figure_dir

DT = torch.float64
A = 6.077


def main(out_dir=None, seed=0):
    torch.manual_seed(seed)

    # ---- coherent reciprocal-space map around (1 1 1) ----
    coords, elements, _ = cdse_supercell((10, 10, 4), dtype=DT)
    H, L, I = reciprocal_space_map(coords, elements, a=A, h0=1.0,
                                   qx_range=(-0.35, 0.35), qz_range=(0.55, 1.45),
                                   n_qx=140, n_qz=220)
    I = I / I.max()

    # ---- phase retrieval round-trip ----
    n = 28
    support = torch.zeros(n, n, dtype=DT)
    support[7:21, 9:17] = 1.0
    truth = (support * (1.0 + 0.5 * torch.rand(n, n, dtype=DT))) \
        * torch.exp(1j * 0.3 * torch.randn(n, n, dtype=DT))
    measured = bcdi_forward(truth)
    init = truth + 0.12 * (torch.randn(n, n, dtype=DT) + 1j * torch.randn(n, n, dtype=DT))
    rec = phase_retrieval(measured, support, init=init, steps=900, lr=0.01)
    rec_mod = rec["psi"].abs()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(14, 4.3))
    im = axL.pcolormesh(H.numpy(), L.numpy(), torch.log10(I + 1e-6).numpy(),
                        shading="auto", cmap="inferno")
    axL.set_xlabel("h (r.l.u.)")
    axL.set_ylabel("l (r.l.u.)")
    axL.set_title("Coherent RSM near (1 1 1)\nfinite-size streak along l")
    fig.colorbar(im, ax=axL, label="log10 |A|^2")

    axM.imshow(truth.abs().numpy(), cmap="viridis")
    axM.set_title("true object modulus")
    axM.axis("off")
    axR.imshow(rec_mod.numpy(), cmap="viridis")
    axR.set_title("phase-retrieved modulus")
    axR.axis("off")

    fig.tight_layout()
    out_dir = figure_dir(out_dir)
    out_path = os.path.join(out_dir, "coherent_rsm_and_phase_retrieval.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
