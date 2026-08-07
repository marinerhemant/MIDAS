"""Tutorial: thickness fringes of a few-layer CdSe nanoplatelet.

Reproduces the canonical "fringe count encodes the monolayer number" picture:
scan the continuous l index along the (1 1 l) rod for N3 = 3, 4, 5 unit cells
out of plane and plot the Laue oscillations.

Run:
    python -m midas_2d.examples.tutorial_npl_fringes        # or run this file
Saves the figure under the figure directory (dev/paper/figures/ in a clone, ./figures otherwise).
"""
from __future__ import annotations

import os

import torch

from midas_2d import build_crystal_tensor, rod_intensity
from midas_2d.viz import plot_rod
from midas_2d.examples._figures import figure_dir

DT = torch.float64


def main(out_dir=None):
    ct = build_crystal_tensor()  # zinc-blende CdSe

    # Zoom on the allowed (1 1 1) Bragg peak so the Laue oscillations are clear:
    # a platelet of N3 cells shows exactly N3-1 fringe minima per Bragg period.
    l = torch.linspace(0.55, 1.45, 4000, dtype=DT)
    hkl = torch.stack([torch.full_like(l, 1.0), torch.full_like(l, 1.0), l], dim=-1)

    curves, labels = [], []
    for n3 in (3, 4, 5):
        N = torch.tensor([1.0e4, 1.0e4, float(n3)], dtype=DT)
        I = rod_intensity(ct, hkl, N, wavelength_A=1.0, apply_lp=False)
        # Normalise each curve to its peak so the fringe *count* is the story,
        # not the N^2 peak-height scaling.
        curves.append(I / I.max())
        labels.append(f"N3 = {n3} cells  ->  {n3 - 1} fringe minima")

    out_dir = figure_dir(out_dir)
    out_path = os.path.join(out_dir, "cdse_npl_thickness_fringes.png")
    plot_rod(l, curves, labels, out_path, logy=False,
             title="CdSe nanoplatelet (1 1 1) Laue oscillations -- fringe count = N3 - 1")
    print(f"saved: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
