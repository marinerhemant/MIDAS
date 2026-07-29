"""Showcase: amortised ML inference + a realistic detector image.

Left  -- AMORTISED INFERENCE: train a small MLP on rocking curves rendered by
the differentiable forward, then read thickness (and out-of-plane MSD) off
held-out patterns in a single network pass.  Parity plot: predicted vs true N3.

Right -- DETECTOR REALISM: project the coherent diffraction of a CdSe platelet
onto a flat area detector (Ewald-correct) and add Poisson photon noise -- the
kind of frame a real measurement produces, ready to fit with a Poisson
likelihood.

Saves a two-panel figure to dev/paper/figures/.
"""
from __future__ import annotations

import math
import os

import torch

from midas_2d import (
    add_poisson_noise,
    build_crystal_tensor,
    cdse_supercell,
    coherent_intensity,
    make_dataset,
    project_to_detector,
    train_surrogate,
)

DT = torch.float64
A = 6.077


def main(out_dir=None, seed=1):
    torch.manual_seed(seed)

    # ---- amortised inference --------------------------------------------------
    ct = build_crystal_tensor()
    X, Y = make_dataset(ct, n=500, n_points=48, seed=seed)
    model, info = train_surrogate(X, Y, epochs=300, lr=2e-3, seed=seed)
    n3_true = info["val_true"][:, 0].numpy()
    n3_pred = info["val_pred"][:, 0].numpy()

    # ---- detector image -------------------------------------------------------
    coords, elements, _ = cdse_supercell((8, 8, 4), dtype=DT)
    # Grid the two in-plane (transverse-to-beam) reciprocal directions H, K at
    # fixed L so the pattern spreads across BOTH detector axes.
    g = torch.linspace(-3.2, 3.2, 200, dtype=DT)
    H, K = torch.meshgrid(g, g, indexing="xy")
    L = torch.zeros_like(H)
    hkl = torch.stack([H, K, L], dim=-1)
    qv = (2 * math.pi / A) * hkl
    I = coherent_intensity(coords, elements, qv)
    pix, valid = project_to_detector(qv, wavelength_A=0.5, distance_mm=150.0,
                                     pixel_mm=0.2, beam_center=(0.0, 0.0))
    counts = add_poisson_noise(I, photons_per_peak=2e4)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    lim = [n3_true.min() - 0.3, n3_true.max() + 0.3]
    axL.plot(lim, lim, "k--", alpha=0.6)
    axL.scatter(n3_true, n3_pred, s=14, alpha=0.6)
    axL.set_xlabel("true N3 (cells)")
    axL.set_ylabel("MLP-predicted N3")
    axL.set_title(f"Amortised inference\n(val MAE: N3={float(info['val_mae'][0]):.2f}, "
                  f"u_perp={float(info['val_mae'][1]):.3f})")
    axL.set_aspect("equal")
    axL.grid(alpha=0.3)

    sc = axR.scatter(pix[..., 0].flatten().numpy(), pix[..., 1].flatten().numpy(),
                     c=torch.log10(counts + 1).flatten().numpy(), s=6, cmap="inferno")
    axR.set_xlabel("detector x (px)")
    axR.set_ylabel("detector y (px)")
    axR.set_title("Coherent pattern on a flat detector\n(Ewald-projected + Poisson noise)")
    axR.set_aspect("equal")
    fig.colorbar(sc, ax=axR, label="log10(counts+1)")

    fig.tight_layout()
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dev", "paper", "figures")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ml_amortized_and_detector.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")
    print(f"val MAE: {info['val_mae'].tolist()}")
    return out_path


if __name__ == "__main__":
    main()
