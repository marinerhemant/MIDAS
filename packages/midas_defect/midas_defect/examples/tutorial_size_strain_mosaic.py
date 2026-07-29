"""Showcase (WS-2): size / microstrain / mosaic from HEDM spot shapes.

Left  -- Williamson-Hall: radial spot width^2 (minus instrument) vs q^2 is a
         straight line whose INTERCEPT gives crystallite size and SLOPE gives
         microstrain; the differentiable fit recovers both, plus mosaic from
         the azimuthal widths.
Right -- a crystallite-size DISTRIBUTION recovered from a broadened radial line
         profile (mixture deconvolution).

Saves a two-panel figure to dev/paper/figures/.
"""
from __future__ import annotations

import os

import torch

from midas_defect.peakshape import (
    azimuthal_width,
    fit_size_strain_mosaic,
    radial_width,
    recover_size_distribution,
    size_broadened_profile,
    size_width_q,
    williamson_hall,
)

DT = torch.float64


def main(out_dir=None, seed=0):
    torch.manual_seed(seed)

    # ---- Williamson-Hall separation (with noise) ----
    D_true, eps_true, mosaic_true = 250.0, 4e-3, 2e-3
    w_inst, w_inst_az = 0.006, 1e-3
    q = torch.linspace(2, 9, 22, dtype=DT)
    w_rad = radial_width(q, w_size=size_width_q(D_true), eps=eps_true, w_inst=w_inst)
    w_rad_noisy = w_rad * (1 + 0.02 * torch.randn_like(w_rad))
    w_az = azimuthal_width(torch.tensor(mosaic_true, dtype=DT), w_inst_az=w_inst_az, n=len(q))
    w_az_noisy = w_az * (1 + 0.02 * torch.randn_like(w_az))

    rec = fit_size_strain_mosaic(q, w_rad_noisy, w_az_noisy, w_inst=w_inst,
                                 w_inst_az=w_inst_az, steps=1500, lr=0.02)
    w_fit = radial_width(q, w_size=size_width_q(torch.tensor(rec["D"], dtype=DT)),
                         eps=rec["eps"], w_inst=w_inst)

    # ---- size distribution ----
    dq = torch.linspace(-0.12, 0.12, 240, dtype=DT)
    D_grid = [120.0, 180.0, 250.0, 350.0, 500.0]
    w_dist_true = torch.tensor([0.1, 0.25, 0.4, 0.18, 0.07], dtype=DT)
    obs = sum(wk * size_broadened_profile(dq, D) for wk, D in zip(w_dist_true, D_grid))
    rec_d = recover_size_distribution(dq, obs, D_grid, steps=2000, lr=0.05)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    x = (q ** 2).numpy()
    y = (w_rad_noisy ** 2 - w_inst ** 2).numpy()
    axL.plot(x, y, "o", ms=5, label="measured (width$^2$ - inst$^2$)")
    axL.plot((q ** 2).numpy(), (w_fit ** 2 - w_inst ** 2).numpy(), "r-",
             label=f"fit: D={rec['D']:.0f} A (true {D_true:.0f}), "
                   f"eps={rec['eps']*1e3:.2f}e-3 (true {eps_true*1e3:.1f}e-3)")
    axL.axhline(size_width_q(torch.tensor(D_true)).item() ** 2, color="0.6", ls=":",
                label="size-only intercept")
    axL.set_xlabel("q$^2$  (A$^{-2}$)")
    axL.set_ylabel("radial width$^2$ - inst$^2$")
    axL.set_title(f"Williamson-Hall: size (intercept) + microstrain (slope)\n"
                  f"mosaic from azimuthal width = {rec['mosaic']*1e3:.2f}e-3 rad "
                  f"(true {mosaic_true*1e3:.1f}e-3)")
    axL.legend(fontsize=8); axL.grid(alpha=0.3)

    xi = torch.arange(len(D_grid)).numpy()
    axR.bar(xi - 0.18, (w_dist_true / w_dist_true.sum()).numpy(), width=0.35,
            label="true", color="0.6")
    axR.bar(xi + 0.18, rec_d["weights"].numpy(), width=0.35, label="recovered",
            color="r", alpha=0.8)
    axR.set_xticks(xi); axR.set_xticklabels([f"{int(d)}" for d in D_grid])
    axR.set_xlabel("crystallite size D (A)"); axR.set_ylabel("population fraction")
    axR.set_title("Size distribution from broadened spot profile")
    axR.legend(fontsize=8); axR.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dev", "paper", "figures")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "size_strain_mosaic.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")
    print(f"recovered D={rec['D']:.1f} eps={rec['eps']:.2e} mosaic={rec['mosaic']:.2e}")
    return out_path


if __name__ == "__main__":
    main()
