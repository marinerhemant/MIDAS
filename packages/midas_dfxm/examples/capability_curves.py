"""Capability curves — the quantitative paper spine that needs no real data.

Item #4 of the post-Phase-5 roadmap. Systematic studies the differentiable forward
makes cheap, each an honest, reviewer-proof capability statement:

  (a) strain-tensor identifiability vs reflection count (rank / condition number),
  (b) strain recovery error vs SNR: direct per-voxel LSQ vs curvature-regularised,
  (c) Burgers-sign recovery margin vs weak-beam offset.

Saves midas_dfxm/dev/paper/figures/capability_curves.png and prints the numbers.
Run:  export KMP_DUPLICATE_LIB_OK=TRUE; python examples/capability_curves.py
"""
from __future__ import annotations

import os

import torch

from midas_dfxm import (
    aligned_resolution,
    reference_q_nom,
    recover_strain_direct,
    recover_strain_regularised,
    strain_design_matrix,
    strain_identifiability,
    voxel_intensity,
    GoniometerSetting,
)
from midas_dfxm.detect import _normalize, match_residual, weak_beam_stack
from midas_dfxm.dislocation import cubic_stiffness, dislocation_deformation_field, stroh_dislocation

DT = torch.float64
CU = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)


def identifiability_vs_reflections():
    """(a) Add reflections one at a time; report rank and condition number."""
    order = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 0), (0, 2, 2), (2, 0, 2), (2, 2, 2), (1, 3, 1)]
    rows = []
    for k in range(1, len(order) + 1):
        info = strain_identifiability(order[:k], dtype=DT)
        rows.append((k, info["rank"], info["cond"]))
    return rows


def recovery_vs_snr():
    """(b) Strain recovery error, direct vs regularised, across noise levels."""
    n = 40
    x = torch.linspace(-1, 1, n, dtype=DT)
    eps6 = torch.zeros(n, 6, dtype=DT)
    eps6[:, 0] = 1e-3 * torch.sin(2 * x)
    eps6[:, 1] = -5e-4 * x
    eps6[:, 5] = 3e-4 * torch.cos(x)
    refl = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 0), (0, 2, 2), (2, 0, 2), (2, 2, 2)]
    M = strain_design_matrix(refl, dtype=DT)
    clean = (eps6 @ M.T).T
    out = []
    for noise in [1e-4, 3e-4, 6e-4, 1e-3, 2e-3]:
        ed, er = [], []
        for seed in range(4):
            torch.manual_seed(seed)
            meas = clean + noise * torch.randn_like(clean)
            d = recover_strain_direct(meas, refl)
            r = recover_strain_regularised(meas, refl, shape=(n, 1, 1),
                                           lambda_smooth=3.0, steps=1200, lr=3e-2)
            ed.append((d - eps6).abs().mean())
            er.append((r - eps6).abs().mean())
        out.append((noise, float(torch.stack(ed).mean()), float(torch.stack(er).mean())))
    return out


def sign_margin_vs_offset():
    """(c) Burgers-sign discrimination margin vs weak-beam rocking offset."""
    xs = torch.linspace(-8, 8, 24, dtype=DT)
    gx, gy, gz = torch.meshgrid(xs, xs, torch.zeros(1, dtype=DT), indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    PLANE, B = (1, 1, 1), (1, -1, 0)
    line = torch.linalg.cross(torch.tensor(PLANE, dtype=DT), torch.tensor(B, dtype=DT))
    refl = [(2, -2, 0), (2, 0, 0), (0, 2, 0), (1, 1, 1)]

    def edge(sign):
        b = tuple(sign * v for v in B)
        return stroh_dislocation(CU, burgers=b, slip_normal=PLANE, line=line, core_radius_um=0.4)

    out = []
    for off in [0.0, 0.01, 0.02, 0.04, 0.08]:
        obs = _normalize(weak_beam_stack(dislocation_deformation_field(pts, edge(+1)), refl, offset_deg=off))
        r_right = float(match_residual(obs, weak_beam_stack(
            dislocation_deformation_field(pts, edge(+1)), refl, offset_deg=off)))
        r_wrong = float(match_residual(obs, weak_beam_stack(
            dislocation_deformation_field(pts, edge(-1)), refl, offset_deg=off)))
        out.append((off, r_right, r_wrong))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "..", "dev", "paper", "figures")
    os.makedirs(outdir, exist_ok=True)

    ident = identifiability_vs_reflections()
    snr = recovery_vs_snr()
    sign = sign_margin_vs_offset()

    print("(a) identifiability vs #reflections  [k, rank, cond]:")
    for k, rank, cond in ident:
        print(f"    k={k}  rank={rank}  cond={cond:.2f}" if cond != float("inf")
              else f"    k={k}  rank={rank}  cond=inf (deficient)")
    print("(b) strain recovery error vs noise  [noise, err_direct, err_reg]:")
    for noise, ed, er in snr:
        print(f"    noise={noise:.0e}  direct={ed:.2e}  reg={er:.2e}  ratio={er/ed:.2f}")
    print("(c) sign margin vs offset  [offset_deg, res_right, res_wrong]:")
    for off, rr, rw in sign:
        print(f"    off={off:.2f}  right={rr:.2e}  wrong={rw:.2e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ks = [r[0] for r in ident]
        ax[0].plot(ks, [r[1] for r in ident], "o-")
        ax[0].axhline(6, ls="--", c="gray")
        ax[0].set_xlabel("# reflections"); ax[0].set_ylabel("strain-tensor rank")
        ax[0].set_title("(a) identifiability — rank 6 = full tensor")

        ns = [r[0] for r in snr]
        ax[1].loglog(ns, [r[1] for r in snr], "o-", label="direct LSQ")
        ax[1].loglog(ns, [r[2] for r in snr], "s-", label="curvature-regularised")
        ax[1].set_xlabel("noise std"); ax[1].set_ylabel("strain recovery error")
        ax[1].set_title("(b) reg wins at low SNR"); ax[1].legend()

        offs = [r[0] for r in sign]
        ax[2].semilogy(offs, [max(r[2], 1e-12) for r in sign], "s-", label="wrong sign residual")
        ax[2].semilogy(offs, [max(r[1], 1e-16) for r in sign], "o-", label="right sign residual")
        ax[2].set_xlabel("weak-beam offset (deg)"); ax[2].set_ylabel("match residual")
        ax[2].set_title("(c) Burgers-sign margin"); ax[2].legend()

        fig.tight_layout()
        out = os.path.join(outdir, "capability_curves.png")
        fig.savefig(out, dpi=140)
        print(f"saved figure -> {os.path.normpath(out)}")
    except ImportError:
        print("matplotlib unavailable; numbers printed above")


if __name__ == "__main__":
    main()
