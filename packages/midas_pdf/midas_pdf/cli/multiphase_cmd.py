"""``midas-pdf-multiphase`` — multi-phase PDF refinement (Rev 14/15).

Fits a weighted sum ``G_total(r) = Σ w_i · G_i(r) · γ_i(r)`` against an
observed ``.gr`` file, given two or more starting CIF structures.  Weights
auto-renormalise; per-phase optional finite-size damping via ``--diameter``.

Example::

    midas-pdf-multiphase --cif Ni.cif Cu.cif \\
                          --gr NiCu_mix.gr \\
                          --r-min 1.5 --r-max 8.0 \\
                          --u-iso 0.006 \\
                          --steps 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

from ._common import (fallback_sigma, load_two_column, maybe_warn_fallback_sigma, print_json)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="midas-pdf-multiphase",
        description="Weighted multi-phase small-box PDF refinement.")
    p.add_argument("--cif", type=Path, nargs="+", required=True,
                    help="CIF files for each phase (2+)")
    p.add_argument("--gr", type=Path, required=True,
                    help="observed G(r) (2- or 3-column ASCII)")
    p.add_argument("--r-min", type=float, default=1.5)
    p.add_argument("--r-max", type=float, default=20.0,
                    help="upper r-fit limit (Å); default 20 — typical PDF fits go 20–30 Å")
    p.add_argument("--u-iso", type=float, default=0.006,
                    help="starting isotropic U for every phase")
    p.add_argument("--scale", type=float, default=1.0,
                    help="starting per-phase scale (uniform)")
    p.add_argument("--weights", type=float, nargs="+", default=None,
                    help="starting mixing weights (default: uniform)")
    p.add_argument("--diameter", type=float, nargs="+", default=None,
                    help="per-phase finite-size damping D (Å); pass one per CIF")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    args = p.parse_args(argv)

    if len(args.cif) < 2:
        p.error("at least two CIFs required")

    from midas_pdf.cif import read_cif_to_crystal
    from midas_pdf.structure import build_pair_list
    from midas_pdf.multi_phase import refine_multi_phase

    crystals = [read_cif_to_crystal(c).to_torch() for c in args.cif]
    r_np, G_np, sigma_np = load_two_column(args.gr)
    mask = (r_np >= args.r_min) & (r_np <= args.r_max)
    r = torch.tensor(r_np[mask], dtype=torch.float64)
    G = torch.tensor(G_np[mask], dtype=torch.float64)
    sigma_slice = sigma_np[mask] if sigma_np is not None else None
    sigma, sigma_fallback = fallback_sigma(G, sigma_slice)
    maybe_warn_fallback_sigma(sigma_fallback, args.gr)

    pairs_list = [build_pair_list(c, r_max=args.r_max + 1.0) for c in crystals]

    if args.weights is not None:
        if len(args.weights) != len(crystals):
            p.error(f"--weights length {len(args.weights)} != number of CIFs "
                    f"{len(crystals)}")
        init_weights = torch.tensor(args.weights, dtype=torch.float64)
    else:
        init_weights = None

    if args.diameter is not None:
        if len(args.diameter) != len(crystals):
            p.error(f"--diameter length {len(args.diameter)} != number of CIFs "
                    f"{len(crystals)}")
        diameters = torch.tensor(args.diameter, dtype=torch.float64)
    else:
        diameters = None

    res = refine_multi_phase(
        crystals, r, G, pairs_list,
        sigma_obs=sigma,
        init_a=[float(c.lattice_params[0].detach()) for c in crystals],
        init_u_iso=args.u_iso,
        init_scale=args.scale,
        init_weights=init_weights,
        diameters_A=diameters,
        steps=args.steps,
        lr=args.lr,
    )

    payload = {
        "cif": [str(c) for c in args.cif],
        "gr": str(args.gr),
        "r_range": [args.r_min, args.r_max],
        "n_data_points": int(mask.sum()),
        "n_phases": len(crystals),
        "fitted": res.fitted,
        "uncertainty": res.uncertainty,
        "chi2_reduced": res.chi2_reduced,
        "final_loss": res.loss_history[-1] if res.loss_history else None,
    }
    print_json(payload)
    return 0


if __name__ == "__main__":                             # pragma: no cover
    sys.exit(main())
