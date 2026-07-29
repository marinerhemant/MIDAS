"""``midas-pdf-refine`` — small-box PDF refinement against a ``.gr`` file.

Usage::

    midas-pdf-refine --cif Ni.cif --gr Ni.gr \\
                     --r-max 8.0 --u-iso 0.006 --steps 400

The ``.gr`` file is two- or three-column ASCII (``r  G(r)  [sigma]``).
Outputs a JSON summary of the refined ``(a, u_iso, scale)``, their
Hessian-based uncertainties, and $\\chi^2/\\mathrm{ndof}$.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from ._common import (fallback_sigma, load_two_column, maybe_warn_fallback_sigma, print_json)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="midas-pdf-refine",
        description="Small-box PDF refinement of a Crystal against a G(r) file.")
    p.add_argument("--cif", type=Path, required=True,
                    help="starting Crystal structure (CIF)")
    p.add_argument("--gr", type=Path, required=True,
                    help="observed G(r): 2- or 3-column ASCII (r G [sigma])")
    p.add_argument("--r-min", type=float, default=1.5)
    p.add_argument("--r-max", type=float, default=20.0,
                    help="upper r-fit limit (Å); default 20 — typical PDF fits go 20–30 Å")
    p.add_argument("--u-iso", type=float, default=0.006,
                    help="initial isotropic displacement U_iso (Å²)")
    p.add_argument("--scale", type=float, default=1.0,
                    help="initial overall scale")
    p.add_argument("--steps", type=int, default=400,
                    help="LBFGS iterations")
    p.add_argument("--bg-order", type=int, default=0,
                    help="polynomial background order (0 = constant)")
    p.add_argument("--posterior-samples", type=int, default=0,
                    help="draw N Laplace-posterior samples (0 = off)")
    p.add_argument("--json", action="store_true", default=True,
                    help="print JSON summary to stdout (default)")
    args = p.parse_args(argv)

    from midas_pdf.cif import read_cif_to_crystal
    from midas_pdf.structure import (
        build_pair_list, pdffit_gr, refine_structure,
    )

    crystal = read_cif_to_crystal(args.cif).to_torch()
    r_np, G_np, sigma_np = load_two_column(args.gr)
    mask = (r_np >= args.r_min) & (r_np <= args.r_max)
    r = torch.tensor(r_np[mask], dtype=torch.float64)
    G = torch.tensor(G_np[mask], dtype=torch.float64)
    sigma_slice = sigma_np[mask] if sigma_np is not None else None
    sigma, sigma_fallback = fallback_sigma(G, sigma_slice)
    maybe_warn_fallback_sigma(sigma_fallback, args.gr)

    pairs = build_pair_list(crystal, r_max=args.r_max + 1.0)
    res = refine_structure(
        crystal, r, G, pairs, sigma_obs=sigma,
        init_a=float(crystal.lattice_params[0].detach()),
        init_u_iso=args.u_iso, init_scale=args.scale,
        bg_order=args.bg_order, steps=args.steps,
        n_posterior_samples=args.posterior_samples,
    )

    payload = {
        "cif": str(args.cif),
        "gr": str(args.gr),
        "r_range": [args.r_min, args.r_max],
        "n_data_points": int(mask.sum()),
        "fitted": res.fitted,
        "uncertainty": res.uncertainty,
        "chi2_reduced": res.chi2_reduced,
        "final_loss": res.loss,
    }
    if res.posterior is not None:
        payload["posterior_summary"] = {
            k: {"mean": float(res.posterior[f"{k}_samples"].mean()),
                 "std":  float(res.posterior[f"{k}_samples"].std())}
            for k in ("a", "u_iso", "scale")
            if f"{k}_samples" in res.posterior
        }
    print_json(payload)
    return 0


if __name__ == "__main__":                             # pragma: no cover
    sys.exit(main())
