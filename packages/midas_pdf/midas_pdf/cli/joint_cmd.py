"""``midas-pdf-joint`` — joint SAXS + PDF (add ``--sans`` for three-way).

Usage::

    midas-pdf-joint --cif Ni.cif --gr Ni.gr --saxs Ni_saxs.dat \\
                    --init-diameter 100 --shape sphere --polydispersity 0.05

    # Three-way with SANS added:
    midas-pdf-joint --cif Ni.cif --gr Ni.gr --saxs Ni_saxs.dat \\
                    --sans Ni_sans.dat --init-scale-sans 0.35

Inputs are 2- or 3-column ASCII (``x  y  [sigma]``). Emits JSON with
fitted parameters, uncertainties, and per-channel χ².
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
        prog="midas-pdf-joint",
        description="Joint SAXS+PDF (add --sans for three-way SAXS+SANS+PDF).")
    p.add_argument("--cif", type=Path, required=True)
    p.add_argument("--gr", type=Path, required=True,
                    help="PDF G(r) file: r  G(r)  [sigma]")
    p.add_argument("--saxs", type=Path, required=True,
                    help="SAXS I(Q) file: Q  I(Q)  [sigma]")
    p.add_argument("--sans", type=Path, default=None,
                    help="SANS I(Q) file (optional; enables three-way)")
    p.add_argument("--shape", choices=("sphere", "ellipsoid", "cylinder"),
                    default="sphere")
    p.add_argument("--polydispersity", type=float, default=0.05)
    p.add_argument("--n-poly-nodes", type=int, default=11)
    p.add_argument("--r-min", type=float, default=1.5)
    p.add_argument("--r-max", type=float, default=20.0,
                    help="upper r-fit limit (Å); default 20 — typical PDF fits go 20–30 Å")
    p.add_argument("--init-a", type=float, default=None)
    p.add_argument("--init-u-iso", type=float, default=0.006)
    p.add_argument("--init-diameter", type=float, default=100.0)
    p.add_argument("--init-scale-saxs", type=float, default=1.0)
    p.add_argument("--init-scale-sans", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.5)
    args = p.parse_args(argv)

    from midas_pdf.cif import read_cif_to_crystal
    from midas_pdf.structure import build_pair_list
    from midas_pdf.saxs import (
        SAXSModel, joint_refine, joint_refine_three_way,
    )

    crystal = read_cif_to_crystal(args.cif).to_torch()

    # Load PDF
    r_np, G_np, sigma_G_np = load_two_column(args.gr)
    mask = (r_np >= args.r_min) & (r_np <= args.r_max)
    r = torch.tensor(r_np[mask], dtype=torch.float64)
    G = torch.tensor(G_np[mask], dtype=torch.float64)
    sigma_slice = sigma_G_np[mask] if sigma_G_np is not None else None
    sigma_G, sigma_G_fallback = fallback_sigma(G, sigma_slice)
    maybe_warn_fallback_sigma(sigma_G_fallback, args.gr)
    pairs = build_pair_list(crystal, r_max=args.r_max + 2.0)

    # Load SAXS
    q_np, I_np, sigma_I_np = load_two_column(args.saxs)
    q_saxs = torch.tensor(q_np, dtype=torch.float64)
    I_saxs = torch.tensor(I_np, dtype=torch.float64)
    sigma_I_saxs = (torch.tensor(sigma_I_np, dtype=torch.float64)
                    if sigma_I_np is not None
                    else 0.05 * I_saxs.abs())

    saxs_model = SAXSModel(shape=args.shape,
                            polydispersity=args.polydispersity,
                            n_poly_nodes=args.n_poly_nodes)

    if args.sans is not None:
        # Three-way
        q_sans_np, I_sans_np, sigma_I_sans_np = load_two_column(args.sans)
        q_sans = torch.tensor(q_sans_np, dtype=torch.float64)
        I_sans = torch.tensor(I_sans_np, dtype=torch.float64)
        sigma_I_sans = (torch.tensor(sigma_I_sans_np, dtype=torch.float64)
                        if sigma_I_sans_np is not None
                        else 0.05 * I_sans.abs())
        sans_model = SAXSModel(shape=args.shape,
                                polydispersity=args.polydispersity,
                                n_poly_nodes=args.n_poly_nodes)
        res = joint_refine_three_way(
            crystal_tensor=crystal, r_pdf=r, G_obs=G, pairs=pairs,
            sigma_G=sigma_G,
            q_saxs=q_saxs, I_saxs=I_saxs, sigma_I_saxs=sigma_I_saxs,
            saxs_model=saxs_model,
            q_sans=q_sans, I_sans=I_sans, sigma_I_sans=sigma_I_sans,
            sans_model=sans_model,
            init_a=args.init_a, init_u_iso=args.init_u_iso,
            init_diameter_A=args.init_diameter,
            init_scale_saxs=args.init_scale_saxs,
            init_scale_sans=args.init_scale_sans,
            n_steps=args.steps, lr=args.lr,
        )
        payload = {
            "mode": "three_way",
            "n_r_points": int(r.numel()),
            "n_q_saxs_points": int(q_saxs.numel()),
            "n_q_sans_points": int(q_sans.numel()),
            "fitted": res.fitted, "uncertainty": res.uncertainty,
            "chi2_pdf":   res.chi2_pdf,
            "chi2_saxs":  res.chi2_saxs,
            "chi2_sans":  res.chi2_sans,
            "chi2_total": res.chi2_total,
        }
    else:
        # Two-way SAXS + PDF
        res = joint_refine(
            crystal_tensor=crystal, r_pdf=r, G_obs=G, pairs=pairs,
            sigma_G=sigma_G,
            q_saxs=q_saxs, I_saxs=I_saxs, sigma_I=sigma_I_saxs,
            saxs_model=saxs_model,
            init_a=args.init_a, init_u_iso=args.init_u_iso,
            init_diameter_A=args.init_diameter,
            init_scale_saxs=args.init_scale_saxs,
            n_steps=args.steps, lr=args.lr,
        )
        payload = {
            "mode": "two_way",
            "n_r_points": int(r.numel()),
            "n_q_saxs_points": int(q_saxs.numel()),
            "fitted": res.fitted, "uncertainty": res.uncertainty,
            "chi2_pdf":   res.chi2_pdf,
            "chi2_saxs":  res.chi2_saxs,
            "chi2_total": res.chi2_total,
        }
    print_json(payload)
    return 0


if __name__ == "__main__":                             # pragma: no cover
    sys.exit(main())
