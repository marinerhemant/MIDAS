"""``midas-pdf-coreshell`` — core-shell nanoparticle PDF refinement (Rev 14/15).

Fits two nested-sphere phases with volume-fraction-weighted G(r) contributions
against an observed ``.gr`` file.

Example::

    midas-pdf-coreshell --core-cif Ni.cif --shell-cif NiO.cif \\
                         --gr coreshell.gr \\
                         --r-core 30 --shell-thickness 10 \\
                         --steps 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

from ._common import (fallback_sigma, load_two_column, maybe_warn_fallback_sigma, print_json)

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-pdf"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



def main(argv: Optional[Sequence[str]] = None) -> int:
    p = _midas_make_parser(
        prog="midas-pdf-coreshell",
        description="Core-shell PDF refinement (two-phase, volume-fraction-weighted).")
    p.add_argument("--core-cif", type=Path, required=True,
                    help="core-phase CIF")
    p.add_argument("--shell-cif", type=Path, required=True,
                    help="shell-phase CIF")
    p.add_argument("--gr", type=Path, required=True,
                    help="observed G(r): 2- or 3-column ASCII")
    p.add_argument("--r-min", type=float, default=1.5)
    p.add_argument("--r-max", type=float, default=20.0,
                    help="upper r-fit limit (Å); default 20 — typical PDF fits go 20–30 Å")
    p.add_argument("--u-iso-core", type=float, default=0.006)
    p.add_argument("--u-iso-shell", type=float, default=0.010)
    p.add_argument("--r-core", type=float, default=30.0,
                    help="starting core radius (Å)")
    p.add_argument("--shell-thickness", type=float, default=10.0,
                    help="starting shell thickness (Å)")
    p.add_argument("--pin-geometry", action="store_true",
                    help="freeze R_core and shell_thickness at their init "
                         "(recommended when SAXS has already determined them)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    args = p.parse_args(argv)

    from midas_pdf.cif import read_cif_to_crystal
    from midas_pdf.structure import build_pair_list
    from midas_pdf.multi_phase import refine_core_shell, core_shell_pdf_gr

    core = read_cif_to_crystal(args.core_cif).to_torch()
    shell = read_cif_to_crystal(args.shell_cif).to_torch()

    r_np, G_np, sigma_np = load_two_column(args.gr)
    mask = (r_np >= args.r_min) & (r_np <= args.r_max)
    r = torch.tensor(r_np[mask], dtype=torch.float64)
    G = torch.tensor(G_np[mask], dtype=torch.float64)
    sigma_slice = sigma_np[mask] if sigma_np is not None else None
    sigma, sigma_fallback = fallback_sigma(G, sigma_slice)
    maybe_warn_fallback_sigma(sigma_fallback, args.gr)

    core_pairs = build_pair_list(core, r_max=args.r_max + 1.0)
    shell_pairs = build_pair_list(shell, r_max=args.r_max + 1.0)

    res = refine_core_shell(
        core, shell, r, G, core_pairs, shell_pairs,
        sigma_obs=sigma,
        init_a_core=float(core.lattice_params[0].detach()),
        init_a_shell=float(shell.lattice_params[0].detach()),
        init_u_iso_core=args.u_iso_core,
        init_u_iso_shell=args.u_iso_shell,
        init_R_core_A=args.r_core,
        init_shell_thickness_A=args.shell_thickness,
        steps=args.steps,
        lr=args.lr,
    )

    payload = {
        "core_cif": str(args.core_cif),
        "shell_cif": str(args.shell_cif),
        "gr": str(args.gr),
        "r_range": [args.r_min, args.r_max],
        "n_data_points": int(mask.sum()),
        "pin_geometry": bool(args.pin_geometry),
        "fitted": res.fitted,
        "uncertainty": res.uncertainty,
        "chi2_reduced": res.chi2_reduced,
        "volume_fractions": {"core": res.volume_fractions[0],
                              "shell": res.volume_fractions[1]},
        "final_loss": res.loss_history[-1] if res.loss_history else None,
    }
    print_json(payload)
    return 0


if __name__ == "__main__":                             # pragma: no cover
    sys.exit(main())
