"""``midas-pdf-rmc`` — reverse Monte Carlo refinement CLI.

Usage::

    midas-pdf-rmc --cif seed.cif --gr target.gr \\
                  --size 4 --moves 5000 \\
                  --min-distance 1.5 --temperature 1.0 \\
                  --output refined.cif

Reads a starting Crystal from a CIF, expands to an ``(N, N, N)``
supercell, and runs :func:`midas_pdf.rmc.rmc_refine` against a target
$G(r)$ from a two- or three-column ASCII file.  Emits a JSON summary
of initial vs final $\\chi^2$, acceptance ratio, coordination number
in the first shell, and (optionally) writes the final supercell as a
$P1$ CIF.

Supported move mixtures via ``--move-types``:

    displace          only single-atom Gaussian displacements (default)
    displace+cluster  50/50 mix of single + block moves
    all               single + cluster + rigid-rotation
    gc                grand-canonical (adds insert / remove)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from ._common import load_two_column, print_json

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



def _build_moves(kind: str, sigma_A: float, cluster_radius_A: float,
                  species: str) -> List:
    from midas_pdf.rmc import (
        DisplaceMove, ClusterDisplaceMove, RigidRotationMove,
        InsertMove, RemoveMove,
    )
    if kind == "displace":
        return [DisplaceMove(sigma_A=sigma_A)]
    if kind == "displace+cluster":
        return [DisplaceMove(sigma_A=sigma_A),
                ClusterDisplaceMove(radius_A=cluster_radius_A, sigma_A=sigma_A)]
    if kind == "all":
        return [DisplaceMove(sigma_A=sigma_A),
                ClusterDisplaceMove(radius_A=cluster_radius_A, sigma_A=sigma_A),
                RigidRotationMove(radius_A=cluster_radius_A,
                                    sigma_rad=sigma_A / 2.0)]
    if kind == "gc":
        return [DisplaceMove(sigma_A=sigma_A),
                InsertMove(species=species),
                RemoveMove(species=species)]
    raise SystemExit(f"unknown --move-types {kind!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = _midas_make_parser(
        prog="midas-pdf-rmc",
        description="Reverse Monte Carlo against a G(r) target.")
    p.add_argument("--cif", type=Path, required=True,
                    help="starting Crystal (CIF) — expanded to supercell")
    p.add_argument("--gr", type=Path, required=True,
                    help="target G(r): 2- or 3-column ASCII (r G [sigma])")
    p.add_argument("--size", type=int, default=4,
                    help="supercell dimension (N × N × N)")
    p.add_argument("--r-min", type=float, default=1.5)
    p.add_argument("--r-max", type=float, default=None,
                    help="r_max for pair evaluation (default = half of cell)")
    p.add_argument("--moves", type=int, default=2000,
                    help="number of Metropolis moves")
    p.add_argument("--move-types",
                    choices=("displace", "displace+cluster", "all", "gc"),
                    default="displace")
    p.add_argument("--sigma-A", type=float, default=0.05,
                    help="Gaussian displacement width per move (Å)")
    p.add_argument("--cluster-radius", type=float, default=3.0,
                    help="cluster-move neighbour radius (Å)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--chemical-potential", type=float, default=0.0,
                    help="μ for GC moves (0 = no bias)")
    p.add_argument("--min-distance", type=float, default=None,
                    help="hard-sphere veto (Å) before χ² evaluation")
    p.add_argument("--u-iso", type=float, default=0.005,
                    help="isotropic displacement for the Gaussian broadening")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None,
                    help="optional CIF path for the final supercell (P1)")
    p.add_argument("--first-shell-window", type=float, nargs=2,
                    default=(2.0, 2.9),
                    help="r range for first-shell coordination diagnostic")
    args = p.parse_args(argv)

    from midas_pdf.cif import read_cif_to_crystal, write_supercell_to_cif
    from midas_pdf.rmc import (
        Supercell, rmc_refine, coordination_number,
    )

    crystal = read_cif_to_crystal(args.cif).to_torch()
    sc = Supercell.from_crystal(crystal, size=(args.size, args.size, args.size))
    L = float(sc.cell[0, 0])

    # Load target
    r_np, G_np, sigma_np = load_two_column(args.gr)
    r_max = args.r_max if args.r_max is not None else L / 2.0 - 0.5
    mask = (r_np >= args.r_min) & (r_np <= r_max)
    r = torch.tensor(r_np[mask], dtype=torch.float64)
    G = torch.tensor(G_np[mask], dtype=torch.float64)
    sigma_G = (torch.tensor(sigma_np[mask], dtype=torch.float64)
               if sigma_np is not None else None)

    # Species for GC / insert moves (uses the first unique species)
    species = sc.species_set()[0] if sc.species_set() else "X"

    moves = _build_moves(args.move_types, args.sigma_A,
                          args.cluster_radius, species)

    result = rmc_refine(
        sc, r, G, sigma_G=sigma_G, moves=moves,
        n_moves=args.moves, u_iso=args.u_iso,
        r_max_pairs=r_max + 1.0,
        temperature=args.temperature,
        chemical_potential=args.chemical_potential,
        min_distance_A=args.min_distance, seed=args.seed,
    )

    # Coordination in the first shell
    shell_lo, shell_hi = args.first_shell_window
    Z_before = coordination_number(sc.__class__(species=list(sc.species),
                                                  positions=sc.positions.clone(),
                                                  cell=sc.cell.clone()),
                                     r_shell=(shell_lo, shell_hi))
    Z_after = coordination_number(sc, r_shell=(shell_lo, shell_hi))

    payload = {
        "cif": str(args.cif),
        "gr": str(args.gr),
        "supercell": {
            "n_atoms": sc.n_atoms,
            "size": args.size,
            "cell_A": [L, float(sc.cell[1, 1]), float(sc.cell[2, 2])],
        },
        "moves": {
            "kind": args.move_types,
            "n_attempted": result.n_moves,
            "n_accepted":  result.n_accepted,
            "acceptance_ratio": result.acceptance_ratio,
            "sigma_A": args.sigma_A,
            "cluster_radius_A": args.cluster_radius,
        },
        "chi2": {
            "initial": result.initial_chi2,
            "final":   result.final_chi2,
            "reduction_percent": 100.0 * (1.0 - result.final_chi2
                                                / max(result.initial_chi2, 1e-30)),
        },
        "coordination_first_shell": {
            "window_A": [shell_lo, shell_hi],
            "Z_after_mean": Z_after["Z_mean"],
            "Z_after_std":  Z_after["Z_std"],
        },
    }
    if args.output is not None:
        write_supercell_to_cif(sc, args.output, name=f"rmc_{args.cif.stem}")
        payload["output_cif"] = str(args.output)
    print_json(payload)
    return 0


if __name__ == "__main__":                             # pragma: no cover
    sys.exit(main())
