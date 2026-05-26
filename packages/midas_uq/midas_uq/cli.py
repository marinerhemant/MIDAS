"""Command-line interface for midas-uq.

Usage:
    midas-uq half-half  <args>     # K-split half-half UQ
    midas-uq jackknife  <args>     # per-observation jackknife
    midas-uq laplace    <args>     # Hessian-based Laplace covariance
    midas-uq version

For now the CLI focuses on the FF/pf spot-based workflow with MIDAS
SpotMatrix.csv + Grains.csv inputs. NF-mode CLI is exposed under
`--mode nf` with image-stack inputs (HDF5 / numpy) and is documented in
`examples/nf_frame_split.py`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import __version__


DEG2RAD = math.pi / 180.0


# --------------------------------------------------------------------- IO
def _parse_grains_csv(path: Path) -> dict:
    header_cols = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("%GrainID"):
                header_cols = line.lstrip("%").strip().split("\t")
            elif line.startswith("%"):
                continue
            else:
                vals = line.strip().split("\t")
                if not vals[0]:
                    continue
                rows.append(vals)
    arr = np.array([[float(v) for v in row] for row in rows], dtype=np.float64)
    return {col: arr[:, i] for i, col in enumerate(header_cols)}


def _parse_spot_matrix(path: Path) -> dict:
    data = np.loadtxt(path, skiprows=1)
    return {
        "grain_id":  data[:, 0].astype(int),
        "omega_deg": data[:, 2],
        "eta_deg":   data[:, 6],
        "theta_deg": data[:, 10],
    }


def _ps_to_geom_keys(path: Path) -> dict:
    """Sniff a paramstest.txt / stem.txt for the keys we need to build
    HEDMGeometry. Handles both Park22-style (semicolons) and Ti-7Al-style
    formats."""
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(";").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            k, v = parts
            v = v.rstrip(";").strip()
            cfg.setdefault(k, v)

    def f(key, dflt=None):
        if key not in cfg: return dflt
        return float(cfg[key].split()[0].rstrip(";"))

    npx = int(f("NrPixels") or 0)
    return {
        "Lsd": f("Lsd"),
        "px": f("px"),
        "BC": [float(x) for x in cfg["BC"].split()[:2]],
        "Wavelength": f("Wavelength"),
        "LatticeConstant": [float(x) for x in cfg["LatticeConstant"].split()[:6]],
        "SpaceGroup": int(f("SpaceGroup")),
        "OmegaStart": f("OmegaStart"),
        "OmegaStep": f("OmegaStep"),
        "MinEta": f("MinEta") or 6.0,
        "NrPixelsY": npx or int(f("NrPixelsY") or 2048),
        "NrPixelsZ": npx or int(f("NrPixelsZ") or 2048),
    }


def _build_hkls_via_midas_hkls(
    space_group_number: int,
    lattice_constants: list[float],
    wavelength_A: float,
    two_theta_max_deg: float,
):
    """Generate (hkls_cart, thetas, hkls_int) via midas-hkls (pure Python).

    Replaces the legacy `GetHKLList` C-binary path. `lattice_constants`
    is the standard 6-vector [a, b, c, alpha, beta, gamma].
    """
    from midas_diffract import hkls_for_forward_model
    from midas_hkls import SpaceGroup, Lattice
    a, b, c, alpha, beta, gamma = lattice_constants
    sg = SpaceGroup.from_number(int(space_group_number))
    # Pick the most-general lattice constructor available; for cubic
    # systems midas-hkls.Lattice.for_system("cubic", a=...) is the
    # standard path. Fall back to direct constructor if needed.
    if abs(a - b) < 1e-9 and abs(b - c) < 1e-9 \
            and abs(alpha - 90) < 1e-9 and abs(beta - 90) < 1e-9 \
            and abs(gamma - 90) < 1e-9:
        lat = Lattice.for_system("cubic", a=a)
    else:
        lat = Lattice(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    return hkls_for_forward_model(
        sg, lat,
        wavelength_A=wavelength_A,
        two_theta_max_deg=two_theta_max_deg,
    )


def _build_model_from_paths(param_path: Path, hkls_path: Optional[Path] = None,
                            two_theta_max_deg: float = 15.0):
    """Build HEDMForwardModel from a paramstest.txt (and optional hkls.csv).

    If `hkls_path` is given and exists, reflection list is parsed from the
    legacy `hkls.csv` (column convention: int hkl in cols 0-2, g-vector
    in cols 5-7, theta-degrees in col 8). Otherwise the reflection list
    is generated via `midas-hkls` from the param file's `LatticeConstant`,
    `SpaceGroup`, and `Wavelength` — pure Python, no GetHKLList needed.
    """
    from midas_diffract import HEDMForwardModel, HEDMGeometry
    cfg = _ps_to_geom_keys(param_path)
    cfg["n_frames"] = int(round(360.0 / abs(cfg["OmegaStep"])))

    if hkls_path is not None and Path(hkls_path).exists():
        hkls_arr = np.loadtxt(hkls_path, skiprows=1)
        hkls_int = torch.tensor(hkls_arr[:, 0:3], dtype=torch.float64)
        hkls_cart = torch.tensor(hkls_arr[:, 5:8], dtype=torch.float64)
        thetas = torch.tensor(hkls_arr[:, 8] * DEG2RAD, dtype=torch.float64)
    else:
        hkls_cart, thetas, hkls_int = _build_hkls_via_midas_hkls(
            cfg["SpaceGroup"], cfg["LatticeConstant"],
            wavelength_A=cfg["Wavelength"],
            two_theta_max_deg=two_theta_max_deg,
        )

    geom = HEDMGeometry(
        Lsd=cfg["Lsd"], y_BC=cfg["BC"][0], z_BC=cfg["BC"][1],
        px=cfg["px"], omega_start=cfg["OmegaStart"], omega_step=cfg["OmegaStep"],
        n_frames=cfg["n_frames"],
        n_pixels_y=cfg["NrPixelsY"], n_pixels_z=cfg["NrPixelsZ"],
        min_eta=cfg["MinEta"], wavelength=cfg["Wavelength"],
        flip_y=True,
    )
    return HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geom,
        hkls_int=hkls_int, device=torch.device("cpu"),
    )


def _build_grain_state(grains: dict, idx: int):
    from ._common import GrainState
    import numpy as np
    R = np.array([
        [grains["O11"][idx], grains["O12"][idx], grains["O13"][idx]],
        [grains["O21"][idx], grains["O22"][idx], grains["O23"][idx]],
        [grains["O31"][idx], grains["O32"][idx], grains["O33"][idx]],
    ], dtype=np.float64)
    Phi = math.acos(max(-1.0, min(1.0, R[2, 2])))
    sinP = math.sin(Phi)
    if abs(sinP) > 1e-6:
        phi1 = math.atan2(R[0, 2], -R[1, 2])
        phi2 = math.atan2(R[2, 0], R[2, 1])
    else:
        phi1 = math.atan2(R[1, 0], R[0, 0])
        phi2 = 0.0
    euler = torch.tensor([phi1, Phi, phi2], dtype=torch.float64)
    latc = torch.tensor(
        [grains["a"][idx], grains["b"][idx], grains["c"][idx],
         grains["alpha"][idx], grains["beta"][idx], grains["gamma"][idx]],
        dtype=torch.float64,
    )
    pos = torch.tensor(
        [grains["X"][idx], grains["Y"][idx], grains["Z"][idx]],
        dtype=torch.float64,
    )
    return GrainState(euler, latc, pos)


def _grain_observations(spots: dict, grain_id: int):
    mask = spots["grain_id"] == grain_id
    if not mask.any():
        return None
    two_theta = 2.0 * spots["theta_deg"][mask] * DEG2RAD
    eta = spots["eta_deg"][mask] * DEG2RAD
    omega = spots["omega_deg"][mask] * DEG2RAD
    return torch.tensor(
        np.stack([two_theta, eta, omega], axis=1),
        dtype=torch.float64,
    )


# --------------------------------------------------------------------- handlers
def cmd_half_half(args):
    from .spots import half_half_spots
    grains = _parse_grains_csv(Path(args.grains))
    spots = _parse_spot_matrix(Path(args.spot_matrix))
    model = _build_model_from_paths(
        Path(args.params),
        Path(args.hkls) if args.hkls else None,
        two_theta_max_deg=args.two_theta_max,
    )

    out_rows = []
    grain_ids = [int(g) for g in grains["GrainID"]]
    if args.max_grains:
        grain_ids = grain_ids[: args.max_grains]
    for gid in grain_ids:
        idx = list(grains["GrainID"]).index(gid)
        state = _build_grain_state(grains, idx)
        obs = _grain_observations(spots, gid)
        if obs is None or obs.shape[0] < 2 * args.min_spots:
            continue
        res = half_half_spots(
            model, state, obs,
            n_splits=args.n_splits, seed=args.seed,
            phase_steps=(args.phase1, args.phase2, args.phase3),
        )
        out_rows.append({
            "grain_id": gid,
            "n_spots": res.n_spots,
            "n_splits": res.n_splits,
            "misori_median_deg": res.misori_median_deg,
            "misori_p90_deg": res.misori_p90_deg,
            "lattice_median_A": res.lattice_median_A,
            "lattice_p90_A": res.lattice_p90_A,
        })
        if args.verbose:
            print(f"grain {gid}: mis_med={res.misori_median_deg:.4f}° "
                  f"lat_med={res.lattice_median_A:.2e}Å")

    if out_rows:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
        print(f"wrote {len(out_rows)} rows -> {out_path}")


def cmd_jackknife(args):
    from .spots import jackknife_spots
    grains = _parse_grains_csv(Path(args.grains))
    spots = _parse_spot_matrix(Path(args.spot_matrix))
    model = _build_model_from_paths(
        Path(args.params),
        Path(args.hkls) if args.hkls else None,
        two_theta_max_deg=args.two_theta_max,
    )

    idx = list(grains["GrainID"]).index(args.grain_id)
    state = _build_grain_state(grains, idx)
    obs = _grain_observations(spots, args.grain_id)
    if obs is None:
        sys.exit(f"No spots for grain {args.grain_id}")
    res = jackknife_spots(
        model, state, obs,
        phase_steps=(args.phase1, args.phase2, args.phase3),
        verbose=args.verbose,
    )
    out_path = Path(args.out) if args.out else Path(f"jackknife_grain_{args.grain_id}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "influence_mis_deg", "influence_lat_A"])
        w.writeheader()
        for k in range(obs.shape[0]):
            w.writerow({
                "k": k,
                "influence_mis_deg": res.influence_misori_deg[k],
                "influence_lat_A": res.influence_lat_A[k],
            })
    print(f"wrote per-spot influence -> {out_path}")
    top = res.top_k(10, by="misori")
    print(f"top-10 influence spots: {top.tolist()}")


def cmd_laplace(args):
    from .laplace import laplace_covariance
    grains = _parse_grains_csv(Path(args.grains))
    spots = _parse_spot_matrix(Path(args.spot_matrix))
    model = _build_model_from_paths(
        Path(args.params),
        Path(args.hkls) if args.hkls else None,
        two_theta_max_deg=args.two_theta_max,
    )

    idx = list(grains["GrainID"]).index(args.grain_id)
    state = _build_grain_state(grains, idx)
    obs = _grain_observations(spots, args.grain_id)
    cfg = _ps_to_geom_keys(Path(args.params))
    sigma_vec = torch.tensor([
        0.5 * cfg["px"] / cfg["Lsd"],     # 2theta noise floor
        0.5 / 500.0,                       # eta
        0.25 * abs(cfg["OmegaStep"]) * DEG2RAD,  # omega
    ], dtype=torch.float64)
    res = laplace_covariance(
        model, state, obs, sigma_vec,
        refine_first=True, n_mc_samples=args.n_mc,
    )
    out_path = Path(args.out) if args.out else Path(f"laplace_grain_{args.grain_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "grain_id": args.grain_id,
            "condition_number": res.condition_number,
            "eigenvalues": res.eigenvalues.tolist(),
            "misori_p95_deg": res.misori_p95_deg,
            "lattice_p95_A": res.lattice_p95_A,
            "covariance_9x9": res.covariance.tolist(),
        }, f, indent=2)
    print(f"wrote -> {out_path}")
    print(f"  Laplace misori P95 = {res.misori_p95_deg:.5f}°")
    print(f"  Laplace lattice P95 = {res.lattice_p95_A:.2e}Å")
    print(f"  Hessian cond = {res.condition_number:.2e}")


# --------------------------------------------------------------------- main
def _add_common_args(p):
    p.add_argument("--params", required=True, help="paramstest.txt path")
    p.add_argument("--hkls", default=None,
                   help=("(optional) legacy hkls.csv path. If omitted, "
                         "the reflection list is generated on the fly via "
                         "midas-hkls from the param file's LatticeConstant, "
                         "SpaceGroup, and Wavelength."))
    p.add_argument("--two-theta-max", type=float, default=15.0,
                   help=("2-theta cutoff (deg) for midas-hkls-generated "
                         "reflection lists. Default 15."))
    p.add_argument("--grains", required=True, help="Grains.csv path")
    p.add_argument("--spot-matrix", required=True, help="SpotMatrix.csv path")
    p.add_argument("--phase1", type=int, default=10)
    p.add_argument("--phase2", type=int, default=10)
    p.add_argument("--phase3", type=int, default=8)
    p.add_argument("--verbose", action="store_true")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="midas-uq",
        description=("Cross-validation based uncertainty quantification for "
                     "HEDM grain refinement."),
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_hh = sub.add_parser("half-half", help="K-split half-half UQ")
    _add_common_args(p_hh)
    p_hh.add_argument("--n-splits", type=int, default=5)
    p_hh.add_argument("--seed", type=int, default=0)
    p_hh.add_argument("--max-grains", type=int, default=0)
    p_hh.add_argument("--min-spots", type=int, default=12)
    p_hh.add_argument("--out", default="uq_half_half.csv")
    p_hh.set_defaults(func=cmd_half_half)

    p_jk = sub.add_parser("jackknife", help="Per-spot leave-one-out influence")
    _add_common_args(p_jk)
    p_jk.add_argument("--grain-id", type=int, required=True)
    p_jk.add_argument("--out", default=None)
    p_jk.set_defaults(func=cmd_jackknife)

    p_lp = sub.add_parser("laplace", help="Hessian-based Laplace covariance")
    _add_common_args(p_lp)
    p_lp.add_argument("--grain-id", type=int, required=True)
    p_lp.add_argument("--n-mc", type=int, default=2000)
    p_lp.add_argument("--out", default=None)
    p_lp.set_defaults(func=cmd_laplace)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
