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

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-uq"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



DEG2RAD = math.pi / 180.0


# --------------------------------------------------------------------- IO
def _parse_grains_csv(path: Path) -> dict:
    """Read a ``Grains.csv`` of any width into ``{column name: array}``.

    Delegates to the canonical name-resolving reader in
    ``midas_process_grains.io``. This used to look for a header line starting
    with ``%GrainID`` only. ``midas_process_grains.io.csv`` writes ``%ID``
    (every mode except ``c_parity``), so on those files ``header_cols`` stayed
    ``None`` and the very next line raised
    ``TypeError: 'NoneType' object is not iterable`` — the shipped CLI simply
    could not read half the corpus.

    The grain-id column is normalised to ``GrainID`` whichever token the file
    carries. On a legacy 21-column file the per-grain lattice is not stored, so
    it is reconstructed from the header lattice and the Voigt strain block for
    use as a refinement SEED (the refiner recovers the rest).
    """
    from midas_process_grains.io import read_grains_csv

    t = read_grains_csv(path)
    cols = {name: t.column(name) for name in t.columns}
    cols["GrainID"] = np.asarray(t.ids, dtype=np.float64)

    if "a" not in cols:
        # Legacy 21-column layout: cols 13-18 are E11..E23 (dimensionless
        # strain-gauge form), not a b c alpha beta gamma. Seed the lattice from
        # the header reference lattice: a = a0 (1 + E11), etc.
        if t.strain_voigt is None or t.lattice_parameter is None:
            raise ValueError(
                f"{path}: no per-grain lattice (a b c alpha beta gamma) and no "
                f"E11..E23 + '%\tLattice Parameter:' preamble to derive one "
                f"from. Header has {t.columns!r}."
            )
        e = np.asarray(t.strain_voigt, dtype=np.float64)
        if np.nanmax(np.abs(e[:, :3])) > 0.1:
            # Refuse to guess: a microstrain-scaled block here would produce a
            # lattice off by orders of magnitude and a silently hopeless fit.
            raise ValueError(
                f"{path}: E11..E33 reach {np.nanmax(np.abs(e[:, :3])):.3g}, "
                f"which is not a dimensionless strain. Cannot derive a lattice "
                f"seed; re-run process_grains to get the a/b/c columns."
            )
        nom = np.asarray(t.lattice_parameter, dtype=np.float64)
        for k, name in enumerate(("a", "b", "c")):
            cols[name] = nom[k] * (1.0 + e[:, k])
        for k, name in enumerate(("alpha", "beta", "gamma")):
            cols[name] = np.full(t.n_grains, nom[3 + k])
    return cols


def _parse_spot_matrix(path: Path) -> dict:
    """Read a ``SpotMatrix.csv`` of either width (12 legacy / 28 expanded).

    ``read_spot_matrix`` drops the ``Matched == 0`` rows by default: those are
    reflections a grain was predicted to produce and that were never observed
    (~3.3 % of rows on real data). They carry ``-1`` in ``SpotID``/``RingNr``
    and NaN in every observed column.

    Leaving them in did not crash — ``_associate``'s ``min_d < max_dist``
    is False for a NaN distance, so the fit itself was unaffected — but every
    COUNT downstream was wrong: ``n_spots`` in the half-half output was
    overstated, the ``--min-spots`` gate admitted grains with fewer real spots
    than asked for, ``half_half_spots`` split ``randperm(n)`` over rows that
    were not all observations so the two halves held unequal amounts of real
    evidence, and ``jackknife_spots``' per-row influence index no longer
    lined up with an observed spot.

    η is recomputed as ``atan2(-YLab, ZLab)`` rather than read from the
    SpotMatrix ``Eta`` column. That column is not reliably an angle. Measured:
    on ``nfdev_jul26_20id_ff/report_au/SpotMatrix.csv`` it agrees with the
    lab-frame value to 5e-7 deg, but on
    ``demk_ff_Cu_results/LayerNr_1/SpotMatrix.csv`` (written by the Python
    ProcessGrains) it holds MICRONS — ±5e4, tracking YLab — which this CLI was
    then multiplying by π/180 and handing to the spot-matching loss. ``dev/paper/scripts/run_proto9_park22.py`` in this same
    package already recomputes it for exactly this reason; the shipped CLI did
    not. Recomputing is correct in both cases — where the column really is
    degrees it reproduces it, because MIDAS defines η that way.
    """
    from midas_process_grains.io import read_spot_matrix

    t = read_spot_matrix(path, matched_only=True)
    return {
        "grain_id":  np.asarray(t.grain_id, dtype=int),
        "spot_id":   np.asarray(t.spot_id, dtype=int),
        "omega_deg": t.omega,
        "eta_deg":   np.degrees(np.arctan2(-t.y_lab, t.z_lab)),
        "theta_deg": t.theta,
        "n_rows_total":     t.n_rows_total,
        "n_rows_unmatched": t.n_rows_unmatched,
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


def _grain_spot_ids(spots: dict, grain_id: int) -> np.ndarray:
    """MIDAS SpotIDs for a grain, in the SAME row order as
    :func:`_grain_observations`.

    The jackknife reports influence per observation index ``k``; without this
    there is no way back from ``k`` to the spot it names. (With unmatched rows
    left in, ``k`` did not even index an observation.)
    """
    return spots["spot_id"][spots["grain_id"] == grain_id]


def _report_spot_filter(spots: dict) -> None:
    """State how many predicted-but-never-found rows were dropped.

    Silently dropping them is how the counts got out of step in the first
    place, so the number is printed rather than assumed to be zero.
    """
    n_un = spots.get("n_rows_unmatched", 0)
    if n_un:
        print(f"SpotMatrix: {spots['n_rows_total']} rows, {n_un} unmatched "
              f"(Matched == 0) dropped, {len(spots['grain_id'])} observations")


# --------------------------------------------------------------------- handlers
def cmd_half_half(args):
    from .spots import half_half_spots
    grains = _parse_grains_csv(Path(args.grains))
    spots = _parse_spot_matrix(Path(args.spot_matrix))
    _report_spot_filter(spots)
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
    _report_spot_filter(spots)
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
    # Row k of `obs` is spot_ids[k]. Emit both: an influence ranking that can
    # only be read as a row index is not actionable, and before the
    # Matched == 0 rows were filtered out the index did not even name an
    # observation.
    spot_ids = _grain_spot_ids(spots, args.grain_id)
    res = jackknife_spots(
        model, state, obs,
        phase_steps=(args.phase1, args.phase2, args.phase3),
        verbose=args.verbose,
    )
    out_path = Path(args.out) if args.out else Path(f"jackknife_grain_{args.grain_id}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["k", "spot_id", "influence_mis_deg", "influence_lat_A"])
        w.writeheader()
        for k in range(obs.shape[0]):
            w.writerow({
                "k": k,
                "spot_id": int(spot_ids[k]),
                "influence_mis_deg": res.influence_misori_deg[k],
                "influence_lat_A": res.influence_lat_A[k],
            })
    print(f"wrote per-spot influence -> {out_path}")
    top = res.top_k(10, by="misori")
    print(f"top-10 influence spots (SpotID): "
          f"{[int(spot_ids[k]) for k in top.tolist()]}")


def cmd_laplace(args):
    from .laplace import laplace_covariance
    grains = _parse_grains_csv(Path(args.grains))
    spots = _parse_spot_matrix(Path(args.spot_matrix))
    _report_spot_filter(spots)
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
    parser = _midas_make_parser(
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
