"""Geometry multi-start for the multipoint calibration (MIDAS issue #16).

Both multipoint drivers used to start every local search from the paramfile
geometry. The soft path's multi-start perturbed the geometry only on trials
after the first, by ~0.3 of a tolerance, and ``NumIterations`` defaults to 1,
so in practice the geometry start WAS the paramfile value. On real data (AlON
NF, 100 voxels) the objective is multimodal in the wedge: starts at
-0.05/0/0.09/0.15/0.30 deg converged to 0.029/0.043/0.086/0.089/0.103 deg with
overlap 30.63/31.20/31.80/30.86/28.85, and the reported answer (from starts
near 0) was not the best basin.

This module supplies the two pieces the drivers share:

* :func:`build_geometry_starts` -- the seed, a DETERMINISTIC scan of each
  weakly-determined parameter across its tolerance box (the wedge when it is
  refined; optionally the tilts), and random starts drawn uniformly inside the
  tolerance boxes of every refined geometry parameter with a fixed RNG seed.
* :func:`analyse_basins` -- groups the trial end points into basins and sets
  ``multimodal`` when two or more DISTINCT basins score within a stated
  relative margin of the best. Distinct means some geometry parameter differs
  by more than ``basin_frac`` times its tolerance half-width.

Geometry vectors use the hard path's layout (``tx, ty, tz, Lsd[0],
dLsd[1..], ybc[..], zbc[..], [wedge]``); :func:`geometry_layout` builds the
names, seed values and tolerance half-widths from a :class:`FitParams`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Scan and random starts are kept this far inside the box, so a start is
#: never exactly on a bound (where the tanh reparameterisation has no gradient).
BOX_FILL = 0.9


def geometry_layout(p) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """``(names, seed, halfwidth)`` of the refined geometry, hard-path layout."""
    nL = p.n_distances
    names = (["tx", "ty", "tz", "Lsd[0]"]
             + [f"dLsd[{i}]" for i in range(1, nL)]
             + [f"ybc[{i}]" for i in range(nL)]
             + [f"zbc[{i}]" for i in range(nL)]
             + (["wedge"] if p.refine_wedge else []))
    seed = [p.tx, p.ty, p.tz, p.Lsd[0]]
    seed += [p.Lsd[i] - p.Lsd[i - 1] for i in range(1, nL)]
    seed += list(p.ybc) + list(p.zbc)
    hw = [p.tilts_tol] * 3 + [p.lsd_tol] + [p.lsd_rel_tol] * (nL - 1)
    hw += [p.bc_tol_a] * nL + [p.bc_tol_b] * nL
    if p.refine_wedge:
        seed.append(p.wedge)
        hw.append(p.wedge_tol)
    return names, np.asarray(seed, float), np.abs(np.asarray(hw, float))


def scanned_params(p) -> List[str]:
    """The parameters given a deterministic scan: the wedge when refined (it
    is known to be multimodal on real data), plus the tilts on request."""
    out = []
    if p.refine_wedge:
        out.append("wedge")
    if p.multipoint_scan_tilts:
        out += ["tx", "ty", "tz"]
    return out


def scan_fractions(n: int) -> List[float]:
    """``n`` offsets in units of the half-width, spread over
    ``[-BOX_FILL, +BOX_FILL]`` with 0 (the seed, already a start) removed, and
    ordered outermost first so that truncating the list keeps the spread."""
    if n <= 0:
        return []
    grid = np.linspace(-BOX_FILL, BOX_FILL, n + (1 if n % 2 else 0))
    grid = [float(g) for g in grid if abs(g) > 1e-9]
    return sorted(grid, key=lambda g: (-abs(g), g))[:n]


@dataclass
class GeomStart:
    label: str
    x: np.ndarray            # full geometry vector (hard-path layout)


def build_geometry_starts(
    names: Sequence[str],
    seed: np.ndarray,
    halfwidth: np.ndarray,
    *,
    scan: Sequence[str],
    n_scan: int,
    n_total: int,
    rng_seed: int,
) -> List[GeomStart]:
    """Seed, then the deterministic scan, then random starts.

    ``n_total`` is the total number of starts. If it is smaller than
    ``1 + len(scan) * n_scan`` the scan is truncated (outermost points kept);
    if larger, the remainder are random starts, each coordinate uniform in
    ``seed +/- BOX_FILL * halfwidth`` (parameters with zero tolerance are left
    at the seed), drawn from ``np.random.default_rng(rng_seed)`` so the run is
    reproducible.
    """
    seed = np.asarray(seed, float)
    hw = np.asarray(halfwidth, float)
    starts = [GeomStart("seed", seed.copy())]
    fr = scan_fractions(n_scan)
    for name in scan:
        if name not in names:
            continue
        j = list(names).index(name)
        if hw[j] <= 0:
            continue
        for f in fr:
            x = seed.copy()
            x[j] = seed[j] + f * hw[j]
            starts.append(GeomStart(f"scan {name}={x[j]:.6g}", x))
    n_total = max(1, int(n_total))
    if len(starts) > n_total:
        # Keep the seed; interleave the parameters' scans so each keeps its
        # outermost points.
        scan_starts = starts[1:]
        per = max(1, len(fr))
        order = sorted(range(len(scan_starts)),
                       key=lambda i: (i % per, i // per))
        starts = [starts[0]] + [scan_starts[i] for i in order][:n_total - 1]
    rng = np.random.default_rng(rng_seed)
    k = 0
    while len(starts) < n_total:
        u = rng.uniform(-BOX_FILL, BOX_FILL, size=seed.shape)
        starts.append(GeomStart(f"random {k}", seed + u * hw))
        k += 1
    return starts


def default_n_starts(p) -> int:
    """``MultipointGeomStarts`` if set (>0), else the seed plus the full scan,
    and never fewer than ``NumIterations`` (which used to be the number of
    multi-start trials). With ``RefineWedge 0`` and the default
    ``NumIterations 1`` this is 1: the old single-start behaviour."""
    if p.multipoint_geom_starts > 0:
        return int(p.multipoint_geom_starts)
    n_scan = len(scanned_params(p)) * len(scan_fractions(p.multipoint_geom_scan))
    return max(1 + n_scan, int(p.num_iterations))


def analyse_basins(
    trials: List[dict],
    names: Sequence[str],
    halfwidth: np.ndarray,
    *,
    basin_frac: float,
    rel_margin: float,
) -> dict:
    """Group trial END points into basins and decide ``multimodal``.

    Each trial is a dict with ``end`` (geometry vector, hard-path layout) and
    ``end_frac_overlap`` (higher is better). Trials are visited best first; a
    trial joins the first basin whose representative (its best member) is
    within ``basin_frac * halfwidth`` in EVERY parameter with a non-zero
    tolerance, else it founds a new basin. A basin is COMPETITIVE when
    ``best - its_best <= rel_margin * |best|``. ``multimodal`` is True when two
    or more distinct basins are competitive.
    """
    hw = np.asarray(halfwidth, float)
    live = hw > 0
    thr = basin_frac * hw
    order = sorted(range(len(trials)),
                   key=lambda i: -float(trials[i]["end_frac_overlap"]))
    basins: List[dict] = []
    for i in order:
        x = np.asarray(trials[i]["end"], float)
        for b in basins:
            d = np.abs(x - b["_rep"])
            if np.all(d[live] <= thr[live]):
                b["trials"].append(i)
                break
        else:
            basins.append(dict(_rep=x, trials=[i],
                               best_frac_overlap=float(
                                   trials[i]["end_frac_overlap"])))
    best = basins[0]["best_frac_overlap"] if basins else float("nan")
    out_basins = []
    for b in basins:
        gap = best - b["best_frac_overlap"]
        comp = bool(gap <= rel_margin * abs(best))
        out_basins.append(dict(
            geometry={n: float(v) for n, v in zip(names, b["_rep"])},
            best_frac_overlap=b["best_frac_overlap"],
            relative_gap_to_best=(gap / abs(best)) if best else float("inf"),
            competitive=comp,
            trials=b["trials"],
        ))
    n_comp = sum(b["competitive"] for b in out_basins)
    # Which parameters actually separate the competitive basins.
    comp_reps = [b["_rep"] for b, ob in zip(basins, out_basins)
                 if ob["competitive"]]
    ambiguous: List[str] = []
    if len(comp_reps) >= 2:
        R = np.stack(comp_reps)
        spread = R.max(0) - R.min(0)
        ambiguous = [n for j, n in enumerate(names)
                     if live[j] and spread[j] > thr[j]]
    return dict(
        multimodal=bool(n_comp >= 2),
        n_basins=len(out_basins),
        n_competitive_basins=int(n_comp),
        ambiguous_params=ambiguous,
        basins=out_basins,
        criteria=dict(
            basin_frac=float(basin_frac),
            basin_rule=("distinct basins differ by > basin_frac * tolerance "
                        "half-width in at least one refined geometry "
                        "parameter"),
            rel_margin=float(rel_margin),
            margin_rule=("a basin is competitive when (best - basin) frac "
                         "overlap <= rel_margin * best"),
        ),
    )


def geom_dict(names: Sequence[str], x: np.ndarray) -> Dict[str, float]:
    return {n: float(v) for n, v in zip(names, np.asarray(x, float))}


def print_multimodal_warning(analysis: dict, objective: str) -> None:
    """The loud warning. Printed unconditionally when ``multimodal``."""
    if not analysis.get("multimodal"):
        return
    bar = "!" * 72
    c = analysis["criteria"]
    print(bar)
    print(f"WARNING: the multipoint geometry is NOT UNIQUELY DETERMINED "
          f"({objective} objective).")
    print(f"  {analysis['n_competitive_basins']} distinct basins score within "
          f"{100 * c['rel_margin']:.1f}% of the best (distinct = > "
          f"{c['basin_frac']:g} x tolerance apart); separated in: "
          f"{', '.join(analysis['ambiguous_params']) or '(none listed)'}")
    for b in analysis["basins"]:
        if not b["competitive"]:
            continue
        g = ", ".join(f"{k}={v:.5g}" for k, v in b["geometry"].items()
                      if k in analysis["ambiguous_params"])
        print(f"    overlap {b['best_frac_overlap']:.6f}  "
              f"(-{100 * b['relative_gap_to_best']:.2f}%)  {g}")
    print("  The reported geometry is the best basin found, but the data do "
          "not")
    print("  clearly prefer it. Check it against an independent measurement "
          "(calibrant,")
    print("  FF, more/wider-spread voxels) before adopting it.")
    print(bar, flush=True)
