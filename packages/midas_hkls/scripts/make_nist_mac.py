"""Regenerate midas_hkls/data/nist_mac.json: NIST XCOM mass attenuation coefficients via xraylib, with EVERY
absorption edge bracketed.

Why this exists: the previously shipped table (232 energies) bracketed only a few K edges (e.g. Fe, Ni, Cu); 158 edges
across 92 elements were not bracketed (grid gaps up to 5.9%), so log-log interpolation smeared the edge jump and mu was
wrong by up to the full jump within a few percent of every such edge (found 2026-09-24: the Zn K edge at 9.659 keV was
interpolated between 9.44 and 10.0 keV). This generator keeps every original energy (values bit-identical: the table was
produced by xraylib.CS_Total, verified to 0.0 relative difference) and, for every K, L1-L3 and M1-M5 edge of
Z = 1-92 inside the grid range, LOCATES the jump in CS_Total itself and adds two points just either side of it, so
interpolation never crosses an edge.

Re-running it on the shipped table reproduces that table exactly (the base grid is kept in _meta.base_energy_keV).

usage: python scripts/make_nist_mac.py [out_path]     (needs xraylib; writes midas_hkls/data/nist_mac.json by default)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EDGE_EPS = 1e-5          # brackets at the LOCATED jump x (1 -/+ EDGE_EPS)
SEARCH = 0.02            # search +/-2% around xraylib.EdgeEnergy for the jump in CS_Total itself: the two can differ by
                         # more than 0.1% (Rn L3: EdgeEnergy 14.610 keV, CS jump just above 14.617), so bracketing the
                         # tabulated edge energy misses some jumps (6 of 156 in a first version)
MIN_JUMP = 1.05          # smaller steps are not edges worth bracketing
REFINE_TOL = 5e-3        # adaptive refinement: insert geometric midpoints until log-log interpolation matches CS_Total
                         # to REFINE_TOL at every midpoint, for every element (the curve bends just above some edges)
MAX_REFINE = 14


def main():
    import xraylib as xr
    here = Path(__file__).resolve().parents[1] / "midas_hkls" / "data" / "nist_mac.json"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else here
    old = json.loads(here.read_text())
    # the base grid is the ORIGINAL 232-energy table, carried in _meta so a re-run from the shipped table is idempotent
    E0 = np.asarray(old["_meta"].get("base_energy_keV", old["_energy_keV"]), float)
    lo, hi = float(E0.min()), float(E0.max())
    shells = [xr.K_SHELL, xr.L1_SHELL, xr.L2_SHELL, xr.L3_SHELL, xr.M1_SHELL, xr.M2_SHELL, xr.M3_SHELL, xr.M4_SHELL, xr.M5_SHELL]
    def cs(Z, e):
        try:
            return float(xr.CS_Total(Z, float(e)))
        except ValueError:               # beyond xraylib's spline range: 0, as in the original table
            return 0.0

    def locate_jump(Z, e0):
        """Energy of the discontinuity in CS_Total near e0, or None if there is no jump >= MIN_JUMP."""
        grid = np.exp(np.linspace(np.log(max(e0 * (1 - SEARCH), lo)), np.log(min(e0 * (1 + SEARCH), hi)), 801))
        v = np.array([cs(Z, x) for x in grid])
        if np.any(v <= 0):
            return None
        r = v[1:] / v[:-1]
        k = int(np.argmax(r))
        if r[k] < MIN_JUMP:
            return None
        a, b = grid[k], grid[k + 1]
        for _ in range(60):
            m = 0.5 * (a + b)
            if cs(Z, m) / cs(Z, a) >= MIN_JUMP:
                b = m
            else:
                a = m
        return 0.5 * (a + b)

    edges = []
    for Z in range(1, 93):
        for sh in shells:
            try:
                e = float(xr.EdgeEnergy(Z, sh))
            except ValueError:
                continue
            if not (lo < e < hi):
                continue                  # windows near the table limits are CLIPPED, not skipped (Zn L3 at 1.02 keV)
            j = locate_jump(Z, e)
            if j is not None:
                edges.append(j)
    extra = np.concatenate([np.asarray(edges) * (1 - EDGE_EPS), np.asarray(edges) * (1 + EDGE_EPS)])
    E = np.unique(np.concatenate([E0, extra]))

    E = E[(E >= lo) & (E <= hi)]
    syms = [k for k in old if not k.startswith("_")]
    Zs = [int(old[k]["Z"]) for k in syms]
    vals = np.array([[cs(Z, e) for e in E] for Z in Zs])
    n_refined = 0
    for it in range(MAX_REFINE):
        mid = np.sqrt(E[:-1] * E[1:])
        vm = np.array([[cs(Z, e) for e in mid] for Z in Zs])
        need = np.zeros(len(mid), bool)
        for k in range(len(Zs)):
            a, b, t = vals[k, :-1], vals[k, 1:], vm[k]
            ok = (a > 0) & (b > 0) & (t > 0)
            est = np.sqrt(a * b)                          # log-log interpolation at the geometric midpoint
            need |= ok & (np.abs(est / np.where(ok, t, 1.0) - 1) > REFINE_TOL)
        need &= (E[1:] / E[:-1] - 1) > 2 * EDGE_EPS       # never split an edge bracket
        if not need.any():
            break
        n_refined += int(need.sum())
        E = np.concatenate([E, mid[need]]); o = np.argsort(E); E = E[o]
        vals = np.concatenate([vals, vm[:, need]], axis=1)[:, o]
    table = {"_meta": dict(old["_meta"]), "_energy_keV": E.tolist()}
    table["_meta"].update(base_energy_keV=E0.tolist(), n_energy_points=int(len(E)), edges_bracketed=int(len(edges)), edge_eps=EDGE_EPS,
                          refine_tol=REFINE_TOL, points_added_by_refinement=n_refined,
                          generator="packages/midas_hkls/scripts/make_nist_mac.py",
                          note="every K, L1-L3, M1-M5 edge in range: the jump located in CS_Total itself, bracketed at jump*(1-/+edge_eps)")
    for k, sym in enumerate(syms):
        table[sym] = dict(Z=Zs[k], mu_rho=vals[k].tolist())
    # the original energies must be reproduced exactly
    idx = np.searchsorted(E, E0)
    idx_old = np.searchsorted(np.asarray(old["_energy_keV"], float), E0)
    for sym, v in old.items():
        if sym.startswith("_"):
            continue
        a = np.asarray(v["mu_rho"])[idx_old]; b = np.asarray(table[sym]["mu_rho"])[idx]
        if not np.array_equal(a, b):
            raise SystemExit(f"{sym}: regenerated values at the original energies differ from the shipped table")
    out.write_text(json.dumps(table))
    print(f"wrote {out}: {len(E)} energies ({len(E0)} original, {len(edges)} edges bracketed, {n_refined} added by refinement)")


if __name__ == "__main__":
    main()
