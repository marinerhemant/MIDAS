"""Choose the geometry and the known phases of a series from its own frames.

Beamtimes mix detector distances and alloys; a log may not record which distance a
series used; one reference structure cannot fit every alloy; and the dominant phase
can change during a series (e.g. bcc before a laser pass, fcc after it).

For frame blocks spread through the series, every (geometry, reference) pair is fitted
(:func:`fit_matrix_scale`). A fit COUNTS only if it is consistent: at least
``min_rings`` rings whose per-ring scales agree within ``spread_tol`` -- a real phase
gives one scale from every ring, a pattern that merely catches neighbouring peaks of
other phases does not. Among consistent fits the ranking is by COMPLETENESS (the
fraction of the reference's lines present), not by ring count: a line-dense reference
matches more rings by chance, and a reference whose lines are a subset-coincidence of
the true one (a cubic cell sqrt(2) larger, say) is incomplete.

Known phases = every block's best fit, plus consistent, >= ``min_completeness``
fits with >= 2 accepted rings that do not coincide with that block's best pattern.
Two references whose fitted rings coincide (>= 80 % within ``coincide_tol``) are ONE
phase -- e.g. two reference cells of the same structure type at different scales --
but two patterns of one structure type whose fitted cells differ by more than
``merge_tol`` are two phases. The full ranking is returned so the choice is visible
and can be overridden.

The reference list is finite. A phase outside it is reported as the nearest
reference whose lines happen to coincide with some of its lines. So every block
also lists its sharp lines that no known phase explains (``unexplained``): a
non-empty list means the phase set is incomplete, whatever the labels say.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from midas_integrate_v2.streaming.snapshot_profile import fit_matrix_scale, ring_profile, sharp_peaks

from .geometry import build_maps, fit_window, matrix_lines
from .io import list_frames, load_mask, sum_frames


def _key(r):
    return (r["completeness"], r["n_rings"], -r["scale_spread"])


def select_setup(frames: str, geometries: Dict[str, str], matrices: Dict[str, str], *,
                 flip: Optional[str] = "ud", mask: Optional[str] = None, invalid_below: float = 0.0,
                 n_frames: int = 25, first: int = 0, tth_step: float = 0.005, n_fit_lines: int = 8,
                 scale_grid=(0.98, 1.05, 0.0002), min_rings: int = 3, spread_tol: float = 0.003,
                 min_completeness: float = 0.75, coincide_tol: float = 0.004,
                 merge_tol: float = 0.03, rescale_tol: float = 0.01, ring_tol: float = 0.004,
                 near_spread: float = 0.004,
                 blocks: Optional[Sequence[float]] = None) -> dict:
    """``blocks``: fractional positions (0 = start, 1 = end) of the frame blocks to try;
    default: only ``first``. ``merge_tol``: two proportional patterns best in different
    blocks are one phase only if their fitted cells differ by less than this (thermal /
    composition drift, not a different phase). ``rescale_tol``: relative scale allowed
    between two phases' blocks in the sub-pattern test (they may be at different
    temperatures). ``ring_tol``: see :func:`fit_matrix_scale`. ``near_spread``: a more complete
    pattern that narrowly fails the spread gate (<= this) still vetoes fits whose rings all
    lie on its lines (wrong patterns scatter far more)."""
    all_files = list_frames(frames)
    starts = [first] if blocks is None else sorted(
        {int(round(f * max(len(all_files) - n_frames, 0))) for f in blocks})
    ranking: List[dict] = []
    lines_of: Dict[tuple, np.ndarray] = {}
    full_of: Dict[tuple, np.ndarray] = {}
    peaks_of: Dict[tuple, list] = {}
    for st in starts:
        _rank_block(all_files[st:st + n_frames], st, geometries, matrices, flip, mask, invalid_below,
                    tth_step, n_fit_lines, scale_grid, min_rings, spread_tol, ring_tol, ranking, lines_of,
                    full_of, peaks_of)
    ok = [r for r in ranking if r["ok"]]
    if not ok:
        return dict(best=None, phases=[], phase_info=[], geometry=None, ranking=_sorted(ranking),
                    blocks_tried=starts, n_frames_block=n_frames)
    # geometry: best mean of per-block best completeness
    per_geom = {}
    for g in geometries:
        per_block = [max((r for r in ok if r["geometry"] == g and r["block_start"] == st), key=_key, default=None)
                     for st in starts]
        vals = [b["completeness"] for b in per_block if b is not None]
        per_geom[g] = (np.mean(vals) * len(vals) / len(starts) if vals else 0.0, per_block)
    geom = max(per_geom, key=lambda g: per_geom[g][0])
    # ---- known phases -------------------------------------------------------------
    # candidates: every block's best fit, and consistent, complete fits with >= 2 rings not
    # on that block's best pattern. Then two reductions:
    #  (1) proportional patterns that are best in DIFFERENT blocks are one phase evolving
    #      (temperature / composition) -- e.g. two fcc reference cells, cold vs hot;
    #  (2) a phase whose fitted rings lie >= 80 % on another known phase's full line list
    #      is a sub-pattern of it (e.g. bcc with a_fcc = sqrt(2) a_bcc) and is dropped.
    rel_grid = np.arange(1 - rescale_tol, 1 + rescale_tol + 1e-9, 0.0002)

    def contained(a, b):
        """a's ACCEPTED rings lie on b's full line list, for some relative scale within
        +/- rescale_tol (the two may have been chosen in blocks at different temperatures)."""
        fa = lines_of[(geom, a["name"])]
        fa = (fa[np.array(a["rings"], int)] if a["rings"] else fa) * a["scale"]
        fb = full_of[(geom, b["name"])] * b["scale"]
        return max(np.mean([np.min(np.abs(x / (fb * r) - 1)) <= coincide_tol for x in fa])
                   for r in rel_grid) >= 0.8

    def as_c(r):
        return dict(name=r["matrix"], scale=r["scale"], rings=r["ring_index"])

    cand: List[dict] = []                   # {name, scale, block, primary}
    for st in starts:
        # a block's primary: best consistent fit that is not a sub-pattern of another
        # consistent fit in the same block (bcc lines all lie on fcc lines at sqrt(2) a)
        blk = [r for r in ok if r["geometry"] == geom and r["block_start"] == st]
        # identical accepted rings (mutual containment): the pattern predicting fewer lines
        # wins (bcc over the fcc at sqrt(2) a whose odd lines are unobserved)
        full = [r for r in blk if not any(
            o is not r and contained(as_c(r), as_c(o))
            and (not contained(as_c(o), as_c(r)) or o["n_full"] < r["n_full"]) for o in blk)]
        # a fit whose rings all lie on a MORE COMPLETE pattern of this block (consistent or not:
        # a hot block's split rings can fail the spread gate) is not evidence for itself
        # (e.g. bcc 110/220/400 = fcc 111/222/422 when the fcc has 8 of 8 rings)
        blk_all = [r for r in ranking if r["geometry"] == geom and r["block_start"] == st
                   and np.isfinite(r["scale_coarse"]) and r["completeness"] >= min_completeness
                   and np.isfinite(r["scale_spread"]) and r["scale_spread"] <= near_spread]
        full = [r for r in full if not any(
            o["matrix"] != r["matrix"] and o["completeness"] > r["completeness"]
            and contained(as_c(r), dict(name=o["matrix"], scale=o["scale_coarse"], rings=o["ring_index"]))
            for o in blk_all)]
        prim = max(full, key=_key, default=None)
        if prim is None:
            continue
        cand.append(dict(name=prim["matrix"], scale=prim["scale"], block=st, primary=True,
                         completeness=prim["completeness"], rings=prim["ring_index"]))
        prim_lines = lines_of[(geom, prim["matrix"])] * prim["scale"]
        for r in ok:
            if (r["geometry"] != geom or r["block_start"] != st or r["matrix"] == prim["matrix"]
                    or r["completeness"] < min_completeness):
                continue
            own = lines_of[(geom, r["matrix"])] * r["scale"]
            accepted = own[np.array(r["ring_index"], int)] if r["ring_index"] else np.array([])
            distinct = [d for d in accepted if np.min(np.abs(d / prim_lines - 1)) > coincide_tol]
            if len(distinct) >= 2:
                cand.append(dict(name=r["matrix"], scale=r["scale"], block=st, primary=False,
                                 completeness=r["completeness"], rings=r["ring_index"]))

    def proportional(a, b):
        la, lb = np.sort(lines_of[(geom, a)])[::-1], np.sort(lines_of[(geom, b)])[::-1]
        na, nb = la / la[0], lb / lb[0]
        return np.mean([np.min(np.abs(x / nb - 1)) <= coincide_tol for x in na]) >= 0.8

    def same_cell(a, b):
        da = lines_of[(geom, a["name"])][0] * a["scale"]
        db = lines_of[(geom, b["name"])][0] * b["scale"]
        return abs(da / db - 1) <= merge_tol

    kept: List[dict] = []
    for c in cand:
        dup = None
        for k in kept:
            if c["name"] == k["name"] or (proportional(c["name"], k["name"]) and c["block"] != k["block"]
                                          and same_cell(c, k)):
                dup = k
                break
        if dup is None:
            kept.append(dict(c, blocks=[c["block"]]))
        else:
            dup["blocks"].append(c["block"])
    final = [k for k in kept
             if not any(o is not k and contained(k, o) and not contained(o, k) for o in kept)]
    # mutual containment (identical line sets): keep the one predicting fewer lines
    nfull = {k["name"]: len(full_of[(geom, k["name"])]) for k in final}
    final = sorted(final, key=lambda k: nfull[k["name"]])
    phases: List[str] = []
    phase_info: List[dict] = []
    for k in final:
        if any(contained(k, o) and contained(o, k) for o in final if o["name"] in phases):
            continue
        phases.append(k["name"])
        # cell from the block where this phase's fit is most complete (a weak early block
        # must not set it); every block's cell is listed
        own = [r for r in ok if r["geometry"] == geom and r["matrix"] == k["name"]]
        src = max(own, key=_key)
        phase_info.append(dict(name=k["name"], cell_a=round(src["cell_a"], 5), scale=round(src["scale"], 6),
                               block_start=src["block_start"], blocks=k["blocks"], primary=k["primary"],
                               completeness=src["completeness"], scale_spread=round(src["scale_spread"], 6),
                               cells_by_block={int(r["block_start"]): round(r["cell_a"], 5) for r in own}))
    best = max((r for r in ok if r["geometry"] == geom), key=_key)
    # each known phase may be hotter or colder in another block (up to merge_tol)
    thermal = np.arange(1 - merge_tol, 1 + merge_tol + 1e-9, 0.0002)
    unexplained = _unexplained(peaks_of, geom, starts, phase_info, full_of, thermal, coincide_tol)
    return dict(best=best, phases=phases, phase_info=phase_info, geometry=geom, ranking=_sorted(ranking),
                blocks_tried=starts, n_frames_block=n_frames, unexplained=unexplained,
                n_unexplained=max((len(u) for u in unexplained.values()), default=0),
                rule=(f"consistent = rings >= {min_rings} and per-ring scale MAD <= {spread_tol}; rank by "
                      f"completeness; extra phase if consistent, completeness >= {min_completeness}, "
                      f">= 2 rings not within {coincide_tol} of the block's best pattern; proportional "
                      f"patterns best in different blocks with cells within {merge_tol} = one phase; "
                      f"sub-patterns (>= 80 % of accepted rings on another phase's lines, relative scale "
                      f"free within {rescale_tol}) dropped; rings off the others by > {ring_tol} dropped; "
                      f"cell from the most complete block; unexplained = sharp lines on no known phase"))


def _unexplained(peaks_of, geom, starts, phase_info, full_of, rel_grid, tol):
    """Per block: sharp lines (d, snr) that no known phase explains. In each block each phase
    takes up to two relative scales (within the given range: hot and cold parts of the sample
    can share a block), each explaining at least two lines."""
    out = {}
    for st in starts:
        pk = peaks_of.get((geom, st), [])
        d = np.array([p["d"] for p in pk])
        if d.size == 0:
            out[st] = []
            continue
        explained = np.zeros(d.size, bool)
        for ph in phase_info:
            fb = full_of[(geom, ph["name"])] * ph["scale"]
            for _ in range(2):          # up to two scales per phase: hot and cold parts of one block
                best = max((np.array([np.min(np.abs(x / (fb * r) - 1)) <= tol for x in d]) & ~explained
                            for r in rel_grid), key=lambda m: m.sum())
                if best.sum() < 2:
                    break
                explained |= best
        out[st] = [dict(d=round(float(p["d"]), 5), snr=round(float(p["snr"]), 1))
                   for p, e in zip(pk, explained) if not e]
    return out


def _sorted(ranking):
    return sorted(ranking, key=lambda r: (-r["ok"], -r["completeness"], -r["n_rings"]))


def _rank_block(files, start, geometries, matrices, flip, mask, invalid_below, tth_step,
                n_fit_lines, scale_grid, min_rings, spread_tol, ring_tol, ranking, lines_of, full_of,
                peaks_of):
    from midas_hkls.io.cif import read_cif
    img = sum_frames(files, flip, load_mask(mask, flip), invalid_below)
    ok = img >= 0
    grid = np.arange(*scale_grid)
    for gname, gpath in geometries.items():
        maps = build_maps(gpath, tth_step)
        tc, prof = ring_profile(img, ok, maps.tth, tth_step, tth0=maps.tth0)
        tmax = float(np.nanmax(maps.tth[ok])) if ok.any() else float(np.nanmax(maps.tth))
        # only lines strong enough to matter (>= 20 sigma): the list flags a missing phase
        pk = sharp_peaks(tc, prof / len(files), max_fwhm=fit_window(None, maps), min_sigma=20.0)
        for p in pk:
            p["d"] = maps.wavelength / (2 * np.sin(np.radians(p["tth"] / 2)))
        peaks_of[(gname, start)] = pk
        for mname, cif in matrices.items():
            lines = matrix_lines(cif, maps.wavelength, tmax)
            fit = np.sort(lines)[::-1][:n_fit_lines]
            lines_of[(gname, mname)] = fit
            full_of[(gname, mname)] = np.sort(lines)[::-1]
            mf = fit_matrix_scale(tc, prof / len(files), fit, maps.wavelength, scale_grid=grid,
                                  window_deg=fit_window(None, maps), min_rings=min_rings, ring_tol=ring_tol,
                                  ring_tol_deg=maps.pixel_deg)
            inside = np.isfinite(mf.scale) and grid[0] + 0.002 < mf.scale < grid[-1] - 0.002
            a_ref = read_cif(cif).lattice.a
            ranking.append(dict(geometry=gname, matrix=mname, n_rings=mf.n_rings, n_lines=mf.n_lines,
                                completeness=round(mf.completeness, 4),
                                scale=float(mf.scale), scale_spread=float(mf.scale_spread),
                                cell_a=float(mf.scale * a_ref), block_start=int(start),
                                ring_index=sorted(int(k) for k in mf.per_ring_scale), n_full=int(len(lines)),
                                scale_coarse=float(mf.scale_coarse),
                                ok=bool(inside and mf.consistent(spread_tol, min_rings))))
