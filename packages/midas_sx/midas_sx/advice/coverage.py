"""Coverage checks: rotation range and Friedel pairs, completeness and redundancy, the next setting, resolution.

Contract: see :mod:`midas_sx.advice` (measured numbers of THIS data only; the rule is stated in the finding and defined in ``THRESHOLDS``; ``gain`` only from a
measured or simulated source that is named; ``scope`` says what the check does not cover; ``CheckUnavailable`` when the quantity cannot be measured).

Inputs read from each run (a :class:`~midas_sx.workflow.Reduction` or any object with these attributes): ``geom`` (the refined Geometry; ``omega_first_deg`` is the centre
of frame 0), ``U`` (3x3, q_sample = U B h), ``obs`` (table with h k l), ``orientation`` (an OrientationResult, or None when the geometry was not refined). The merged data
come from ``ctx.merged()`` (columns h k l n F2 sigF2).

Definitions used here (stated once):

* *expected* reflections: the unique (Laue class, Friedel mates merged) reflections with d >= the detector's corner d_min (largest 2theta over the four corners; smallest over the runs) AND at
  least ``ring_coverage_min`` of whose Debye ring (by azimuth) lies on the detector of at least one run (a ring with only a sliver on the detector cannot be reached by most orientations). This denominator is GEOMETRIC: it counts everything the detector can see, including reflections that are intrinsically
  weak, so a good data set does not reach 100 % on it. ``completeness`` therefore also reports the figure among the strong half (calculated |F|^2 above the median) when the crystal
  has atoms.
* *reachable* reflections: expected ones with a predicted Ewald crossing on the detector inside the scan of at least one run (refined geometry and U). Expected but not reachable is
  an acquisition hole (rotation range / orientation); reachable but not observed is a detection problem (weak, masked, crowded).
* a Friedel *pair* is an unordered {h, -h} of signed indices, both with a crossing in the same run.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..assign import reciprocal_basis
from ..geometry import Scan, spot_directions
from ..predict import enumerate_hkl, predicted_crossings
from ..reduce import _transmission
from ..workflow import _corner_tth_max_deg
from . import THRESHOLDS, CheckUnavailable, Finding, register_check

# coverage-only thresholds, registered into the shared table at import (same convention as advice.THRESHOLDS: value, reason)
THRESHOLDS.update({
    "friedel_min_pairs": (30, "fewer in-run Friedel pairs than this cannot constrain tx and the wedge independently of the orientation fit (A39 runs: 35, 57, 47 observed; simulation: ~100 pairs give 0.01 deg)"),
    "direction_gap": (0.15, "a polar sector (angle between the reciprocal vector and the rotation axis) whose n>=1 completeness is this far below the overall figure is a hole in the rotation coverage"),
    "direction_gap_limiting": (0.5, "a sector completeness below this makes the direction hole limiting"),
    "multiplicity_n1_fraction": (0.2, "fraction of observed unique reflections seen only once above which scaling and rejection have too little to work with"),
    "resolution_signal_ratio": (1.3, "the I/sigma limit lies at d greater than this times the detector d_min: the data run out of signal before the detector edge"),
    "ring_coverage_min": (0.5, "a reflection counts as expected only if at least this fraction of its Debye ring (by azimuth) lies on the detector; the corner d_min counts rings of which only a sliver is on the detector and overstates what any orientation can give (a swung detector makes the inscribed-edge value meaningless)"),
    "setting_min_gain": (5, "the best candidate setting must add at least this many unique reflections to be recommended"),
})

_STEP_DEG = 15.0


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# shared products
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _scan_of(run) -> Scan:
    g = run.geom
    return Scan(float(g.omega_first_deg) - 0.5 * float(g.omega_step_deg), float(g.omega_step_deg), int(g.n_frames))


def _d_min_run(run) -> float:
    g = run.geom
    return float(g.wavelength_A / (2.0 * math.sin(math.radians(_corner_tth_max_deg(g) / 2.0))))


def _cache(ctx) -> dict:
    return ctx.options.setdefault("_coverage_cache", {})


def _rep_keys(sg, hkl: np.ndarray) -> np.ndarray:
    return np.array([sg.asu_representative(int(a), int(b), int(c)) for a, b, c in hkl], dtype=int).reshape(-1, 3)


def _ring_coverage(run, tth_deg: np.ndarray, n_eta: int = 36, grid: int = 160) -> np.ndarray:
    """Fraction of the azimuth (``n_eta`` bins) at which a ring of scattering angle ``tth_deg`` has detector pixels (a ``grid`` x ``grid`` subsample, band +-0.3 deg)."""
    g = run.geom
    rows = np.linspace(0.0, float(g.n_pix_z) - 1.0, grid)
    cols = np.linspace(0.0, float(g.n_pix_y) - 1.0, grid)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    _, tth, eta = spot_directions(rr.ravel(), cc.ravel(), g)
    tth = np.asarray(tth, float)
    eb = np.clip(((np.asarray(eta, float) + 180.0) / 360.0 * n_eta).astype(int), 0, n_eta - 1)
    out = np.zeros(len(tth_deg))
    order = np.argsort(tth)
    ts, es = tth[order], eb[order]
    for i, t in enumerate(tth_deg):
        a, b = np.searchsorted(ts, t - 0.3), np.searchsorted(ts, t + 0.3)
        if b > a:
            out[i] = len(np.unique(es[a:b])) / float(n_eta)
    return out


def _expected(ctx) -> dict:
    """Signed expected reflections, their unique class, d, and the unique list. Cached.

    Expected = unique reflections with d >= the corner d_min of the detector AND at least ``ring_coverage_min`` of their Debye ring on the detector of at least one run."""
    c = _cache(ctx)
    if "expected" in c:
        return c["expected"]
    if not ctx.runs:
        raise CheckUnavailable("no runs")
    d_corner = min(_d_min_run(r) for r in ctx.runs.values())
    d_target = ctx.options.get("d_min")                                  # a resolution TARGET (assess(..., d_min=...)): the expected set stops there
    d_cut = max(d_corner, float(d_target)) if d_target else d_corner
    sg, lat = ctx.space_group, ctx.lattice
    hkl = enumerate_hkl(lat, d_cut, sg)
    rep = _rep_keys(sg, hkl)
    uniq, inv = np.unique(rep, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    d_u = np.array([lat.d_spacing(int(a), int(b), int(c_)) for a, b, c_ in uniq])
    wl = float(next(iter(ctx.runs.values())).geom.wavelength_A)
    tth_u = np.degrees(2.0 * np.arcsin(np.clip(wl / (2.0 * d_u), 0.0, 1.0)))
    cov = np.max([_ring_coverage(r, tth_u) for r in ctx.runs.values()], axis=0)
    keep_u = cov >= THRESHOLDS["ring_coverage_min"][0]
    new_idx = -np.ones(len(uniq), int)
    new_idx[keep_u] = np.arange(int(keep_u.sum()))
    keep_h = keep_u[inv]
    hkl, rep, inv = hkl[keep_h], rep[keep_h], new_idx[inv[keep_h]]
    uniq, d_u = uniq[keep_u], d_u[keep_u]
    d_signed = np.array([lat.d_spacing(int(a), int(b), int(c_)) for a, b, c_ in hkl])
    out = dict(d_min=float(d_u.min()) if len(d_u) else d_corner, d_corner=d_corner, d_ref=d_cut, hkl=hkl, rep=rep, uniq=uniq, inv=inv, d_signed=d_signed, d_u=d_u,
               n_dropped_partial_rings=int((~keep_u).sum()))
    c["expected"] = out
    return out


def _predicted(ctx, name: str, scan: Optional[Scan] = None, hkl: Optional[np.ndarray] = None, tag: str = "own") -> pd.DataFrame:
    """Predicted crossings of ``hkl`` (default: all expected signed reflections) for run ``name`` (its refined U, geom and scan unless ``scan`` is given). Cached by ``tag``."""
    c = _cache(ctx)
    key = ("pred", name, tag)
    if key in c:
        return c[key]
    run = ctx.runs[name]
    ex = _expected(ctx)
    p = predicted_crossings(np.asarray(run.U, float), ctx.lattice, ex["hkl"] if hkl is None else hkl, run.geom, scan or _scan_of(run))
    c[key] = p
    return p


def _accessible(ctx, run, p: pd.DataFrame) -> np.ndarray:
    """Boolean mask of the predicted crossings ``p`` of ``run`` that the sample environment lets through: all of them without ``Acquisition.dac``; with it, those whose anvil
    transmission along the diffracted direction at their omega is positive (the package's own opening test, ``reduce._transmission``; the reduction rejects the rest as outside_opening)."""
    dac = getattr(ctx.acq, "dac", None)
    if dac is None or not len(p):
        return np.ones(len(p), bool)
    s, _, _ = spot_directions(p["row"].to_numpy(float), p["col"].to_numpy(float), run.geom)
    return np.asarray(_transmission(np.asarray(s, float), p["omega"].to_numpy(float), dac), float) > 0.0


def _merged_n(ctx) -> Dict[tuple, int]:
    """Unique-class key -> number of retained observations in the merged data."""
    m = ctx.merged().merged
    if m is None or len(m) == 0:
        raise CheckUnavailable("no merged reflections")
    rep = _rep_keys(ctx.space_group, m[["h", "k", "l"]].to_numpy(int))
    out: Dict[tuple, int] = {}
    for key, n in zip(map(tuple, rep), m["n"].to_numpy(int)):
        out[key] = out.get(key, 0) + int(n)
    return out


def _strong_half(ctx, ex) -> Optional[np.ndarray]:
    """Boolean per unique reflection: calculated |F|^2 above the median among the expected ones; None when the crystal has no atoms."""
    c = _cache(ctx)
    if "strong" in c:
        return c["strong"]
    atoms = getattr(ctx.crystal, "atoms", None)
    out = None
    if atoms:
        from midas_hkls.structure_factor import f2_normalised
        f2 = f2_normalised(ctx.lattice, ctx.space_group, atoms, ex["uniq"])
        out = f2 > np.median(f2)
    c["strong"] = out
    return out


def _shell_edges(d: np.ndarray, n: int = 6) -> np.ndarray:
    """Shell edges in d with equal volume in 1/d^3 (about equal numbers of reflections), from the largest to the smallest d."""
    s3 = np.linspace((1.0 / d.max()) ** 3, (1.0 / d.min()) ** 3 * 1.000001, n + 1)
    return s3 ** (-1.0 / 3.0)


def _fmt(x, nd=3):
    return float(np.round(x, nd))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. rotation range and Friedel pairs
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _pairs(p: pd.DataFrame) -> int:
    s = set(zip(p.h.astype(int), p.k.astype(int), p.l.astype(int)))
    return sum(1 for t in s if (-t[0], -t[1], -t[2]) in s) // 2


def _observed_pairs(obs: pd.DataFrame) -> int:
    s = set(zip(obs.h.astype(int), obs.k.astype(int), obs.l.astype(int)))
    return sum(1 for t in s if (-t[0], -t[1], -t[2]) in s) // 2


@register_check("scan_range")
def check_scan_range(ctx) -> List[Finding]:
    if not ctx.runs:
        raise CheckUnavailable("no runs")
    ex = _expected(ctx)
    thr_range = THRESHOLDS["friedel_min_range_deg"][0]
    thr_pairs = THRESHOLDS["friedel_min_pairs"][0]
    per = {}
    for name, run in ctx.runs.items():
        rng = float(run.geom.n_frames) * abs(float(run.geom.omega_step_deg))
        pred = _predicted(ctx, name)
        pairs_pred = _pairs(pred)
        pairs_obs = _observed_pairs(run.obs) if getattr(run, "obs", None) is not None and len(run.obs) else None
        per[name] = dict(range_deg=_fmt(rng, 1), pairs_predicted=int(pairs_pred), pairs_observed=None if pairs_obs is None else int(pairs_obs))
    short = [n for n, v in per.items() if v["range_deg"] < thr_range]
    if not short:
        return []
    # what a 180+ deg (here 360 deg) scan of the best-covered short run would give, same step, same orientation: a prediction
    ref_name = max(short, key=lambda n: per[n]["range_deg"])
    run = ctx.runs[ref_name]
    step = abs(float(run.geom.omega_step_deg))
    ref = {}
    if ctx.options.get("reference_360", True):
        s0 = _scan_of(run)
        scan360 = Scan(s0.omega_start, float(run.geom.omega_step_deg), int(round(360.0 / step)))
        ref_pairs = _pairs(_predicted(ctx, ref_name, scan=scan360, tag="ref360"))
        ref = dict(pairs_predicted_360deg_same_orientation=int(ref_pairs))
    refined = any(getattr(r, "orientation", None) is not None for r in ctx.runs.values())
    refined = bool(ctx.options.get("geometry_refined", refined))
    have = [v["pairs_observed"] if v["pairs_observed"] is not None else v["pairs_predicted"] for n, v in per.items() if n in short]
    n_pairs_min = int(min(have)) if have else 0
    # cross-run mates: signed hkl of one run whose mate is observed in ANOTHER run (needs the unknown relative rotation of the runs to be used; reported, not used here)
    cross = None
    names = list(ctx.runs)
    if len(names) > 1 and all(getattr(ctx.runs[n], "obs", None) is not None for n in names):
        sets = {n: set(zip(ctx.runs[n].obs.h.astype(int), ctx.runs[n].obs.k.astype(int), ctx.runs[n].obs.l.astype(int))) for n in names}
        allsigned = set().union(*sets.values())
        nin = sum(1 for t in allsigned if any(t in sets[a] and (-t[0], -t[1], -t[2]) in sets[b] for a in names for b in names if a != b))
        cross = dict(signed_hkl_with_mate_in_another_run=int(nin), signed_hkl_observed=int(len(allsigned)))
    limited_by_env = ctx.acq.max_range_deg is not None and all(per[n]["range_deg"] >= 0.9 * ctx.acq.max_range_deg for n in short)
    limiting = refined and n_pairs_min < thr_pairs and not limited_by_env
    measured = dict(per_run=per, short_runs=short, **ref, geometry_refined_from_data=refined, fewest_in_run_pairs_in_short_runs=n_pairs_min)
    if cross:
        measured["cross_run"] = cross
    if ctx.acq.max_range_deg is not None:
        measured["sample_environment_max_range_deg"] = ctx.acq.max_range_deg
    action = ("Scan at least 180 deg (360 deg if the sample environment allows): every reflection then has its Friedel mate in the same run, which gives an estimate of the detector tilt tx and "
              "the rotation-axis wedge that does not depend on the orientation fit, plus an internal check of absorption/centring. ")
    if limited_by_env:
        action = ("The scan range is set by the sample environment (max_range_deg given): this is not a choice. Compensate with more orientations of the crystal (see 'next_setting') "
                  "and a calibrant/standard crystal to fix the geometry, because without in-run mates the geometry rests on the orientation fit alone.")
    elif limiting:
        action += "Needed here: the wedge/tilts were fitted from the data and only %d in-run pairs (fewest over the short runs) exist to check them (fewer than %d)." % (n_pairs_min, thr_pairs)
    elif not refined:
        action += "Not urgent here: the geometry was not fitted from the data."
    else:
        action += "Not urgent here: %d in-run pairs (fewest over the short runs) exist, enough to check the fitted wedge/tilts (at least %d); a longer scan would still make the check complete." % (n_pairs_min, thr_pairs)
    gain = ("simulated (validation/real/PREREG_friedel_geometry.md, positive control): about 1500 pairs of a 360 deg scan give tx +-0.002 deg and wedge +-0.001 deg; about 100 pairs give +-0.01 deg. "
            "On the real L-alanine runs (170 deg, 35-57 in-run pairs) the pair estimates agreed with the orientation fit within 0.035 deg (INCONCLUSIVE by the registered rule).")
    return [Finding(
        id="scan_range", severity="limiting" if limiting else "note",
        title="rotation range shorter than 180 deg: few or no in-run Friedel pairs",
        measured=measured,
        threshold=f"range < {thr_range:g} deg means (almost) no in-run Friedel pairs; limiting only if the geometry was refined from the data AND fewer than {thr_pairs} in-run pairs exist and the range is not set by the sample environment",
        evidence="run.geom (omega_step, n_frames), predicted_crossings of all expected signed hkl with the run's refined U and geometry (pairs = unordered {h,-h} both crossing), run.obs for observed pairs",
        action=action, gain=gain,
        scope="Predicted pair counts are geometric (a crossing on the detector); observed counts are what survived integration. Pairs between different runs need the unknown relative rotation of the runs and are reported, not used.")]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. completeness, redundancy, direction holes
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _coverage_numbers(ctx) -> dict:
    c = _cache(ctx)
    if "cov" in c:
        return c["cov"]
    ex = _expected(ctx)
    nmap = _merged_n(ctx)
    uniq = ex["uniq"]
    n_obs = np.array([nmap.get(tuple(u), 0) for u in uniq])
    # reachable: some signed equivalent has a predicted crossing in some run
    reach = np.zeros(len(uniq), bool)
    key_to_idx = {tuple(u): i for i, u in enumerate(uniq)}
    hkl_idx = {tuple(h): i for i, h in enumerate(ex["hkl"])}
    cross_any = np.zeros(len(uniq), bool)          # some crossing exists, whether or not the cell lets it through
    for name in ctx.runs:
        p = _predicted(ctx, name)
        if len(p):
            idx = np.array([hkl_idx[(int(a), int(b), int(c_))] for a, b, c_ in zip(p.h, p.k, p.l)])
            cross_any[np.unique(ex["inv"][idx])] = True
            acc = _accessible(ctx, ctx.runs[name], p)
            if acc.any():
                reach[np.unique(ex["inv"][idx[acc]])] = True
    # polar angle between q (sample frame) and the rotation axis; the most favourable presentation over runs and signed equivalents
    A = np.asarray(reciprocal_basis(ctx.lattice), float)
    psi_best = np.zeros(len(uniq))
    for run in ctx.runs.values():
        q = (np.asarray(run.U, float) @ A @ ex["hkl"].T).T
        cosz = np.abs(q[:, 2]) / np.linalg.norm(q, axis=1)
        psi = np.degrees(np.arccos(np.clip(cosz, 0.0, 1.0)))            # 0 = along the axis, 90 = equatorial
        np.maximum.at(psi_best, ex["inv"], psi)
    out = dict(n_obs=n_obs, reach=reach, cross_any=cross_any, aperture=getattr(ctx.acq, "dac", None) is not None, psi_best=psi_best, strong=_strong_half(ctx, ex))
    c["cov"] = out
    return out


@register_check("completeness_redundancy")
def check_completeness(ctx) -> List[Finding]:
    ex = _expected(ctx)
    cv = _coverage_numbers(ctx)
    n_obs, reach, psi, strong = cv["n_obs"], cv["reach"], cv["psi_best"], cv["strong"]
    N_geom = len(n_obs)
    if cv["aperture"]:                                  # a declared cell aperture: the denominator is what the scans can physically reach (Acquisition.dac)
        keep = reach.copy()
        n_blocked = int((cv["cross_any"] & ~reach).sum())
        n_obs, psi, reach = n_obs[keep], psi[keep], reach[keep]
        strong = None if strong is None else strong[keep]
        ex = dict(ex, d_u=ex["d_u"][keep])
    else:
        n_blocked = None
    N = len(n_obs)
    ctx.numbers["completeness"] = dict(n_expected=int(N_geom), n_accessible=int(N), blocked_by_aperture=n_blocked, frac_n_ge_1=round(float(np.mean(n_obs >= 1)), 4), frac_n_ge_2=round(float(np.mean(n_obs >= 2)), 4),
                                       mean_multiplicity=round(float(np.mean(n_obs[n_obs >= 1])), 2) if (n_obs >= 1).any() else None)      # reported also when no finding is raised
    c1 = float((n_obs >= 1).mean())
    c2 = float((n_obs >= 2).mean())
    thr1, thr2 = THRESHOLDS["completeness_n1"][0], THRESHOLDS["completeness_n2"][0]
    findings: List[Finding] = []
    edges = _shell_edges(ex["d_u"])
    shells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ex["d_u"] <= lo) & (ex["d_u"] > hi)
        if m.any():
            shells.append(dict(d_range=f"{lo:.2f}-{hi:.2f}", expected=int(m.sum()), n1=_fmt((n_obs[m] >= 1).mean(), 3), n2=_fmt((n_obs[m] >= 2).mean(), 3)))
    base = dict(detector_corner_d_min_A=_fmt(ex["d_corner"]), effective_d_min_A=_fmt(ex["d_min"]), unique_dropped_partial_rings=ex["n_dropped_partial_rings"], expected_unique=int(N_geom), accessible_unique=int(N), blocked_by_aperture=n_blocked, reachable_unique=int(reach.sum()), observed_n1=int((n_obs >= 1).sum()), observed_n2=int((n_obs >= 2).sum()),
                completeness_n1=_fmt(c1), completeness_n2=_fmt(c2), by_shell=shells)
    if strong is not None:
        s1, s2 = float((n_obs[strong] >= 1).mean()), float((n_obs[strong] >= 2).mean())
        base.update(strong_half_completeness_n1=_fmt(s1), strong_half_completeness_n2=_fmt(s2))
    else:
        s1 = s2 = None
        base["strong_half_completeness"] = "not computed: the crystal has no atoms"
    below = (c2 < thr2) or (c1 < thr1)
    if below:
        strong_below = (s2 is not None and s2 < thr2)
        lim = strong_below or (s2 is None and c2 < thr2)
        n_unreach = int(N - reach.sum())
        n_undetected = int(reach.sum() - ((n_obs >= 1) & reach).sum())
        base.update(expected_but_not_reachable=n_unreach, reachable_but_not_observed=n_undetected)
        if n_unreach >= n_undetected:
            action = ("Most of the missing reflections never cross the Ewald sphere inside the scans taken: add rotation runs at other crystal orientations (see 'next_setting' for the "
                      "predicted best one) or lengthen the scan; exposure will not help.")
        else:
            action = ("Most of the missing reflections do cross the Ewald sphere inside the scans but were not observed: they are weak, masked or crowded. Check the 'systematic_floor' and "
                      "'resolution_limit' findings before changing anything; a longer exposure helps only if the floor is counting-limited.")
        findings.append(Finding(
            id="completeness", severity="limiting" if lim else "note",
            title="unique reflections inside the detector d_min are missing" if lim else "completeness below the bar on the geometric denominator only (the missing ones are the weak half)",
            measured=base,
            threshold=f"completeness n>=2 < {thr2:g} or n>=1 < {thr1:g} on the geometric denominator; limiting only if the strong half (calculated |F|^2 above the median) is also below {thr2:g}",
            evidence="enumerate_hkl to the detector d_min, Laue-unique classes (SpaceGroup.asu_representative), merged n from ctx.merged(); reachable = predicted_crossings of any run",
            action=action,
            scope="The denominator is geometric (everything the detector can see, intrinsically weak reflections included), so 100 % is not attainable; the strong-half figure uses calculated |F|^2 of the supplied crystal. "
                  "Anomalous differences are not modelled: Friedel mates count as one reflection."))
    # redundancy
    m = ctx.merged().merged
    nn = m["n"].to_numpy(int)
    n1f = float((nn == 1).mean())
    mean_m = float(nn.mean())
    hist = {"1": int((nn == 1).sum()), "2": int((nn == 2).sum()), "3": int((nn == 3).sum()), "4": int((nn == 4).sum()), "5-8": int(((nn >= 5) & (nn <= 8)).sum()), "9+": int((nn >= 9).sum())}
    thr_m, thr_f = THRESHOLDS["mean_multiplicity"][0], THRESHOLDS["multiplicity_n1_fraction"][0]
    if mean_m < thr_m or n1f > thr_f:
        lim = mean_m < thr_m and n1f > thr_f
        findings.append(Finding(
            id="redundancy", severity="limiting" if lim else "note",
            title="low redundancy: scaling, outlier rejection and the error model have little to work with",
            measured=dict(mean_multiplicity=_fmt(mean_m, 2), fraction_seen_once=_fmt(n1f), multiplicity_histogram=hist, observed_unique=int(len(nn))),
            threshold=f"mean multiplicity < {thr_m:g} or more than {thr_f:g} of the observed unique reflections seen once; limiting if both",
            evidence="merged n (retained observations per unique reflection) from ctx.merged()",
            action="Collect more observations of each reflection: another run (see 'next_setting' for whether a repeat or a re-orientation adds more) or a longer scan. The smooth scale, the robust merge and the systematic-floor estimate all need equivalents spread over each run.",
            gain="measured on L-alanine (3 runs, mean multiplicity ~5, 10 % seen once): scale_smooth + robust merge took R_int 15.7 -> 6.8 % (CLAIM_floor.md); no measurement exists for the low-redundancy case.",
            scope="n counts retained observations after the robust merge's rejection; Friedel mates are merged."))
    # direction holes
    if len(ctx.runs) >= 1:
        sectors = []
        for lo, hi in ((0, 30), (30, 60), (60, 90.001)):
            m_ = (psi >= lo) & (psi < hi)
            if m_.sum() >= 10:
                sectors.append(dict(psi_deg=f"{lo}-{min(hi, 90):g}", expected=int(m_.sum()), n1=_fmt((n_obs[m_] >= 1).mean(), 3)))
        if sectors:
            worst = min(sectors, key=lambda s: s["n1"])
            gap = THRESHOLDS["direction_gap"][0]
            if worst["n1"] < c1 - gap:
                lim = worst["n1"] < THRESHOLDS["direction_gap_limiting"][0]
                findings.append(Finding(
                    id="direction_holes", severity="limiting" if lim else "note",
                    title="reflections near the rotation axis are missing (a cone around the axis)",
                    measured=dict(worst_sector=worst, sectors=sectors, overall_n1=_fmt(c1)),
                    threshold=f"a polar sector (angle of q from the rotation axis, most favourable presentation over the runs and signed equivalents) with n>=1 completeness more than {gap:g} below the overall figure; limiting below {THRESHOLDS['direction_gap_limiting'][0]:g}",
                    evidence="q_sample = U B h per run (psi = angle to the stage axis), unique classes keep their largest psi over runs and equivalents; merged n from ctx.merged()",
                    action="Add a run with the crystal re-oriented so that the axis points elsewhere in reciprocal space (re-mount, or tilt with chi/phi): reflections near the rotation axis are the ones a single rotation cannot reach.",
                    scope="Sectors are defined in the rotation-stage frame of each run; a goniometer that moves the axis between runs is accounted for only through each run's refined U."))
    return findings


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. next setting
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _rotations(step_deg: float):
    from midas_stress.orientation import axis_angle_to_orient_mat as aa
    cands = [("identity", 0.0, np.eye(3))]
    for axis_name, axis in (("X (beam)", (1.0, 0.0, 0.0)), ("Y (horizontal, across the beam)", (0.0, 1.0, 0.0))):
        a = step_deg
        while a < 360.0 - 1e-9:
            cands.append((axis_name, a, np.asarray(aa(axis, a), float)))
            a += step_deg
    return cands


@register_check("next_setting")
def check_next_setting(ctx) -> List[Finding]:
    if not ctx.options.get("suggest_setting", True):
        raise CheckUnavailable("suggest_setting=False")
    ex = _expected(ctx)
    cv = _coverage_numbers(ctx)
    n_obs = cv["n_obs"]
    N = len(n_obs)
    acc_mask = cv["reach"] if cv["aperture"] else np.ones(N, bool)       # with a declared cell aperture only the accessible classes can be gained
    N_acc = int(acc_mask.sum())
    missing = np.flatnonzero((n_obs < 2) & acc_mask)         # unique reflections one more run could bring to n >= 2
    min_gain = THRESHOLDS["setting_min_gain"][0]
    c2 = float((n_obs[acc_mask] >= 2).mean()) if N_acc else 1.0
    if len(missing) < min_gain or c2 >= THRESHOLDS["completeness_n2"][0]:
        return []
    t0 = time.time()
    # base run: the widest scan; the extra run is assumed to repeat its scan (same geometry, same omega range) at a different crystal orientation
    base_name = max(ctx.runs, key=lambda n: float(ctx.runs[n].geom.n_frames) * abs(float(ctx.runs[n].geom.omega_step_deg)))
    run = ctx.runs[base_name]
    scan = _scan_of(run)
    pos_of = -np.ones(N, int)
    pos_of[missing] = np.arange(len(missing))
    miss_mask = pos_of[ex["inv"]] >= 0
    hkl_m = ex["hkl"][miss_mask]
    pos_m = pos_of[ex["inv"][miss_mask]]
    hkl_idx = {tuple(h): i for i, h in enumerate(hkl_m)}
    n_m = n_obs[missing]
    U0 = np.asarray(run.U, float)
    rows = []
    for axis_name, ang, Rm in _rotations(float(ctx.options.get("setting_step_deg", _STEP_DEG))):
        p = predicted_crossings(Rm @ U0, ctx.lattice, hkl_m, run.geom, scan)
        if len(p) and cv["aperture"]:
            p = p[_accessible(ctx, run, p)]
        if len(p):
            idx = np.array([hkl_idx[(int(a), int(b), int(c_))] for a, b, c_ in zip(p.h, p.k, p.l)])
            cnt = np.bincount(pos_m[idx], minlength=len(missing))      # predicted crossings per missing unique reflection in the extra run
        else:
            cnt = np.zeros(len(missing), int)
        rows.append(dict(axis=axis_name, angle_deg=ang, gain_n2=int(((n_m + cnt) >= 2).sum()), new_seen=int(((n_m == 0) & (cnt >= 1)).sum())))
    df = pd.DataFrame(rows)
    ident = df[df.axis == "identity"].iloc[0]
    cand = df[df.axis != "identity"]
    best = cand.sort_values(["gain_n2", "new_seen", "angle_deg"], ascending=[False, False, True]).iloc[0]
    if int(max(best.gain_n2, ident.gain_n2)) < min_gain:
        return []
    reorient_helps = (int(best.gain_n2) - int(ident.gain_n2)) >= min_gain or int(best.new_seen) > int(ident.new_seen)
    n2_now = int(((n_obs >= 2) & acc_mask).sum())
    pick = best if reorient_helps else ident
    pred_c2 = (n2_now + int(pick.gain_n2)) / max(N_acc, 1)
    s_acc = None if cv["strong"] is None else (cv["strong"] & acc_mask)
    lim = c2 < THRESHOLDS["completeness_n2"][0] and (s_acc is None or not s_acc.any() or float((n_obs[s_acc] >= 2).mean()) < THRESHOLDS["completeness_n2"][0])
    measured = dict(base_run=base_name, best_rotation=dict(axis=best.axis, angle_deg=float(best.angle_deg)), predicted_gain_n2_best_rotation=int(best.gain_n2),
                    predicted_new_reflections_best_rotation=int(best.new_seen), repeat_same_orientation_control=dict(gain_n2=int(ident.gain_n2), new_reflections=int(ident.new_seen)),
                    unique_below_n2_now=int(len(missing)), completeness_n2_now=_fmt(c2), predicted_completeness_n2_after_one_more_run=_fmt(pred_c2),
                    candidates_evaluated=int(len(df)), seconds=_fmt(time.time() - t0, 1))
    if reorient_helps:
        action = (f"Re-orient the crystal by about {best.angle_deg:g} deg about the {best.axis} axis of the rotation stage at omega = 0 (a goniometer tilt/chi or a re-mount) and collect one more run with the same scan. "
                  f"PREDICTED: it brings {int(best.gain_n2)} of the {len(missing)} unique reflections below n = 2 to n >= 2 and reaches {int(best.new_seen)} not seen so far (a repeat of the old orientation: {int(ident.gain_n2)} and {int(ident.new_seen)}).")
        title = "one more run at a re-oriented crystal is predicted to add the most missing reflections"
    else:
        action = (f"One more run adds as much at the SAME orientation as at any tilt tried (PREDICTED: {int(ident.gain_n2)} of the {len(missing)} unique reflections below n = 2 reach n >= 2; best tilted setting "
                  f"{int(best.gain_n2)}): the reflections are reachable, they are just seen too few times. Repeat the scan, or re-orient the crystal if you also want independent absorption/centring information.")
        title = "the missing reflections are reachable: they need more observations, not new orientations"
    return [Finding(
        id="next_setting", severity="limiting" if lim else "note", title=title, measured=measured,
        threshold=f"recommended when completeness n>=2 is below {THRESHOLDS['completeness_n2'][0]:g} and one more run is predicted to bring at least {min_gain} unique reflections to n >= 2",
        evidence="predicted_crossings of the unique reflections below n = 2 (all signed equivalents) for the refined U of the widest run rotated by candidate rotations about the stage X and Y axes in steps of the "
                 "option setting_step_deg (15 deg), midas_stress axis_angle_to_orient_mat, same geometry and omega range; a reflection's predicted new observations = its crossings in the extra run",
        action=action,
        gain="PREDICTED from geometry only (a forecast, not a measurement): every crossing on the detector is assumed to be observed, so the real gain is smaller by the fraction of weak or crowded reflections.",
        scope="The sign and zero of the goniometer angle are those of the refined U of the first run in the stage frame; check the direction against your goniometer. Absorption by the environment at the new orientation, "
              "shadowing and collisions are not modelled.")]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. resolution limit
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@register_check("resolution_limit")
def check_resolution_limit(ctx) -> List[Finding]:
    ex = _expected(ctx)
    m = ctx.merged().merged
    if m is None or len(m) == 0:
        raise CheckUnavailable("no merged reflections")
    m = m[(m.sigF2 > 0)].copy()
    if len(m) < 20:
        raise CheckUnavailable("fewer than 20 merged reflections with a sigma")
    lat = ctx.lattice
    d = np.array([lat.d_spacing(int(a), int(b), int(c_)) for a, b, c_ in m[["h", "k", "l"]].to_numpy(int)])
    isg = (m.F2 / m.sigF2).to_numpy(float)
    edges = _shell_edges(d, 8)
    shells, med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (d <= lo) & (d > hi)
        if s.sum() >= 5:
            shells.append(dict(d_range=f"{lo:.2f}-{hi:.2f}", n=int(s.sum()), median_I_over_sigma=_fmt(np.median(isg[s]), 1)))
            med.append((hi, float(np.median(isg[s]))))
    if not shells:
        raise CheckUnavailable("too few reflections per shell")
    lim_val = THRESHOLDS["i_over_sigma_limit"][0]
    d_lim = None
    for (hi, v) in med:
        if v < lim_val:
            d_lim = hi
            break
    ratio_thr = THRESHOLDS["resolution_signal_ratio"][0]
    d_obs_min = float(d.min())
    # survivorship guard (PREREG_advice.md Amendment 3): the median I/sigma above is taken over OBSERVED reflections, which are the strong ones. Count the expected classes that were never seen.
    cv = _coverage_numbers(ctx)
    acc = cv["reach"] if cv["aperture"] else np.ones(len(cv["n_obs"]), bool)
    d_acc, n_acc = ex["d_u"][acc], cv["n_obs"][acc]
    shell_n1 = []
    if len(d_acc) >= 30:
        e6 = _shell_edges(d_acc, 6)
        for lo, hi in zip(e6[:-1], e6[1:]):
            sm = (d_acc <= lo) & (d_acc > hi)
            if sm.any():
                shell_n1.append((hi, lo, float((n_acc[sm] >= 1).mean()), int(sm.sum())))
    if shell_n1 and shell_n1[-1][2] < 0.5:
        d_sig = next(hi for hi, lo, c1_, nn_ in shell_n1 if c1_ < 0.5 and all(x[2] < 0.5 for x in shell_n1[shell_n1.index((hi, lo, c1_, nn_)):]))
        return [Finding(
            id="resolution_signal_limited", severity="limiting",
            title="the data run out of signal before the detector edge: most of the expected reflections in the outermost shell were never detected",
            measured=dict(detector_corner_d_min_A=_fmt(ex["d_corner"]), effective_d_min_A=_fmt(ex["d_min"]), smallest_d_observed_A=_fmt(d_obs_min), resolution_limit_A=_fmt(d_sig),
                          outer_shell_n1_completeness=_fmt(shell_n1[-1][2], 3), shells_n1_completeness=[dict(d_range=f"{lo:.2f}-{hi:.2f}", expected=nn_, n1=_fmt(c1_, 3)) for hi, lo, c1_, nn_ in shell_n1],
                          median_I_over_sigma_of_observed=[sh["median_I_over_sigma"] for sh in shells]),
            threshold="the outermost of 6 equal-volume shells of the expected (accessible) classes has n>=1 completeness below 0.5; this overrides a high median I/sigma of the reflections that WERE observed",
            evidence="n>=1 completeness by shell of the expected unique classes (coverage numbers, aperture-aware) against the merged data; median F2/sigF2 of the observed ones listed for comparison",
            action="A longer exposure, a brighter beam, a lower background or a lower temperature raise the signal of the high-order reflections; more redundancy helps only if the floor is counting-limited (see "
                   "'systematic_floor'). The detector is not the limit here: it sees these reflections, the data do not contain them.",
            scope="completeness here counts reflections that are intrinsically weak (heavy displacement, high order) as missing; a crystal that simply does not diffract that far gives the same finding.")]
    measured = dict(detector_corner_d_min_A=_fmt(ex["d_corner"]), effective_d_min_A=_fmt(ex["d_min"]), smallest_d_observed_A=_fmt(d_obs_min), shells=shells, i_over_sigma_limit=lim_val)
    note_sigma = "the merged sigma includes the error-model term when the merge was robust, so I/sigma is bounded by about 1/g"
    ref = ex["d_ref"]                                          # the detector corner d_min, or the resolution target when one was given
    if d_lim is None:
        return [Finding(
            id="resolution_detector_limited", severity="note",
            title="the data reach the detector edge with signal to spare: the detector coverage cuts the resolution",
            measured=dict(measured, resolution_limit="beyond the smallest d observed"),
            threshold=f"median I/sigma stays >= {lim_val:g} in every shell of the merged data, down to d = {d_obs_min:.2f} A (detector corner d_min {ref:.2f} A, effective d_min {ex['d_min']:.2f} A)",
            evidence="median F2/sigF2 of the merged unique reflections by shell (equal volume in 1/d^3 over the observed d range), ctx.merged(); d from the lattice",
            action="If higher resolution is wanted (anisotropic displacement parameters, light atoms next to heavy ones), move the detector closer or swing it to higher 2theta; the signal is there.",
            scope=f"{note_sigma}. Reflections not observed at all are not in the median (see the completeness finding for them)."),
        ]
    ratio = d_lim / ref
    if ratio > ratio_thr:
        return [Finding(
            id="resolution_signal_limited", severity="limiting",
            title="the data run out of signal before the detector edge",
            measured=dict(measured, resolution_limit_A=_fmt(d_lim), ratio_to_detector_corner_d_min=_fmt(ratio, 2)),
            threshold=f"the median I/sigma falls below {lim_val:g} at d = {d_lim:.2f} A, more than {ratio_thr:g} x the detector corner d_min ({ref:.2f} A)",
            evidence="median F2/sigF2 of the merged unique reflections by shell (equal volume in 1/d^3 over the observed d range), ctx.merged()",
            action="A longer exposure or more redundancy raises I/sigma ONLY if the data are counting-limited: check the 'systematic_floor' finding first (if the floor exceeds the counting prediction, more exposure will not help). "
                   "Other levers: a smaller crystal-to-detector distance does not add signal; lower background (beam stop, air path, mount) and a lower temperature do.",
            scope=f"{note_sigma}."),
        ]
    return [Finding(
        id="resolution_detector_limited", severity="note",
        title="signal and detector edge agree: the resolution limit is about the detector's d_min",
        measured=dict(measured, resolution_limit_A=_fmt(d_lim), ratio_to_detector_corner_d_min=_fmt(ratio, 2)),
        threshold=f"the median I/sigma falls below {lim_val:g} at d = {d_lim:.2f} A, within {ratio_thr:g} x the detector corner d_min ({ref:.2f} A)",
        evidence="median F2/sigF2 of the merged unique reflections by shell (equal volume in 1/d^3 over the observed d range), ctx.merged()",
        action="For more resolution both more signal (exposure, redundancy, background) and a closer or swung detector are needed.",
        scope=f"{note_sigma}.")]
