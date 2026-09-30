"""Validate the column-content pipeline on synthetic columns of KNOWN content, at YOUR geometry, cell, wedge, kernel,
crowding and spread classes -- before reading real data. The recall table this returns is the completeness statement.

    spec = ValidationSpec(geometries=[geom_a, geom_b], a=..., c=..., sg=..., kernel=measured_kernel, ...)
    out  = validate(spec, outdir, workers=30)      # stage A (round 0 + fit + residual spots), B (g_disc), C (pipeline), D (gates)

The gates default to the values registered for the monochromatic implementation (manuals/column-content/ENVELOPE.md §2);
pass your own ``gates`` if your preregistration differs. Write the preregistration BEFORE running this.

Generalised from the development driver used for the registered validation (``mono_val.py``, 2026-09-27). With the
default spec it builds the identical columns (same seeds, same draws, same order).
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from midas_defect.geometry import Geometry

DEFAULT_GATES = dict(V1_false_ub=0.10, V2_recall=0.90, V3_share_err=0.20, V4_point_deg=0.05, V6_unexplained=0.15,
                     V7_spread_recall=0.85, V8_spread_spearman=0.6)


@dataclass
class ValidationSpec:
    geometries: Sequence[Geometry]                       # cycled over columns (e.g. two wedges)
    a: float
    c: float
    sg: int
    kernel_sigmas: Tuple[float, float, float]            # MEASURED (sig_frame, sig_rad, sig_tan), e.g. estimate_kernel
    b: Optional[float] = None                            # None = tetragonal (a = b)
    hkl_max: Tuple[int, int, int] = (8, 8, 30)
    n_columns: int = 240
    n_domains: Sequence[int] = (1, 2, 3, 4)              # equal blocks of columns
    seed0: int = 40000
    near_frac: float = 0.4
    near_deltas_deg: Sequence[float] = (1.0, 3.0, 14.0)
    spread_mix: Sequence[Tuple[str, float, Sequence[float]]] = (("point", 0.35, (0.0,)), ("cloud", 0.30, (0.1, 0.3)),
                                                                ("streak", 0.35, (0.2, 0.5, 1.0)))
    brightness_max: float = 31.6                          # log-uniform brightness on [1, brightness_max]
    kernel_scale_range: Tuple[float, float] = (0.9, 1.1)
    peak_range: Tuple[float, float] = (5e3, 1e5)
    foreign_frac: float = 0.5
    foreign_cell: Optional[Tuple[float, int, Tuple[int, int, int]]] = (3.567, 227, (6, 6, 6))   # (a cubic, sg, hkl_max)
    powder_two_theta_deg: Sequence[float] = ()
    threshold: float = 200.0
    K: int = 24
    fit_kw: dict = field(default_factory=lambda: dict(n_iter=300, lr=3e-4, inits=(0.05, 0.4)))
    find_kw: dict = field(default_factory=lambda: dict(seed_from_nominal=True))
    search_fn: Optional[Callable] = None                 # (df, geom, q, I) -> [(U, n)]; default find_domains(a, c, sg)
    gates: dict = field(default_factory=lambda: dict(DEFAULT_GATES))


def _cells(spec):
    from midas_defect.synthetic import _b_matrix, _hkl_candidates
    B = _b_matrix(spec.a, spec.b or spec.a, spec.c)
    return B, _hkl_candidates(*spec.hkl_max, spec.sg)


def build_column(spec: ValidationSpec, i: int):
    """Deterministic column i: (frames, truth dict, geom, B, hkl, nominal_kernel)."""
    import torch
    from . import DomainSpec, GaussKernel3D, rotvec_to_matrix, synthetic_column
    from midas_defect.synthetic import _b_matrix, _hkl_candidates
    rng = np.random.default_rng(spec.seed0 + i)
    g = spec.geometries[i % len(spec.geometries)]
    B, hkl = _cells(spec)
    block = max(spec.n_columns // len(spec.n_domains), 1)
    N = spec.n_domains[min(i // block, len(spec.n_domains) - 1)]

    def rrot():
        M = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(M) < 0:
            M[:, 0] *= -1
        return M
    specs, kinds, partners, deltas = [], [], [], []
    for d in range(N):
        if d > 0 and rng.uniform() < spec.near_frac:
            j = int(rng.integers(0, d)); dl = float(rng.choice(list(spec.near_deltas_deg)))
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
            U = rotvec_to_matrix(torch.as_tensor(ax * math.radians(dl), dtype=torch.float64)).numpy() @ specs[j].U
            kinds.append("near"); partners.append(j); deltas.append(dl)
        else:
            U = rrot(); kinds.append("random"); partners.append(-1); deltas.append(0.0)
        m = rng.uniform(); acc = 0.0; sp, par = spec.spread_mix[-1][0], float(spec.spread_mix[-1][2][0])
        for name, frac, pars in spec.spread_mix:
            acc += frac
            if m < acc:
                sp, par = name, (0.0 if name == "point" else float(rng.choice(list(pars)))); break
        specs.append(DomainSpec(U=U, brightness=float(np.exp(rng.uniform(0, math.log(spec.brightness_max)))),
                                spread=sp, param_deg=par))
    s_true = float(rng.uniform(*spec.kernel_scale_range))
    ks = spec.kernel_sigmas
    ktrue = GaussKernel3D(ks[0] * s_true, ks[1] * s_true, ks[2] * s_true, g.bcz_px, g.bcy_px)
    foreign = None
    if spec.foreign_cell is not None and rng.uniform() < spec.foreign_frac:
        fa, fsg, fh = spec.foreign_cell
        foreign = dict(U=rrot(), B=_b_matrix(fa, fa, fa), hkl_all=_hkl_candidates(*fh, fsg),
                       brightness=float(np.exp(rng.uniform(0, math.log(spec.brightness_max)))))
    frames, truth = synthetic_column(specs, B=B, hkl_all=hkl, geom=g, kernel=ktrue,
                                     peak_counts=float(np.exp(rng.uniform(math.log(spec.peak_range[0]), math.log(spec.peak_range[1])))),
                                     foreign=foreign, powder_two_theta_deg=tuple(spec.powder_two_theta_deg), seed=spec.seed0 + i)
    rms = []
    for rv in truth.comp_rotvecs:
        d = rv - rv.mean(0); rms.append(math.degrees(math.sqrt(float(np.trace(d.T @ d / len(rv))))))
    T = dict(i=i, N=N, geom=i % len(spec.geometries), U=[u.tolist() for u in truth.U], share=truth.share.tolist(),
             spread=truth.spread, param=truth.param_deg.tolist(), rms_deg=rms, kind=kinds, partner=partners, delta=deltas,
             foreign=foreign is not None, foreign_share=truth.foreign_share, s_true=s_true)
    return frames, T, g, B, hkl, GaussKernel3D(*ks, g.bcz_px, g.bcy_px)


def _search(spec, df, g, q, I):
    from .pipeline import search
    if spec.search_fn is not None:
        return spec.search_fn(df, g, q, I)
    return search(df, g, q, I, a=spec.a, c=spec.c, sg=spec.sg, find_kw=spec.find_kw)


def _stage_a(args):
    spec, i, out = args
    import torch
    torch.set_num_threads(1)
    from . import ColumnFit, ingest_column, spots_to_q
    from midas_defect.ingest import find_blobs_3d
    try:
        frames, T, g, B, hkl, knom = build_column(spec, i)
        ing = ingest_column(frames, g, threshold=spec.threshold)
        sols = _search(spec, ing.spots, g, ing.q, ing.intensity)
        res = dict(i=i, truth=T, n_spots=len(ing.spots), round0=[dict(U=np.asarray(U).tolist(), n=int(n)) for U, n in sols])
        if sols:
            fit = ColumnFit(ing.sub, ing.mask, ing.labels, [U for U, _ in sols], B, hkl, g, knom, K=spec.K)
            fit.fit(**spec.fit_kw)
            res["report0"] = fit.report()
            o = find_blobs_3d(fit.residual_stack(), ing.mask[0], threshold=spec.threshold)
            rdf = o[0] if isinstance(o, tuple) else o
            np.savez_compressed(f"{out}/resid_{i:03d}.npz", q=spots_to_q(rdf, g),
                                **{k: (np.asarray(rdf[c_], float) if len(rdf) else np.zeros(0))
                                   for k, c_ in (("I", "integrated"), ("row", "row"), ("col", "col"), ("frame", "frame"))})
        json.dump(res, open(f"{out}/A_{i:03d}.json", "w"), default=float)
        return dict(i=i, ok=True)
    except Exception as e:
        json.dump(dict(i=i, error=repr(e)), open(f"{out}/A_{i:03d}.json", "w"))
        return dict(i=i, error=repr(e))


def _stage_c(args):
    spec, i, out, gd = args
    import torch
    torch.set_num_threads(1)
    from . import ingest_column, run_column
    try:
        frames, T, g, B, hkl, knom = build_column(spec, i)
        r = run_column(frames, g, a=spec.a, c=spec.c, sg=spec.sg, B=B, hkl_all=hkl, kernel=knom, g_disc=gd, K=spec.K,
                       fit_kw=spec.fit_kw, find_kw=spec.find_kw, ingest_kw=dict(threshold=spec.threshold),
                       search_fn=spec.search_fn)
        json.dump(dict(i=i, truth=T, rounds=r.rounds, report=r.report, U=[u.tolist() for u in r.U]),
                  open(f"{out}/C_{i:03d}.json", "w"), default=float)
        return dict(i=i, ok=True)
    except Exception as e:
        json.dump(dict(i=i, error=repr(e)), open(f"{out}/C_{i:03d}.json", "w"))
        return dict(i=i, error=repr(e))


def measure_gate(spec: ValidationSpec, out: str, n_draws: int = 5):
    import pandas as pd
    from .pipeline import discovery_gate
    Q, I, D = [], [], []
    for i in range(spec.n_columns):
        p = f"{out}/resid_{i:03d}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        if len(z["q"]) < 3:
            continue
        Q.append(z["q"]); I.append(z["I"]); D.append(pd.DataFrame(dict(row=z["row"], col=z["col"], frame=z["frame"], integrated=z["I"])))
    gd, info = discovery_gate(Q, I, D, spec.geometries[0], a=spec.a, c=spec.c, sg=spec.sg, n_draws=n_draws, find_kw=spec.find_kw)
    info.update(g_disc=gd, n_columns=len(Q))
    json.dump(info, open(f"{out}/gdisc.json", "w"), indent=1, default=float)
    return gd, info


def evaluate(spec: ValidationSpec, out: str) -> dict:
    """Frozen gates + reported tables. Matching tolerance: 0.5 deg + 2 sigma (cloud) or + half-width (streak)."""
    from scipy.stats import chi2, spearmanr
    from .pipeline import misorientation_deg
    G = spec.gates
    rows, v3, v4, v8, ue_nf, ue_f, pairs, floor, inits = [], [], [], [], [], [], [], [], []
    n_false = nf = errs = false_in_foreign = 0
    missing = []
    for i in range(spec.n_columns):
        p = f"{out}/C_{i:03d}.json"
        if not os.path.exists(p):
            missing.append(i); continue
        r = json.load(open(p))
        if "error" in r:
            errs += 1; continue
        T = r["truth"]; nf += 1
        rep = r.get("report") or dict(orientations=[], unexplained_flux_frac=1.0)
        (ue_f if T["foreign"] else ue_nf).append(rep.get("unexplained_flux_frac", 1.0)); inits.append(rep.get("init_won"))
        TU = [np.array(u) for u in T["U"]]
        tol = [0.5 + (2 * p_ if s == "cloud" else p_ if s == "streak" else 0.0) for s, p_ in zip(T["spread"], T["param"])]
        rec = rep.get("orientations", [])
        A = np.array([[misorientation_deg(o["U"], t, spec.sg) for t in TU] for o in rec]) if rec else np.zeros((0, len(TU)))
        M = A < np.array(tol)[None, :] if len(A) else np.zeros((0, len(TU)), bool)
        nfl = int((~M.any(1)).sum()) if len(A) else 0
        n_false += nfl; false_in_foreign += nfl if T["foreign"] else 0
        for k in range(len(TU)):
            found = bool(len(A) and M[:, k].any())
            rows.append(dict(i=i, share=T["share"][k], spread=T["spread"][k], param=T["param"][k], N=T["N"], geom=T["geom"], found=found))
            if not found:
                continue
            j = int(np.argmin(np.where(M[:, k], A[:, k], np.inf)))
            if int(M[j].sum()) != 1:
                continue
            o = rec[j]
            if T["share"][k] >= 0.05:
                v3.append(abs(o["share"] - T["share"][k]) / T["share"][k])
            if T["spread"][k] == "point":
                v4.append(float(A[j, k]))
                if o.get("spread_rms_deg") is not None:
                    floor.append(o["spread_rms_deg"])
            elif T["share"][k] >= 0.05 and o.get("spread_rms_deg") is not None:
                v8.append((T["rms_deg"][k], o["spread_rms_deg"], T["spread"][k], T["param"][k]))
        for k, (kind, pj, dl) in enumerate(zip(T["kind"], T["partner"], T["delta"])):
            if kind == "near" and len(A):
                pairs.append((dl, bool(M[:, k].any() and M[:, pj].any() and int(np.argmin(A[:, k])) != int(np.argmin(A[:, pj])))))
    big = [x for x in rows if x["share"] >= 0.05]; sp = [x for x in big if x["spread"] != "point"]
    recall = float(np.mean([x["found"] for x in big])) if big else None
    srec = float(np.mean([x["found"] for x in sp])) if sp else None
    ub = 0.5 * chi2.ppf(0.95, 2 * (n_false + 1)) / max(nf, 1)
    rho = float(spearmanr([x[0] for x in v8], [x[1] for x in v8]).correlation) if len(v8) > 3 else None
    E = dict(columns=nf, errors=errs, missing=missing, gates=G,
             V1=dict(false=n_false, per_column=n_false / max(nf, 1), ub95=ub, false_in_foreign_columns=false_in_foreign, passed=ub <= G["V1_false_ub"]),
             V2=dict(n=len(big), recall=recall, passed=recall is not None and recall >= G["V2_recall"]),
             V3=dict(n=len(v3), median=float(np.median(v3)) if v3 else None, passed=bool(v3) and float(np.median(v3)) <= G["V3_share_err"]),
             V4=dict(n=len(v4), median_deg=float(np.median(v4)) if v4 else None, passed=bool(v4) and float(np.median(v4)) <= G["V4_point_deg"]),
             V6=dict(n=len(ue_nf), median=float(np.median(ue_nf)) if ue_nf else None, foreign_median=float(np.median(ue_f)) if ue_f else None,
                     passed=bool(ue_nf) and float(np.median(ue_nf)) <= G["V6_unexplained"]),
             V7=dict(n=len(sp), recall=srec, passed=srec is not None and srec >= G["V7_spread_recall"]),
             V8=dict(n=len(v8), spearman=rho, passed=rho is not None and rho >= G["V8_spread_spearman"]))
    E["read"] = "VALIDATED" if all(E[k]["passed"] for k in ("V1", "V2", "V3", "V4", "V6", "V7", "V8")) else "NOT VALIDATED"
    if missing:                  # an unfinished run must not read as a validated one scored on the columns that finished
        E["read"] = f"INCOMPLETE ({len(missing)} of {spec.n_columns} columns missing)"
    bins = [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.01)]
    E["recall_by_share"] = {f"{lo}-{hi}": dict(n=len(s), recall=float(np.mean([x["found"] for x in s])) if s else None)
                            for lo, hi in bins for s in [[x for x in rows if lo <= x["share"] < hi]]}
    E["recall_by_class"] = {f"{c}_{p}": dict(n=len(s), recall=float(np.mean([x["found"] for x in s])) if s else None)
                            for c, p in sorted({(x["spread"], x["param"]) for x in big}) for s in [[x for x in big if x["spread"] == c and x["param"] == p]]}
    E["recall_by_N"] = {str(n): float(np.mean([x["found"] for x in big if x["N"] == n])) for n in sorted({x["N"] for x in big})}
    E["recall_by_geometry"] = {str(k): float(np.mean([x["found"] for x in big if x["geom"] == k])) for k in sorted({x["geom"] for x in big})}
    E["pair_resolution"] = {str(d): dict(n=len(s), frac_two=float(np.mean(s)) if s else None)
                            for d in sorted({p_[0] for p_ in pairs}) for s in [[p_[1] for p_ in pairs if p_[0] == d]]}
    E["point_spread_floor_deg"] = dict(median=float(np.median(floor)) if floor else None, q90=float(np.percentile(floor, 90)) if floor else None)
    E["spread_by_class"] = {f"{c}_{p}": dict(n=len(s), true_med=float(np.median([x[0] for x in s])), rec_med=float(np.median([x[1] for x in s])))
                            for c, p in sorted({(x[2], x[3]) for x in v8}) for s in [[x for x in v8 if x[2] == c and x[3] == p]]}
    E["init_won"] = {str(k): inits.count(k) for k in set(inits)}
    json.dump(E, open(f"{out}/eval.json", "w"), indent=1, default=float)
    return E


def validate(spec: ValidationSpec, out: str, *, workers: int = 30, stages: str = "ABCD") -> dict:
    """Run the registered stages. Returns the evaluation dict (stage D) or the stage-B gate info if C/D are skipped."""
    os.makedirs(out, exist_ok=True)
    result = {}
    if "A" in stages:
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_stage_a, [(spec, i, out) for i in range(spec.n_columns)], chunksize=1))
    if "B" in stages:
        gd, info = measure_gate(spec, out); result["gdisc"] = info
        if gd is None:
            result["stop"] = "no discovery gate reaches the chance bound; report, do not run stage C"
            return result
    if "C" in stages:
        gd = json.load(open(f"{out}/gdisc.json"))["g_disc"]
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_stage_c, [(spec, i, out, gd) for i in range(spec.n_columns)], chunksize=1))
    if "D" in stages:
        result = evaluate(spec, out)
    return result
