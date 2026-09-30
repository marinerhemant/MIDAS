"""The column-content pipeline for one raster point (monochromatic rotation data).

    ingest (mask, sector background, 3-D blobs)  ->  round 0 = find_domains on the detected spots
    repeat (<= max_rounds):  joint fit of ALL current orientations (ColumnFit) -> residual -> spots -> find_domains
                             -> a solution is NEW if > new_min_deg from every current orientation AND its claim count
                                exceeds g_disc -> add, refit on the ORIGINAL data; stop when a round adds none.

``g_disc`` must be MEASURED (:func:`discovery_gate`), never assumed: the residual after a joint fit is not the frame the
round-0 search was tuned on, and chance solutions on residuals differ (the Laue implementation measured 0.46 chance
solutions per frame at its round-0 gate on residuals).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from midas_defect.domains import find_domains
from midas_defect.geometry import Geometry, detector_angle_maps, pixel_to_qlab, qlab_to_qsample
from midas_defect.ingest import build_mask, find_blobs_3d, subtract_background
from .fit import ColumnFit


def misorientation_deg(U1, U2, space_group: int) -> float:
    from midas_stress.orientation import misorientation_om
    r = misorientation_om(np.asarray(U1, float).ravel().tolist(), np.asarray(U2, float).ravel().tolist(), int(space_group))
    return math.degrees(r[0] if isinstance(r, (list, tuple)) else r)


@dataclass
class Ingested:
    sub: np.ndarray
    mask: np.ndarray        # True = excluded (midas_defect convention)
    labels: np.ndarray
    spots: object           # DataFrame from find_blobs_3d
    q: np.ndarray           # (n, 3) sample-frame q of every spot
    intensity: np.ndarray


def spots_to_q(df, geom: Geometry, *, omega_sign: int = 1):
    """Spot table -> sample-frame q (1/A): omega = omega_sign * (omega_first + frame * step)."""
    if len(df) == 0:
        return np.zeros((0, 3))
    rows = np.asarray(df["row"], float); cols = np.asarray(df["col"], float); fr = np.asarray(df["frame"], float)
    qlab = pixel_to_qlab(rows, cols, geom, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(omega_sign * (geom.omega_first_deg + fr * geom.omega_step_deg)), dtype=torch.float64)
    return qlab_to_qsample(qlab, om, math.radians(geom.wedge_deg)).numpy()


def ingest_column(frames, geom: Geometry, *, threshold: float = 200.0, mask: Optional[np.ndarray] = None,
                  omega_sign: int = 1, sat_level: Optional[float] = None, blob_kw: Optional[dict] = None) -> Ingested:
    """midas_defect ingest for one point. ``mask`` (True = excluded) defaults to build_mask(frames).mask.
    Saturated voxels (>= sat_level) are added to the voxel mask used by the fit."""
    tth, az = detector_angle_maps(geom)
    m2 = build_mask(frames).mask if mask is None else np.asarray(mask, bool)
    sub = subtract_background(frames, np.asarray(tth), np.asarray(az), m2)
    sub = sub[0] if isinstance(sub, tuple) else sub
    out = find_blobs_3d(sub, m2, threshold=threshold, return_labels=True, **(blob_kw or {}))
    df, labels = out[0], out[-1]
    vox = np.broadcast_to(m2, sub.shape).copy()
    if sat_level is not None:
        vox |= np.asarray(frames) >= sat_level
    return Ingested(sub=sub, mask=vox, labels=labels, spots=df, q=spots_to_q(df, geom, omega_sign=omega_sign),
                    intensity=np.asarray(df["integrated"], float) if len(df) else np.zeros(0))


def search(ing_or_df, geom, q, intensity, *, a, c, sg, find_kw=None):
    df = ing_or_df
    if len(df) == 0:
        return []
    ds = find_domains(q, intensity, np.asarray(df["row"], float), np.asarray(df["col"], float), np.asarray(df["frame"], float),
                      a=a, c=c, space_group_number=sg, **(find_kw or {}))
    return [(d.U, d.n) for d in ds.domains]


def discovery_gate(residual_spots_q, intensity, df, geom, *, a, c, sg, gates=(5, 7, 9, 11, 13), n_draws=20,
                   ub_max=0.10, rng_seed=0, find_kw=None):
    """Measured discovery gate: scramble residual spot directions (|q| kept), run the same search, and return the smallest
    gate whose chance-solution rate has a Poisson 95% upper bound <= ub_max per column. residual_spots_q may be a list
    over columns (one per synthetic/real point); counts are pooled."""
    from scipy.stats import chi2
    rng = np.random.default_rng(rng_seed)
    Qs = residual_spots_q if isinstance(residual_spots_q, list) else [residual_spots_q]
    Is = intensity if isinstance(intensity, list) else [intensity]
    Ds = df if isinstance(df, list) else [df]
    counts = {g: 0 for g in gates}; n = 0
    for q, I, d in zip(Qs, Is, Ds):
        for _ in range(n_draws):
            v = rng.normal(size=q.shape); v /= np.linalg.norm(v, axis=1, keepdims=True)
            qs = v * np.linalg.norm(q, axis=1, keepdims=True)
            sols = search(d, geom, qs, I, a=a, c=c, sg=sg, find_kw=find_kw)
            for g in gates:
                counts[g] += sum(1 for _, nn in sols if nn > g)
            n += 1
    ub = {g: 0.5 * chi2.ppf(0.95, 2 * (counts[g] + 1)) / max(n, 1) for g in gates}
    ok = [g for g in gates if ub[g] <= ub_max]
    return (min(ok) if ok else None), dict(counts=counts, ub95=ub, n_columns_x_draws=n)


@dataclass
class ColumnResult:
    rounds: List[dict] = field(default_factory=list)
    U: List[np.ndarray] = field(default_factory=list)
    round_found: List[int] = field(default_factory=list)
    report: Optional[dict] = None


def run_column(frames, geom: Geometry, *, a: float, c: float, sg: int, B, hkl_all, kernel, g_disc: int,
               ing: Optional[Ingested] = None, max_rounds: int = 3, new_min_deg: float = 1.0, K: int = 24,
               fit_kw=None, find_kw=None, ingest_kw=None, omega_sign: int = 1,
               search_fn=None) -> ColumnResult:
    """search_fn(df, geom, q, intensity) -> [(U, n)] replaces the default find_domains(a, c, sg) search in round 0
    AND discovery (find_domains handles tetragonal-type cells only; pass your own for other lattices)."""
    ing = ing or ingest_column(frames, geom, omega_sign=omega_sign, **(ingest_kw or {}))
    res = ColumnResult()
    srch = search_fn or (lambda d, g_, q_, I_: search(d, g_, q_, I_, a=a, c=c, sg=sg, find_kw=find_kw))
    for U, n in srch(ing.spots, geom, ing.q, ing.intensity):
        res.U.append(np.asarray(U, float)); res.round_found.append(0)
    for rnd in range(max_rounds + 1):
        if not res.U:
            res.rounds.append(dict(round=rnd, n_orientations=0)); break
        fit = ColumnFit(ing.sub, ing.mask, ing.labels, res.U, B, hkl_all, geom, kernel, K=K, omega_sign=omega_sign)
        fit.fit(**(fit_kw or {}))
        rep = fit.report()
        for o, rf in zip(rep["orientations"], res.round_found):
            o["round_found"] = rf
        res.report = rep
        info = dict(round=rnd, n_orientations=len(res.U), unexplained=rep["unexplained_flux_frac"])
        if rnd == max_rounds:
            res.rounds.append(info); break
        resid = fit.residual_stack()
        out = find_blobs_3d(resid, ing.mask[0] if ing.mask.ndim == 3 else ing.mask, threshold=(ingest_kw or {}).get("threshold", 200.0))
        rdf = out[0] if isinstance(out, tuple) else out
        rq = spots_to_q(rdf, geom, omega_sign=omega_sign)
        added = 0
        for U, n in srch(rdf, geom, rq, np.asarray(rdf["integrated"], float) if len(rdf) else np.zeros(0)):
            if n > g_disc and all(misorientation_deg(U, V, sg) > new_min_deg for V in res.U):
                res.U.append(np.asarray(U, float)); res.round_found.append(rnd + 1); added += 1
        info["added"] = added; res.rounds.append(info)
        if added == 0:
            break
    return res
