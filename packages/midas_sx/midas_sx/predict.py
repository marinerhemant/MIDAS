"""Predicted reflections and twin-pair measurement targets (PREREGISTER A26).

``predicted_crossings`` lists every Ewald crossing of a set of reflections for a given orientation, geometry and scan, as (frame, row, col)
by the production maps (:func:`midas_sx.orient.predict_positions`). ``twin_targets`` pairs each reflection ``h`` of domain 1 with the reflection
``T h`` of the second twin component (domain 2 = domain 1 rotated 90 deg about c) and turns the pair into measurement targets:

* ``blend``  - the two components are closer than ``d_res_px`` pixels (and ``d_res_frames`` frames): ONE target at their midpoint, the
  integration box widened by half their separation, both components counted in the model (flags (1, 1));
* ``c1`` / ``c2`` - the components are resolved, or only one of them crosses: one target per component (flags (1, 0) / (0, 1)).

A target is a *prediction*, not a detected blob: it is integrated whether or not the blob finder saw it, and its corrections are evaluated at the
calculated crossing by :func:`midas_sx.reduce.reduce_spots`.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from midas_defect.geometry import Geometry

from .assign import ewald_crossings
from midas_hkls import Lattice

from .assign import reciprocal_basis
from .geometry import Scan
from .orient import predict_positions

__all__ = ["enumerate_hkl", "predicted_crossings", "twin_targets", "TWIN_90_C"]

#: (h, k, l) -> (k, -h, l): the second component of a 90 deg twin about c (see `twin_partner_check` in the tests for the sense)
TWIN_90_C: Tuple[Tuple[int, int, int], ...] = ((0, 1, 0), (-1, 0, 0), (0, 0, 1))


def enumerate_hkl(lattice: Lattice, d_min: float, space_group=None) -> np.ndarray:
    """All (h, k, l) != 0 with d >= d_min (both signs), systematic absences removed when ``space_group`` is given."""
    Bi = np.linalg.norm(np.asarray(lattice.reciprocal_cartesian_vectors(), float), axis=1)       # |a*|, |b*|, |c*| (1/A)
    nmax = [int(math.floor(1.0 / (d_min * b))) + 1 for b in Bi]
    h, k, l = np.meshgrid(*[np.arange(-n, n + 1) for n in nmax], indexing="ij")
    hkl = np.stack([h.ravel(), k.ravel(), l.ravel()], axis=1)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    q = (np.asarray(reciprocal_basis(lattice)) @ hkl.T).T
    hkl = hkl[np.linalg.norm(q, axis=1) <= 1.0 / d_min]
    if space_group is not None:
        keep = np.array([not space_group.is_systematically_absent(int(a), int(b), int(c)) for a, b, c in hkl])
        hkl = hkl[keep]
    return hkl.astype(int)


def predicted_crossings(U: np.ndarray, lattice: Lattice, hkl: np.ndarray, geom: Geometry, scan: Scan, *, margin_px: float = 3.0) -> pd.DataFrame:
    """Every Ewald crossing of ``hkl`` inside the scan and on the detector. Columns: i (row of ``hkl``), h, k, l, omega, frame, row, col."""
    hkl = np.asarray(hkl, int)
    q_s = 2.0 * np.pi * (np.asarray(U, float) @ reciprocal_basis(lattice) @ hkl.T).T
    idx, om = [], []
    lo, hi = scan.omega_start, scan.omega_end
    for i in range(len(hkl)):
        for w in np.degrees(ewald_crossings(q_s[i], geom.wavelength_A, geom.wedge_deg)):
            for shift in (-360.0, 0.0, 360.0):
                if min(lo, hi) <= w + shift <= max(lo, hi):
                    idx.append(i); om.append(w + shift)
    if not idx:
        return pd.DataFrame(columns=["i", "h", "k", "l", "omega", "frame", "row", "col"])
    idx, om = np.array(idx), np.array(om)
    fr, ro, co, ok = predict_positions(U, lattice, hkl[idx], om, geom, scan)
    n_r, n_c = int(geom.n_pix_z), int(geom.n_pix_y)
    keep = ok & np.isfinite(ro) & np.isfinite(co)
    keep &= (ro > margin_px) & (ro < n_r - margin_px) & (co > margin_px) & (co < n_c - margin_px)
    return pd.DataFrame(dict(i=idx[keep], h=hkl[idx[keep], 0], k=hkl[idx[keep], 1], l=hkl[idx[keep], 2], omega=om[keep], frame=fr[keep],
                             row=ro[keep], col=co[keep]))


def twin_targets(U1: np.ndarray, U2: np.ndarray, lattice: Lattice, geom: Geometry, scan: Scan, hkl: np.ndarray, *,
                 twin_matrix=TWIN_90_C, d_res_px: float = 6.0, d_res_frames: float = 3.0, d_max_px: float = 25.0) -> pd.DataFrame:
    """Measurement targets for a two-component twin; see the module docstring. ``hkl`` are DOMAIN-1 indices; the partner index is ``T h``.

    Columns: frame, row, col, extra_px, extra_frames, kind (blend | c1 | c2), h k l (index of the first component, or of the component for c2),
    p0 p1 p2 (partner index T h, the domain-2 index of the second component), f1, f2 (component flags), sep_px, and the individual predicted component
    positions frame1 row1 col1 / frame2 row2 col2 (NaN where that component does not cross).
    """
    hkl = np.asarray(hkl, int)
    T = np.asarray(twin_matrix, int)
    part = hkl @ T.T
    c1 = predicted_crossings(U1, lattice, hkl, geom, scan)
    c2 = predicted_crossings(U2, lattice, part, geom, scan)
    rows = []
    by2 = {i: g for i, g in c2.groupby("i")}
    used2 = set()
    for _, r1 in c1.iterrows():
        i = int(r1.i)
        g = by2.get(i)
        best = None
        if g is not None and len(g):
            d = np.hypot(g.row.to_numpy() - r1.row, g.col.to_numpy() - r1.col)
            df = np.abs(g.frame.to_numpy() - r1.frame)
            ok = (d < d_max_px) & (df < 4.0 * d_res_frames)
            if ok.any():
                j = int(np.argmin(np.where(ok, d + df, np.inf)))
                best = (g.index[j], float(d[j]), float(df[j]))
        base = dict(h=int(r1.h), k=int(r1.k), l=int(r1.l), p0=int(part[i, 0]), p1=int(part[i, 1]), p2=int(part[i, 2]))
        if best is not None and best[1] < d_res_px and best[2] < d_res_frames:
            r2 = c2.loc[best[0]]; used2.add(best[0])
            rows.append(dict(base, frame=0.5 * (r1.frame + r2.frame), row=0.5 * (r1.row + r2.row), col=0.5 * (r1.col + r2.col),
                             extra_px=0.5 * best[1], extra_frames=0.5 * best[2], kind="blend", f1=1, f2=1, sep_px=best[1],
                             frame1=r1.frame, row1=r1.row, col1=r1.col, frame2=r2.frame, row2=r2.row, col2=r2.col))
        else:
            rows.append(dict(base, frame=r1.frame, row=r1.row, col=r1.col, extra_px=0.0, extra_frames=0.0, kind="c1", f1=1, f2=0,
                             sep_px=best[1] if best else np.nan, frame1=r1.frame, row1=r1.row, col1=r1.col))
            if best is not None:
                r2 = c2.loc[best[0]]; used2.add(best[0])
                rows.append(dict(base, frame=r2.frame, row=r2.row, col=r2.col, extra_px=0.0, extra_frames=0.0, kind="c2", f1=0, f2=1, sep_px=best[1],
                                 frame2=r2.frame, row2=r2.row, col2=r2.col))
    for j, r2 in c2.iterrows():                      # second-component crossings with no first-component partner
        if j not in used2:
            i = int(r2.i)
            rows.append(dict(h=int(hkl[i, 0]), k=int(hkl[i, 1]), l=int(hkl[i, 2]), p0=int(part[i, 0]), p1=int(part[i, 1]), p2=int(part[i, 2]),
                             frame=r2.frame, row=r2.row, col=r2.col, extra_px=0.0, extra_frames=0.0, kind="c2", f1=0, f2=1, sep_px=np.nan,
                             frame2=r2.frame, row2=r2.row, col2=r2.col))
    out = pd.DataFrame(rows)
    for c in ("frame1", "row1", "col1", "frame2", "row2", "col2"):
        if c not in out:
            out[c] = np.nan
    return out
