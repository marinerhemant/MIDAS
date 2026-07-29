"""Full n*G/3 satellite ladder + fundamental-vs-9R-satellite decontamination.

``satellite_radial_excess`` answers "is there discrete 9R excess at G/3 / 2G/3?".
This module builds the WHOLE ladder along the activated <111> (n = +-1 .. +-n_max)
and then **decontaminates** each rung against the predicted reflections of *all*
indexed grains -- the step that separates genuine forbidden-gap 9R satellites from
ordinary fundamentals that merely land on the rod:

  * n = +-3  ->  the carrier's own 111  (fundamental, on the rod)
  * n = +-6  ->  the carrier's own 222  (fundamental, NOT a 220: 6*G/3 = 2*G_111
                 = G_222 = 2 sqrt 3 B0, whereas G_220 = 2 sqrt 2 B0 sits near n=5)
  * n = +-1, 2, 4, 5  ->  FCC-forbidden gaps; if no grain's reflection lands within
                 ``tol`` they are genuine 9R satellites.

For each rung the nearest reflection over all grains is found by full 3-D vector
distance (not just |q|): the 5G/3-vs-220 coincidence is purely radial and vanishes
in 3-D, so radial proximity is not enough -- this is the decontamination that the
G/3,2G/3-only excess metric cannot do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..bragg_diffuse import enumerate_hkls

__all__ = ["SatelliteLadder", "build_satellite_ladder", "decontaminate_ladder"]


@dataclass
class SatelliteLadder:
    """The measured n*G/3 ladder. ``rungs`` is a list of per-rung dicts with keys
    n (signed int), q_target, q_obs (signed projection), intensity, n_voxels, and --
    after :func:`decontaminate_ladder` -- classification ('fundamental'|'9R-satellite')
    plus nearest {grain, hkl, dist_inv_A}."""

    axis: NDArray[np.floating]
    G_magnitude: float
    rungs: list
    metadata: dict = field(default_factory=dict)


def _unit(v) -> NDArray[np.floating]:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _g_crystal(hkls: NDArray[np.integer], crystal) -> NDArray[np.floating]:
    a = float(crystal.lattice.a)
    c = float(crystal.lattice.c)
    tp = 2.0 * math.pi
    G = np.empty(hkls.shape, dtype=np.float64)
    G[:, 0] = tp * hkls[:, 0] / a
    G[:, 1] = tp * hkls[:, 1] / a
    G[:, 2] = tp * hkls[:, 2] / c
    return G


def _orientations(orientations) -> NDArray[np.floating]:
    U = np.asarray(orientations, dtype=np.float64)
    if U.ndim == 2 and U.shape[-1] == 9:
        U = U.reshape(-1, 3, 3)
    elif U.shape == (3, 3):
        U = U.reshape(1, 3, 3)
    return U


def build_satellite_ladder(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    axis_dir: NDArray[np.floating],
    G_magnitude: float,
    *,
    n_max: int = 6,
    tube_parallel: float = 0.12,
    perp_max: float = 0.25,
    min_voxels: int = 20,
) -> SatelliteLadder:
    """Measure the satellite at each rung n = +-1 .. +-n_max along ``axis_dir``.

    Returns a :class:`SatelliteLadder`; rungs with fewer than ``min_voxels`` in the
    selection tube are omitted. ``q_obs`` is the intensity-weighted *signed*
    projection onto the axis (so +n and -n Friedel mates are kept separate).
    """
    qs = np.asarray(qs, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    axis = _unit(axis_dir)
    g3 = G_magnitude / 3.0

    proj = qs @ axis
    perp = np.linalg.norm(qs - proj[:, None] * axis, axis=1)
    rungs = []
    for n in range(-n_max, n_max + 1):
        if n == 0:
            continue
        qt = n * g3
        sel = (np.abs(proj - qt) < tube_parallel) & (perp < perp_max) & (vals > 0) & np.isfinite(vals)
        if int(sel.sum()) < min_voxels:
            continue
        w = vals[sel]
        rungs.append({
            "n": n,
            "q_target": float(qt),
            "q_obs": float(np.average(proj[sel], weights=w)),
            "q_centroid": np.average(qs[sel], axis=0, weights=w),
            "intensity": float(w.sum()),
            "n_voxels": int(sel.sum()),
        })
    return SatelliteLadder(axis=axis, G_magnitude=float(G_magnitude), rungs=rungs,
                           metadata={"n_max": n_max, "g3": float(g3)})


def decontaminate_ladder(
    ladder: SatelliteLadder,
    orientations,
    crystal,
    *,
    tol_inv_A: float = 0.20,
    q_max_inv_A: float = 8.5,
) -> SatelliteLadder:
    """Label each rung 'fundamental' or '9R-satellite' by full-3D decontamination.

    Forward every grain's allowed reflections (U @ G_crystal) and, for each rung
    point ``q_obs * axis``, find the nearest reflection by 3-D vector distance. A
    rung within ``tol_inv_A`` of *any* grain's reflection is a fundamental (the
    nearest grain+hkl is recorded); otherwise it is a genuine forbidden-gap 9R
    satellite. Mutates and returns ``ladder`` (adds 'classification' and 'nearest'
    to each rung).

    Notes
    -----
    Uses 3-D vector distance on purpose: the 5G/3 (|q|=4.99) vs 220 (|q|=4.89)
    overlap is a radial coincidence that disappears in 3-D, so only the full-vector
    test correctly keeps 5G/3 as a satellite.

    Orientation convention (IMPORTANT). ``orientations`` must be in the package's
    ``U @ G_crystal`` convention (the same as :mod:`midas_defect.bragg_diffuse` /
    :mod:`midas_defect.defect_tests` and ``seed_index.predict_q_from_U``): the
    sample-frame reflection is ``U @ (2*pi*hkl/a)``. A **raw MIDAS ``Grains.csv``
    orientation matrix must be transposed** before passing it here (its rows are
    crystal axes in the sample frame, so the sample-frame reflection is ``OM.T @
    G_crystal``). Passing un-transposed OMs silently mislabels every fundamental as
    a satellite (the carrier's own 111/222 land ~0.3-0.6 1/A off). Likewise the
    ``axis`` used to build the ladder must be ``U @ <111>``.
    """
    axis = _unit(ladder.axis)
    U = _orientations(orientations)
    hkls = enumerate_hkls(crystal, q_max_inv_A=q_max_inv_A)
    Gc = _g_crystal(hkls, crystal)
    # all grains' reflections in the sample frame, with (grain, hkl-row) provenance
    pts = []
    prov = []
    for gi in range(U.shape[0]):
        P = (U[gi] @ Gc.T).T
        pts.append(P)
        prov.append(np.column_stack([np.full(len(P), gi), np.arange(len(P))]))
    allP = np.concatenate(pts, axis=0)
    allprov = np.concatenate(prov, axis=0)

    for rung in ladder.rungs:
        # decontaminate against the rung's full 3-D centroid (the empirical satellite
        # axis is a few deg off the crystallographic <111>, an error that grows with
        # |q|; the on-axis projection q_obs*axis would over-distance the fundamentals).
        pt = rung.get("q_centroid")
        pt = rung["q_obs"] * axis if pt is None else np.asarray(pt, dtype=np.float64)
        d = np.linalg.norm(allP - pt, axis=1)
        j = int(np.argmin(d))
        gi, hi = int(allprov[j, 0]), int(allprov[j, 1])
        dist = float(d[j])
        rung["nearest"] = {
            "grain": gi,
            "hkl": tuple(int(x) for x in hkls[hi]),
            "dist_inv_A": dist,
        }
        rung["classification"] = "fundamental" if dist < tol_inv_A else "9R-satellite"
    ladder.metadata["tol_inv_A"] = float(tol_inv_A)
    ladder.metadata["n_satellites"] = sum(
        1 for r in ladder.rungs if r.get("classification") == "9R-satellite"
    )
    ladder.metadata["n_fundamentals"] = sum(
        1 for r in ladder.rungs if r.get("classification") == "fundamental"
    )
    return ladder
