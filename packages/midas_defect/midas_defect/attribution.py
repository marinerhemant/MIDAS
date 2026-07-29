"""Attribution guard: refuse per-variant claims on shared reciprocal directions.

The 2026-06 demk failure root cause: a diffuse feature (the 9R satellite) lives on
the parent/twin **coincident** <111> (the twin plane). FF-HEDM integrates over a
grain's volume and has no spatial channel, so a feature on a reciprocal direction
shared by two variants **cannot** be attributed to one variant. Per-variant /
per-grain numbers computed anyway are projection-geometry artifacts (see
AUDIT_2026-06-23.md). This module detects coincident directions and *raises* rather
than letting a caller manufacture a per-variant result FF physics cannot support.

Hard limit: separating a shared-plane feature parent-vs-twin needs pf-HEDM (spatial).
No software fixes it; this guard makes the package say so instead of approximating.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["AttributionError", "hkl_family_dirs", "coincident_axes",
           "assert_variant_attributable"]


class AttributionError(RuntimeError):
    """Raised when a per-variant/per-grain claim is requested for a feature on a
    reciprocal direction shared by two or more variants (FF cannot attribute it)."""


def hkl_family_dirs(hkl=(1, 1, 1)) -> NDArray[np.floating]:
    """Unit vectors of the full signed <hkl> family (cubic), deduplicated by sign."""
    h, k, l = hkl
    seen = set()
    out = []
    for p in set(itertools.permutations((abs(h), abs(k), abs(l)))):
        for s in itertools.product((1, -1), repeat=3):
            v = (p[0] * s[0], p[1] * s[1], p[2] * s[2])
            if v == (0, 0, 0):
                continue
            key = min(v, tuple(-x for x in v))  # fold ±
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
    D = np.array(out, dtype=float)
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def coincident_axes(
    reference_OMs,
    hkl=(1, 1, 1),
    *,
    tol_deg: float = 10.0,
) -> NDArray[np.floating]:
    """Sample-frame directions shared (within ``tol_deg``) by >=2 reference variants.

    For each pair of reference orientations, a <hkl> direction of one that lies within
    ``tol_deg`` of a <hkl> direction of the other is "coincident" — a reciprocal
    direction both variants scatter into. A feature there is not variant-attributable.

    Parameters
    ----------
    reference_OMs : sequence of (3,3)
        Variant reference orientations (crystal->sample), e.g. [parent, twin].
    hkl : the family that carries the feature (default <111> for FCC 9R / SF).
    tol_deg : coincidence tolerance (degrees). **Must be >= the deformation mosaic
        spread**: in a deformed sample two single reference orientations coincide only
        at the mosaic scale (demk: ~8.5 deg), so a too-tight tol falsely reports "not
        shared" and lets an artifact through. Default 10 deg is deliberately
        conservative — a false refusal is far safer than manufacturing a per-variant
        number FF cannot support.

    Returns
    -------
    (M, 3) unit directions in the sample frame (deduplicated).
    """
    refs = [np.asarray(R, dtype=float) for R in reference_OMs]
    if len(refs) < 2:
        return np.zeros((0, 3))
    F = hkl_family_dirs(hkl)
    ct = math.cos(math.radians(tol_deg))
    found = []
    for a, b in itertools.combinations(range(len(refs)), 2):
        Da = (refs[a] @ F.T).T
        Db = (refs[b] @ F.T).T
        Da /= np.linalg.norm(Da, axis=1, keepdims=True)
        Db /= np.linalg.norm(Db, axis=1, keepdims=True)
        dots = np.abs(Da @ Db.T)             # (nF, nF)
        ia, ib = np.where(dots >= ct)
        for i in ia:
            d = Da[i] * (1.0 if Da[i, 2] >= 0 else -1.0)
            found.append(d)
    if not found:
        return np.zeros((0, 3))
    F2 = np.array(found)
    _, idx = np.unique(np.round(F2, 3), axis=0, return_index=True)
    return F2[idx]


def assert_variant_attributable(
    feature_axis_sample,
    reference_OMs,
    hkl=(1, 1, 1),
    *,
    tol_deg: float = 5.0,
    what: str = "this feature",
) -> None:
    """Raise ``AttributionError`` if ``feature_axis_sample`` is on a coincident axis.

    Call this at the top of any per-variant/per-grain routine that attributes a diffuse
    feature to a variant. If the feature direction (sample frame) coincides with a
    parent/twin-shared <hkl>, FF-HEDM cannot attribute it and we refuse.
    """
    axis = np.asarray(feature_axis_sample, dtype=float)
    axis = axis / np.linalg.norm(axis)
    shared = coincident_axes(reference_OMs, hkl, tol_deg=tol_deg)
    if shared.shape[0] == 0:
        return
    ang = np.degrees(np.arccos(np.clip(np.abs(shared @ axis).max(), 0.0, 1.0)))
    if ang <= tol_deg:
        raise AttributionError(
            f"{what} lies on a parent/twin-shared <{''.join(str(x) for x in hkl)}> "
            f"direction ({ang:.1f} deg from a coincident axis, tol {tol_deg:.1f}). "
            "FF-HEDM integrates over grain volume and cannot attribute a shared "
            "reciprocal direction to one variant — per-variant/per-grain numbers here "
            "are projection-geometry artifacts (see AUDIT_2026-06-23.md). "
            "Spatial separation requires pf-HEDM."
        )
