"""Family-level Bragg-peak asterism -- symmetry-safe, sample-frame, isolated peaks.

:mod:`local_decomposition` and :mod:`second_moment` measure asterism **per indexed
grain**. When an indexer over-fragments a deformed orientation family into many
sub-grains (common for a bent mosaic), per-grain asterism measures the *fragment*
size, not the family's true lattice curvature -- averaging it back by family
recovers the indexer's granularity, not the physics. This module measures the arc at
the **family** level directly.

Two pitfalls it avoids (both learned the hard way on demk Cu-Al):

1. **Cubic symmetry.** Grains within one family share a small *disorientation* but may
   sit in different symmetry settings, so a fixed crystallographic index (e.g.
   ``(2 0 0)``) points in different sample-frame directions for different grains
   (up to 90 deg apart). Grouping voxels by crystallographic ``hkl`` therefore mixes
   symmetry variants and inflates the arc spuriously. The fix here is to work purely
   in the **sample frame**: the caller supplies the family's physical reflection
   directions (e.g. ``reflection_directions`` of one family member), and each voxel is
   assigned to its nearest *physical* direction. See :func:`family_asterism_arc`.

2. **Fault / relrod contamination.** Reflections on or near the active
   ``<111>`` (the 111 itself, and the 220/311 relrod anchors) are contaminated by
   stacking-fault relrod and polytype-satellite intensity that masquerades as
   tangential spread. Use **isolated** reflections -- for FCC the ``<200>`` set,
   which carries no relrod anchor and is off the fault ladder.

The arc is the intensity-weighted RMS angular deviation of the diffuse intensity from
each reflection's (refined) direction -- a direct, |q|-independent measure of lattice
curvature (asterism). It is robust to the angular cutoff and to which family member
supplies the reference directions. Pure NumPy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["reflection_directions", "family_asterism_arc"]


def reflection_directions(
    orientation: NDArray[np.floating],
    hkls: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Sample-frame unit directions of crystal reflections ``hkls`` for ``orientation``.

    ``orientation`` is a 3x3 matrix in the ``U @ G`` convention (sample vector =
    ``OM @ crystal_vector``); ``hkls`` is ``(M, 3)``. Returns ``(M, 3)`` unit vectors
    ``normalize(OM @ hkl)``. For a cubic crystal these are the physical directions of
    the reflections regardless of the reflection's magnitude, suitable as the
    ``center_dirs`` argument of :func:`family_asterism_arc`. Using one family member's
    directions is sufficient (the family shares them up to the mosaic).
    """
    OM = np.asarray(orientation, dtype=np.float64).reshape(3, 3)
    hk = np.atleast_2d(np.asarray(hkls, dtype=np.float64))
    d = (OM @ hk.T).T
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-12)


def family_asterism_arc(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    center_dirs: NDArray[np.floating],
    *,
    max_angle_deg: float = 20.0,
    min_voxels_per_reflection: int = 50,
) -> dict:
    """Intensity-weighted azimuthal asterism arc (deg) of one orientation family.

    Given all diffuse/near-Bragg voxels ``(qs, vals)`` belonging to a single family,
    near a set of **isolated** Bragg reflections whose sample-frame unit directions are
    ``center_dirs`` ``(K, 3)``, assign each voxel to its nearest direction (within
    ``max_angle_deg``), refine each reflection's direction to the intensity-weighted
    mean of its voxels, and return the intensity-weighted RMS angular deviation pooled
    over reflections -- the family asterism arc.

    Working in the sample frame with caller-supplied physical directions makes this
    **symmetry-safe** (no crystallographic-``hkl`` variant mixing); choosing ``<200>``
    (or otherwise isolated) reflections for ``center_dirs`` keeps it **relrod/fault-free**.

    Parameters
    ----------
    qs : (N, 3) voxel q-vectors (sample frame). Normalized internally.
    vals : (N,) intensities (weights).
    center_dirs : (K, 3) sample-frame reflection directions (need not be unit).
    max_angle_deg : a voxel farther than this from every direction is dropped.
    min_voxels_per_reflection : reflections with fewer assigned voxels are skipped.

    Returns
    -------
    dict with
        arc_deg : float -- pooled intensity-weighted RMS angular arc (NaN if none).
        per_reflection : (K,) per-direction arc in deg (NaN where skipped).
        n_voxels : (K,) voxels used per direction.
        n_reflections_used : int.
    """
    q = np.asarray(qs, dtype=np.float64)
    w = np.asarray(vals, dtype=np.float64)
    C = np.asarray(center_dirs, dtype=np.float64).reshape(-1, 3)
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    K = C.shape[0]

    per = np.full(K, np.nan)
    nvox = np.zeros(K, dtype=int)
    if q.shape[0] == 0:
        return dict(arc_deg=float("nan"), per_reflection=per, n_voxels=nvox,
                    n_reflections_used=0)

    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    good = np.isfinite(w) & (w > 0)
    qn, w = qn[good], w[good]

    cosall = qn @ C.T                             # (N, K)
    ci = np.argmax(cosall, axis=1)
    best = np.degrees(np.arccos(np.clip(cosall[np.arange(len(qn)), ci], -1.0, 1.0)))
    near = best < max_angle_deg
    qn, w, ci = qn[near], w[near], ci[near]

    tot_w = 0.0
    tot_var = 0.0
    used = 0
    for c in range(K):
        m = ci == c
        n = int(m.sum())
        nvox[c] = n
        if n < min_voxels_per_reflection or w[m].sum() <= 0:
            continue
        md = np.average(qn[m], axis=0, weights=w[m])
        md = md / np.maximum(np.linalg.norm(md), 1e-12)
        ang = np.degrees(np.arccos(np.clip(qn[m] @ md, -1.0, 1.0)))
        wc = w[m]
        arc_c = np.sqrt(np.sum(wc * ang ** 2) / wc.sum())
        per[c] = arc_c
        tot_var += np.sum(wc * ang ** 2)
        tot_w += wc.sum()
        used += 1

    arc = float(np.sqrt(tot_var / tot_w)) if tot_w > 0 else float("nan")
    return dict(arc_deg=arc, per_reflection=per, n_voxels=nvox,
                n_reflections_used=used)
