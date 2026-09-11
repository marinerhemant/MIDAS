"""Friedel-pair intensity asymmetry per grain.

Centrosymmetric crystals satisfy Friedel's law |F(hkl)| = |F(-h-k-l)|. An
observed intensity asymmetry can come from defect-induced anomalous scattering,
unresolved twin overlap, or detector/absorption systematics -- but check the
integration first. If the intensities are connected-component label sums at a
fixed absolute threshold, the dominant "asymmetry" is usually segmentation:
label volume goes as the amplitude raised to a power of up to ~5, so two mates
of slightly different brightness are cut at different fractions of their own
peak. On the demk L5 reference sample Friedel pairs came out 0.32-1.82 apart in
label sum while their intensity per voxel held at 0.965
(``manuals/defect/ENVELOPE.md`` section 16). Measure the slope with
:func:`midas_defect.segmentation_bias.label_volume_slope` and censor both mates
identically with :func:`midas_defect.segmentation_bias.mirrored_dead_masks`
before attributing an asymmetry to the sample.

To decide which spots ARE Friedel mates, use :func:`classify_q_pair`.

The asymmetry metric is

    A(hkl) = |I(hkl) - I(-h-k-l)| / (I(hkl) + I(-h-k-l))

ranging in [0, 1].
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


def friedel_pair_asymmetry(
    intensity_per_grain_reflection: Mapping[tuple[int, tuple[int, int, int]], float],
) -> dict:
    """Friedel-pair asymmetry per (grain, hkl-pair).

    Parameters
    ----------
    intensity_per_grain_reflection
        Mapping ``(grain_idx, (h, k, l)) -> intensity``. The function looks
        up ``(grain_idx, (-h, -k, -l))`` for each entry; unpaired entries
        are skipped.

    Returns
    -------
    dict with keys
        ``asymmetry_per_pair``  (n_pairs,) float in [0, 1]
        ``grain_per_pair``      (n_pairs,) int
        ``hkl_per_pair``        (n_pairs, 3) int  (the +hkl member)
        ``mean_asymmetry``      float
        ``median_asymmetry``    float
    """
    seen: set[tuple[int, tuple[int, int, int]]] = set()
    pair_grain: list[int] = []
    pair_hkl: list[tuple[int, int, int]] = []
    pair_A: list[float] = []

    for (g, hkl), I_plus in intensity_per_grain_reflection.items():
        # Avoid double-counting by canonicalising to the positive Friedel mate.
        if hkl in seen:
            continue
        neg = (-hkl[0], -hkl[1], -hkl[2])
        key_neg = (g, neg)
        if key_neg not in intensity_per_grain_reflection:
            continue
        I_minus = intensity_per_grain_reflection[key_neg]
        total = I_plus + I_minus
        if total <= 0:
            continue
        A = abs(I_plus - I_minus) / total
        pair_grain.append(g)
        pair_hkl.append(hkl)
        pair_A.append(float(A))
        seen.add(hkl)
        seen.add(neg)

    A_arr = np.asarray(pair_A, dtype=float)
    return {
        "asymmetry_per_pair": A_arr,
        "grain_per_pair": np.asarray(pair_grain, dtype=int),
        "hkl_per_pair": np.asarray(pair_hkl, dtype=int).reshape(-1, 3),
        "mean_asymmetry": float(A_arr.mean()) if A_arr.size else float("nan"),
        "median_asymmetry": float(np.median(A_arr)) if A_arr.size else float("nan"),
    }


def classify_q_pair(
    q_a: ArrayLike,
    q_b: ArrayLike,
    *,
    q_rel_tol: float = 0.01,
    cos_tol: float = 1e-3,
) -> dict:
    """Classify two spots of a rotation scan from their SAMPLE-frame q vectors.

    One reciprocal-lattice vector gives four spots per 360 degrees: ``q`` and
    ``-q`` each cross the Ewald sphere twice. With the omega rotation undone, the
    two crossings of one ``q`` share a vector and Friedel mates are antipodal.
    For two spots of equal ``|q|``:

    * ``'crossing'``  -- unit vectors parallel,     ``dot >= 1 - cos_tol``
    * ``'friedel'``   -- unit vectors antiparallel, ``dot <= -(1 - cos_tol)``
    * ``'unrelated'`` -- anything else, including two different members of one
      ``{hkl}`` family at the same ``|q|``

    **Decide by the sign of the dot product, not by the omega separation.** On
    the demk L5 reference sample the two crossings of the satellites ran
    712.5-726.8 frames apart on a 1440-frame scan while a genuine Friedel pair sat
    at 713.7, so "half a turn apart" does not separate the two kinds. And do not
    assign pairs from which side of the beam centre the spots fall on: splitting
    four same-``|q|`` spots by detector column averaged two ``<220>`` variants
    59 degrees apart (``manuals/defect/phase-4-rods.md``).

    Parameters
    ----------
    q_a, q_b
        ``(3,)`` scattering vectors in the SAMPLE frame, in consistent units.
    q_rel_tol
        Largest ``| |q_a| - |q_b| | / mean(|q|)`` for the two to count as the same
        reflection; above it the result is ``'unrelated'``.
    cos_tol
        Tolerance on the unit-vector dot product, in (0, 1). The default ``1e-3``
        accepts about 2.6 degrees of centroid scatter; the demk L5 pairs sat
        within ``4e-4`` of +/-1.

    Returns
    -------
    dict with keys
        ``kind``            ``'crossing'``, ``'friedel'`` or ``'unrelated'``
        ``dot``             float, dot product of the unit vectors
        ``axis_angle_deg``  float in [0, 90], angle between the two as undirected axes
        ``q_rel_diff``      float, ``| |q_a| - |q_b| | / mean(|q|)``
    """
    a = np.asarray(q_a, dtype=float).ravel()
    b = np.asarray(q_b, dtype=float).ravel()
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("q_a and q_b must each be 3-vectors")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if not (np.isfinite(na) and np.isfinite(nb)) or na == 0.0 or nb == 0.0:
        raise ValueError("q vectors must be finite and non-zero")
    if not 0.0 < cos_tol < 1.0:
        raise ValueError(f"cos_tol must lie in (0, 1), got {cos_tol!r}")
    dot = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    q_rel = abs(na - nb) / (0.5 * (na + nb))
    axis_angle = float(np.degrees(np.arccos(min(1.0, abs(dot)))))
    if q_rel > q_rel_tol:
        kind = "unrelated"
    elif dot >= 1.0 - cos_tol:
        kind = "crossing"
    elif dot <= -(1.0 - cos_tol):
        kind = "friedel"
    else:
        kind = "unrelated"
    return {"kind": kind, "dot": dot, "axis_angle_deg": axis_angle, "q_rel_diff": float(q_rel)}


__all__ = ["classify_q_pair", "friedel_pair_asymmetry"]
