"""Friedel-pair intensity asymmetry per grain.

Centrosymmetric crystals satisfy Friedel's law |F(hkl)| = |F(-h-k-l)|. Any
observed intensity asymmetry comes from defect-induced anomalous scattering,
unresolved twin overlap, or detector/absorption systematics. The asymmetry
metric is

    A(hkl) = |I(hkl) - I(-h-k-l)| / (I(hkl) + I(-h-k-l))

ranging in [0, 1].
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray


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


__all__ = ["friedel_pair_asymmetry"]
