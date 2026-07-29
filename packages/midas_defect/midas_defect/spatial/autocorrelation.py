"""Distance-binned Pearson correlation of a per-grain scalar field.

For matched-pair statistics in matrix-twin populations, the autocorrelation
is the simplest non-parametric probe of a stress/strain length scale: the
distance bin where r(d) drops out of the noise marks the characteristic
correlation length.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def epsilon_autocorrelation(
    eps_eq_per_grain: NDArray[np.floating],
    pos: NDArray[np.floating],
    variant_labels: NDArray[np.intp] | None = None,
    distance_bins: NDArray[np.floating] | None = None,
) -> dict:
    """Distance-binned Pearson r of a per-grain scalar field.

    Parameters
    ----------
    eps_eq_per_grain : (n_grains,) scalar (e.g. von-Mises strain). NaN-tolerant.
    pos : (n_grains, 3) coordinates in any consistent unit (typically um).
    variant_labels : optional; if given, only intra-variant pairs are included.
    distance_bins : 1-D edges. Defaults to ``[0, 25, 50, 100, 150, 200, 300, 500, 1000]``.

    Returns
    -------
    dict with
        bin_centers       (n_bins,)
        pearson_r_per_bin (n_bins,)  NaN where < 3 pairs in the bin
        n_pairs_per_bin   (n_bins,)
    """
    e = np.asarray(eps_eq_per_grain, dtype=float)
    p = np.asarray(pos, dtype=float)
    if e.shape[0] != p.shape[0]:
        raise ValueError(f"length mismatch: eps {e.shape[0]} vs pos {p.shape[0]}")

    if distance_bins is None:
        distance_bins = np.array([0.0, 25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0])
    edges = np.asarray(distance_bins, dtype=float)

    n = e.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    if variant_labels is not None:
        v = np.asarray(variant_labels, dtype=int)
        intra = v[iu] == v[ju]
        iu = iu[intra]
        ju = ju[intra]

    d = np.linalg.norm(p[iu] - p[ju], axis=1)
    e_i = e[iu]
    e_j = e[ju]
    finite = np.isfinite(e_i) & np.isfinite(e_j)
    d = d[finite]
    e_i = e_i[finite]
    e_j = e_j[finite]

    n_bins = edges.size - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_per_bin = np.full(n_bins, np.nan)
    n_per_bin = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = (d >= edges[b]) & (d < edges[b + 1])
        m = int(mask.sum())
        n_per_bin[b] = m
        if m < 3:
            continue
        ei = e_i[mask]
        ej = e_j[mask]
        if ei.std() < 1e-30 or ej.std() < 1e-30:
            continue
        r_per_bin[b] = float(np.corrcoef(ei, ej)[0, 1])

    return {
        "bin_centers": centers,
        "pearson_r_per_bin": r_per_bin,
        "n_pairs_per_bin": n_per_bin,
    }


__all__ = ["epsilon_autocorrelation"]
