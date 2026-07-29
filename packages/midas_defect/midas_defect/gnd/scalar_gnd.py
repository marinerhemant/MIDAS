"""Scalar GND density from inter-grain misorientation.

For a population of grains the Kröner / Ashby relation gives a useful first
approximation to the GND content as

    rho_GND ~ < theta_ij / (d_ij b) >

where the average is over neighbour pairs (i, j), ``theta_ij`` is the small-
angle misorientation in radians, ``d_ij`` is the spatial separation, and
``b`` is the Burgers magnitude. The full Nye tensor (:mod:`nye_tensor`) is
the proper rank-2 generalization; this scalar form is what you reach for
when each grain has only a handful of neighbours.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import CrystalPhase
from ..variants.kmeans_fz import _PHASE_TO_SPACE_GROUP


def scalar_gnd_from_inter_grain_misorientation(
    OM: NDArray[np.floating],
    pos: NDArray[np.floating],
    burgers_length: float,
    k_NN: int = 5,
    variant_labels: NDArray[np.intp] | None = None,
    phase: CrystalPhase = CrystalPhase.FCC,
) -> NDArray[np.floating]:
    """Per-grain scalar GND density (m^-2).

    Parameters
    ----------
    OM, pos : (n_grains, 3, 3) and (n_grains, 3)
    burgers_length : Burgers vector magnitude in metres (e.g. 2.57e-10 for Cu).
    k_NN : number of nearest neighbours per grain (excluding self).
    variant_labels : optional; if given, kNN search is restricted to grains
                     of the same variant as the query grain.
    phase : crystal system; sets the symmetry for the misorientation.

    Returns
    -------
    rho_GND_per_grain : (n_grains,) m^-2; NaN where there are no in-variant
                        neighbours within reach.
    """
    import midas_stress.orientation as o
    from scipy.spatial import cKDTree

    OM = np.asarray(OM, dtype=float)
    pos = np.asarray(pos, dtype=float)
    n = OM.shape[0]
    if pos.shape[0] != n:
        raise ValueError(f"length mismatch: OM {n} vs pos {pos.shape[0]}")
    sg = _PHASE_TO_SPACE_GROUP[phase]

    if variant_labels is None:
        var = np.zeros(n, dtype=int)
    else:
        var = np.asarray(variant_labels, dtype=int)

    out = np.full(n, np.nan, dtype=float)
    for lbl in sorted(set(int(x) for x in np.unique(var))):
        idx = np.where(var == lbl)[0]
        if idx.size < 2:
            continue
        tree = cKDTree(pos[idx])
        k_eff = min(k_NN + 1, idx.size)
        dists, nn = tree.query(pos[idx], k=k_eff)
        if k_eff == 1:
            continue
        dists = dists[:, 1:]
        nn = nn[:, 1:]
        for li, g in enumerate(idx):
            nbrs = idx[nn[li]]
            ang = []
            for t in nbrs:
                a, _ = o.misorientation_om(OM[g].ravel(), OM[t].ravel(), space_group=sg)
                ang.append(float(a))
            ang = np.asarray(ang) / 1.0  # radians
            d_si = dists[li] * 1e-6  # um -> m  (assume pos in um)
            rho = (ang / (d_si * burgers_length)).mean()
            out[g] = rho
    return out


__all__ = ["scalar_gnd_from_inter_grain_misorientation"]
