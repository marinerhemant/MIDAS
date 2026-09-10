"""Map-level orientation statistics: KAM, GROD, medoids and axis means.

These summarise a MAP of orientations rather than a pair, so they live apart
from :mod:`midas_stress.orientation` and its dual numpy/torch inner loops.
Everything here is numpy and returns **radians**, matching the rest of
midas_stress (``.mic`` files and the ``GetMisorientation`` CLI use degrees;
this module does not).

Written after hand-rolling all four of these in an analysis and getting two of
them wrong -- see the individual docstrings for which, and how.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["mean_axis", "axis_spread", "medoid_orientation", "grod", "kam",
           "grid_neighbours", "group_orientations"]


def mean_axis(v: np.ndarray) -> np.ndarray:
    """Mean direction of AXIS data, where ``v`` and ``-v`` mean the same thing.

    Returns the leading eigenvector of the orientation tensor
    ``T = <v v^T>``, normalised.

    Do NOT use ``v.mean(axis=0)`` for axes. A crystal direction carries no
    sign: half the c-axes of one texture component can come back as ``-c``, and
    the vector mean then partially cancels. Measured on a real S5 map, the
    vector mean reported a 22.6 deg spread where the tensor gives 7.9 deg --
    the difference was entirely sign bookkeeping, not physics.

    Parameters
    ----------
    v : (n, 3) array_like

    Returns
    -------
    (3,) unit vector.
    """
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    if len(v) == 0:
        raise ValueError("no vectors given")
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    T = (v[:, :, None] * v[:, None, :]).mean(axis=0)
    w = np.linalg.eigh(T)[1][:, -1]
    return w / np.linalg.norm(w)


def axis_spread(v: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """Angle of each axis from ``reference`` (default :func:`mean_axis`), radians.

    Sign-insensitive: uses ``|v . ref|``, so an axis and its negative give the
    same angle and the result is in ``[0, pi/2]``.
    """
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    r = mean_axis(v) if reference is None else np.asarray(reference, float)
    r = r / np.linalg.norm(r)
    return np.arccos(np.clip(np.abs(v @ r), 0.0, 1.0))


def _pairwise(oms: np.ndarray, space_group: int) -> np.ndarray:
    """Full pairwise misorientation matrix in radians. O(n^2) -- small maps only."""
    from .orientation import misorientation_om
    n = len(oms)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = float(misorientation_om(oms[i], oms[j],
                                                        space_group)[0])
    return M


def medoid_orientation(oms: np.ndarray, space_group: int) -> int:
    """Index of the orientation with the smallest total misorientation to the rest.

    A medoid, not a mean. Averaging orientation matrices elementwise is not an
    orientation, and quaternion averaging needs a consistent hemisphere AND a
    symmetry choice per element; the medoid needs neither and is always a real
    measured orientation. Use it as the GROD reference.
    """
    oms = np.asarray(oms, dtype=float).reshape(-1, 3, 3)
    if len(oms) == 0:
        raise ValueError("no orientations given")
    return int(np.argmin(_pairwise(oms, space_group).sum(axis=1)))


def grod(oms: np.ndarray, space_group: int, *,
         reference: Optional[np.ndarray] = None) -> np.ndarray:
    """Grain reference orientation deviation, radians, one per orientation.

    ``reference`` defaults to the :func:`medoid_orientation` of ``oms``.

    .. warning::

       GROD is only interpretable ABOVE the orientation measurement floor. On a
       real S5 map the GROD median was 0.72 deg against a per-domain sigma(U) of
       roughly 0.70 deg RMS, i.e. the map was mostly noise, with one genuinely
       deformed region at 3-3.5 deg. Always quote sigma(U) beside a GROD map;
       a colour scale starting at zero will otherwise render noise as structure.
    """
    from .orientation import misorientation_om
    oms = np.asarray(oms, dtype=float).reshape(-1, 3, 3)
    ref = oms[medoid_orientation(oms, space_group)] if reference is None \
        else np.asarray(reference, float).reshape(3, 3)
    return np.array([float(misorientation_om(o, ref, space_group)[0]) for o in oms])


def kam(oms: np.ndarray, neighbours: np.ndarray, space_group: int, *,
        max_angle: Optional[float] = None) -> np.ndarray:
    """Kernel average misorientation, radians, one per orientation.

    ``neighbours`` is an ``(n, k)`` integer array of neighbour indices, ``-1``
    padding where a site has fewer than ``k``. :func:`grid_neighbours` builds
    one for a raster.

    ``max_angle`` (radians) EXCLUDES neighbour pairs above it. Set it, and set
    it below the smallest wall misorientation in the map. KAM is a measure of
    LOCAL lattice curvature; a pair straddling a domain boundary measures the
    boundary instead, and one such neighbour swamps the average. Measured on
    S5: within-domain neighbours sat at 0.44 deg while cross-domain pairs were
    11-14 deg, so leaving them in turns a KAM map into a smeared domain map.

    Sites left with no admissible neighbour return NaN rather than 0 -- zero is
    a legitimate KAM value and must not be manufactured by a missing neighbour.
    """
    from .orientation import misorientation_om
    oms = np.asarray(oms, dtype=float).reshape(-1, 3, 3)
    nb = np.asarray(neighbours, dtype=int)
    if nb.ndim != 2 or len(nb) != len(oms):
        raise ValueError(f"neighbours must be (n, k) with n = {len(oms)}; "
                         f"got {nb.shape}")
    out = np.full(len(oms), np.nan)
    for i in range(len(oms)):
        acc = []
        for j in nb[i]:
            if j < 0 or j == i:
                continue
            w = float(misorientation_om(oms[i], oms[j], space_group)[0])
            if max_angle is not None and w > max_angle:
                continue
            acc.append(w)
        if acc:
            out[i] = float(np.mean(acc))
    return out


def grid_neighbours(index: Sequence[int], shape, *,
                    connectivity: int = 4) -> np.ndarray:
    """Neighbour table for sites on a regular raster.

    ``index`` holds the flat raster index (``row * n_cols + col``) of each
    orientation, ``shape`` is ``(n_rows, n_cols)``. Returns ``(n, k)`` with
    ``-1`` where a neighbour is off-grid or not measured, ready for :func:`kam`.

    Only sites present in ``index`` become neighbours, so unindexed positions
    break the kernel rather than being silently treated as coincident.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    nrow, ncol = int(shape[0]), int(shape[1])
    idx = np.asarray(index, dtype=int)
    where = {int(p): i for i, p in enumerate(idx)}
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = np.full((len(idx), len(steps)), -1, dtype=int)
    for i, p in enumerate(idx):
        r, c = divmod(int(p), ncol)
        for k, (dr, dc) in enumerate(steps):
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrow and 0 <= cc < ncol:
                out[i, k] = where.get(rr * ncol + cc, -1)
    return out


def group_orientations(oms: np.ndarray, space_group: int, tol_rad: float, *,
                       priority: Optional[np.ndarray] = None):
    """Greedy grouping of orientations by misorientation. Radians.

    Returns ``(labels, reps)``: ``labels[i]`` is the group of orientation ``i``,
    and ``reps`` holds the index of each group's representative. Groups are
    relabelled by DESCENDING size, so label 0 is always the largest.

    Representatives are taken in order of descending ``priority`` (default:
    input order); a new representative is opened whenever an orientation is at
    least ``tol_rad`` from every existing one. Pass the reflection count as
    ``priority`` to seed groups from the best-determined orientations.

    .. warning::

       **The result depends on the order representatives are picked in, and
       group sizes near a tie are not stable.** On a real S5 map the largest
       group came out at 236 members under one ordering and 234 under an
       equally defensible one. Quote group sizes as approximate, and never
       build a claim on a difference of a few members between groups. This is a
       greedy partition, not a clustering with a defined optimum -- if the
       structure matters, check that the answer survives a change of ``tol_rad``
       and of ``priority``.

    Parameters
    ----------
    oms : (n, 3, 3) array_like
    space_group : int
    tol_rad : float
        Misorientation below which two orientations join the same group.
    priority : (n,) array_like, optional
    """
    from .orientation import misorientation_om

    oms = np.asarray(oms, dtype=float).reshape(-1, 3, 3)
    n = len(oms)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int)
    order = (np.arange(n) if priority is None
             else np.argsort(-np.asarray(priority, dtype=float)))
    reps: list[int] = []
    for i in order:
        if all(float(misorientation_om(oms[i], oms[r], space_group)[0]) >= tol_rad
               for r in reps):
            reps.append(int(i))
    lab = np.array([int(np.argmin([
        float(misorientation_om(o, oms[r], space_group)[0]) for r in reps]))
        for o in oms])
    size = np.bincount(lab, minlength=len(reps))
    remap = np.zeros(len(reps), int)
    remap[np.argsort(-size)] = np.arange(len(reps))
    return remap[lab], np.asarray(reps, int)[np.argsort(-size)]
