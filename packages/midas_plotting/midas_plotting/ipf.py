"""Inverse-pole-figure colouring.

Colour encodes which crystal direction is parallel to a chosen sample axis, so
one grain is one colour and boundaries appear as colour discontinuities. That is
the property an Euler-to-RGB dump does NOT have: two orientations a fraction of
a degree apart can land on very different Euler triplets (and hence very
different colours) near the gimbal-lock line, which makes a single grain look
like several.

Symmetry operators come from :mod:`midas_stress`, never hand-listed here.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["ipf_rgb", "sym_matrices", "CUBIC", "HEXAGONAL"]

CUBIC = "cubic"
HEXAGONAL = "hexagonal"

# Laue class per space-group range, for the ones MIDAS actually reconstructs.
# Deliberately explicit rather than clever: a wrong guess here silently
# recolours a map without any other symptom.
_SG_LAUE = [
    (195, 230, CUBIC),
    (168, 194, HEXAGONAL),
]


def laue_class(space_group: int) -> str:
    """Laue family used for the IPF triangle.

    Raises for space groups whose triangle is not implemented, rather than
    falling back to cubic -- a silent fallback would produce a plausible-looking
    but meaningless map.
    """
    for lo, hi, name in _SG_LAUE:
        if lo <= int(space_group) <= hi:
            return name
    raise NotImplementedError(
        f"IPF colouring for space group {space_group} is not implemented "
        f"(have: cubic 195-230, hexagonal 168-194). Refusing to guess."
    )


def sym_matrices(space_group: int) -> np.ndarray:
    """``(n_sym, 3, 3)`` proper-rotation operators from midas_stress."""
    from midas_stress.orientation import make_symmetries, quat_to_orient_mat

    n, quats = make_symmetries(int(space_group))
    q = np.asarray(quats)[: int(n)]
    return np.stack([np.asarray(quat_to_orient_mat(qi)).reshape(3, 3) for qi in q])


def _reduce_cubic(d: np.ndarray) -> np.ndarray:
    """Fold directions into the standard [001]-[101]-[111] triangle."""
    d = np.abs(d)
    d = np.sort(d, axis=-1)                     # u <= v <= w
    return d


def _rgb_cubic(d: np.ndarray) -> np.ndarray:
    u, v, w = d[:, 0], d[:, 1], d[:, 2]
    rgb = np.stack([w - v, (v - u) * np.sqrt(2.0), u * np.sqrt(3.0)], axis=1)
    return rgb


def _rgb_hexagonal(d: np.ndarray) -> np.ndarray:
    """Standard [0001]-[10-10]-[2-1-10] triangle.

    ``d`` is Cartesian with c along +z. After symmetry reduction the
    representative has ``dz >= 0`` and azimuth in ``[0, 30]`` degrees.
    """
    dz = np.abs(d[:, 2])
    planar = np.hypot(d[:, 0], d[:, 1])
    phi = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
    phi = np.minimum(phi % 60.0, 60.0 - (phi % 60.0))     # fold to [0, 30]
    t = np.clip(phi / 30.0, 0.0, 1.0)
    return np.stack([dz, planar * (1.0 - t), planar * t], axis=1)


def ipf_rgb(
    euler: np.ndarray,
    space_group: int = 225,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    *,
    gamma: float = 0.5,
) -> np.ndarray:
    """RGB per orientation for the crystal direction parallel to ``axis``.

    Parameters
    ----------
    euler : (N, 3) array
        Bunge ZXZ Euler angles in **radians** -- the MIDAS ``.mic`` convention.
    space_group : int
        Used for the symmetry operators and to pick the triangle.
    axis : length-3
        Sample-frame direction. ``(0,0,1)`` gives the usual IPF-Z.
    gamma : float
        Perceptual lift applied as ``rgb ** gamma``. 0.5 (sqrt) matches the
        common convention; 1.0 disables it.

    Returns
    -------
    (N, 3) float array in [0, 1].
    """
    from midas_stress.orientation import euler_to_orient_mat_batch

    euler = np.asarray(euler, dtype=float).reshape(-1, 3)
    if euler.size == 0:
        return np.zeros((0, 3))
    g = np.asarray(euler_to_orient_mat_batch(euler)).reshape(-1, 3, 3)
    return ipf_rgb_from_matrix(g, space_group, axis, gamma=gamma)


def ipf_rgb_from_matrix(
    orient_mat: np.ndarray,
    space_group: int = 225,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    *,
    gamma: float = 0.5,
) -> np.ndarray:
    """RGB per orientation, from ``(N, 3, 3)`` orientation matrices.

    The same colouring as :func:`ipf_rgb`, entered from the matrix rather than
    from Euler angles. Far-field ``Grains.csv`` carries both (``O11..O33`` and
    ``Eul0..2``); this avoids a needless matrix -> Euler -> matrix round trip,
    which is lossy near the gimbal-lock configurations of the ZXZ convention.
    """
    g = np.asarray(orient_mat, dtype=float).reshape(-1, 3, 3)
    if g.size == 0:
        return np.zeros((0, 3))

    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n == 0:
        raise ValueError("axis must be non-zero")
    a = a / n

    d = np.einsum("nij,j->ni", g, a)                 # crystal dir of the axis
    return direction_rgb(d, space_group, gamma=gamma)


def direction_rgb(
    dirs: np.ndarray, space_group: int = 225, *, gamma: float = 0.5,
) -> np.ndarray:
    """RGB for **crystal directions** -- the colouring core.

    ``dirs`` is ``(N, 3)`` in crystal coordinates; it is normalised here.
    Both :func:`ipf_rgb` and the legend drawn by
    ``midas_plotting.ff.ipf_legend`` go through this, so the key on a figure
    is guaranteed to match the colours in the map beside it. A legend computed
    by a separate copy of the triangle maths is a legend that eventually lies.
    """
    d = np.asarray(dirs, dtype=float).reshape(-1, 3)
    if d.size == 0:
        return np.zeros((0, 3))
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.divide(d, nrm, out=np.zeros_like(d), where=nrm > 0)

    fam = laue_class(space_group)
    sym = sym_matrices(space_group)
    d = np.einsum("sij,nj->nsi", sym, d)             # every equivalent

    if fam == CUBIC:
        red = _reduce_cubic(d)
        pick = np.argmax(red[:, :, 2], axis=1)       # closest to [001]
        red = red[np.arange(red.shape[0]), pick]
        red /= np.linalg.norm(red, axis=1, keepdims=True)
        rgb = _rgb_cubic(red)
    else:
        dd = d.copy()
        dd[:, :, 2] = np.abs(dd[:, :, 2])
        pick = np.argmax(dd[:, :, 2], axis=1)        # closest to [0001]
        red = dd[np.arange(dd.shape[0]), pick]
        red /= np.linalg.norm(red, axis=1, keepdims=True)
        rgb = _rgb_hexagonal(red)

    rgb = np.clip(rgb, 0.0, None)
    mx = rgb.max(axis=1, keepdims=True)
    rgb = np.where(mx > 0, rgb / mx, rgb)
    return np.clip(rgb ** float(gamma), 0.0, 1.0)
