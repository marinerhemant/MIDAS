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

__all__ = ["ipf_rgb", "sym_matrices", "CUBIC", "HEXAGONAL", "TRIGONAL"]

CUBIC = "cubic"
HEXAGONAL = "hexagonal"
TRIGONAL = "trigonal"

# Laue class per space-group range, for the ones MIDAS actually reconstructs.
# Deliberately explicit rather than clever: a wrong guess here silently
# recolours a map without any other symptom.
_SG_LAUE = [
    (195, 230, CUBIC),
    (168, 194, HEXAGONAL),
    (143, 167, TRIGONAL),
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
        f"(have: cubic 195-230, hexagonal 168-194, trigonal 143-167). Refusing to guess."
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


def _trigonal_sector_deg(sym: np.ndarray) -> float:
    """Azimuthal fundamental-sector width, measured from the operators.

    Counts the distinct upper-hemisphere azimuths a generic direction is sent
    to by the group plus the Laue centre. For -3m that is 6 images, so 60 deg
    -- twice the hexagonal 30. Measured rather than hard-coded so a change in
    the operator set cannot silently mis-scale the colour ramp.
    """
    g = np.array([0.3411, 0.1297, 0.4271])       # generic, no special azimuth
    g = g / np.linalg.norm(g)
    both = np.concatenate([sym @ g, -(sym @ g)], axis=0)
    up = both[both[:, 2] >= -1e-12]
    phi = np.sort(np.degrees(np.arctan2(up[:, 1], up[:, 0])) % 360.0)
    keep = [phi[0]]
    for a in phi[1:]:
        if a - keep[-1] > 1e-6:
            keep.append(a)
    return 360.0 / len(keep)


def _rgb_trigonal(d_all: np.ndarray, sym: np.ndarray) -> np.ndarray:
    """Standard [0001]-[10-10]-[01-10] triangle for Laue class -3m.

    D3 has 6 proper rotations (3-fold about c, three 2-fold in the basal
    plane) against 6/mmm's 12, so the sector spans 60 deg, not 30. Colouring
    R-3m with the hexagonal triangle folds by a 6-fold axis the crystal does
    not have and silently gives distinct orientations the same colour.

    The azimuth is NOT hand-folded. Where MIDAS puts the 2-fold axes is a
    convention, and assuming one sits at azimuth 0 gives a colouring that is
    not symmetry invariant (measured: colour moved by up to 0.99 under
    ``g -> S.g``). Instead take the whole orbit, add ``-d`` for the Laue
    centre, keep the upper hemisphere and pick the smallest azimuth --
    canonical whatever the operator convention.

    ``d_all`` is ``(n, n_sym, 3)``, every symmetry image of each direction.
    """
    both = np.concatenate([d_all, -d_all], axis=1)              # (n, 2s, 3)
    phi = np.degrees(np.arctan2(both[:, :, 1], both[:, :, 0])) % 360.0
    phi = np.where(both[:, :, 2] >= -1e-12, phi, np.inf)        # upper only
    k = np.argmin(phi, axis=1)
    idx = np.arange(both.shape[0])
    rep, phi_c = both[idx, k], phi[idx, k]
    phi_c = np.where(np.isfinite(phi_c), phi_c, 0.0)
    rep = rep / np.linalg.norm(rep, axis=1, keepdims=True)

    dz = np.abs(rep[:, 2])
    planar = np.hypot(rep[:, 0], rep[:, 1])
    t = np.clip(phi_c / _trigonal_sector_deg(sym), 0.0, 1.0)
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

    # TRANSPOSE. MIDAS orientation matrices map CRYSTAL -> LAB (v_lab = g v_crystal),
    # so the crystal direction parallel to the sample axis `a` is g^T a, not g a.
    # This is not cosmetic: with g a, the colour is NOT symmetry-invariant for this
    # convention -- the 24 equally valid representations of one grain spread over
    # 0.96 in RGB, so a map's colour depended on which variant the indexer happened
    # to store. Two grains agreeing to 0.5 deg came out 0.42 apart in RGB.
    # Fixed 2026-09-03; see tests::test_ipf_colour_is_symmetry_invariant.
    d = np.einsum("nji,j->ni", g, a)                 # crystal dir of the axis = g^T a
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
    elif fam == TRIGONAL:
        rgb = _rgb_trigonal(d, sym)
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
