"""Proper rotation groups in the crystal Cartesian frame, and plane normals.

What this adds to :mod:`midas_hkls.space_group`, which already carries all 230
space groups as integer Seitz operators:

* the **proper rotation group as actual rotations** — orthogonal 3x3 matrices in
  a Cartesian frame, which is what anything geometric needs (orientation
  averaging, generalised spherical harmonics, pole figures, texture);
* **Cartesian plane normals** of a {hkl} family, which for every system but cubic
  requires the reciprocal metric. In hcp Ti the angle between (10-10) and (0001)
  is not what the raw index triple suggests.

Two things are structural here rather than optional, and both were established by
measurement rather than assumed:

**Improper operations map to -R; they are not discarded.** Friedel's law makes an
X-ray measurement centrosymmetric, so the symmetry a diffraction experiment can
recover is the **Laue** group (point group + inversion), not the crystal's own
point group. The Laue group is ``{+-R}``, whose proper half is ``R`` where
``det R = +1`` and ``-R`` where ``det R = -1``. Dropping the improper half instead
under-symmetrises the **73** space groups that carry improper operations but no
inversion centre (of 230: 92 are centrosymmetric, where -R is already present and
nothing changes; 65 are Sohncke, with no improper operations to drop; 73 are
neither, and those break). Pm (#6) collapses to order 1 instead of 2, Pmm2 (#25)
to 2 instead of 4. ``tests/test_point_group.py`` counts all three classes so the
split cannot drift.

**Group elements are keyed on the matrix, never on a quaternion.** ``q`` and
``-q`` are the same rotation and the sign tie-break is unstable exactly at
``w = 0`` -- the 180-degree elements, of which every one of these groups has
several. Keying a closure on quaternions closed the octahedral group to 28
elements instead of 24. Matrices have no double cover.

Deriving the group from the space group (rather than from a hand-written table
keyed by crystal system) also keeps the **setting**: monoclinic unique-axis b
against unique-axis c, and the rhombohedral settings, come out right for free.

numpy only -- no scipy, no torch. Rotations are returned as ``(n, 3, 3)`` arrays;
callers that want quaternions or Euler angles convert at their own boundary.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .lattice import Lattice
from .space_group import SpaceGroup
from .tables import laue_class_for

__all__ = [
    "EXPECTED_ORDER",
    "LAUE_TO_PROPER_GROUP",
    "PROPER_POINT_GROUP_GENERATORS",
    "as_lattice",
    "laue_class",
    "plane_normal",
    "plane_normals",
    "point_group_rotations",
    "proper_group_symbol",
    "proper_rotations_from_space_group",
]


# The 11 proper (rotation-only) point groups, by generator. These are also
# exactly the 11 Laue classes' proper halves, so this list is complete for
# diffraction: the 32 crystallographic point groups collapse to these 11 once
# Friedel's law is applied.
PROPER_POINT_GROUP_GENERATORS: dict[str, tuple[tuple[tuple[float, float, float], float], ...]] = {
    "1":   (),
    "2":   (((0, 0, 1), 180.0),),
    "222": (((0, 0, 1), 180.0), ((1, 0, 0), 180.0)),
    "4":   (((0, 0, 1), 90.0),),
    "422": (((0, 0, 1), 90.0), ((1, 0, 0), 180.0)),
    "3":   (((0, 0, 1), 120.0),),
    "32":  (((0, 0, 1), 120.0), ((1, 0, 0), 180.0)),
    "6":   (((0, 0, 1), 60.0),),
    "622": (((0, 0, 1), 60.0), ((1, 0, 0), 180.0)),
    "23":  (((0, 0, 1), 180.0), ((1, 1, 1), 120.0)),
    "432": (((0, 0, 1), 90.0), ((1, 1, 1), 120.0)),
}

#: Order of each proper point group. Asserted on closure, so a generator typo is
#: an error rather than a quietly smaller group.
EXPECTED_ORDER: dict[str, int] = {
    "1": 1, "2": 2, "222": 4, "4": 4, "422": 8, "3": 3, "32": 6,
    "6": 6, "622": 12, "23": 12, "432": 24,
}

#: The 11 Laue classes are in 1:1 correspondence with the 11 proper rotation
#: groups, so this is a complete mapping for every one of the 230 space groups.
LAUE_TO_PROPER_GROUP: dict[str, str] = {
    "-1": "1", "2/m": "2", "mmm": "222", "4/m": "4", "4/mmm": "422",
    "-3": "3", "-3m": "32", "6/m": "6", "6/mmm": "622",
    "m-3": "23", "m-3m": "432",
}

_CLOSURE_LIMIT = 200


def as_lattice(lattice) -> Lattice:
    """Accept a :class:`~midas_hkls.lattice.Lattice` or a 6-tuple.

    The 6-tuple form is ``(a, b, c, alpha, beta, gamma)`` with angles in degrees,
    which is how lattice parameters arrive from every MIDAS parameter file.
    """
    if isinstance(lattice, Lattice):
        return lattice
    vals = tuple(float(v) for v in lattice)
    if len(vals) != 6:
        raise ValueError(
            f"lattice must be a Lattice or (a, b, c, alpha, beta, gamma); "
            f"got {len(vals)} values")
    return Lattice(*vals)


def _key(m: np.ndarray) -> tuple:
    """Canonical hashable key for a rotation matrix.

    ``+ 0.0`` normalises negative zero, which otherwise makes ``-0.0`` and
    ``0.0`` hash to different keys and silently inflates a closed group.
    """
    return tuple(np.round(m, 8).ravel() + 0.0)


def _axis_angle_matrix(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    """Rotation matrix from an axis (need not be unit) and an angle in degrees."""
    u = np.asarray(axis, dtype=float)
    n = np.linalg.norm(u)
    if n == 0:
        raise ValueError("rotation axis must be non-zero")
    u = u / n
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) * c + s * K + (1.0 - c) * np.outer(u, u)


def point_group_rotations(symbol: str) -> np.ndarray:
    """All proper rotations of a point group, by closing its generators.

    Parameters
    ----------
    symbol
        One of the 11 proper point-group symbols, e.g. ``'432'``, ``'622'``.

    Returns
    -------
    numpy.ndarray
        ``(order, 3, 3)`` rotation matrices, identity first.

    Notes
    -----
    This is the *canonical-setting* group. For a real space group use
    :func:`proper_rotations_from_space_group`, which derives the same group in
    the cell's actual setting.
    """
    if symbol not in PROPER_POINT_GROUP_GENERATORS:
        raise KeyError(f"unknown proper point group {symbol!r}; "
                       f"have {sorted(PROPER_POINT_GROUP_GENERATORS)}")
    gens = [_axis_angle_matrix(ax, ang)
            for ax, ang in PROPER_POINT_GROUP_GENERATORS[symbol]]
    ident = np.eye(3)
    elems: dict[tuple, np.ndarray] = {_key(ident): ident}
    frontier = [ident]
    while frontier:
        new = []
        for r in frontier:
            for g in gens:
                c = g @ r
                k = _key(c)
                if k not in elems:
                    elems[k] = c
                    new.append(c)
        frontier = new
        if len(elems) > _CLOSURE_LIMIT:
            raise RuntimeError(
                f"point group {symbol} did not close (> {_CLOSURE_LIMIT} "
                "elements) -- generator axis or angle wrong?")
    out = np.array(list(elems.values()))
    expected = EXPECTED_ORDER.get(symbol)
    if expected is not None and len(out) != expected:
        raise AssertionError(
            f"{symbol}: closed to {len(out)} elements, expected {expected}")
    return out


def _space_group(spec) -> SpaceGroup:
    if isinstance(spec, SpaceGroup):
        return spec
    if isinstance(spec, (int, np.integer)):
        return SpaceGroup.from_number(int(spec))
    s = str(spec)
    try:
        return SpaceGroup.from_hm(s)
    except Exception:
        return SpaceGroup.from_hall(s)


def proper_rotations_from_space_group(spec, lattice) -> np.ndarray:
    """Proper rotation group of ANY space group, in the crystal Cartesian frame.

    Parameters
    ----------
    spec
        A space-group number (1-230), a Hermann-Mauguin symbol, a Hall symbol,
        or an existing :class:`~midas_hkls.space_group.SpaceGroup`.
    lattice
        :class:`~midas_hkls.lattice.Lattice` or ``(a, b, c, alpha, beta,
        gamma)``. **Required** -- the space group's operations are integer
        matrices in the lattice basis, and an integer matrix is not a rotation
        until it is conjugated through the cell (see
        :meth:`~midas_hkls.lattice.Lattice.cartesian_vectors`).

    Returns
    -------
    numpy.ndarray
        ``(order, 3, 3)`` rotation matrices of the **Laue group's proper half**.

    Raises
    ------
    AssertionError
        If a conjugated operation is not orthogonal, which means the lattice
        passed does not belong to the space group passed (a hexagonal group with
        ``gamma = 90``, say). This is a free correctness gate and it fires
        loudly rather than returning a plausible non-group.
    """
    sg = _space_group(spec)
    lat = as_lattice(lattice)
    M = lat.cartesian_vectors().T           # columns are a1, a2, a3
    Minv = np.linalg.inv(M)
    elems: dict[tuple, np.ndarray] = {}
    for op in sg.symmetry_operations():
        R = np.asarray(op.R, dtype=float).reshape(3, 3)
        # Friedel => the recoverable group is the LAUE group. Map improper
        # operations onto their proper partner instead of dropping them; see the
        # module docstring for the 73 space groups that depend on this.
        if int(round(np.linalg.det(R))) == -1:
            R = -R
        Rc = M @ R @ Minv
        if not np.allclose(Rc @ Rc.T, np.eye(3), atol=1e-8):
            raise AssertionError(
                f"space group {spec}: operation {op.to_xyz()} is not orthogonal "
                f"in Cartesian after conjugation -- lattice "
                f"(a={lat.a}, b={lat.b}, c={lat.c}, alpha={lat.alpha}, "
                f"beta={lat.beta}, gamma={lat.gamma}) is inconsistent with the "
                f"space group")
        elems.setdefault(_key(Rc), Rc)
    if not elems:
        raise AssertionError(f"space group {spec}: no proper rotations found")
    return np.array(list(elems.values()))


def laue_class(spec) -> str:
    """Laue class symbol (one of the 11) for a space group."""
    if isinstance(spec, (int, np.integer)):
        return laue_class_for(int(spec))
    return _space_group(spec).laue_class


def proper_group_symbol(spec) -> str:
    """Proper point-group symbol of a space group's Laue class.

    Useful as a cross-check: ``point_group_rotations(proper_group_symbol(225))``
    and ``proper_rotations_from_space_group(225, cubic_lattice)`` must be the
    same set of matrices, because a cubic cell's Cartesian frame *is* the
    canonical one.
    """
    return LAUE_TO_PROPER_GROUP[laue_class(spec)]


# ----------------------------------------------------------- plane normals
def plane_normal(hkl, lattice=None) -> np.ndarray:
    """Unit normal of (hkl) in the crystal Cartesian frame.

    Omit ``lattice`` only for cubic, where (hkl) already *is* a Cartesian
    direction. That shortcut is wrong for every other system -- it is the
    specific error that makes cubic texture code silently unusable on hcp.
    """
    h = np.asarray(hkl, dtype=float)
    if h.shape[-1] != 3:
        raise ValueError(f"hkl must have 3 components, got shape {h.shape}")
    g = h if lattice is None else h @ as_lattice(lattice).reciprocal_cartesian_vectors()
    n = np.linalg.norm(g)
    if n == 0:
        raise ValueError("(000) has no plane normal")
    return g / n


def plane_normals(hkl, rotations: np.ndarray, lattice=None) -> np.ndarray:
    """Unique symmetry-equivalent unit normals of {hkl}, **both signs**.

    Parameters
    ----------
    hkl
        Miller indices of one member of the family.
    rotations
        ``(n, 3, 3)`` proper rotation group, from
        :func:`proper_rotations_from_space_group` or
        :func:`point_group_rotations`.
    lattice
        As :func:`plane_normal`.

    Returns
    -------
    numpy.ndarray
        ``(m, 3)`` unit normals, sorted for reproducibility.

    Notes
    -----
    Both signs are included because diffraction cannot separate ``h`` from
    ``-h``. That is the same fact that makes only even-order harmonics
    measurable, so a caller expanding a pole figure and a caller counting family
    multiplicity must agree on it -- hence it is applied here, once.
    """
    g0 = plane_normal(hkl, lattice)
    rots = np.asarray(rotations, dtype=float)
    if rots.ndim != 3 or rots.shape[1:] != (3, 3):
        raise ValueError(f"rotations must be (n, 3, 3), got {rots.shape}")
    fam = {tuple(np.round(v, 9) + 0.0) for v in rots @ g0}
    fam |= {tuple(np.round(-np.array(v), 9) + 0.0) for v in fam}
    arr = np.array(sorted(fam))
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)
