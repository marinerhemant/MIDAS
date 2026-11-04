"""Slip and twin systems for hexagonal-close-packed crystals.

All families are defined in 4-index Miller-Bravais notation for clarity and
materialized to crystal-Cartesian unit vectors via :func:`hcp_systems` at the
caller's ``c_over_a``. Twin shear and geometry depend on ``c/a``, so explicit
parameterization is unavoidable.

Crystal-Cartesian frames
------------------------
Two frames are in use, 30 degrees about c apart (``HCP_FRAMES``):

``"midas"`` (default)
    The frame of MIDAS ``Grains.csv`` orientation matrices (crystal -> lab) and
    of ``midas_hkls``' ``B`` matrix (a port of the C ``CorrectHKLsLatC``):
    ``g = B (h, k, l)`` with **a\* along x**, b\* in the xy plane, c\* along z;
    real-space a at -30 deg (150 deg), b at 90 deg, c || z. Use it for any
    orientation that comes from MIDAS (``Grains.csv``, ``midas-index`` /
    ``midas-fit-grain``, ``midas_diffract`` forward model).

``"crystallographic"``
    The textbook frame, real-space a1 along x::

        a1 = a (1, 0, 0)
        a2 = a (-1/2, sqrt(3)/2, 0)
        c  = a * (c/a) (0, 0, 1)

    Use it for orientations expressed in that frame, e.g. the ones this
    package solves for itself on the ``midas_hkls.Lattice`` seed path
    (``rows.py`` / ``selfcal.py`` build ``B`` from
    ``Lattice.reciprocal_cartesian_vectors``) until ``refine_lattice`` moves
    them to the MIDAS frame.

The 12 hexagonal symmetry operators and all disorientation ANGLES are the same
in both frames, so an angle-only check cannot tell them apart; plane normals,
slip/twin directions and rotation AXES are rotated by 30 deg about c. Before
2026-10-20 only the crystallographic frame existed, so the tables were 30 deg
off for MIDAS orientations (basal <a> directions at |cos| = 0.866 from the real
a axes; Schmid factors off by 0.02-0.04 on average, up to 0.12).

References
----------
Partridge 1967, *Metall. Rev.* 12:169 (HCP slip and twinning families,
{10-12} tensile, {11-22} compressive).
Christian & Mahajan 1995, *Prog. Mater. Sci.* 39:1-157 (twin geometry and
``c/a``-dependent shear magnitudes).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from midas_stress.twin_relations import HCP_FRAMES  # crystal Cartesian frames; see the module docstring

# Rotation by -30 deg about z: takes the textbook (a1 along x) frame to the MIDAS
# (a* along x) frame. Checked against midas_hkls.frame_rotation in the tests.
_C30, _S30 = np.cos(np.pi / 6.0), np.sin(np.pi / 6.0)
_RZ_M30 = np.array([[_C30, _S30, 0.0], [-_S30, _C30, 0.0], [0.0, 0.0, 1.0]])


def _to_frame(vec: NDArray[np.floating], frame: str) -> NDArray[np.floating]:
    """Rotate a textbook-frame vector (or rows) into ``frame``."""
    if frame not in HCP_FRAMES:
        raise ValueError(f"frame must be one of {HCP_FRAMES}, got {frame!r}")
    if frame == "crystallographic":
        return vec
    return vec @ _RZ_M30.T

# ----------------------------------------------------------------------------
# 4-index <-> 3-index conversions (directions only; planes use h, k, l with
# the i index dropped because h + k + i = 0).
# ----------------------------------------------------------------------------

def bravais_to_miller(u: int, v: int, t: int, w: int) -> tuple[int, int, int]:
    """4-index direction [u v t w] -> 3-index direction (U, V, W).

    Uses the convention V_phys = (u - t) a1 + (v - t) a2 + w c, which gives
    U = u - t, V = v - t, W = w. Requires u + v + t = 0.
    """
    if u + v + t != 0:
        raise ValueError(f"Miller-Bravais constraint violated: u+v+t={u+v+t}")
    return (u - t, v - t, w)


def miller_to_bravais(U: int, V: int, W: int) -> tuple[int, int, int, int]:
    """3-index direction (U, V, W) -> 4-index [u v t w].

    Inverse of :func:`bravais_to_miller`. Uses the rational form
    u = (2U - V)/3, v = (2V - U)/3, t = -(u + v), w = W and then rescales to
    the smallest integer 4-tuple.
    """
    num_u, num_v = 2 * U - V, 2 * V - U
    num_w = 3 * W  # absorb /3 into a common factor of 3
    num_t = -(num_u + num_v)
    g = np.gcd.reduce(np.abs([num_u, num_v, num_t, num_w]))
    if g == 0:
        return (0, 0, 0, 0)
    return tuple(int(x // g) for x in (num_u, num_v, num_t, num_w))  # type: ignore[return-value]


# ----------------------------------------------------------------------------
# Geometry helpers (return unit vectors in crystal-Cartesian).
# ----------------------------------------------------------------------------

def direction_bravais_to_cart(
    u: float, v: float, t: float, w: float, c_over_a: float,
    *, frame: str = "midas",
) -> NDArray[np.floating]:
    """Unit Cartesian direction of 4-index Miller-Bravais [u v t w] in ``frame``
    (``"midas"`` default, or ``"crystallographic"``; see the module docstring)."""
    if not np.isclose(u + v + t, 0.0):
        raise ValueError(f"Miller-Bravais constraint violated: u+v+t={u+v+t}")
    U, V, W = u - t, v - t, w
    vec = np.array([U - V / 2.0, V * np.sqrt(3) / 2.0, W * c_over_a], dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError(f"Zero-length direction [{u} {v} {t} {w}]")
    return _to_frame(vec / norm, frame)


def plane_normal_bravais_to_cart(
    h: float, k: float, i: float, l: float, c_over_a: float,
    *, frame: str = "midas",
) -> NDArray[np.floating]:
    """Unit Cartesian normal of 4-index Miller-Bravais plane (h k i l) in ``frame``.

    With h + k + i = 0 the normal in the ``"crystallographic"`` frame is
    proportional to (h, (h + 2k)/sqrt(3), l a/c) (Cullity & Stock, *Elements of
    X-Ray Diffraction*, App. 4); in the ``"midas"`` frame (default) it is that
    vector rotated by -30 deg about c, i.e. ((2h + k)/sqrt(3), k, l a/c) = B (h, k, l).
    """
    if not np.isclose(h + k + i, 0.0):
        raise ValueError(f"Miller-Bravais plane constraint violated: h+k+i={h+k+i}")
    vec = np.array(
        [float(h), (h + 2.0 * k) / np.sqrt(3), float(l) / c_over_a], dtype=float
    )
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError(f"Zero-length normal ({h} {k} {i} {l})")
    return _to_frame(vec / norm, frame)


# ----------------------------------------------------------------------------
# Slip / twin family tables, 4-index Miller-Bravais (plane, direction).
# ----------------------------------------------------------------------------

# Basal (0001)<11-20>: 1 plane * 3 a-directions = 3 systems.
BASAL_SLIP_4INDEX: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = [
    ((0, 0, 0, 1), (1, 1, -2, 0)),
    ((0, 0, 0, 1), (-2, 1, 1, 0)),
    ((0, 0, 0, 1), (1, -2, 1, 0)),
]

# Prismatic {1-100}<11-20>: 3 planes * 1 a-direction in each = 3 systems.
PRISMATIC_SLIP_4INDEX = [
    ((1, 0, -1, 0), (1, -2, 1, 0)),
    ((0, 1, -1, 0), (-2, 1, 1, 0)),
    ((-1, 1, 0, 0), (1, 1, -2, 0)),
]

# Pyramidal <a> {1-101}<11-20>: 6 systems.
PYRAMIDAL_A_SLIP_4INDEX = [
    ((1, 0, -1, 1), (1, -2, 1, 0)),
    ((0, 1, -1, 1), (-2, 1, 1, 0)),
    ((-1, 1, 0, 1), (1, 1, -2, 0)),
    ((-1, 0, 1, 1), (1, -2, 1, 0)),
    ((0, -1, 1, 1), (-2, 1, 1, 0)),
    ((1, -1, 0, 1), (1, 1, -2, 0)),
]

# Pyramidal <c+a> {11-22}<11-2-3>: 6 systems.
PYRAMIDAL_CA_SLIP_4INDEX = [
    ((1, 1, -2, 2), (1, 1, -2, -3)),
    ((-1, 2, -1, 2), (-1, 2, -1, -3)),
    ((-2, 1, 1, 2), (-2, 1, 1, -3)),
    ((-1, -1, 2, 2), (-1, -1, 2, -3)),
    ((1, -2, 1, 2), (1, -2, 1, -3)),
    ((2, -1, -1, 2), (2, -1, -1, -3)),
]

# Tensile twin {10-12}<-1011>: 6 systems (active under c-axis tension if c/a < sqrt(3)).
TWIN_TENSILE_4INDEX = [
    ((1, 0, -1, 2), (-1, 0, 1, 1)),
    ((0, 1, -1, 2), (0, -1, 1, 1)),
    ((-1, 1, 0, 2), (1, -1, 0, 1)),
    ((-1, 0, 1, 2), (1, 0, -1, 1)),
    ((0, -1, 1, 2), (0, 1, -1, 1)),
    ((1, -1, 0, 2), (-1, 1, 0, 1)),
]

# Compressive twin {11-22}<11-2-3>: 6 systems (active under c-axis compression).
TWIN_COMPRESSIVE_4INDEX = PYRAMIDAL_CA_SLIP_4INDEX.copy()


_FAMILY_TABLE: dict[str, list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]] = {
    "basal_a":       BASAL_SLIP_4INDEX,
    "prismatic_a":   PRISMATIC_SLIP_4INDEX,
    "pyramidal_a":   PYRAMIDAL_A_SLIP_4INDEX,
    "pyramidal_ca":  PYRAMIDAL_CA_SLIP_4INDEX,
    "twin_tensile":  TWIN_TENSILE_4INDEX,
    "twin_compressive": TWIN_COMPRESSIVE_4INDEX,
}


def hcp_systems(family: str, c_over_a: float, *, frame: str = "midas") -> NDArray[np.floating]:
    """Materialize an HCP slip/twin family to crystal-Cartesian unit vectors.

    ``frame`` is ``"midas"`` (default: the frame of MIDAS ``Grains.csv``
    orientations) or ``"crystallographic"`` (textbook a1 along x); see the
    module docstring. Pass the one the orientation matrices you will rotate
    these systems with live in.

    Parameters
    ----------
    family
        One of ``"basal_a"``, ``"prismatic_a"``, ``"pyramidal_a"``,
        ``"pyramidal_ca"``, ``"twin_tensile"``, ``"twin_compressive"``.
    c_over_a
        Lattice ratio. Mg = 1.624, Ti = 1.587, Zn = 1.856, ideal = sqrt(8/3).

    Returns
    -------
    (n_sys, 2, 3) array
        ``out[i, 0]`` plane normal, ``out[i, 1]`` slip/twin direction; both
        unit-norm in crystal-Cartesian. ``n . b = 0`` for every row.
    """
    if family not in _FAMILY_TABLE:
        raise KeyError(
            f"unknown HCP family {family!r}; expected one of {sorted(_FAMILY_TABLE)}"
        )
    rows = _FAMILY_TABLE[family]
    out = np.empty((len(rows), 2, 3), dtype=float)
    for idx, (plane, direction) in enumerate(rows):
        n = plane_normal_bravais_to_cart(*plane, c_over_a=c_over_a, frame=frame)
        b = direction_bravais_to_cart(*direction, c_over_a=c_over_a, frame=frame)
        if abs(float(np.dot(n, b))) > 1e-9:
            raise AssertionError(
                f"non-orthogonal HCP system {family}[{idx}]: "
                f"plane={plane} direction={direction} c/a={c_over_a}"
            )
        out[idx, 0] = n
        out[idx, 1] = b
    return out


def gamma_twin_hcp_tensile(c_over_a: float) -> float:
    """Twin shear magnitude for {10-12}<-1011> tensile twinning.

    From Christian & Mahajan 1995: gamma = |c^2 - 3 a^2| / (sqrt(3) a c),
    rewritten in c/a as |(c/a)^2 - 3| / (sqrt(3) (c/a)). Vanishes at the
    ideal ratio sqrt(3) and reverses sense across it.
    """
    return abs(c_over_a**2 - 3.0) / (np.sqrt(3) * c_over_a)


def gamma_twin_hcp_compressive(c_over_a: float) -> float:
    """Twin shear magnitude for {11-22}<11-2-3> compressive twinning.

    Yoo 1981: gamma = (2 (c/a)^2 - 3) / (sqrt(3) c/a)  (approximate).
    """
    return abs(2.0 * c_over_a**2 - 3.0) / (np.sqrt(3) * c_over_a)


__all__ = [
    "BASAL_SLIP_4INDEX",
    "PRISMATIC_SLIP_4INDEX",
    "PYRAMIDAL_A_SLIP_4INDEX",
    "PYRAMIDAL_CA_SLIP_4INDEX",
    "TWIN_COMPRESSIVE_4INDEX",
    "TWIN_TENSILE_4INDEX",
    "bravais_to_miller",
    "HCP_FRAMES",
    "direction_bravais_to_cart",
    "gamma_twin_hcp_compressive",
    "gamma_twin_hcp_tensile",
    "hcp_systems",
    "miller_to_bravais",
    "plane_normal_bravais_to_cart",
]
