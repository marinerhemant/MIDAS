"""The two crystal Cartesian frames in this code base, and the rotation between them.

For any cell whose angles are not all 90 degrees there are two natural ways to put
the lattice into Cartesian axes, and both are used here:

``textbook`` (:class:`midas_hkls.lattice.Lattice`)
    real-space ``a`` along x, ``b`` in the xy plane. For hexagonal cells that puts
    a1 along x.

``midas`` (:func:`midas_b_matrix`, a port of the C ``CorrectHKLsLatC``)
    **a\\* along x**, b\\* in the xy plane, c\\* completing the right-handed set; the
    reciprocal vector of (h, k, l) is ``B @ (h, k, l)``. MIDAS ``Grains.csv``
    orientation matrices (crystal -> lab), ``hkls.csv`` g vectors and everything
    ``midas-index`` / ``midas-fit-grain`` produce live in this frame. For a
    hexagonal cell real-space a is at -30 deg (150 deg) and b at 90 deg.

They differ by a fixed rotation (:func:`frame_rotation`): none for cubic, tetragonal
and orthorhombic cells (all three angles 90 deg), 30 deg about c for hexagonal and
trigonal-hex cells, ``|angle - 90|`` about the unique axis for monoclinic cells, and a
general rotation for rhombohedral-axes and triclinic cells. Symmetry operators and
disorientation ANGLES do not see the difference for hexagonal cells; plane normals,
slip and twin directions and rotation AXES do, so anything built in one frame and
applied to an orientation expressed in the other is silently off by that rotation.
"""
from __future__ import annotations

import numpy as np

from .lattice import Lattice
from .nf_hkls import _b_matrix
from .point_group import as_lattice

__all__ = ["midas_b_matrix", "frame_rotation"]


def midas_b_matrix(lattice) -> np.ndarray:
    """``B`` with ``g_cart = B @ (h, k, l)`` in the MIDAS crystal frame (a* along x).

    ``lattice`` is a :class:`~midas_hkls.lattice.Lattice` or ``(a, b, c, alpha,
    beta, gamma)`` with angles in degrees. Columns are a*, b*, c* in 1/Angstrom
    (no 2*pi).
    """
    return _b_matrix(as_lattice(lattice))


def frame_rotation(lattice) -> np.ndarray:
    """Rotation ``R`` taking vectors from the textbook frame to the MIDAS frame.

    ``v_midas = R @ v_textbook``, where the textbook frame is the one of
    :class:`~midas_hkls.lattice.Lattice` (real a along x, b in xy). ``R`` is a
    proper rotation (``R R^T = I``, ``det = +1``), the identity when all three
    cell angles are 90 deg.

    Use it to bring slip systems, twin normals or axes written in the textbook
    frame into the frame of a MIDAS orientation matrix::

        n_midas = n_textbook @ frame_rotation(cell).T
    """
    lat = as_lattice(lattice)
    A_text = np.asarray(lat.cartesian_vectors(), float).T          # columns a, b, c (textbook)
    B = midas_b_matrix(lat)                                        # columns a*, b*, c* (MIDAS)
    A_midas = np.linalg.inv(B).T                                   # columns a, b, c (MIDAS)
    R = A_midas @ np.linalg.inv(A_text)
    # The two frames hold the same lattice, so R must be a rotation; scrub the
    # float noise so it is exactly orthogonal.
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:                       # pragma: no cover - cannot happen for a right-handed cell
        raise ValueError("textbook and MIDAS frames have opposite handedness")
    return R
