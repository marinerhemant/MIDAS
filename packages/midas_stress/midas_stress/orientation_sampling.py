"""Uniform sampling of orientation space (numpy + scipy; importing it pulls in ``midas_stress.orientation``, hence torch, about 1.4 s).

``uniform_rotations(n)`` gives n quasi-uniform rotations of SO(3); ``orientation_grid(step_deg, space_group)`` keeps one representative per
symmetry class of the Laue group of ``space_group``, so the grid covers orientation space modulo the crystal symmetry with a covering radius of
about ``step_deg``. Rotations are matrices U (n, 3, 3) in the convention of ``quat_to_orient_mat``; the symmetry operators are
``make_symmetries`` (the canonical list ``misorientation_om`` uses), so the grid and the misorientation agree on what "the same orientation" is.
"""
from __future__ import annotations

import math

import numpy as np

from .orientation import make_symmetries, quat_to_orient_mat

# calibrated so that the 95th percentile of the covering distance of the full-SO(3) grid is about step_deg
# (tests/test_orientation_sampling.py::test_covering_distance)
_GRID_DENSITY = 1.3


def _super_fibonacci(n: int) -> np.ndarray:
    """n quasi-uniform unit quaternions (Alexa 2022, 'Super-Fibonacci spirals'), scalar part first; q and -q are the same rotation."""
    phi = math.sqrt(2.0)
    psi = 1.533751168755204288118041
    s = np.arange(n) + 0.5
    r = np.sqrt(s / n)
    R = np.sqrt(1.0 - s / n)
    a, b = 2 * math.pi * s / phi, 2 * math.pi * s / psi
    return np.stack([r * np.sin(a), r * np.cos(a), R * np.sin(b), R * np.cos(b)], axis=1)


def _quats_to_matrices(q: np.ndarray) -> np.ndarray:
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c),
        2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b),
        2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d], axis=1).reshape(-1, 3, 3)


def uniform_rotations(n: int) -> np.ndarray:
    """(n, 3, 3) quasi-uniform rotation matrices covering all of SO(3)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return _quats_to_matrices(_super_fibonacci(int(n)))


def laue_rotations(space_group: int) -> np.ndarray:
    """(n, 3, 3) proper rotations of the Laue class of ``space_group``, the matrix form of ``make_symmetries``."""
    _, quats = make_symmetries(int(space_group))
    return np.array([np.array(quat_to_orient_mat(q), dtype=float).reshape(3, 3) for q in quats])


def orientation_grid(step_deg: float = 3.0, space_group: int = 1) -> np.ndarray:
    r"""Orientations covering SO(3) modulo the Laue group of ``space_group``, with covering radius of about ``step_deg``.

    Quasi-uniform sampling of SO(3), then ONE representative per symmetry class: a rotation R is kept only if it is the member of its class
    ``{R s}`` (s over the proper rotations of the Laue group) with the smallest rotation angle, ``tr(R) >= tr(R s)`` for all s. A group of order
    N keeps about 1/N of SO(3).
    """
    if not step_deg > 0:
        raise ValueError("step_deg must be > 0")
    rho = math.radians(step_deg)
    n_full = int(math.ceil(_GRID_DENSITY * math.pi / (rho - math.sin(rho))))
    R = uniform_rotations(n_full)
    ops = laue_rotations(space_group)
    if len(ops) > 1:
        tr = np.einsum("nij,sji->ns", R, ops)
        keep = np.einsum("nii->n", R) >= tr.max(axis=1) - 1e-12
        R = R[keep]
    return R
