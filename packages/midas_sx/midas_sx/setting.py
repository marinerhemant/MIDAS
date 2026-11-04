"""Axis setting of a lattice found without a cell.

``midas_hkls.ab_initio`` returns a basis in whatever order and handedness it finds; the reduction needs U such that ``q_sample = U B h`` for the REFERENCE cell and
space group (axis order and directions of the CIF). :func:`match_setting` finds the signed axis permutation that makes the found lattice the reference one.
"""
from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np

from midas_hkls import Lattice

from .assign import reciprocal_basis

__all__ = ["match_setting"]


def match_setting(UB: np.ndarray, lattice_ref: Lattice, *, max_orthogonality_error: float = 0.03) -> Tuple[np.ndarray, np.ndarray, float]:
    """``UB`` (3, 3): columns are the found reciprocal basis vectors in the sample frame (1/d). Returns ``(U, P, err)``: ``P`` the signed permutation matrix (columns of UB
    reordered / flipped, determinant +1) and ``U = UB P B_ref^-1`` the nearest rotation; ``err`` = max |U U^T - I| before the polar decomposition, the figure of merit (a wrong
    permutation, or a lattice that is not the reference, gives errors of order 0.1 and more). Raises ValueError above ``max_orthogonality_error``.

    Mirror-equivalent signed permutations (all of them for an orthorhombic mmm lattice) give the same intensities; the first best one is returned. NOT valid for a chiral lattice setting question: for a lower-symmetry lattice (monoclinic, triclinic) the mirror image is a different setting and only the proper ones are considered."""
    B = reciprocal_basis(lattice_ref)
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            P = np.zeros((3, 3))
            for j, (i, s_) in enumerate(zip(perm, signs)):
                P[i, j] = s_
            U0 = np.asarray(UB, float) @ P @ np.linalg.inv(B)
            if np.linalg.det(U0) < 0:                                      # a reflection is orthogonal too: the orthogonality error alone cannot tell it from a rotation
                continue
            err = float(np.abs(U0 @ U0.T - np.eye(3)).max())
            if best is None or err < best[2] - 1e-12:
                best = (U0, P, err)
    U0, P, err = best
    if err > max_orthogonality_error:
        raise ValueError(f"no signed permutation makes the found lattice the reference cell (best orthogonality error {err:.3f} > {max_orthogonality_error})")
    u, _, vt = np.linalg.svd(U0)
    return u @ vt, P, err
