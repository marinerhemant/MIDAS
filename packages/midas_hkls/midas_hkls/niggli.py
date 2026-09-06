"""Niggli reduction — the canonical form of a lattice.

Two bases describe the same lattice if and only if their **Niggli-reduced** cells
agree. That is the property this module exists for: an ab-initio indexer, or any
two people working from different settings, can produce wildly different-looking
cells for one lattice, and there is no way to compare them until both are reduced.

Measured need: indexing the same La₃Ni₂O₇ lattice at different tolerances
returned an I-tetragonal cell, an F-orthorhombic one in the √2 setting, and a
C-monoclinic one. All three are the same lattice. Reduced, they agree.

Algorithm
---------
Křivý & Gruber, *Acta Cryst.* **A32** 297 (1976), with the standard corrections
(Grosse-Kunstleve, Sauter & Adams, *Acta Cryst.* **A60** 1 (2004)). It works on
the metric scalars

    A = a·a,  B = b·b,  C = c·c,  ξ = 2b·c,  η = 2a·c,  ζ = 2a·b

and applies eight conditions until none fires. The transformation is tracked so
the reduced basis can be recovered, not just its metric.

Note on the transformation matrices: basis vectors are **columns**, so
``(A @ M)[:, j] = Σ_k A[:, k] M[k, j]``. Step 5 is ``c → c − s·b`` and therefore
sets ``M[1, 2] = −s``. Writing ``M[2, 1]`` instead modifies **b**, and the
algorithm then oscillates forever between steps 1/2 and 5/6/7 without ever
converging — which is exactly what a transposed matrix looks like from outside.

Tolerance
---------
Every comparison is against ``eps``, taken relative to the cell size. Exact
arithmetic would loop forever on a cell that is metrically degenerate to within
floating point; a tolerance both terminates and expresses the real question,
which is whether two lengths are equal *to the precision of the measurement*.
Pass the ``eps`` your data justifies rather than accepting the default silently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["NiggliCell", "niggli_reduce", "same_lattice"]


@dataclass
class NiggliCell:
    """A Niggli-reduced cell and the transformation that produced it."""
    cell: Tuple[float, ...]
    transformation: np.ndarray        # integer, A_reduced = A_input @ T
    n_iterations: int
    converged: bool

    def __str__(self) -> str:
        c = self.cell
        return (f"Niggli {c[0]:.4f} {c[1]:.4f} {c[2]:.4f} | "
                f"{c[3]:.3f} {c[4]:.3f} {c[5]:.3f}"
                + ("" if self.converged else "  [DID NOT CONVERGE]"))


def _scalars(A_mat: np.ndarray) -> Tuple[float, ...]:
    G = A_mat.T @ A_mat
    return (G[0, 0], G[1, 1], G[2, 2], 2 * G[1, 2], 2 * G[0, 2], 2 * G[0, 1])


def _cell_of(A_mat: np.ndarray) -> Tuple[float, ...]:
    from .ub_refine import cell_from_metric
    return cell_from_metric(A_mat.T @ A_mat)


def niggli_reduce(cell: Sequence[float], *, eps_rel: float = 1e-9,
                  max_iter: int = 500) -> NiggliCell:
    """Reduce a cell to its unique Niggli form.

    ``cell`` is ``(a, b, c, alpha, beta, gamma)`` with angles in degrees.
    Returns the reduced cell and the integer transformation ``T`` with
    ``A_reduced = A_input @ T``.

    Raises ``ValueError`` on a degenerate cell. Sets ``converged=False`` rather
    than looping forever if ``max_iter`` is hit — which in practice means ``eps``
    is too tight for the numbers.
    """
    from .lattice import Lattice
    c0 = tuple(float(v) for v in cell)
    lat = Lattice(a=c0[0], b=c0[1], c=c0[2], alpha=c0[3], beta=c0[4], gamma=c0[5])
    A0 = np.asarray(lat.cartesian_vectors(), float).T
    if abs(np.linalg.det(A0)) < 1e-12:
        raise ValueError("degenerate cell: zero volume")

    T = np.eye(3, dtype=np.int64)
    eps = eps_rel * max(_scalars(A0)[:3])

    def gt(x, y):  return x > y + eps
    def eq(x, y):  return abs(x - y) <= eps
    def lt(x, y):  return x < y - eps

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        A_mat = A0 @ T
        Aa, Bb, Cc, xi, eta, zeta = _scalars(A_mat)

        # 1 / 2 — order the axes
        if gt(Aa, Bb) or (eq(Aa, Bb) and gt(abs(xi), abs(eta))):
            T = T @ np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], np.int64)
            continue
        if gt(Bb, Cc) or (eq(Bb, Cc) and gt(abs(eta), abs(zeta))):
            T = T @ np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], np.int64)
            continue

        # 3 / 4 — make the products consistently signed
        n_neg = sum(1 for v in (xi, eta, zeta) if lt(v, 0.0))
        n_zero = sum(1 for v in (xi, eta, zeta) if eq(v, 0.0))
        if (n_neg % 2) == 0 and n_zero == 0:          # all-positive form
            if n_neg:
                d = np.diag([-1 if lt(xi, 0) else 1,
                             -1 if lt(eta, 0) else 1,
                             -1 if lt(zeta, 0) else 1]).astype(np.int64)
                if np.linalg.det(d) < 0:
                    d = -d
                T = T @ d
                continue
        else:                                          # all-non-positive form
            f = [1, 1, 1]
            for idx, v in enumerate((xi, eta, zeta)):
                if gt(v, 0.0):
                    f[idx] = -1
            if np.prod(f) < 0:
                # flip the one that is closest to zero
                j = int(np.argmin([abs(v) for v in (xi, eta, zeta)]))
                f[j] = -f[j]
            if f != [1, 1, 1]:
                T = T @ np.diag(f).astype(np.int64)
                continue

        # 5, 6, 7 — reduce the off-diagonals
        if gt(abs(xi), Bb) or (eq(xi, Bb) and lt(2 * eta, zeta)) \
                or (eq(xi, -Bb) and lt(zeta, 0.0)):
            s = 1 if xi > 0 else -1
            T = T @ np.array([[1, 0, 0], [0, 1, -s], [0, 0, 1]], np.int64)  # c -= s*b
            continue
        if gt(abs(eta), Aa) or (eq(eta, Aa) and lt(2 * xi, zeta)) \
                or (eq(eta, -Aa) and lt(zeta, 0.0)):
            s = 1 if eta > 0 else -1
            T = T @ np.array([[1, 0, -s], [0, 1, 0], [0, 0, 1]], np.int64)  # c -= s*a
            continue
        if gt(abs(zeta), Aa) or (eq(zeta, Aa) and lt(2 * xi, eta)) \
                or (eq(zeta, -Aa) and lt(eta, 0.0)):
            s = 1 if zeta > 0 else -1
            T = T @ np.array([[1, -s, 0], [0, 1, 0], [0, 0, 1]], np.int64)  # b -= s*a
            continue

        # 8 — the final all-negative condition
        if lt(xi + eta + zeta + Aa + Bb, 0.0) or \
                (eq(xi + eta + zeta + Aa + Bb, 0.0) and gt(2 * (Aa + eta) + zeta, 0.0)):
            T = T @ np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], np.int64)
            continue

        converged = True
        break

    A_red = A0 @ T
    if np.linalg.det(A_red) < 0:
        T = T @ np.diag([1, 1, -1]).astype(np.int64)
        A_red = A0 @ T
    return NiggliCell(cell=_cell_of(A_red), transformation=T,
                      n_iterations=it, converged=converged)


def same_lattice(cell_a: Sequence[float], cell_b: Sequence[float], *,
                 rel_len: float = 1e-3, abs_ang: float = 0.05) -> bool:
    """Do two cells describe the **same lattice**, in any setting?

    Reduces both and compares invariants. This is the only correct way to ask:
    comparing axis lengths directly says "no" for two settings of one lattice,
    which is how a correct ab-initio result gets reported as a failure.

    The comparison is on **sorted lengths and sorted |90° − angle|**, not on the
    reduced cells element by element, and that is deliberate. A Niggli cell is
    either type I (all angles acute) or type II (all obtuse), and the two are
    supplements. When an angle sits near 90° — extremely common, since it happens
    whenever the lattice is nearly orthogonal — measurement noise decides the
    type, so two determinations of one lattice legitimately reduce to different
    types: 79.9° in one and 100.1° in the other. Comparing the deviation from
    90° is invariant to that flip; comparing the angles directly is not, and
    reports the same lattice as two.
    """
    ra = niggli_reduce(cell_a).cell
    rb = niggli_reduce(cell_b).cell
    la, lb = sorted(ra[:3]), sorted(rb[:3])
    for x, y in zip(la, lb):
        if abs(x - y) > rel_len * max(abs(x), abs(y), 1e-12):
            return False
    da = sorted(abs(v - 90.0) for v in ra[3:])
    db = sorted(abs(v - 90.0) for v in rb[3:])
    return all(abs(x - y) <= abs_ang for x, y in zip(da, db))
