"""Direct & reciprocal lattice geometry."""
from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, degrees, pi, radians, sin, sqrt
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Lattice:
    a: float    # Å
    b: float
    c: float
    alpha: float  # degrees
    beta: float
    gamma: float

    def __post_init__(self) -> None:
        if min(self.a, self.b, self.c) <= 0:
            raise ValueError("lattice constants must be positive")
        for ang in (self.alpha, self.beta, self.gamma):
            if not (0 < ang < 180):
                raise ValueError("lattice angles must be in (0, 180) degrees")

    def metric_tensor(self) -> np.ndarray:
        ca = cos(radians(self.alpha))
        cb = cos(radians(self.beta))
        cg = cos(radians(self.gamma))
        return np.array([
            [self.a * self.a,        self.a * self.b * cg, self.a * self.c * cb],
            [self.a * self.b * cg,   self.b * self.b,      self.b * self.c * ca],
            [self.a * self.c * cb,   self.b * self.c * ca, self.c * self.c],
        ])

    def volume(self) -> float:
        return float(sqrt(np.linalg.det(self.metric_tensor())))

    def reciprocal_metric_tensor(self) -> np.ndarray:
        return np.linalg.inv(self.metric_tensor())

    def cartesian_vectors(self) -> np.ndarray:
        """Direct lattice vectors as ROWS, embedded in a Cartesian frame.

        Convention: **a1 along x, a2 in the xy plane with positive y, a3 with
        positive z** — the standard crystallographic embedding.

        The metric tensors above are enough for scalar quantities (d-spacings,
        angles), because those are basis-independent. Anything with a *direction*
        needs this: a symmetry operation from a space group is an integer matrix
        in the LATTICE basis, and an integer matrix is not a rotation (the
        hexagonal 6-fold has entries in {0, ±1} and is emphatically not
        orthogonal). Conjugating through this embedding, ``R_cart = M R M^-1``
        with ``M`` the columns of the direct vectors, is what turns it into one.
        """
        al, be, ga = radians(self.alpha), radians(self.beta), radians(self.gamma)
        v1 = np.array([self.a, 0.0, 0.0])
        v2 = np.array([self.b * cos(ga), self.b * sin(ga), 0.0])
        cx = self.c * cos(be)
        cy = self.c * (cos(al) - cos(be) * cos(ga)) / sin(ga)
        cz2 = self.c * self.c - cx * cx - cy * cy
        if cz2 <= 0:
            raise ValueError(
                f"degenerate cell: a={self.a} b={self.b} c={self.c} "
                f"alpha={self.alpha} beta={self.beta} gamma={self.gamma} "
                "do not close a positive-volume parallelepiped")
        return np.array([v1, v2, [cx, cy, sqrt(cz2)]])

    def reciprocal_cartesian_vectors(self) -> np.ndarray:
        """Reciprocal lattice vectors as ROWS, in the same Cartesian frame.

        The 2π convention is dropped: these are the crystallographic
        ``b_i = (a_j x a_k) / V`` reciprocal vectors, so ``|b_i| = 1/d`` for the
        corresponding plane. Callers that only want a *direction* (a plane
        normal) normalise anyway, so the convention cancels there.
        """
        d = self.cartesian_vectors()
        vol = float(np.dot(d[0], np.cross(d[1], d[2])))
        return np.array([np.cross(d[1], d[2]), np.cross(d[2], d[0]),
                         np.cross(d[0], d[1])]) / vol

    def reciprocal(self) -> "Lattice":
        Gstar = self.reciprocal_metric_tensor()
        a_star = sqrt(Gstar[0, 0])
        b_star = sqrt(Gstar[1, 1])
        c_star = sqrt(Gstar[2, 2])
        alpha_s = degrees(np.arccos(Gstar[1, 2] / (b_star * c_star)))
        beta_s  = degrees(np.arccos(Gstar[0, 2] / (a_star * c_star)))
        gamma_s = degrees(np.arccos(Gstar[0, 1] / (a_star * b_star)))
        return Lattice(a_star, b_star, c_star, alpha_s, beta_s, gamma_s)

    def d_spacing(self, h: int, k: int, l: int) -> float:
        """Compute d_hkl in Å using 1/d^2 = h_i G*_ij h_j."""
        Gstar = self.reciprocal_metric_tensor()
        v = np.array([h, k, l], dtype=float)
        inv_d2 = float(v @ Gstar @ v)
        if inv_d2 <= 0:
            return float("inf")
        return 1.0 / sqrt(inv_d2)

    def two_theta_deg(self, h: int, k: int, l: int, wavelength_A: float) -> float:
        """Bragg 2θ in degrees for a reflection (returns NaN if outside Bragg cutoff)."""
        d = self.d_spacing(h, k, l)
        s = wavelength_A / (2.0 * d)
        if not (-1.0 <= s <= 1.0):
            return float("nan")
        return 2.0 * degrees(asin(s))

    @classmethod
    def for_system(cls, system: str, *, a: float, b: float | None = None, c: float | None = None,
                   alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0) -> "Lattice":
        """Build a lattice with the symmetry constraints of the given crystal system."""
        sysname = system.lower()
        if sysname == "cubic":
            return cls(a, a, a, 90.0, 90.0, 90.0)
        if sysname == "tetragonal":
            if c is None:
                raise ValueError("tetragonal requires c")
            return cls(a, a, c, 90.0, 90.0, 90.0)
        if sysname == "orthorhombic":
            if b is None or c is None:
                raise ValueError("orthorhombic requires a, b, c")
            return cls(a, b, c, 90.0, 90.0, 90.0)
        if sysname == "hexagonal" or sysname == "trigonal":
            if c is None:
                raise ValueError("hexagonal/trigonal requires c (or use rhombohedral)")
            return cls(a, a, c, 90.0, 90.0, 120.0)
        if sysname == "monoclinic":
            if b is None or c is None:
                raise ValueError("monoclinic requires a, b, c, beta")
            return cls(a, b, c, 90.0, beta, 90.0)
        if sysname == "triclinic":
            if b is None or c is None:
                raise ValueError("triclinic requires a, b, c, alpha, beta, gamma")
            return cls(a, b, c, alpha, beta, gamma)
        raise ValueError(f"unknown crystal system: {system}")
