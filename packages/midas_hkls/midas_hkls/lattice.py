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


# ---------------------------------------------------------------------
#  Powder lattice refinement from observed d-spacings
# ---------------------------------------------------------------------

#: Which reciprocal-metric-tensor terms each crystal system is allowed, as
#: (name, coefficient function of (h, k, l)).  1/d^2 is *linear* in these,
#: which is what makes the refinement below a direct least squares with no
#: starting guess and no iteration.
_QUAD_TERMS = {
    "cubic":        [("A", lambda h, k, l: h * h + k * k + l * l)],
    "tetragonal":   [("A", lambda h, k, l: h * h + k * k),
                     ("C", lambda h, k, l: l * l)],
    "hexagonal":    [("A", lambda h, k, l: h * h + h * k + k * k),
                     ("C", lambda h, k, l: l * l)],
    "trigonal":     [("A", lambda h, k, l: h * h + h * k + k * k),
                     ("C", lambda h, k, l: l * l)],
    "orthorhombic": [("A", lambda h, k, l: h * h), ("B", lambda h, k, l: k * k),
                     ("C", lambda h, k, l: l * l)],
    "monoclinic":   [("A", lambda h, k, l: h * h), ("B", lambda h, k, l: k * k),
                     ("C", lambda h, k, l: l * l), ("E", lambda h, k, l: h * l)],
    "triclinic":    [("A", lambda h, k, l: h * h), ("B", lambda h, k, l: k * k),
                     ("C", lambda h, k, l: l * l), ("D", lambda h, k, l: h * k),
                     ("E", lambda h, k, l: h * l), ("F", lambda h, k, l: k * l)],
}


@dataclass(frozen=True)
class PowderLatticeFit:
    """Result of :func:`refine_lattice_from_d_spacings`."""
    lattice: "Lattice"
    system: str
    n_reflections: int
    rms_strain: float              #: RMS of (d_obs - d_calc)/d_calc
    max_abs_strain: float
    residual_strain: np.ndarray    #: per-reflection (d_obs - d_calc)/d_calc
    d_calc: np.ndarray
    condition_number: float
    sigma: dict                    #: 1-sigma on each refined length, Å


def refine_lattice_from_d_spacings(
    hkls,
    d_obs,
    system: str,
    *,
    weights=None,
) -> PowderLatticeFit:
    """Refine lattice constants directly from observed d-spacings.

    This is the *powder* determination of the cell: it uses only measured
    ring positions and Miller indices, so it does not depend on any
    per-grain/per-voxel refinement and cannot form a feedback loop with one.
    That matters when the cell is being used as the strain-free reference —
    recovering it from refined per-grain cells returns roughly the mean of
    whatever reference those refinements were started from.

    Because ``1/d^2 = h_i G*_ij h_j`` is **linear** in the reciprocal metric
    tensor, the symmetry-allowed components solve as a direct least squares.
    There is no starting guess and no iteration, so the answer is determined
    by the data alone.

    Parameters
    ----------
    hkls : array-like (N, 3)
        Miller indices of the observed reflections.
    d_obs : array-like (N,)
        Observed d-spacings, Å.
    system : str
        ``cubic``, ``tetragonal``, ``hexagonal``, ``trigonal``,
        ``orthorhombic``, ``monoclinic`` (b-unique) or ``triclinic``.
    weights : array-like (N,), optional
        Per-reflection weights (e.g. 1/sigma^2 on 1/d^2). Default uniform.

    Returns
    -------
    PowderLatticeFit

    Notes
    -----
    The fit is on ``1/d^2``; ``rms_strain`` is reported as a *relative*
    d-spacing residual so it is directly comparable to a calibrant strain.
    A cell that reproduces its own rings should sit near the measurement
    floor; a large ``rms_strain`` means the indexing, the wavelength or the
    distance is wrong, not that the cell needs more parameters.

    ``d_obs`` inherits any error in wavelength or sample-detector distance —
    those are degenerate with an overall cell scale exactly as they are in a
    powder calibration, so the cell is only as good as the geometry it came
    from.
    """
    key = str(system).strip().lower()
    if key not in _QUAD_TERMS:
        raise ValueError(
            f"unknown crystal system {system!r}; expected one of "
            f"{sorted(_QUAD_TERMS)}")
    terms = _QUAD_TERMS[key]

    hkls = np.asarray(hkls, dtype=float).reshape(-1, 3)
    d_obs = np.asarray(d_obs, dtype=float).reshape(-1)
    if hkls.shape[0] != d_obs.size:
        raise ValueError("hkls and d_obs must have the same length")
    if np.any(d_obs <= 0):
        raise ValueError("d_obs must be positive")
    if d_obs.size < len(terms):
        raise ValueError(
            f"{key} needs at least {len(terms)} reflections, got {d_obs.size}")

    w = (np.ones_like(d_obs) if weights is None
         else np.asarray(weights, dtype=float).reshape(-1))
    sw = np.sqrt(w)

    M = np.column_stack([[fn(*hkl) for hkl in hkls] for _name, fn in terms])
    y = 1.0 / d_obs ** 2
    sol, *_ = np.linalg.lstsq(M * sw[:, None], y * sw, rcond=None)
    sv = np.linalg.svd(M * sw[:, None], compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")

    q = dict(zip([n for n, _ in terms], sol))
    if np.any(np.asarray(list(q.values()))[:1] <= 0):
        raise ValueError("refinement returned a non-positive metric term; "
                         "check the hkl assignment")

    def _len(v):
        if v <= 0:
            raise ValueError("non-positive metric term; check hkl assignment")
        return 1.0 / sqrt(v)

    if key == "cubic":
        a = _len(q["A"]); lat = Lattice(a, a, a, 90.0, 90.0, 90.0)
        lengths = {"a": a}
    elif key == "tetragonal":
        a, c = _len(q["A"]), _len(q["C"])
        lat = Lattice(a, a, c, 90.0, 90.0, 90.0); lengths = {"a": a, "c": c}
    elif key in ("hexagonal", "trigonal"):
        # A = 4 / (3 a^2)
        a = sqrt(4.0 / (3.0 * q["A"])); c = _len(q["C"])
        lat = Lattice(a, a, c, 90.0, 90.0, 120.0); lengths = {"a": a, "c": c}
    elif key == "orthorhombic":
        a, b, c = _len(q["A"]), _len(q["B"]), _len(q["C"])
        lat = Lattice(a, b, c, 90.0, 90.0, 90.0)
        lengths = {"a": a, "b": b, "c": c}
    else:
        raise NotImplementedError(
            f"{key} refines the metric tensor but the cell-parameter "
            "back-conversion is not implemented yet; use the returned "
            "metric terms directly")

    d_calc = np.array([lat.d_spacing(*hkl) for hkl in hkls.astype(int)])
    resid = (d_obs - d_calc) / d_calc
    rms = float(np.sqrt(np.mean(resid ** 2)))

    # 1-sigma on each length, propagated from the linear fit
    dof = max(d_obs.size - len(terms), 1)
    s2 = float(np.sum(w * (y - M @ sol) ** 2) / dof)
    try:
        cov = s2 * np.linalg.inv((M * w[:, None]).T @ M)
        sig_q = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sig_q = np.full(len(terms), np.nan)
    # length = k * q^-1/2  =>  dlength/dq = -length / (2 q)
    sigma = {}
    for (name, _fn), sq in zip(terms, sig_q):
        if name == "A" and key in ("hexagonal", "trigonal"):
            sigma["a"] = abs(lengths["a"] / (2 * q["A"])) * sq
        elif name == "A":
            sigma["a"] = abs(lengths["a"] / (2 * q["A"])) * sq
        elif name == "B" and "b" in lengths:
            sigma["b"] = abs(lengths["b"] / (2 * q["B"])) * sq
        elif name == "C" and "c" in lengths:
            sigma["c"] = abs(lengths["c"] / (2 * q["C"])) * sq

    return PowderLatticeFit(
        lattice=lat, system=key, n_reflections=int(d_obs.size),
        rms_strain=rms, max_abs_strain=float(np.max(np.abs(resid))),
        residual_strain=resid, d_calc=d_calc,
        condition_number=cond, sigma=sigma,
    )
