"""Symmetry-constrained and joint multi-domain cell refinement.

:func:`midas_hkls.refine_ub_from_gvectors` refines a **free triclinic** cell in
closed form. That is the right default when you want to discover a lattice, and
the wrong tool when you want to measure a small distortion of a lattice you
already know, because the six free parameters absorb noise into exactly the
quantity being measured. Measured on real La3Ni2O7 DAC data (2026-09-03): six
extra free-triclinic parameters bought only 3 % in fit rms, and the ~1.6 %
"orthorhombic split" they produced was reproduced in full by bootstrap
resampling at *fixed* geometry -- it was noise wearing the shape of an effect.

Two things fix that, and both are here.

**Constrain the cell to its crystal system.** Fit ``a, b, c`` with the angles
pinned at 90 for orthorhombic, ``a, c`` for tetragonal, and so on. The
distortion you are measuring is then the only place it can go.

**Refine several domains jointly against ONE cell.** Domains of the same phase
share a cell but not an orientation. A joint fit multiplies the reflection count
against the same cell parameters, which is what actually buys precision -- on
that same data the binding limit was counting statistics, needing N ~ 1055
reflections against the 71 one domain supplied.

Both return bootstrap uncertainties as well as analytic ones, because the
analytic covariance assumes iid isotropic Gaussian error and rotation data
violate that: error across ``q`` (set by the omega step) is typically ~2x the
error along it (which sets cell lengths), and residuals are outlier-heavy. On
the data above the analytic sigma was ~2x the bootstrap one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .lattice import Lattice
from .ub_refine import cell_from_metric

__all__ = ["FREE_PARAMS", "ConstrainedFit", "DomainData",
           "refine_cell_constrained", "refine_cell_joint", "split_with_error"]

#: Free cell parameters per crystal system, and how to expand them to a, b, c,
#: alpha, beta, gamma. Everything else is fixed by symmetry.
FREE_PARAMS = {
    "triclinic":    ("a", "b", "c", "alpha", "beta", "gamma"),
    "monoclinic":   ("a", "b", "c", "beta"),
    "orthorhombic": ("a", "b", "c"),
    "tetragonal":   ("a", "c"),
    "hexagonal":    ("a", "c"),
    "rhombohedral": ("a", "alpha"),
    "cubic":        ("a",),
}


def _expand(system: str, p: Sequence[float]) -> Tuple[float, ...]:
    """Free parameters -> the full (a, b, c, alpha, beta, gamma)."""
    s = system
    if s == "cubic":
        a, = p; return (a, a, a, 90., 90., 90.)
    if s == "tetragonal":
        a, c = p; return (a, a, c, 90., 90., 90.)
    if s == "hexagonal":
        a, c = p; return (a, a, c, 90., 90., 120.)
    if s == "rhombohedral":
        a, al = p; return (a, a, a, al, al, al)
    if s == "orthorhombic":
        a, b, c = p; return (a, b, c, 90., 90., 90.)
    if s == "monoclinic":
        a, b, c, be = p; return (a, b, c, 90., be, 90.)
    if s == "triclinic":
        return tuple(float(v) for v in p)
    raise ValueError(f"unknown crystal system {system!r}")


def _collapse(system: str, cell: Sequence[float]) -> np.ndarray:
    """(a, b, c, alpha, beta, gamma) -> just the free parameters."""
    a, b, c, al, be, ga = (float(v) for v in cell)
    return np.array({"cubic": [a], "tetragonal": [a, c], "hexagonal": [a, c],
                     "rhombohedral": [a, al], "orthorhombic": [a, b, c],
                     "monoclinic": [a, b, c, be],
                     "triclinic": [a, b, c, al, be, ga]}[system], float)


def _B_of(cell: Sequence[float], two_pi: bool = False) -> np.ndarray:
    lat = Lattice(a=cell[0], b=cell[1], c=cell[2],
                  alpha=cell[3], beta=cell[4], gamma=cell[5])
    B = np.asarray(lat.reciprocal_cartesian_vectors(), float).T
    return B * (2*math.pi) if two_pi else B


def _rotvec_to_R(v: np.ndarray) -> np.ndarray:
    """Rodrigues. Small-angle safe."""
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th)*K + (1-math.cos(th))*(K @ K)


def _R_to_rotvec(R: np.ndarray) -> np.ndarray:
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-12:
        return np.zeros(3)
    if abs(math.pi - th) < 1e-6:                 # near 180 deg
        w, V = np.linalg.eigh((R + np.eye(3)) / 2.0)
        return V[:, int(np.argmax(w))] * th
    return th / (2*math.sin(th)) * np.array(
        [R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])


@dataclass
class DomainData:
    """One domain's indexed reflections."""
    hkl: np.ndarray
    g: np.ndarray
    label: str = ""


@dataclass
class ConstrainedFit:
    system: str
    cell: Tuple[float, ...]
    cell_sigma: Tuple[float, ...]
    rotvecs: List[np.ndarray]
    rms: float
    n_reflections: int
    n_domains: int
    two_pi: bool
    cell_sigma_bootstrap: Optional[Tuple[float, ...]] = None
    n_bootstrap: int = 0
    free_names: Tuple[str, ...] = ()

    def __str__(self) -> str:
        c, s = self.cell, self.cell_sigma
        return (f"{self.system} {c[0]:.4f}({s[0]*1e4:.0f}) "
                f"{c[1]:.4f}({s[1]*1e4:.0f}) {c[2]:.4f}({s[2]*1e4:.0f}) | "
                f"{c[3]:.2f} {c[4]:.2f} {c[5]:.2f}  "
                f"[{self.n_reflections} refl, {self.n_domains} domain(s), "
                f"rms {self.rms:.3e}]")


def _residuals(x, doms, system, nfree, two_pi):
    cell = _expand(system, x[:nfree])
    if min(cell[:3]) <= 0 or not all(1.0 < v < 179.0 for v in cell[3:]):
        return np.full(sum(len(d.hkl) for d in doms) * 3, 1e6)
    B = _B_of(cell, two_pi)
    out = []
    for i, d in enumerate(doms):
        R = _rotvec_to_R(x[nfree + 3*i: nfree + 3*i + 3])
        out.append(((d.hkl @ (R @ B).T) - d.g).ravel())
    return np.concatenate(out)


def _fit(doms, system, cell0, rot0, two_pi, maxfev):
    from scipy.optimize import least_squares
    nfree = len(FREE_PARAMS[system])
    x0 = np.concatenate([_collapse(system, cell0)] + [np.asarray(r, float) for r in rot0])
    res = least_squares(_residuals, x0, args=(doms, system, nfree, two_pi),
                        method="lm", max_nfev=maxfev)
    return res, nfree


def refine_cell_joint(domains: Sequence[DomainData], *, system: str,
                      cell0: Sequence[float],
                      two_pi: bool = False,
                      n_bootstrap: int = 0,
                      rng: Optional[np.random.Generator] = None,
                      maxfev: int = 20000) -> ConstrainedFit:
    """Refine ONE cell of the given crystal system against several domains.

    Each domain gets its own orientation; all share the cell. ``cell0`` is the
    starting cell (a, b, c, alpha, beta, gamma) — only the components the system
    leaves free are refined.

    ``n_bootstrap > 0`` adds a resampling error on the cell parameters. Prefer
    it to the analytic sigma whenever the residuals are anisotropic or
    outlier-heavy, which for rotation data they generally are.
    """
    doms = list(domains)
    if not doms:
        raise ValueError("no domains given")
    if system not in FREE_PARAMS:
        raise ValueError(f"unknown crystal system {system!r}")
    B0 = _B_of(cell0, two_pi)
    rot0 = []
    for d in doms:                      # per-domain orientation by Kabsch
        M = (d.g.T @ (d.hkl @ np.linalg.inv(B0).T))
        Usv, _, Vt = np.linalg.svd(M)
        R = Usv @ np.diag([1., 1., float(np.sign(np.linalg.det(Usv @ Vt)))]) @ Vt
        rot0.append(_R_to_rotvec(R))

    res, nfree = _fit(doms, system, cell0, rot0, two_pi, maxfev)
    cell = _expand(system, res.x[:nfree])
    n = sum(len(d.hkl) for d in doms)
    rms = float(np.sqrt(np.mean(res.fun**2)))

    # analytic sigma on the free parameters, propagated to all six
    dof = max(1, 3*n - len(res.x))
    s2 = float(res.fun @ res.fun) / dof
    try:
        cov = s2 * np.linalg.inv(res.jac.T @ res.jac)
        sig_free = np.sqrt(np.clip(np.diag(cov)[:nfree], 0, None))
    except np.linalg.LinAlgError:
        sig_free = np.full(nfree, np.nan)
    J = np.zeros((6, nfree))
    for j in range(nfree):
        h = max(1e-7, abs(res.x[j]) * 1e-6)
        xp = res.x[:nfree].copy(); xp[j] += h
        xm = res.x[:nfree].copy(); xm[j] -= h
        J[:, j] = (np.array(_expand(system, xp)) - np.array(_expand(system, xm))) / (2*h)
    cell_sigma = tuple(float(v) for v in np.sqrt(np.clip((J**2) @ (sig_free**2), 0, None)))

    boot = None
    if n_bootstrap > 0:
        rng = rng or np.random.default_rng(0)
        keep = []
        for _ in range(int(n_bootstrap)):
            rs = [DomainData(d.hkl[i], d.g[i], d.label) for d in doms
                  for i in [rng.integers(0, len(d.hkl), len(d.hkl))]]
            try:
                r2, _ = _fit(rs, system, cell, [res.x[nfree+3*i:nfree+3*i+3]
                                                for i in range(len(doms))],
                             two_pi, maxfev)
                keep.append(_expand(system, r2.x[:nfree]))
            except Exception:
                continue
        if len(keep) > 4:
            boot = tuple(float(v) for v in np.std(np.array(keep), axis=0, ddof=1))

    return ConstrainedFit(
        system=system, cell=tuple(float(v) for v in cell), cell_sigma=cell_sigma,
        rotvecs=[res.x[nfree+3*i:nfree+3*i+3] for i in range(len(doms))],
        rms=rms, n_reflections=n, n_domains=len(doms), two_pi=two_pi,
        cell_sigma_bootstrap=boot, n_bootstrap=len(keep) if boot else 0,
        free_names=tuple(FREE_PARAMS[system]))


def refine_cell_constrained(hkl, g, *, system: str, cell0, **kw) -> ConstrainedFit:
    """Single-domain convenience wrapper over :func:`refine_cell_joint`."""
    return refine_cell_joint([DomainData(np.asarray(hkl, float),
                                         np.asarray(g, float))],
                             system=system, cell0=cell0, **kw)


def split_with_error(fit: ConstrainedFit) -> Tuple[float, float, float]:
    """The a/b split in per cent, its sigma, and the significance.

    Uses the bootstrap sigma when one was computed, since the analytic one
    assumes an error model rotation data do not obey. Returns ``(split, sigma,
    z)``; ``sigma`` is NaN when a and b are equal by symmetry, because then the
    quantity is not measured but imposed.
    """
    a, b = fit.cell[0], fit.cell[1]
    if fit.system in ("cubic", "tetragonal", "hexagonal", "rhombohedral"):
        return 0.0, float("nan"), float("nan")
    s = fit.cell_sigma_bootstrap or fit.cell_sigma
    split = 200.0 * abs(a - b) / (a + b)
    sig = 200.0 * math.hypot(s[0], s[1]) / (a + b)
    return split, sig, (split / sig if sig > 0 else float("nan"))
