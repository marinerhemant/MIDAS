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
           "refine_cell_constrained", "refine_cell_joint", "refine_cell_radial",
           "refine_cell_joint_robust", "refine_cell_radial_robust",
           "tukey_biweight", "split_with_error"]

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
    #: Every individual bootstrap resample's full expanded cell, (n_kept, 6) array of
    #: (a, b, c, alpha, beta, gamma), or None when n_bootstrap=0 or too few resamples
    #: succeeded. `cell_sigma_bootstrap` is this array's std; kept here too because a
    #: single sigma number cannot show a skewed or multi-modal spread, and "the envelope
    #: IS the result" (this project's own standing lesson) applies to a, b, c individually,
    #: not only to their difference.
    cell_bootstrap_samples: Optional[np.ndarray] = None
    #: Per-domain, per-reflection weight used by this fit (1.0 = fully trusted,
    #: 0.0 = fully excluded). ``None`` for a plain (non-robust, no manual
    #: ``weights=``) call. Set explicitly when a caller passes ``weights=``,
    #: and set to the FINAL converged weights by :func:`refine_cell_joint_robust`
    #: / :func:`refine_cell_radial_robust`.
    weights: Optional[Tuple[np.ndarray, ...]] = None
    #: Number of IRLS iterations the robust wrappers took to converge (0 for
    #: a plain fit, including one called with a fixed ``weights=`` array).
    robust_n_iter: int = 0

    def __str__(self) -> str:
        c, s = self.cell, self.cell_sigma
        return (f"{self.system} {c[0]:.4f}({s[0]*1e4:.0f}) "
                f"{c[1]:.4f}({s[1]*1e4:.0f}) {c[2]:.4f}({s[2]*1e4:.0f}) | "
                f"{c[3]:.2f} {c[4]:.2f} {c[5]:.2f}  "
                f"[{self.n_reflections} refl, {self.n_domains} domain(s), "
                f"rms {self.rms:.3e}]")


def _residuals(x, doms, system, nfree, two_pi, weights=None):
    cell = _expand(system, x[:nfree])
    if min(cell[:3]) <= 0 or not all(1.0 < v < 179.0 for v in cell[3:]):
        return np.full(sum(len(d.hkl) for d in doms) * 3, 1e6)
    B = _B_of(cell, two_pi)
    out = []
    for i, d in enumerate(doms):
        R = _rotvec_to_R(x[nfree + 3*i: nfree + 3*i + 3])
        r = (d.hkl @ (R @ B).T) - d.g
        if weights is not None:
            r = r * weights[i][:, None]
        out.append(r.ravel())
    return np.concatenate(out)


def _fit(doms, system, cell0, rot0, two_pi, maxfev, weights=None):
    from scipy.optimize import least_squares
    nfree = len(FREE_PARAMS[system])
    x0 = np.concatenate([_collapse(system, cell0)] + [np.asarray(r, float) for r in rot0])
    res = least_squares(_residuals, x0, args=(doms, system, nfree, two_pi, weights),
                        method="lm", max_nfev=maxfev)
    return res, nfree


def refine_cell_joint(domains: Sequence[DomainData], *, system: str,
                      cell0: Sequence[float],
                      two_pi: bool = False,
                      n_bootstrap: int = 0,
                      rng: Optional[np.random.Generator] = None,
                      maxfev: int = 20000,
                      weights: Optional[Sequence[np.ndarray]] = None) -> ConstrainedFit:
    """Refine ONE cell of the given crystal system against several domains.

    Each domain gets its own orientation; all share the cell. ``cell0`` is the
    starting cell (a, b, c, alpha, beta, gamma) — only the components the system
    leaves free are refined.

    ``n_bootstrap > 0`` adds a resampling error on the cell parameters. Prefer
    it to the analytic sigma whenever the residuals are anisotropic or
    outlier-heavy, which for rotation data they generally are.

    ``weights``, if given, is one array per domain (same length as that
    domain's ``hkl``/``g``) scaling its residual rows before the sum of
    squares — 1.0 trusts a reflection fully, 0.0 excludes it. This is a fixed,
    caller-supplied weighting; to have the weights themselves determined from
    the data (an outlier that fits badly gets automatically downweighted), use
    :func:`refine_cell_joint_robust` instead, which calls this function
    repeatedly inside an IRLS loop.
    """
    doms = list(domains)
    if not doms:
        raise ValueError("no domains given")
    if system not in FREE_PARAMS:
        raise ValueError(f"unknown crystal system {system!r}")
    if weights is not None:
        weights = [np.asarray(w, float) for w in weights]
        if len(weights) != len(doms):
            raise ValueError(f"weights has {len(weights)} entries, expected {len(doms)} (one per domain)")
        for d, w in zip(doms, weights):
            if w.shape != (len(d.hkl),):
                raise ValueError(f"weights entry has shape {w.shape}, expected ({len(d.hkl)},)")
    B0 = _B_of(cell0, two_pi)
    rot0 = []
    for d in doms:                      # per-domain orientation by Kabsch
        M = (d.g.T @ (d.hkl @ np.linalg.inv(B0).T))
        Usv, _, Vt = np.linalg.svd(M)
        R = Usv @ np.diag([1., 1., float(np.sign(np.linalg.det(Usv @ Vt)))]) @ Vt
        rot0.append(_R_to_rotvec(R))

    res, nfree = _fit(doms, system, cell0, rot0, two_pi, maxfev, weights)
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
    keep_arr = None
    if n_bootstrap > 0:
        rng = rng or np.random.default_rng(0)
        keep = []
        for _ in range(int(n_bootstrap)):
            rs = []
            w_rs = [] if weights is not None else None
            for j, d in enumerate(doms):
                i = rng.integers(0, len(d.hkl), len(d.hkl))
                rs.append(DomainData(d.hkl[i], d.g[i], d.label))
                if weights is not None:
                    w_rs.append(weights[j][i])
            try:
                r2, _ = _fit(rs, system, cell, [res.x[nfree+3*i:nfree+3*i+3]
                                                for i in range(len(doms))],
                             two_pi, maxfev, w_rs)
                keep.append(_expand(system, r2.x[:nfree]))
            except Exception:
                continue
        if len(keep) > 4:
            keep_arr = np.array(keep)
            boot = tuple(float(v) for v in np.std(keep_arr, axis=0, ddof=1))

    return ConstrainedFit(
        system=system, cell=tuple(float(v) for v in cell), cell_sigma=cell_sigma,
        rotvecs=[res.x[nfree+3*i:nfree+3*i+3] for i in range(len(doms))],
        rms=rms, n_reflections=n, n_domains=len(doms), two_pi=two_pi,
        cell_sigma_bootstrap=boot, n_bootstrap=len(keep) if boot else 0,
        free_names=tuple(FREE_PARAMS[system]),
        cell_bootstrap_samples=keep_arr if n_bootstrap > 0 else None,
        weights=tuple(weights) if weights is not None else None)


def _residuals_radial(x, doms, system, nfree, two_pi, weights=None):
    cell = _expand(system, x[:nfree])
    if min(cell[:3]) <= 0 or not all(1.0 < v < 179.0 for v in cell[3:]):
        return np.full(sum(len(d.hkl) for d in doms), 1e6)
    B = _B_of(cell, two_pi)
    out = []
    for i, d in enumerate(doms):
        g_pred_mag = np.linalg.norm(d.hkl @ B.T, axis=1)
        g_obs_mag = np.linalg.norm(d.g, axis=1)
        r = g_pred_mag - g_obs_mag
        if weights is not None:
            r = r * weights[i]
        out.append(r)
    return np.concatenate(out)


def _fit_radial(doms, system, cell0, two_pi, maxfev, weights=None):
    from scipy.optimize import least_squares
    nfree = len(FREE_PARAMS[system])
    x0 = _collapse(system, cell0)
    res = least_squares(_residuals_radial, x0, args=(doms, system, nfree, two_pi, weights),
                        method="lm", max_nfev=maxfev)
    return res, nfree


def refine_cell_radial(domains: Sequence[DomainData], *, system: str,
                       cell0: Sequence[float],
                       two_pi: bool = False,
                       n_bootstrap: int = 0,
                       rng: Optional[np.random.Generator] = None,
                       maxfev: int = 20000,
                       weights: Optional[Sequence[np.ndarray]] = None) -> ConstrainedFit:
    """Refine ONE cell from the RADIAL ``|g|`` residual only — orientation-free.

    Unlike :func:`refine_cell_joint`, no per-domain rotation is fitted (or
    needed) at all: a rotation matrix preserves vector norm exactly, so
    ``|g_pred| = |B(cell) @ hkl|`` depends on the cell alone, never on
    orientation. Orientation is used upstream, to assign each observed spot
    its ``hkl`` in the first place — it plays no role in this residual once
    that assignment is fixed. ``DomainData.g`` still holds the full 3-vector
    (so callers pass the same domains as :func:`refine_cell_joint`); only its
    norm is used here.

    This is the length-only half of the staged refit registered in
    ``PREREGISTER_staged_radial_refit_2026-09-13.md`` (real La3Ni2O7 DAC
    data): the joint fit's residual mixes the radial (length-setting) and
    tangential (omega-step-set, ~2x noisier per this module's own docstring)
    components of ``g_pred - g_obs`` in one isotropic sum of squares. This
    function isolates the radial component only. Use it AFTER orientation
    has already been fixed by a separate fit (e.g.
    ``seed_index.refine_U_from_centroids`` /
    ``seed_index.bootstrap_orientation_uncertainty``), not as a replacement
    for indexing — it cannot determine orientation and does not try to.

    ``ConstrainedFit.rotvecs`` is always ``[]`` here (there is nothing to
    report): this function never fits one.

    ``weights``, same meaning as in :func:`refine_cell_joint`: one fixed
    array per domain, 1.0 trusts a reflection, 0.0 excludes it. For weights
    determined automatically from the data, see :func:`refine_cell_radial_robust`.
    """
    doms = list(domains)
    if not doms:
        raise ValueError("no domains given")
    if system not in FREE_PARAMS:
        raise ValueError(f"unknown crystal system {system!r}")
    if weights is not None:
        weights = [np.asarray(w, float) for w in weights]
        if len(weights) != len(doms):
            raise ValueError(f"weights has {len(weights)} entries, expected {len(doms)} (one per domain)")
        for d, w in zip(doms, weights):
            if w.shape != (len(d.hkl),):
                raise ValueError(f"weights entry has shape {w.shape}, expected ({len(d.hkl)},)")

    res, nfree = _fit_radial(doms, system, cell0, two_pi, maxfev, weights)
    cell = _expand(system, res.x[:nfree])
    n = sum(len(d.hkl) for d in doms)
    rms = float(np.sqrt(np.mean(res.fun**2)))

    dof = max(1, n - nfree)
    s2 = float(res.fun @ res.fun) / dof
    try:
        cov = s2 * np.linalg.inv(res.jac.T @ res.jac)
        sig_free = np.sqrt(np.clip(np.diag(cov), 0, None))
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
    keep_arr = None
    if n_bootstrap > 0:
        rng = rng or np.random.default_rng(0)
        keep = []
        for _ in range(int(n_bootstrap)):
            rs = []
            w_rs = [] if weights is not None else None
            for j, d in enumerate(doms):
                i = rng.integers(0, len(d.hkl), len(d.hkl))
                rs.append(DomainData(d.hkl[i], d.g[i], d.label))
                if weights is not None:
                    w_rs.append(weights[j][i])
            try:
                r2, _ = _fit_radial(rs, system, cell, two_pi, maxfev, w_rs)
                keep.append(_expand(system, r2.x[:nfree]))
            except Exception:
                continue
        if len(keep) > 4:
            keep_arr = np.array(keep)
            boot = tuple(float(v) for v in np.std(keep_arr, axis=0, ddof=1))

    return ConstrainedFit(
        system=system, cell=tuple(float(v) for v in cell), cell_sigma=cell_sigma,
        rotvecs=[], rms=rms, n_reflections=n, n_domains=len(doms), two_pi=two_pi,
        cell_sigma_bootstrap=boot, n_bootstrap=len(keep) if boot else 0,
        free_names=tuple(FREE_PARAMS[system]),
        cell_bootstrap_samples=keep_arr if n_bootstrap > 0 else None,
        weights=tuple(weights) if weights is not None else None)


def tukey_biweight(resid: Sequence[float], c: float = 4.685) -> Tuple[np.ndarray, float]:
    """Tukey biweight (bisquare) robust weights from a scalar residual array.

    ``scale`` is the median-absolute-deviation robust scale
    (``1.4826 * median(|resid - median(resid)|)``, falling back to the
    un-centered MAD, then to ``1.0``, if that vanishes -- e.g. more than half
    the residuals are exactly equal). ``c=4.685`` is the standard choice for
    about 95% efficiency relative to least squares under Gaussian errors. A
    reflection with ``|resid|/scale >= c`` gets weight exactly 0 (fully
    excluded, not merely downweighted); everything else gets a smooth weight
    between 0 and 1 that falls off as the residual grows.
    """
    resid = np.asarray(resid, float)
    scale = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    if scale <= 0:
        scale = 1.4826 * float(np.median(np.abs(resid)))
    if scale <= 0:
        scale = 1.0
    u = resid / (c * scale)
    w = np.where(np.abs(u) < 1, (1 - u**2)**2, 0.0)
    return w, scale


def refine_cell_joint_robust(domains: Sequence[DomainData], *, system: str,
                             cell0: Sequence[float],
                             two_pi: bool = False,
                             n_bootstrap: int = 0,
                             rng: Optional[np.random.Generator] = None,
                             maxfev: int = 20000,
                             tukey_c: float = 4.685,
                             max_iter: int = 10,
                             weight_tol: float = 1e-3) -> ConstrainedFit:
    """:func:`refine_cell_joint`, wrapped in an outer IRLS loop that lets the
    DATA decide which reflections to downweight, instead of a caller having
    to hand-pick a suspicious ``hkl``.

    Every reflection starts at weight 1. After each fit, the per-reflection
    residual vector norm ``|R@B@hkl - g_obs|`` is scored against a Tukey
    biweight (:func:`tukey_biweight`) and reweighted; the loop repeats until
    no weight moves by more than ``weight_tol`` or ``max_iter`` is reached.
    This is standard IRLS / M-estimation.

    **Calibration caveat, checked against synthetic clean (no-outlier) data:**
    ``tukey_biweight``'s ``c=4.685`` reference constant is calibrated for a
    roughly-symmetric, roughly-Gaussian residual. This function's residual is
    a vector NORM -- always non-negative, chi-distributed rather than
    Gaussian even when every reflection's underlying (dx, dy, dz) error truly
    is iid Gaussian. That right skew means a clean domain with NO real
    outliers can still see a substantial minority of reflections (order
    30-35% in a 48-reflection synthetic test with zero planted contamination)
    land at a "low" weight from ordinary distributional shape alone, not
    genuine badness. :func:`refine_cell_radial_robust`'s residual is a signed
    SCALAR (``|g_pred|-|g_obs|``, close to Gaussian for small errors), and
    does not show anywhere near as much of this effect on the same clean
    data -- so do not read "joint flags more reflections than radial" as
    meaning "joint found more real problems"; some of that gap is this
    scoring asymmetry, not the data. Treat a single fit's absolute weight
    values with caution; a reflection downweighted by BOTH the joint and
    radial fits independently is a much stronger signal than either alone.

    **This is not a fix for a systematically skewed reflection population.**
    IRLS protects against individual heavy-tailed outliers -- a contaminated
    or misindexed spot -- by construction; it has no mechanism to catch a
    *population-level* bias (e.g. a domain whose claimed reflections are
    skewed |h|>|k| far more often than |k|>|h|), because that kind of bias is
    shared coherently across most of the domain rather than concentrated in a
    few large residuals. A worked example on real data, where robustifying
    converged two independent estimators TOWARD each other but made both MORE
    significant (not less) on a domain independently known to carry exactly
    that population-level artifact, is in this project's own La3Ni2O7 DAC
    notes (2026-09-14) -- read the result as "some reflections were
    genuinely bad," not as "the robustified answer is now trustworthy."

    The returned fit's ``.weights`` holds the FINAL converged per-domain
    weight arrays (1.0 = fully trusted, 0.0 = fully excluded) and
    ``.robust_n_iter`` the number of IRLS iterations taken. Bootstrap
    uncertainty, if requested, is computed AT those final fixed weights (the
    weights are not re-derived inside each bootstrap resample).
    """
    doms = list(domains)
    if not doms:
        raise ValueError("no domains given")
    w = [np.ones(len(d.hkl)) for d in doms]
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        fit = refine_cell_joint(doms, system=system, cell0=cell0, two_pi=two_pi,
                                maxfev=maxfev, weights=w)
        B = _B_of(fit.cell, two_pi)
        w_new = []
        for i, d in enumerate(doms):
            R = _rotvec_to_R(np.asarray(fit.rotvecs[i]))
            resid = np.linalg.norm((d.hkl @ (R @ B).T) - d.g, axis=1)
            wi, _ = tukey_biweight(resid, c=tukey_c)
            w_new.append(wi)
        delta = max(float(np.max(np.abs(w_new[j] - w[j]))) for j in range(len(doms)))
        w = w_new
        if delta < weight_tol:
            break
    fit = refine_cell_joint(doms, system=system, cell0=cell0, two_pi=two_pi,
                            n_bootstrap=n_bootstrap, rng=rng, maxfev=maxfev, weights=w)
    fit.robust_n_iter = n_iter
    return fit


def refine_cell_radial_robust(domains: Sequence[DomainData], *, system: str,
                              cell0: Sequence[float],
                              two_pi: bool = False,
                              n_bootstrap: int = 0,
                              rng: Optional[np.random.Generator] = None,
                              maxfev: int = 20000,
                              tukey_c: float = 4.685,
                              max_iter: int = 10,
                              weight_tol: float = 1e-3) -> ConstrainedFit:
    """:func:`refine_cell_radial`, with the same IRLS outer loop as
    :func:`refine_cell_joint_robust` -- see its docstring for the full
    explanation and the caveat about population-level bias, which applies
    here identically. The per-reflection score is ``||B@hkl| - |g_obs||``
    (the same magnitude-only residual `refine_cell_radial` fits), so this
    function's outlier judgment is INDEPENDENT of the joint version's --
    comparing the two `.weights` is itself informative (a reflection flagged
    by only one of the two residual definitions is a different kind of
    suspect than one flagged by both).
    """
    doms = list(domains)
    if not doms:
        raise ValueError("no domains given")
    w = [np.ones(len(d.hkl)) for d in doms]
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        fit = refine_cell_radial(doms, system=system, cell0=cell0, two_pi=two_pi,
                                 maxfev=maxfev, weights=w)
        B = _B_of(fit.cell, two_pi)
        w_new = []
        for d in doms:
            g_pred_mag = np.linalg.norm(d.hkl @ B.T, axis=1)
            g_obs_mag = np.linalg.norm(d.g, axis=1)
            resid = np.abs(g_pred_mag - g_obs_mag)
            wi, _ = tukey_biweight(resid, c=tukey_c)
            w_new.append(wi)
        delta = max(float(np.max(np.abs(w_new[j] - w[j]))) for j in range(len(doms)))
        w = w_new
        if delta < weight_tol:
            break
    fit = refine_cell_radial(doms, system=system, cell0=cell0, two_pi=two_pi,
                             n_bootstrap=n_bootstrap, rng=rng, maxfev=maxfev, weights=w)
    fit.robust_n_iter = n_iter
    return fit


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
