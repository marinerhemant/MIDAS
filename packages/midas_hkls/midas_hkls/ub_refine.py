"""Free-triclinic UB refinement from indexed g-vectors, **with uncertainties**.

Given reflections already assigned to integer ``hkl``, the orientation and the
full triclinic cell follow in closed form. Writing ``g = UB·h``, the three rows
of ``UB`` decouple and each is an ordinary linear least squares against the same
normal matrix ``H = Σ hₙhₙᵀ``. No starting guess, no iteration, no local minima,
and no symmetry assumed — six cell parameters and three orientation angles all
free.

The estimator is Paciorek, Meyer, Chambrier & Beitollahi, *Acta Cryst.* **A55**
543 (1999)::

    UB = R H⁻¹ ,   R = Σ rₙ hₙᵀ ,   H = Σ hₙ hₙᵀ

which is what ImageD11's ``indexing.refine`` computes. This implementation adds
the three things a pressure or temperature series needs and that estimator alone
does not give you:

**1. A covariance, and therefore a σ on every cell parameter.** A refined cell
without one cannot answer "did the cell change?", which is the only question a
series asks. ``cell_sigma`` comes from the fit covariance propagated through the
metric-tensor decomposition.

**2. A per-parameter determinability verdict.** With few reflections in a narrow
wedge some parameters are simply not measured — a c-axis is barely constrained
when every accessible reflection has a small l. :attr:`UBFit.determined` says
which, by comparing each σ against the parameter itself, so an undetermined `c`
is reported as undetermined rather than as a number.

**3. Weights.** The unweighted form treats a 10⁶-count reflection and a
200-count one identically. Pass ``weights`` (or ``sigma_g``) and they do not.

Conventions
-----------
``UB`` maps integer Miller indices to scattering vectors in the **sample**
frame, ``g = UB·h``, in whatever reciprocal convention the g-vectors are in
(1/d or 2π/d — the cell comes back in the matching one; see
:attr:`UBFit.q_convention_note`). ``UBI = UB⁻¹`` gives ``h = UBI·g``, and the
**direct** metric tensor is ``G = UBI·UBIᵀ``, from which ``a = √G₀₀`` and
``cos α = G₁₂/(bc)`` — the same decomposition ImageD11 uses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["UBFit", "refine_ub_from_gvectors", "ub_to_cell", "ub_to_u_b",
           "cell_from_metric", "drlv2"]

_CELL_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")


def drlv2(UBI: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Squared distance from ``UBI·g`` to the nearest integer hkl, per reflection.

    ImageD11's scoring metric. A reflection is "indexed at tolerance t" when
    ``sqrt(drlv2) < t``.
    """
    h = np.asarray(g, float) @ np.asarray(UBI, float).T
    return ((h - np.round(h)) ** 2).sum(axis=1)


def cell_from_metric(G: np.ndarray) -> Tuple[float, ...]:
    """Direct metric tensor → ``(a, b, c, alpha, beta, gamma)``, angles in degrees."""
    G = np.asarray(G, float)
    a, b, c = (math.sqrt(max(G[i, i], 0.0)) for i in range(3))
    if min(a, b, c) <= 0:
        raise ValueError("degenerate metric tensor — the cell is not defined")

    def _ang(x):
        return math.degrees(math.acos(max(-1.0, min(1.0, x))))
    return (a, b, c, _ang(G[1, 2] / (b * c)), _ang(G[0, 2] / (a * c)),
            _ang(G[0, 1] / (a * b)))


def ub_to_cell(UB: np.ndarray) -> Tuple[float, ...]:
    """``UB`` → direct cell parameters."""
    UBI = np.linalg.inv(np.asarray(UB, float))
    return cell_from_metric(UBI @ UBI.T)


def ub_to_u_b(UB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``UB`` into orientation ``U`` (in SO(3)) and Busing–Levy ``B``.

    ``B`` is upper triangular by Cholesky of the reciprocal metric, matching
    ImageD11's ``ubitoB``; ``U = UB·B⁻¹`` is then a proper rotation.
    """
    UB = np.asarray(UB, float)
    # We need B with B^T B = G* (the RECIPROCAL metric), because
    # |g|^2 = h^T UB^T UB h = h^T G* h. Cholesky of G* gives it directly;
    # taking Cholesky of the DIRECT metric instead yields a B whose transpose
    # product is inv(L) inv(L)^T, not G*, and U = UB B^-1 is then not a rotation.
    G_star = UB.T @ UB
    B = np.linalg.cholesky(G_star).T
    U = UB @ np.linalg.inv(B)
    # project onto SO(3) — numerically it is a rotation already
    u_, _, vt = np.linalg.svd(U)
    U = u_ @ vt
    if np.linalg.det(U) < 0:
        u_[:, -1] *= -1
        U = u_ @ vt
    return U, B


@dataclass
class UBFit:
    """A free-triclinic UB, its cell, and how well each parameter is determined."""
    UB: np.ndarray
    UBI: np.ndarray
    cell: Tuple[float, ...]
    cell_sigma: Tuple[float, ...]
    U: np.ndarray
    B: np.ndarray
    n_reflections: int
    rms_drlv: float
    sigma_g: float
    cov_ub: np.ndarray                       # (9, 9) for UB flattened row-major
    determined: Dict[str, bool] = field(default_factory=dict)
    q_convention_note: str = (
        "cell is in the convention of the supplied g-vectors: pass g = 1/d for "
        "Angstrom cell parameters, or g = 2*pi/d and divide the cell by 2*pi")

    @property
    def undetermined(self) -> Tuple[str, ...]:
        return tuple(n for n in _CELL_NAMES if not self.determined.get(n, True))

    def __str__(self) -> str:
        body = ", ".join(
            f"{n}={v:.4f}±{s:.4f}" + ("" if self.determined.get(n, True) else " (UNDET)")
            for n, v, s in zip(_CELL_NAMES, self.cell, self.cell_sigma))
        return (f"{body} | n={self.n_reflections}, rms drlv "
                f"{self.rms_drlv:.4f}")


def refine_ub_from_gvectors(hkl, g_obs, *,
                            weights: Optional[Sequence[float]] = None,
                            sigma_g: Optional[float] = None,
                            determined_ratio: float = 0.25,
                            ) -> UBFit:
    """Refine a **free triclinic** UB from assigned reflections, with σ per parameter.

    Parameters
    ----------
    hkl : (N, 3)
        Integer Miller indices. These are taken as given — this routine refines,
        it does not assign.
    g_obs : (N, 3)
        Observed scattering vectors in the sample frame.
    weights : (N,), optional
        Per-reflection weights (e.g. intensity, or 1/σ²). Uniform if omitted.
    sigma_g : float, optional
        Per-component uncertainty on ``g``. If omitted it is **estimated from
        the residuals**, which is honest only when the model is right — supply a
        measured value when you have one.
    determined_ratio : float
        A cell parameter counts as determined when ``σ < determined_ratio × |value|``
        for a length, or ``σ < determined_ratio × 90°`` for an angle.

    Notes
    -----
    Needs at least **3 non-coplanar** reflections; with exactly 3 the fit is
    exact and the residual carries no information, so σ cannot be estimated from
    it — pass ``sigma_g`` in that case or the σ come back as NaN.
    """
    h = np.asarray(hkl, float)
    g = np.asarray(g_obs, float)
    if h.shape != g.shape or h.ndim != 2 or h.shape[1] != 3:
        raise ValueError(f"hkl and g_obs must both be (N, 3); got {h.shape}, {g.shape}")
    n = len(h)
    if n < 3:
        raise ValueError(f"need at least 3 reflections, got {n}")

    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if w.shape != (n,):
        raise ValueError("weights must be (N,)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    # Paciorek: UB = R H^-1, weighted. The three rows of UB decouple and share H.
    # NOTE (2026-09-03): the sigma below assumes iid ISOTROPIC Gaussian error on
    # every g-component. Real HEDM/rotation data violate this: error along q
    # (which sets cell LENGTHS) is much smaller than error across it (set by the
    # omega step). Measured on 2604 nickelate: rms 3.01e-3 radial vs 6.41e-3
    # tangential, a factor 2.1, and residual kurtosis +15 with 5 of 76
    # reflections carrying 64 % of the SSR. The isotropic fit charges that
    # tangential scatter to a, b, c and INFLATES their sigma ~2x -- bootstrap and
    # jackknife both gave half the analytic sigma there, turning a "1.8 sigma,
    # not significant" a/b split into 3.6 sigma. The covariance formula itself is
    # calibrated (500-draw null: sd(z) = 0.991); it is the error MODEL that is
    # wrong. Pass a measured radial sigma_g, or bootstrap, when the difference
    # between two cell lengths is the quantity of interest.
    H = (h * w[:, None]).T @ h
    if np.linalg.matrix_rank(H, tol=1e-9) < 3:
        raise ValueError(
            "the hkl are coplanar (or collinear): H is rank-deficient, so the "
            "cell is not determined in every direction. Index a reflection out "
            "of that plane before refining.")
    Hinv = np.linalg.inv(H)
    R = (g * w[:, None]).T @ h
    UB = R @ Hinv

    resid = g - h @ UB.T
    dof = max(3 * n - 9, 1)
    if sigma_g is None:
        sigma_g = float(np.sqrt((w[:, None] * resid ** 2).sum() / dof)) if 3 * n > 9 \
            else float("nan")

    # cov of each UB row is sigma^2 * Hinv; rows are independent
    cov = np.zeros((9, 9))
    for r in range(3):
        cov[3 * r:3 * r + 3, 3 * r:3 * r + 3] = (sigma_g ** 2) * Hinv

    UBI = np.linalg.inv(UB)
    cell = ub_to_cell(UB)

    # propagate to cell parameters by a numerical Jacobian (9 -> 6)
    if np.isfinite(sigma_g):
        J = np.zeros((6, 9))
        step = 1e-7 * max(1.0, np.abs(UB).max())
        flat = UB.reshape(-1)
        for k in range(9):
            up = flat.copy(); up[k] += step
            dn = flat.copy(); dn[k] -= step
            J[:, k] = (np.array(ub_to_cell(up.reshape(3, 3)))
                       - np.array(ub_to_cell(dn.reshape(3, 3)))) / (2 * step)
        cell_cov = J @ cov @ J.T
        cell_sigma = tuple(float(np.sqrt(max(cell_cov[i, i], 0.0))) for i in range(6))
    else:
        cell_sigma = tuple([float("nan")] * 6)

    determined = {}
    for i, nm in enumerate(_CELL_NAMES):
        ref = abs(cell[i]) if i < 3 else 90.0
        s = cell_sigma[i]
        determined[nm] = bool(np.isfinite(s) and s < determined_ratio * ref)

    U, B = ub_to_u_b(UB)
    return UBFit(UB=UB, UBI=UBI, cell=cell, cell_sigma=cell_sigma, U=U, B=B,
                 n_reflections=n, rms_drlv=float(np.sqrt(drlv2(UBI, g).mean())),
                 sigma_g=float(sigma_g), cov_ub=cov, determined=determined)
