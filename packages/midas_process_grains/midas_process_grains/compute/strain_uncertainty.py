"""Per-grain uncertainty on the two strain tensors MIDAS reports.

``Grains.csv`` carries both conventions and they are computed by different
routes, so their errors come from different places. Propagating one from the
other's covariance would be wrong.

**eFab — Fable-Beaudoin.** A pure function of the refined lattice parameters
(:func:`midas_stress.tensor.lattice_params_to_strain`), so its uncertainty is
the 6x6 lattice block of the per-grain Hessian covariance pushed through the
Jacobian of that function::

    Sigma_eps = J Sigma_latc J^T ,   J = d voigt(eps) / d latc

The Jacobian is taken by autograd on the package's own strain function, so the
two can never drift apart.

**eKen — Kenesei.** A linear least-squares solve ``G eps = b`` over the grain's
own spots (:func:`~midas_process_grains.compute.strain.solve_strain_lstsq`), so
its covariance comes from that solve's normal equations::

    Sigma_eps = s^2 (G^T G)^-1 ,   s^2 = RSS / (n - 6)

with a sandwich form when Tikhonov regularisation is on, because the
regularised estimator is biased and ``(G^T G + aI)^-1`` alone understates it.

Expect sigma(eps_xx) >> sigma(eps_yy), sigma(eps_zz) in FF geometry. That is
not a defect: the beam runs along x, so ``g_x`` is ~0.07 against ~0.7 for the
other two and the xx component is barely constrained — the same weakness the
``regularization`` argument of ``solve_strain_lstsq`` exists to tame. A UQ that
did NOT show it would be the suspicious one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: Voigt order used throughout MIDAS strain code.
VOIGT_ORDER = ("xx", "yy", "zz", "xy", "xz", "yz")
_VOIGT_IJ = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


@dataclass
class StrainSigma:
    """Strain uncertainty for one grain, in Voigt order :data:`VOIGT_ORDER`."""

    sigma_voigt: np.ndarray       # (6,)
    cov_voigt: np.ndarray         # (6, 6)
    sigma_hydrostatic: float      # sigma of tr(eps)/3, with covariance
    sigma_von_mises: Optional[float] = None
    n_spots: Optional[int] = None
    method: str = ""

    def as_dict(self, prefix: str) -> dict:
        out = {f"sigma_{prefix}_{k}": float(v)
               for k, v in zip(VOIGT_ORDER, self.sigma_voigt)}
        out[f"sigma_{prefix}_hydro"] = float(self.sigma_hydrostatic)
        return out


def _hydro_sigma(cov: np.ndarray) -> float:
    """sigma of tr(eps)/3 = (e_xx + e_yy + e_zz)/3, using the covariance.

    The three normal components are strongly correlated in FF geometry, so the
    independent-sum answer is wrong in either direction depending on sign.
    """
    J = np.zeros(6); J[:3] = 1.0 / 3.0
    return float(np.sqrt(max(J @ cov @ J, 0.0)))


def fable_strain_covariance(
    latc: Sequence[float],
    latc_reference: Sequence[float],
    cov_latc: np.ndarray,
) -> StrainSigma:
    """eFab uncertainty, propagated from the lattice-parameter covariance.

    ``cov_latc`` is the (6, 6) lattice block of the per-grain Hessian
    covariance — e.g. ``result.cov[i][3:9, 3:9]`` from
    :func:`~midas_process_grains.compute.position_uncertainty.
    compute_per_grain_parameter_sigma` called with ``return_cov=True``.
    """
    import torch
    from midas_stress.tensor import lattice_params_to_strain

    lat = torch.as_tensor(np.asarray(latc, dtype=np.float64))
    ref = torch.as_tensor(np.asarray(latc_reference, dtype=np.float64))
    C = np.asarray(cov_latc, dtype=np.float64)
    if C.shape != (6, 6):
        raise ValueError(f"cov_latc must be (6, 6); got {C.shape}")

    def _voigt(x: "torch.Tensor") -> "torch.Tensor":
        e = lattice_params_to_strain(x, ref).reshape(3, 3)
        return torch.stack([e[i, j] for i, j in _VOIGT_IJ])

    J = torch.autograd.functional.jacobian(_voigt, lat).detach().cpu().numpy()
    cov = J @ C @ J.T
    return StrainSigma(
        sigma_voigt=np.sqrt(np.maximum(np.diag(cov), 0.0)),
        cov_voigt=cov, sigma_hydrostatic=_hydro_sigma(cov),
        method="fable_beaudoin (propagated from lattice covariance)")


def kenesei_strain_covariance(
    g_obs: np.ndarray,
    ds_obs: np.ndarray,
    ds_0: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    regularization: float = 0.0,
) -> StrainSigma:
    """eKen uncertainty from the least-squares solve's own normal equations.

    ``s^2`` is estimated from the fit residual with ``n - 6`` degrees of
    freedom, so this is data-driven: a grain whose spots disagree gets a wide
    posterior without anyone having to supply a noise level.
    """
    import torch
    from .strain import build_design_matrix

    g = torch.as_tensor(np.asarray(g_obs, dtype=np.float64))
    do = np.asarray(ds_obs, dtype=np.float64)
    d0 = np.asarray(ds_0, dtype=np.float64)
    n = g.shape[0]
    if n < 7:
        raise ValueError(
            f"need >= 7 spots to estimate a variance on 6 parameters; got {n}")

    G = build_design_matrix(g).detach().cpu().numpy()
    b = (do - d0) / np.clip(d0, 1e-30, None)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (n,):
            raise ValueError(f"weights must be ({n},); got {w.shape}")
        G = G * w[:, None]
        b = b * w

    GtG = G.T @ G
    A = GtG + regularization * np.eye(6)
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Ainv = np.linalg.pinv(A)
    eps = Ainv @ (G.T @ b)
    rss = float(((G @ eps - b) ** 2).sum())
    s2 = rss / max(n - 6, 1)

    if regularization > 0:
        # Regularised estimator is biased; its variance is the sandwich
        # (G'G + aI)^-1 G'G (G'G + aI)^-1, NOT (G'G + aI)^-1 alone.
        cov = s2 * (Ainv @ GtG @ Ainv)
    else:
        cov = s2 * Ainv

    return StrainSigma(
        sigma_voigt=np.sqrt(np.maximum(np.diag(cov), 0.0)),
        cov_voigt=cov, sigma_hydrostatic=_hydro_sigma(cov), n_spots=int(n),
        method=("kenesei lstsq (normal equations"
                + (", ridge sandwich)" if regularization > 0 else ")")))


__all__ = ["StrainSigma", "VOIGT_ORDER", "fable_strain_covariance",
           "kenesei_strain_covariance"]
