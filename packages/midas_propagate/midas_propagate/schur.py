"""Per-grain Schur-marginal covariance — paper-1's core math.

Production HEDM reports per-grain posterior sigma assuming the detector
calibration is fixed at its MAP. That's wrong when the calibration was
estimated from finite data (e.g. a CeO2 powder); the calibration's own
posterior Sigma_cc should propagate into each grain's reported sigma.

Paper-1's contribution is to do that propagation correctly via the Schur
complement of the calibration block in the joint Hessian. Notation::

    H_joint = [[H_gg, H_gc], [H_cg, H_cc_total]]

with one big calibration block ``c`` (shared across all grains) and per-grain
blocks ``g_i``. The joint posterior covariance is ``H_joint^{-1}``, whose
``(g_i, g_i)`` diagonal block is the Schur complement::

    Sigma_g_marg = (H_gg - H_gc @ H_cc_total^{-1} @ H_cg)^{-1}

When the calibration is constrained predominantly by the calibrant data
(the sample's contribution to ``H_cc`` is small), we have
``H_cc_total ~= Sigma_cc^{-1}`` and the formula reduces to::

    Sigma_g_marg = (H_gg - H_gc @ Sigma_cc @ H_cg)^{-1}

This module exposes that operation, batched over grains. Companion routine
:func:`per_grain_diagonal_sigma` extracts marginal 1-sigma per refined
parameter from a per-grain covariance stack.

We treat the per-grain blocks as independent (no cross-grain coupling in
the data Hessian), which holds for FF/PF-HEDM where each grain's spots
contribute additively to the joint NLL with no shared per-spot weights.
The calibration block IS shared and is what the Schur complement removes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PerGrainMarginalResult:
    """Output of :func:`per_grain_schur_marginal`.

    Attributes
    ----------
    sigma_gg_frozen : (G, n_g, n_g)
        Per-grain covariance assuming calibration is held at its MAP
        (== ``inv(H_gg)``). Matches what production tools report.
    sigma_gg_calmarg : (G, n_g, n_g)
        Per-grain covariance with calibration marginalised out via Schur.
        This is paper-1's headline quantity.
    info_inflation_eigvals : (G, n_g) or None
        Eigenvalues of ``Sigma_cal_marg @ inv(Sigma_frozen)`` per grain.
        Diagnostic: values > 1 quantify how much calibration uncertainty
        inflates that eigendirection's posterior variance. ``None`` if
        ``with_diagnostics=False`` at the call site.
    cc_rank_used : int
        Effective rank of ``Sigma_cc`` (n_c minus the number of zero
        eigenvalues beneath ``cc_eig_tol``). Reported for transparency
        when Sigma_cc is rank-deficient — common in calibration because
        of structurally unidentifiable parameter combinations (e.g.
        distortion phases whose amplitude companion is ~0).
    """
    sigma_gg_frozen: torch.Tensor
    sigma_gg_calmarg: torch.Tensor
    info_inflation_eigvals: Optional[torch.Tensor]
    cc_rank_used: int


def per_grain_schur_marginal(
    H_gg: torch.Tensor,
    H_gc: torch.Tensor,
    sigma_cc: torch.Tensor,
    *,
    ridge_g: float = 1e-9,
    cc_eig_tol: float = 1e-12,
    with_diagnostics: bool = False,
) -> PerGrainMarginalResult:
    """Per-grain marginal covariance with calibration uncertainty Schur-removed.

    Parameters
    ----------
    H_gg : (G, n_g, n_g)
        Per-grain Hessian blocks of the negative log-posterior at MAP, on
        the grain parameters. Expected positive-semidefinite (the function
        adds ``ridge_g`` to the diagonal for numerical safety).
    H_gc : (G, n_g, n_c)
        Per-grain cross blocks: ``H_gc[k, i, j] = d^2 NLL / d g_i d c_j``
        evaluated at MAP for grain ``k``.
    sigma_cc : (n_c, n_c)
        Calibration covariance from the calibrant-only Laplace fit
        (e.g. ``midas_calibrate_v2.pipelines.bayesian_multi.sigma_cc_at_multi_map``).
        Must be symmetric PSD; may be rank-deficient on unidentifiable
        directions — handled via eigenvalue clipping with ``cc_eig_tol``.
    ridge_g : float
        Diagonal ridge added to ``H_gg`` and to the post-Schur information
        matrix before inversion. Default 1e-9; sized for unit-scale params.
    cc_eig_tol : float
        Eigenvalue cutoff for ``Sigma_cc`` — eigenvalues with
        ``|lambda| < cc_eig_tol * lambda_max`` are zeroed (Moore-Penrose
        pseudoinverse style). Without this, rank-deficient Sigma_cc rows
        (from unidentifiable distortion phases) blow up Sigma_g_marg.
    with_diagnostics : bool
        If True, also compute ``info_inflation_eigvals``. Cheap on small
        n_g; useful for the paper's F3 panel showing per-grain calibration
        sensitivity.

    Returns
    -------
    PerGrainMarginalResult
    """
    if H_gg.ndim != 3 or H_gg.shape[-1] != H_gg.shape[-2]:
        raise ValueError(
            f"H_gg must be (G, n_g, n_g); got shape {tuple(H_gg.shape)}"
        )
    if H_gc.ndim != 3:
        raise ValueError(
            f"H_gc must be (G, n_g, n_c); got shape {tuple(H_gc.shape)}"
        )
    if H_gg.shape[0] != H_gc.shape[0] or H_gg.shape[1] != H_gc.shape[1]:
        raise ValueError(
            f"H_gg and H_gc disagree on (G, n_g): "
            f"H_gg {tuple(H_gg.shape)} vs H_gc {tuple(H_gc.shape)}"
        )
    if sigma_cc.ndim != 2 or sigma_cc.shape[0] != sigma_cc.shape[1]:
        raise ValueError(
            f"sigma_cc must be (n_c, n_c); got shape {tuple(sigma_cc.shape)}"
        )
    if sigma_cc.shape[0] != H_gc.shape[-1]:
        raise ValueError(
            f"sigma_cc dimension {sigma_cc.shape[0]} != n_c from H_gc "
            f"({H_gc.shape[-1]})"
        )

    dtype = H_gg.dtype
    device = H_gg.device
    n_g = H_gg.shape[-1]
    eye_g = torch.eye(n_g, dtype=dtype, device=device)

    # --- Sigma_cc pseudo-clean: zero tiny eigenvalues to handle structural
    # rank-deficiency (e.g. distortion phi with companion amplitude ~0).
    sigma_cc_sym = 0.5 * (sigma_cc + sigma_cc.T)
    eigvals_cc, eigvecs_cc = torch.linalg.eigh(sigma_cc_sym)
    lam_max = float(eigvals_cc.abs().max().item()) if eigvals_cc.numel() > 0 else 0.0
    cutoff = cc_eig_tol * lam_max
    keep = eigvals_cc > cutoff
    eigvals_clipped = torch.where(keep, eigvals_cc,
                                   torch.zeros_like(eigvals_cc))
    sigma_cc_clean = (eigvecs_cc * eigvals_clipped.unsqueeze(0)) @ eigvecs_cc.T
    cc_rank_used = int(keep.sum().item())

    # --- Frozen-cal per-grain covariance: just invert H_gg.
    H_gg_reg = H_gg + ridge_g * eye_g
    sigma_gg_frozen = torch.linalg.inv(H_gg_reg)

    # --- Schur correction: for each grain, subtract H_gc @ Sigma_cc @ H_gc^T
    # from H_gg, then invert. Batched in einsum so we don't unroll Python.
    #   correction[k] = H_gc[k] @ sigma_cc @ H_gc[k]^T
    H_gc_sigma = torch.einsum("gni,ij->gnj", H_gc, sigma_cc_clean)
    correction = torch.einsum("gni,gmi->gnm", H_gc_sigma, H_gc)
    correction = 0.5 * (correction + correction.transpose(-1, -2))
    H_marg = H_gg - correction
    H_marg_reg = H_marg + ridge_g * eye_g
    sigma_gg_calmarg = torch.linalg.inv(H_marg_reg)

    info_inflation_eigvals: Optional[torch.Tensor] = None
    if with_diagnostics:
        # Sigma_marg = inflation @ Sigma_frozen ⇒ inflation = Sigma_marg @ H_gg_reg.
        # Its eigenvalues are >= 1; values >> 1 flag eigendirections where
        # calibration uncertainty dominates the per-grain posterior.
        inflation = torch.einsum("gnm,gml->gnl", sigma_gg_calmarg, H_gg_reg)
        info_inflation_eigvals = torch.linalg.eigvals(inflation).real

    return PerGrainMarginalResult(
        sigma_gg_frozen=sigma_gg_frozen,
        sigma_gg_calmarg=sigma_gg_calmarg,
        info_inflation_eigvals=info_inflation_eigvals,
        cc_rank_used=cc_rank_used,
    )


def profile_unidentifiable(
    sigma_cc: torch.Tensor,
    *,
    var_cap: Optional[float] = None,
    relative_cap: Optional[float] = 1e6,
) -> torch.Tensor:
    """Cap eigenvalues of ``sigma_cc`` to profile out unidentifiable directions.

    The Schur correction assumes the calibration prior is *proper* on every
    direction it propagates through. When ``sigma_cc`` carries directions
    with effectively-infinite variance (e.g. distortion phases whose
    amplitude companion is ~0), those directions should be treated as
    *fixed at MAP* rather than marginalised — otherwise the per-grain
    marginal blows up on couplings the data cannot resolve anyway.

    This helper eigendecomposes ``sigma_cc`` and replaces eigenvalues above
    a cap with zero, projecting out the unidentifiable subspace. Either
    pass an absolute ``var_cap`` (variance ceiling, in units of the parameter
    squared) or a ``relative_cap`` multiplied by the median eigenvalue.

    Parameters
    ----------
    var_cap : float, optional
        Absolute variance ceiling. Eigenvalues > ``var_cap`` are zeroed.
        Takes precedence over ``relative_cap`` when both are given.
    relative_cap : float, default 1e6
        Multiplier on the median positive eigenvalue. Eigenvalues
        > ``relative_cap * median`` are zeroed. The 1e6 default is loose
        enough that no well-conditioned ``sigma_cc`` is touched, but
        catches the production case where Σ_cc has a few directions with
        variances 20+ orders of magnitude larger than the median.
    """
    sym = 0.5 * (sigma_cc + sigma_cc.T)
    eigvals, eigvecs = torch.linalg.eigh(sym)
    if var_cap is None:
        pos = eigvals[eigvals > 0]
        if pos.numel() == 0:
            return torch.zeros_like(sym)
        median = float(pos.median().item())
        var_cap = relative_cap * median
    keep = eigvals < var_cap
    capped = torch.where(keep, eigvals, torch.zeros_like(eigvals))
    return (eigvecs * capped.unsqueeze(0)) @ eigvecs.T


def per_grain_diagonal_sigma(sigma_gg: torch.Tensor) -> torch.Tensor:
    """Per-grain marginal 1-sigma per parameter.

    Returns ``sqrt(diag(sigma_gg))`` per grain, shape ``(G, n_g)``.
    Clamps negative diagonal entries (numerical noise) to zero before
    sqrt — same convention as ``midas_peakfit.laplace_at_map``.
    """
    diag = torch.diagonal(sigma_gg, dim1=-2, dim2=-1)
    return torch.sqrt(diag.clamp(min=0.0))


def sigma_inflation_ratio(
    sigma_gg_frozen: torch.Tensor,
    sigma_gg_calmarg: torch.Tensor,
) -> torch.Tensor:
    """Per-grain, per-parameter ratio of marginal to frozen sigma.

    ``ratio[g, i] = sigma_calmarg[g, i] / sigma_frozen[g, i]``. Always >= 1
    when Sigma_cc is PSD. The headline F3 panel for paper-1 is a histogram
    of this ratio across grains for the strain components.
    """
    s_f = per_grain_diagonal_sigma(sigma_gg_frozen)
    s_m = per_grain_diagonal_sigma(sigma_gg_calmarg)
    return s_m / s_f.clamp(min=1e-30)


__all__ = [
    "PerGrainMarginalResult",
    "per_grain_schur_marginal",
    "per_grain_diagonal_sigma",
    "profile_unidentifiable",
    "sigma_inflation_ratio",
]
