"""Rev-15 Bayesian model comparison (WAIC, LOO-like).

Rev-12/13 gave us posterior samples for single-phase, joint SAXS+PDF, and
three-way SAXS+SANS+PDF refinements.  But posterior samples alone don't
answer *which model is better*.  This module adds WAIC (Watanabe-Akaike
Information Criterion) and a lightweight LOO estimator that operate on
the ``*BayesianResult.G_samples`` / ``I_saxs_samples`` / ``I_sans_samples``
posterior-predictive stacks.

Standard formulation (Gelman & Vehtari):

    lppd  = Σ_i log( 1/S · Σ_s p(y_i | θ_s) )     (log point-wise pred density)
    p_waic = Σ_i Var_s( log p(y_i | θ_s) )
    WAIC  = -2 · (lppd - p_waic)

Lower WAIC is better; the standard error on WAIC comes from the pointwise
contributions.  Two models are compared via ΔWAIC ± SE(ΔWAIC).

For lightweight LOO we compute leave-one-out log-predictive density using
importance sampling (naive — no Pareto smoothing).  This is exact in the
limit S → ∞ and matches PSIS-LOO closely for well-mixed chains; we do
not import ArviZ so the dependency footprint stays minimal.

The module accepts either:
  * pre-computed pointwise log-likelihood ``log_lik`` of shape (S, N), or
  * ``(obs, sigma, mu_samples)`` for a Gaussian likelihood — the common case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import math
import torch


__all__ = [
    "gaussian_pointwise_loglik",
    "waic",
    "loo",
    "compare_models",
    "InformationCriterionResult",
    "ModelComparisonResult",
]


_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class InformationCriterionResult:
    value: float                    # WAIC or LOO
    se: float                       # standard error on the value
    p_eff: float                    # effective number of parameters
    lppd: float                     # log point-wise predictive density
    pointwise: torch.Tensor         # (N,) per-datum contribution
    name: str = "WAIC"


@dataclass
class ModelComparisonResult:
    winner: str
    delta: float                    # WAIC_worse - WAIC_better  (>= 0)
    se_delta: float                 # SE of the difference
    z: float                        # delta / se_delta
    individual: Dict[str, InformationCriterionResult] = field(default_factory=dict)


def gaussian_pointwise_loglik(
    obs: torch.Tensor,
    sigma: torch.Tensor,
    mu_samples: torch.Tensor,
) -> torch.Tensor:
    """Per-sample per-datum Gaussian log-likelihood.

    Parameters
    ----------
    obs : (N,) tensor       observed values
    sigma : (N,) tensor      per-point standard deviation
    mu_samples : (S, N) tensor   posterior samples of the mean

    Returns
    -------
    log_lik : (S, N) tensor
    """
    obs = torch.as_tensor(obs, dtype=torch.float64)
    sigma = torch.as_tensor(sigma, dtype=torch.float64).clamp(min=1e-12)
    mu_samples = torch.as_tensor(mu_samples, dtype=torch.float64)
    if mu_samples.ndim != 2 or mu_samples.shape[1] != obs.numel():
        raise ValueError(
            f"mu_samples must be (S, {obs.numel()}), got {tuple(mu_samples.shape)}"
        )
    residual = obs.unsqueeze(0) - mu_samples                # (S, N)
    z = residual / sigma.unsqueeze(0)
    return -0.5 * (z * z) - torch.log(sigma).unsqueeze(0) - 0.5 * _LOG_2PI


def _lppd_and_pwaic(log_lik: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-point lppd and per-point p_waic vectors."""
    S = log_lik.shape[0]
    # log point-wise predictive density: log(mean(exp(log_lik)))
    #  = logsumexp(log_lik) - log(S)
    lppd_i = torch.logsumexp(log_lik, dim=0) - math.log(S)   # (N,)
    p_waic_i = log_lik.var(dim=0, unbiased=True)             # (N,)
    return lppd_i, p_waic_i


def waic(
    obs: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    mu_samples: Optional[torch.Tensor] = None,
    *,
    log_lik: Optional[torch.Tensor] = None,
    name: str = "WAIC",
) -> InformationCriterionResult:
    """Watanabe-Akaike Information Criterion.

    Pass either ``log_lik`` directly (shape ``(S, N)``) or the Gaussian
    convenience triple ``(obs, sigma, mu_samples)``.
    """
    if log_lik is None:
        if obs is None or sigma is None or mu_samples is None:
            raise ValueError("waic needs either log_lik or (obs, sigma, mu_samples)")
        log_lik = gaussian_pointwise_loglik(obs, sigma, mu_samples)

    lppd_i, p_waic_i = _lppd_and_pwaic(log_lik)
    pointwise = -2.0 * (lppd_i - p_waic_i)                   # (N,)
    value = float(pointwise.sum())
    # SE of WAIC = sqrt(N * var(pointwise))
    N = pointwise.numel()
    se = float(math.sqrt(N * float(pointwise.var(unbiased=True))))
    return InformationCriterionResult(
        value=value,
        se=se,
        p_eff=float(p_waic_i.sum()),
        lppd=float(lppd_i.sum()),
        pointwise=pointwise,
        name=name,
    )


def loo(
    obs: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    mu_samples: Optional[torch.Tensor] = None,
    *,
    log_lik: Optional[torch.Tensor] = None,
    name: str = "LOO",
) -> InformationCriterionResult:
    """Leave-one-out log predictive density via importance sampling.

    For each datum ``i``, the LOO predictive is

        p(y_i | y_{-i}) = 1 / mean_s(1 / p(y_i | θ_s))

    which is the harmonic mean of the per-sample likelihoods.  Naive (no
    Pareto smoothing) — reliable when S is large and no single sample
    dominates the importance weight.  See ``diagnose_loo`` for a warning
    when that assumption is violated.
    """
    if log_lik is None:
        if obs is None or sigma is None or mu_samples is None:
            raise ValueError("loo needs either log_lik or (obs, sigma, mu_samples)")
        log_lik = gaussian_pointwise_loglik(obs, sigma, mu_samples)

    S = log_lik.shape[0]
    # log p(y_i | y_{-i})  ≈  -logsumexp(-log_lik_s) + log S
    loo_i = -torch.logsumexp(-log_lik, dim=0) + math.log(S)  # (N,)
    lppd_i, _ = _lppd_and_pwaic(log_lik)
    p_loo_i = lppd_i - loo_i                                  # per-point p_eff
    pointwise = -2.0 * loo_i
    value = float(pointwise.sum())
    N = pointwise.numel()
    se = float(math.sqrt(N * float(pointwise.var(unbiased=True))))
    return InformationCriterionResult(
        value=value,
        se=se,
        p_eff=float(p_loo_i.sum()),
        lppd=float(loo_i.sum()),
        pointwise=pointwise,
        name=name,
    )


def compare_models(
    models: Dict[str, InformationCriterionResult],
) -> ModelComparisonResult:
    """Pairwise-anchored comparison of two or more models.

    Anchors on the best model (lowest IC value) and reports every other
    model's ΔIC vs it.  SE of the difference uses the pointwise
    contributions — sqrt(N · var(dpw_i - dpw_j)).  Returned struct's
    ``delta`` / ``se_delta`` / ``z`` are for the *second best* vs the
    winner (the tightest comparison).
    """
    if len(models) < 2:
        raise ValueError("compare_models needs at least two models")
    names = list(models.keys())
    ranked = sorted(names, key=lambda n: models[n].value)
    winner = ranked[0]
    runner = ranked[1]
    pw_win = models[winner].pointwise
    pw_run = models[runner].pointwise
    if pw_win.shape != pw_run.shape:
        raise ValueError(
            f"pointwise shapes disagree: {tuple(pw_win.shape)} vs {tuple(pw_run.shape)}"
        )
    diff_pw = pw_run - pw_win
    delta = float(diff_pw.sum())
    N = diff_pw.numel()
    se_delta = float(math.sqrt(N * float(diff_pw.var(unbiased=True))))
    z = delta / max(se_delta, 1e-12)
    return ModelComparisonResult(
        winner=winner,
        delta=delta,
        se_delta=se_delta,
        z=z,
        individual=models,
    )
