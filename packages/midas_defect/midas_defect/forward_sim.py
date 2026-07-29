"""P6 — Differentiable forward model for planar-defect rods.

Models a single planar-defect family in q-space using a closed-form
Hendricks–Teller-like intensity profile. Free parameters:

  * `direction` (3,) — defect-plane normal in q-space (sample frame, unit-norm)
  * `pivot` (3,)     — one point on the rod (typically the q-space anchor)
  * `A`              — overall amplitude
  * `alpha`          — fault probability (∈ (0, 1))
  * `sigma_perp`     — Gaussian half-width perpendicular to the rod (1/Å)
  * `d_layer`        — inter-layer spacing along the rod (Å). Determines the
                        period of the Bragg-comb along the rod direction.

Intensity model:

    q_par   = (q - pivot) · direction
    q_perp² = ||(q - pivot)||² - q_par²

    cross  = exp(-q_perp² / (2 σ_perp²))                        # Gaussian disk
    along  = (1 - α²) / (1 - 2 α cos(2π q_par d_layer) + α²)    # Hendricks-Teller

    I_pred = A · cross · along + baseline

  This is the Lorentzian-comb structure factor of a 1-D layered stack with
  binary random-phase mixing probability α (Hendricks & Teller, 1942 — the
  closed-form, geometric-series result for a single fault type). Key limits:

  * α=0: along reduces to 1 (constant — no positional correlation).
  * α=0.5: peaks have a Lorentzian profile of finite width Γ ~ (1-α)·d_layer.
  * α→1: peaks become δ-functions (high coherence).

  Peak positions: q_par · d_layer = integer (regardless of α). Peak-to-wing
  contrast grows monotonically with α.

  Note: this is NOT the textbook "α=0 sharp / α=0.5 diffuse rod / α=1 sharp"
  picture, which corresponds to a DIFFERENT fault model (random hexagonal
  faults on an FCC lattice). For that model use a different forward — this
  one is the cleanest analytic formula and a good first-pass for the Demk
  rod's α extraction.

All parameters are torch tensors; loss is fully differentiable. Use Adam
for refinement against either L² or Poisson NLL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch

from midas_transforms.device import resolve_device, resolve_dtype


__all__ = [
    "RodParams",
    "rod_intensity",
    "hendricks_teller",
    "fit_rod_params",
    "bootstrap_rod_params",
    "per_hkl_alpha_consistency",
]


@dataclass
class RodParams:
    """Container for a fitted planar-defect rod."""
    direction: np.ndarray         # (3,) unit vector
    pivot:     np.ndarray         # (3,)
    A:         float
    alpha:     float              # fault probability
    sigma_perp: float             # 1/Å
    d_layer:   float              # Å
    baseline:  float
    final_loss: float
    n_voxels:  int
    converged: bool

    def as_dict(self) -> dict:
        return dict(
            direction=self.direction.tolist(),
            pivot=self.pivot.tolist(),
            A=self.A,
            alpha=self.alpha,
            sigma_perp=self.sigma_perp,
            d_layer=self.d_layer,
            baseline=self.baseline,
            final_loss=self.final_loss,
            n_voxels=self.n_voxels,
            converged=self.converged,
        )


# ---------------------------------------------------------------------------
# Core kinematic model
# ---------------------------------------------------------------------------

def hendricks_teller(q_par: torch.Tensor, d_layer: torch.Tensor,
                     alpha: torch.Tensor) -> torch.Tensor:
    """Hendricks–Teller intensity along the rod for a single fault probability.

    .. math::
        I_{HT}(q_\\|) = \\frac{1 - \\alpha^2}
                             {1 - 2 \\alpha \\cos(2 \\pi q_\\| d_{layer}) + \\alpha^2}

    All arguments are torch tensors. `alpha` and `d_layer` are scalars; `q_par`
    is shape (..., ). Returns the same shape as `q_par`.

    Differentiable in all three.
    """
    two_pi = 2.0 * math.pi
    cos_term = torch.cos(two_pi * q_par * d_layer)
    num = 1.0 - alpha * alpha
    den = 1.0 - 2.0 * alpha * cos_term + alpha * alpha
    return num / den.clamp_min(1e-30)


def rod_intensity(
    q: torch.Tensor,                  # (N, 3)
    direction: torch.Tensor,          # (3,)
    pivot: torch.Tensor,              # (3,)
    A: torch.Tensor,                  # scalar
    alpha: torch.Tensor,              # scalar
    sigma_perp: torch.Tensor,         # scalar
    d_layer: torch.Tensor,            # scalar
    baseline: torch.Tensor,           # scalar
) -> torch.Tensor:
    """Full planar-defect rod intensity model. Differentiable in every kw.

    Returns (N,) predicted intensity at each q point.
    """
    rel = q - pivot
    direction_unit = direction / torch.linalg.vector_norm(direction)
    q_par = (rel * direction_unit).sum(dim=-1)
    q_perp_vec = rel - q_par.unsqueeze(-1) * direction_unit
    q_perp2 = (q_perp_vec * q_perp_vec).sum(dim=-1)
    cross = torch.exp(-q_perp2 / (2.0 * sigma_perp * sigma_perp + 1e-30))
    along = hendricks_teller(q_par, d_layer, alpha)
    return A * cross * along + baseline


# ---------------------------------------------------------------------------
# Parameter packing for autograd-friendly refinement
# ---------------------------------------------------------------------------

def _pack_params(direction_init, pivot_init, A_init, alpha_init,
                 sigma_perp_init, d_layer_init, baseline_init,
                 *, dtype, device):
    """Create the leaf tensors with reasonable parameterizations.

    - alpha ∈ (0, 1) via sigmoid(logit_alpha)
    - A, sigma_perp, d_layer, baseline > 0 via softplus(log_*)
    - direction kept as unconstrained 3-vector, normalized in the model
    """
    direction = torch.as_tensor(direction_init, dtype=dtype, device=device
                                 ).clone().requires_grad_(True)
    pivot = torch.as_tensor(pivot_init, dtype=dtype, device=device
                            ).clone().requires_grad_(True)
    # parameterize alpha as sigmoid(logit) ∈ (0, 1)
    logit_alpha = torch.tensor(
        math.log(alpha_init / (1.0 - alpha_init)) if 0 < alpha_init < 1 else 0.0,
        dtype=dtype, device=device,
    ).requires_grad_(True)
    log_A = torch.tensor(math.log(max(A_init, 1e-3)),
                         dtype=dtype, device=device).requires_grad_(True)
    log_sigma = torch.tensor(math.log(max(sigma_perp_init, 1e-3)),
                              dtype=dtype, device=device).requires_grad_(True)
    log_dlayer = torch.tensor(math.log(max(d_layer_init, 1e-3)),
                               dtype=dtype, device=device).requires_grad_(True)
    log_baseline = torch.tensor(math.log(max(baseline_init, 1e-3)),
                                 dtype=dtype, device=device).requires_grad_(True)
    return dict(
        direction=direction, pivot=pivot,
        logit_alpha=logit_alpha,
        log_A=log_A, log_sigma=log_sigma,
        log_dlayer=log_dlayer, log_baseline=log_baseline,
    )


def _unpack_params(p: dict) -> dict:
    alpha = torch.sigmoid(p["logit_alpha"])
    A = torch.exp(p["log_A"])
    sigma_perp = torch.exp(p["log_sigma"])
    d_layer = torch.exp(p["log_dlayer"])
    baseline = torch.exp(p["log_baseline"])
    direction = p["direction"] / torch.linalg.vector_norm(p["direction"])
    return dict(
        direction=direction, pivot=p["pivot"], A=A,
        alpha=alpha, sigma_perp=sigma_perp,
        d_layer=d_layer, baseline=baseline,
    )


# ---------------------------------------------------------------------------
# Public fitter
# ---------------------------------------------------------------------------

def fit_rod_params(
    q: "np.ndarray | torch.Tensor",
    intensity: "np.ndarray | torch.Tensor",
    *,
    direction_init: Sequence[float],
    pivot_init: Sequence[float],
    A_init: float = 1.0,
    alpha_init: float = 0.2,
    sigma_perp_init: float = 0.05,
    d_layer_init: float = 1.0,
    baseline_init: float = 1.0,
    n_steps: int = 400,
    lr: float = 5e-3,
    loss_kind: str = "lsq",                # "lsq" | "poisson"
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    refine_pivot: bool = True,
    refine_direction: bool = True,
) -> RodParams:
    """Refine the rod parameters against measured voxel intensities."""
    device_ = resolve_device(device)
    dtype_  = resolve_dtype(device_, dtype)

    q_t = torch.as_tensor(q, dtype=dtype_, device=device_)
    I_t = torch.as_tensor(intensity, dtype=dtype_, device=device_)
    if q_t.ndim != 2 or q_t.shape[1] != 3:
        raise ValueError(f"q must be shape (N, 3); got {tuple(q_t.shape)}")
    if I_t.shape[0] != q_t.shape[0]:
        raise ValueError("intensity and q size mismatch")

    p = _pack_params(direction_init, pivot_init, A_init, alpha_init,
                     sigma_perp_init, d_layer_init, baseline_init,
                     dtype=dtype_, device=device_)

    params_to_optimize = [p["logit_alpha"], p["log_A"], p["log_sigma"],
                          p["log_dlayer"], p["log_baseline"]]
    if refine_pivot:
        params_to_optimize.append(p["pivot"])
    if refine_direction:
        params_to_optimize.append(p["direction"])

    opt = torch.optim.Adam(params_to_optimize, lr=lr)
    history = []
    for step in range(n_steps):
        opt.zero_grad()
        u = _unpack_params(p)
        pred = rod_intensity(
            q_t, u["direction"], u["pivot"], u["A"], u["alpha"],
            u["sigma_perp"], u["d_layer"], u["baseline"],
        )
        if loss_kind == "lsq":
            loss = ((I_t - pred) ** 2).sum()
        elif loss_kind == "poisson":
            loss = (pred - I_t * torch.log(pred.clamp_min(1e-30))).sum()
        else:
            raise ValueError(f"unknown loss_kind: {loss_kind!r}")
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))

    u = _unpack_params(p)
    converged = len(history) > 2 and history[-1] <= history[0]
    return RodParams(
        direction=u["direction"].detach().cpu().numpy(),
        pivot=u["pivot"].detach().cpu().numpy(),
        A=float(u["A"].detach().cpu()),
        alpha=float(u["alpha"].detach().cpu()),
        sigma_perp=float(u["sigma_perp"].detach().cpu()),
        d_layer=float(u["d_layer"].detach().cpu()),
        baseline=float(u["baseline"].detach().cpu()),
        final_loss=history[-1],
        n_voxels=q_t.shape[0],
        converged=bool(converged),
    )


def bootstrap_rod_params(
    q: "np.ndarray | torch.Tensor",
    intensity: "np.ndarray | torch.Tensor",
    *,
    n_boot: int = 25,
    bootstrap_fraction: float = 0.7,
    seed: int = 0,
    **fit_kwargs,
) -> dict:
    """Bootstrap percentile-CIs on (α, σ_perp, d_layer) by resampling voxels.

    For each of `n_boot` bootstraps, draw a fraction `bootstrap_fraction` of
    the voxels and run `fit_rod_params`. Return per-parameter median + 16/84
    percentile (≈ 1σ Gaussian-equivalent), plus min/max.

    All kwargs are forwarded to `fit_rod_params`.
    """
    q_np = np.asarray(q)
    I_np = np.asarray(intensity)
    n = len(q_np)
    keep = max(20, int(bootstrap_fraction * n))
    rng = np.random.default_rng(seed)
    out_params: dict[str, list] = {
        "alpha": [], "sigma_perp": [], "d_layer": [], "A": [], "baseline": [],
        "direction": [],
    }
    for _ in range(n_boot):
        idx = rng.choice(n, size=keep, replace=False)
        try:
            fit = fit_rod_params(q_np[idx], I_np[idx], **fit_kwargs)
        except Exception:
            continue
        out_params["alpha"].append(fit.alpha)
        out_params["sigma_perp"].append(fit.sigma_perp)
        out_params["d_layer"].append(fit.d_layer)
        out_params["A"].append(fit.A)
        out_params["baseline"].append(fit.baseline)
        out_params["direction"].append(fit.direction)

    def _summ(xs):
        a = np.asarray(xs, dtype=np.float64)
        if a.size == 0:
            return dict(median=None, p16=None, p84=None, min=None, max=None,
                        n=0)
        return dict(
            median=float(np.median(a)),
            p16=float(np.percentile(a, 16)),
            p84=float(np.percentile(a, 84)),
            min=float(a.min()), max=float(a.max()),
            n=int(a.size),
        )

    # direction needs special treatment: report mean unit vector + ang spread
    dirs = np.array(out_params["direction"])
    if len(dirs):
        d_mean = dirs.mean(axis=0); d_mean /= np.linalg.norm(d_mean)
        ang = np.array([
            math.degrees(math.acos(min(1.0, abs(float(d @ d_mean)))))
            for d in dirs
        ])
        dir_summary = dict(
            mean=d_mean.tolist(),
            angle_spread_max_deg=float(ang.max()),
            angle_spread_p84_deg=float(np.percentile(ang, 84)),
        )
    else:
        dir_summary = None

    return dict(
        n_bootstraps=len(out_params["alpha"]),
        keep_per_boot=keep,
        n_total=n,
        alpha=_summ(out_params["alpha"]),
        sigma_perp=_summ(out_params["sigma_perp"]),
        d_layer=_summ(out_params["d_layer"]),
        A=_summ(out_params["A"]),
        baseline=_summ(out_params["baseline"]),
        direction=dir_summary,
    )


def per_hkl_alpha_consistency(
    q: np.ndarray, intensity: np.ndarray, rod_direction: np.ndarray,
    rod_pivot: np.ndarray, *,
    shell_qs: Sequence[float], d_layer_init: float,
    sigma_perp_init: float = 0.025,
    crop_perp: float = 0.10,
    crop_along: float = 0.30,
    min_voxels: int = 40,
    fit_kwargs: Optional[dict] = None,
) -> list[dict]:
    """For each shell-crossing along the rod, fit α independently.

    The rod-line `q(t) = pivot + t * direction` crosses a shell of magnitude R
    at t-values satisfying `|q(t)| = R`. For each such crossing, crop voxels
    in a local box (perp < crop_perp, |t − t_cross| < crop_along) and fit a
    small HT-style model. Returns per-crossing α / d_layer / σ_perp + the
    inferred crossing q-magnitude.

    Used to check that the global HT fit isn't averaging over heterogeneity.
    A consistent rod across shells → α(R) ≈ constant; a strain-broadened
    rod → α apparently grows with R (because the peak widths grow with R).
    """
    direction = np.asarray(rod_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    pivot = np.asarray(rod_pivot, dtype=np.float64)
    fit_kwargs = dict(fit_kwargs or {})
    fit_kwargs.setdefault("n_steps", 500)
    fit_kwargs.setdefault("lr", 1e-2)
    fit_kwargs.setdefault("loss_kind", "lsq")

    rel = q - pivot[None, :]
    t = rel @ direction
    perp = np.linalg.norm(rel - t[:, None] * direction[None, :], axis=1)

    # Find shell crossings:  |pivot + t·direction|^2 = R^2
    pd = float(pivot @ direction)
    p2 = float(pivot @ pivot)
    out = []
    for R in shell_qs:
        disc = pd * pd - (p2 - R * R)
        if disc < 0:
            continue
        sd = math.sqrt(disc)
        for t_cross in (-pd - sd, -pd + sd):
            if not (t.min() - crop_along <= t_cross <= t.max() + crop_along):
                continue
            in_box = (perp < crop_perp) & (np.abs(t - t_cross) < crop_along)
            n_in = int(in_box.sum())
            if n_in < min_voxels:
                continue
            q_local = q[in_box]
            I_local = intensity[in_box]
            try:
                fit = fit_rod_params(
                    q_local, I_local,
                    direction_init=direction.tolist(),
                    pivot_init=(pivot + t_cross * direction).tolist(),
                    A_init=float(I_local.max()),
                    alpha_init=0.25,
                    sigma_perp_init=sigma_perp_init,
                    d_layer_init=d_layer_init,
                    baseline_init=float(np.median(I_local)),
                    refine_direction=False, refine_pivot=False,
                    **fit_kwargs,
                )
                out.append(dict(
                    q_shell=float(R),
                    t_cross=float(t_cross),
                    n_voxels=n_in,
                    alpha=fit.alpha,
                    d_layer=fit.d_layer,
                    sigma_perp=fit.sigma_perp,
                    A=fit.A,
                    baseline=fit.baseline,
                    final_loss=fit.final_loss,
                ))
            except Exception:
                continue
    return out
