"""Joint V + K (+ optional μ, beam) refinement for the Sharma-Offerman V-map.

Wraps :func:`midas_transforms.radius.predicted_spot_intensities` in an
:class:`torch.nn.Module` whose parameters are the refinable handles, then
calls :class:`torch.optim.LBFGS` (or :class:`Adam`) on a log-space residual
loss.  Initialization is the closed-form K from
:func:`refine_K_per_ring_closed_form` plus the user-supplied
:math:`V_{\\rm init}` from :func:`aggregate_per_voxel`.

Parameterizations
-----------------
* ``V[v] = softplus(V_log[v])`` — keeps V strictly positive while allowing
  the unconstrained parameter to range over ℝ.  softplus is smooth and its
  derivative ``sigmoid`` is well-conditioned.
* ``K[r] = exp(K_log[r])`` — strictly positive.
* ``μ = exp(μ_log)`` — strictly positive when refined.
* Beam parameters: refined via the ``nn.Parameter`` flags set when the
  :class:`midas_transforms.geometry.beam.BeamProfile` was constructed
  (``refine=True`` on :class:`TopHat`, ``refine_fwhm=True`` /
  ``refine_offset=True`` on :class:`Gaussian`).  ``refine_beam=False``
  freezes them via :meth:`requires_grad_(False)`.

Loss
----
* ``log_l2``  : mean ``(log I_obs - log I_pred)²`` over valid spots.
* ``huber_log``: Huber on the same log-residual; tolerates outliers.

Both losses are smooth in all parameters; LBFGS converges reliably on
noise-free synthetic to machine precision in a few dozen iterations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from ..geometry.beam import BeamProfile
    from ..geometry.sample import SampleGrid


__all__ = ["RefineResult", "refine_vmap_joint"]


# --------------------------------------------------------- parameterizations


def _softplus_inv(y, *, eps: float = 1e-12) -> "torch.Tensor":
    """Numerically stable inverse of softplus: ``log(exp(y) - 1)`` for y > 0."""
    import torch
    # Use expm1 for accuracy when y is small.
    return torch.log(torch.expm1(torch.clamp(y, min=eps)))


# ----------------------------------------------------------- nn.Module shell


def _build_refine_module(
    V_init: "torch.Tensor",
    K_init: "torch.Tensor",
    mu_init: Optional["torch.Tensor"],
    beam_profile,
    *,
    refine_V: bool,
    refine_K: bool,
    refine_mu: bool,
    refine_beam: bool,
):
    """Pack refinable handles into a single ``nn.Module``."""
    import torch

    class _RefineModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            V_log_init = _softplus_inv(V_init).detach()
            K_log_init = torch.log(K_init).detach()
            if refine_V:
                self.V_log = torch.nn.Parameter(V_log_init.clone())
            else:
                self.register_buffer("V_log", V_log_init.clone())
            if refine_K:
                self.K_log = torch.nn.Parameter(K_log_init.clone())
            else:
                self.register_buffer("K_log", K_log_init.clone())
            if mu_init is not None:
                mu_log_init = torch.log(mu_init).detach()
                if refine_mu:
                    self.mu_log = torch.nn.Parameter(mu_log_init.clone())
                else:
                    self.register_buffer("mu_log", mu_log_init.clone())
            else:
                self.register_buffer("mu_log", torch.zeros(0))
            self.beam = beam_profile
            if not refine_beam:
                for p in self.beam.parameters():
                    p.requires_grad_(False)

        @property
        def V(self) -> "torch.Tensor":
            return torch.nn.functional.softplus(self.V_log)

        @property
        def K(self) -> "torch.Tensor":
            return torch.exp(self.K_log)

        @property
        def mu(self) -> Optional["torch.Tensor"]:
            return torch.exp(self.mu_log) if self.mu_log.numel() > 0 else None

    return _RefineModule()


# ------------------------------------------------------------ loss kernels


def _log_residual(
    I_obs: "torch.Tensor", I_pred: "torch.Tensor",
    *, eps: float = 1e-30,
):
    """Mask invalid spots and return ``log(I_obs) - log(I_pred)`` on the rest."""
    import torch
    valid = (I_obs > eps) & (I_pred > eps)
    safe_obs = torch.where(valid, I_obs, torch.ones_like(I_obs))
    safe_pred = torch.where(valid, I_pred, torch.ones_like(I_pred))
    r = torch.log(safe_obs) - torch.log(safe_pred)
    return r, valid


def _huber_log(r: "torch.Tensor", delta: float = 1.0):
    import torch
    a = r.abs()
    quad = 0.5 * r * r
    lin = delta * (a - 0.5 * delta)
    return torch.where(a <= delta, quad, lin)


# ------------------------------------------------------------ result type


@dataclass
class RefineResult:
    V_voxel:           "torch.Tensor"          # (Nv,) — refined V (softplus-parameterized, > 0)
    K_ring:            "torch.Tensor"          # (R,)  — refined K (exp-parameterized, > 0)
    mu_per_cm:         Optional["torch.Tensor"] = None  # 0-d, only if refined
    beam_profile:      Optional[object] = None           # the (possibly refined) BeamProfile
    loss_history:      Optional["torch.Tensor"] = None   # (n_iters,)
    residuals_per_spot:Optional["torch.Tensor"] = None   # (Ns,) log-residual (0 where invalid)
    n_iterations:      int = 0
    converged:         bool = False


# ----------------------------------------------------------- main API


def refine_vmap_joint(
    V_init: "torch.Tensor",                              # (Nv,)
    K_init: "torch.Tensor",                              # (R,)
    spot_observed_intensity: "torch.Tensor",             # (Ns,)
    spot_ring_idx: "torch.Tensor",                        # (Ns,) int
    spot_grain_idx: "torch.Tensor",                       # (Ns,) int
    spot_scan_pos_um: "torch.Tensor",                     # (Ns,) float
    spot_omega_rad: "torch.Tensor",                       # (Ns,) float
    sample_grid: "SampleGrid",
    beam_profile,
    theoretical_intensity_per_ring: "torch.Tensor",     # (R,)
    *,
    # Geometry mode (forwarded to predicted_spot_intensities)
    scan_axis: str = "pf",
    # Absorption (optional)
    use_absorption: bool = False,
    incident_dirs_per_spot: Optional["torch.Tensor"] = None,
    diffracted_dirs_per_spot: Optional["torch.Tensor"] = None,
    mu_init: Optional["torch.Tensor"] = None,
    # What to refine
    refine_V: bool = True,
    refine_K: bool = True,
    refine_mu: bool = False,
    refine_beam: bool = False,
    # Optimization
    max_iter: int = 100,
    optimizer: str = "lbfgs",                # "lbfgs" | "adam"
    lr: float = 0.5,
    loss_kind: str = "log_l2",                # "log_l2" | "huber_log"
    huber_delta: float = 1.0,
    tolerance: float = 1e-7,
    lbfgs_inner_iter: int = 20,
) -> "RefineResult":
    """Joint torch.optim refinement of V, K (+ optional μ, beam params).

    The forward model is
    :func:`midas_transforms.radius.predicted_spot_intensities`; refinable
    handles are softplus(V), exp(K), exp(μ), and beam parameters that
    were declared as :class:`torch.nn.Parameter` at beam-profile
    construction time (e.g., ``TopHat(width, refine=True)``).

    Returns a :class:`RefineResult` whose tensors live on ``V_init``'s
    device / dtype.
    """
    import torch

    from .forward_model import predicted_spot_intensities

    if use_absorption:
        if (incident_dirs_per_spot is None
                or diffracted_dirs_per_spot is None
                or mu_init is None):
            raise ValueError(
                "use_absorption=True requires incident_dirs_per_spot, "
                "diffracted_dirs_per_spot, and mu_init"
            )
        if refine_mu and mu_init is None:
            raise ValueError("refine_mu=True requires mu_init")

    module = _build_refine_module(
        V_init=V_init, K_init=K_init,
        mu_init=mu_init if use_absorption else None,
        beam_profile=beam_profile,
        refine_V=refine_V, refine_K=refine_K,
        refine_mu=refine_mu, refine_beam=refine_beam,
    )

    # Collect trainable parameters.
    trainable: List["torch.nn.Parameter"] = [
        p for p in module.parameters() if p.requires_grad
    ]
    if not trainable:
        raise ValueError(
            "no trainable parameters — turn on at least one of "
            "refine_V / refine_K / refine_mu / refine_beam"
        )

    def _forward() -> "torch.Tensor":
        kw = dict(
            V_voxel=module.V, K_ring=module.K,
            theoretical_intensity_per_ring=theoretical_intensity_per_ring,
            spot_ring_idx=spot_ring_idx, spot_grain_idx=spot_grain_idx,
            spot_scan_pos_um=spot_scan_pos_um, spot_omega_rad=spot_omega_rad,
            sample_grid=sample_grid, beam_profile=module.beam,
            scan_axis=scan_axis,
        )
        if use_absorption:
            kw.update(
                use_absorption=True,
                incident_dirs_per_spot=incident_dirs_per_spot,
                diffracted_dirs_per_spot=diffracted_dirs_per_spot,
                mu_per_cm=(module.mu if module.mu is not None else mu_init),
            )
        return predicted_spot_intensities(**kw)

    def _compute_loss() -> "torch.Tensor":
        I_pred = _forward()
        r, valid = _log_residual(spot_observed_intensity, I_pred)
        if loss_kind == "log_l2":
            kernel = 0.5 * r * r
        elif loss_kind == "huber_log":
            kernel = _huber_log(r, delta=huber_delta)
        else:
            raise ValueError(f"unknown loss_kind: {loss_kind!r}")
        w = valid.to(kernel.dtype)
        # Mean over valid spots; constant 1 added inside max to guard /0
        n_valid = w.sum().clamp(min=1.0)
        return (kernel * w).sum() / n_valid

    # ---------------------------------------------------------- optimization
    loss_history: List[float] = []
    converged = False
    n_iter = 0

    if optimizer == "lbfgs":
        opt = torch.optim.LBFGS(
            trainable, lr=lr, max_iter=lbfgs_inner_iter,
            tolerance_grad=1e-12, tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            loss = _compute_loss()
            loss.backward()
            return loss

        for it in range(max_iter):
            loss_val = float(opt.step(closure).item())
            loss_history.append(loss_val)
            n_iter = it + 1
            if it > 0:
                rel = abs(loss_history[-2] - loss_val) / max(
                    abs(loss_val), 1e-30
                )
                if rel < tolerance:
                    converged = True
                    break
    elif optimizer == "adam":
        opt = torch.optim.Adam(trainable, lr=lr)
        for it in range(max_iter):
            opt.zero_grad()
            loss = _compute_loss()
            loss.backward()
            opt.step()
            loss_val = float(loss.item())
            loss_history.append(loss_val)
            n_iter = it + 1
            if it > 0:
                rel = abs(loss_history[-2] - loss_val) / max(
                    abs(loss_val), 1e-30
                )
                if rel < tolerance:
                    converged = True
                    break
    else:
        raise ValueError(f"unknown optimizer: {optimizer!r}")

    # ---------------------------------------------------------- result
    with torch.no_grad():
        I_pred_final = _forward()
        r_final, valid = _log_residual(spot_observed_intensity, I_pred_final)
        r_final = torch.where(valid, r_final, torch.zeros_like(r_final))
        V_out = module.V.detach().clone()
        K_out = module.K.detach().clone()
        mu_out = module.mu.detach().clone() if module.mu is not None else None
        loss_t = torch.tensor(
            loss_history, dtype=V_init.dtype, device=V_init.device
        )

    return RefineResult(
        V_voxel=V_out,
        K_ring=K_out,
        mu_per_cm=mu_out,
        beam_profile=module.beam,
        loss_history=loss_t,
        residuals_per_spot=r_final,
        n_iterations=n_iter,
        converged=converged,
    )
