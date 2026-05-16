"""Single-grain refinement entry point.

Public API: :func:`refine_grain`.

The grain state is parameterised as 12 scalars: ``(position[3],
euler[3], lattice[6])``. ``mode`` selects between

* ``"all_at_once"`` — one solver call over all 12 parameters; the
  observed↔predicted association is computed once at entry and held
  fixed (matches the user's spec — "if we fit everything together, we
  don't update spots").
* ``"iterative"`` — four sequential solver calls: position only,
  orientation only, strain only, joint polish. After each call, the
  spot association is recomputed against the updated state. This
  mirrors the C ``FitPosOrStrainsOMP`` default.

All three loss kinds (``pixel``, ``angular``, ``internal_angle``) are
supported via the same residual layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import numpy as np
import torch

from midas_diffract import HEDMForwardModel  # type: ignore

from .config import FitConfig, LossKind
from .matching import MatchResult, associate, ring_slot_lookup
from .observations import ObservedSpots
from .residuals import grain_residuals
from .solvers import closure_kind, get_solver

DEG2RAD = math.pi / 180.0


@dataclass
class GrainFitResult:
    """Output of :func:`refine_grain`."""
    position: torch.Tensor          # (3,) um
    euler: torch.Tensor             # (3,) rad
    lattice: torch.Tensor           # (6,)  a,b,c,alpha,beta,gamma
    final_loss: float
    n_matched: int
    history: list[float]
    converged: bool
    match: MatchResult              # final association
    per_spot_residuals: torch.Tensor  # (S_matched, K_residual)


def _match_with_state(
    model: HEDMForwardModel,
    *,
    pos: torch.Tensor,
    euler: torch.Tensor,
    lattice: torch.Tensor,
    obs: ObservedSpots,
    obs_ring_slot: torch.Tensor,
    pred_ring_slot: torch.Tensor,
    omega_tolerance: float,
    eta_tolerance: float,
) -> MatchResult:
    """Recompute observed↔predicted association at the current state."""
    with torch.no_grad():
        spots = model(euler.view(1, 1, 3), pos.view(1, 1, 3),
                      lattice_params=lattice.view(1, 6))
        # Squeeze (B=1, N=1) leading dims
        def _sq(t):
            while t.dim() > 2 and t.shape[0] == 1:
                t = t.squeeze(0)
                if t.dim() == 0:
                    break
            return t
        return associate(
            obs_ring_nr=obs.ring_nr,
            obs_omega=obs.omega,
            obs_eta=obs.eta,
            pred_ring_slot=pred_ring_slot,
            pred_omega=_sq(spots.omega),
            pred_eta=_sq(spots.eta),
            pred_valid=_sq(spots.valid),
            obs_ring_slot=obs_ring_slot,
            omega_tolerance=omega_tolerance,
            eta_tolerance=eta_tolerance,
        )


def _make_closures(
    *,
    model: HEDMForwardModel,
    obs: ObservedSpots,
    match: MatchResult,
    state_fn,                                    # () -> (pos, euler, lattice)
    px: float, y_BC: float, z_BC: float,
    loss_kind: LossKind,
    active_params: list[torch.Tensor],
):
    """Build a dict of closure flavors keyed by solver-protocol name.

    Returns ``{kind: callable}`` for ``"scalar_with_backward"``,
    ``"scalar_no_backward"``, and ``"residual_no_backward"``. Each variant
    differs only in whether ``backward()`` is called and whether the loss
    or the un-summed residual is returned.

    ``state_fn`` is a callable returning ``(pos, euler, lattice)`` tensors;
    inside it may apply reparameterizations (e.g. sigmoid box bounds) so
    the autograd graph flows back to the *active* parameters even when
    those are not the raw state variables.
    """

    def _residual() -> torch.Tensor:
        pos, euler, lattice = state_fn()
        res = grain_residuals(
            model,
            grain_euler=euler,
            grain_position=pos,
            grain_lattice=lattice,
            obs=obs,
            match=match,
            kind=loss_kind,
            px=px, y_BC=y_BC, z_BC=z_BC,
        )
        return res

    def _scalar_loss(res: torch.Tensor) -> torch.Tensor:
        if res.numel() == 0:
            loss = torch.tensor(1e10, dtype=pos_scaled.dtype, device=pos_scaled.device)
        else:
            loss = (res * res).sum()
        # Make sure every active param touches the autograd graph, even
        # when the loss is independent of it (e.g. position fit under
        # angular/internal_angle losses — a no-op by design).
        nop = torch.zeros((), dtype=loss.dtype, device=loss.device)
        for p in active_params:
            nop = nop + 0.0 * p.sum()
        return loss + nop

    def closure_with_backward() -> torch.Tensor:
        for p in active_params:
            if p.grad is not None:
                p.grad.zero_()
        loss = _scalar_loss(_residual())
        loss.backward()
        return loss

    def closure_no_backward() -> torch.Tensor:
        with torch.no_grad():
            return _scalar_loss(_residual())

    def residual_no_backward() -> torch.Tensor:
        with torch.no_grad():
            return _residual().reshape(-1)

    return {
        "scalar_with_backward": closure_with_backward,
        "scalar_no_backward":   closure_no_backward,
        "residual_no_backward": residual_no_backward,
    }


def refine_grain(
    cfg: FitConfig,
    *,
    model: HEDMForwardModel,
    obs: ObservedSpots,
    init_position: torch.Tensor,    # (3,) um
    init_euler: torch.Tensor,       # (3,) rad
    init_lattice: torch.Tensor,     # (6,)
    pred_ring_slot: torch.Tensor,   # (M,) — ring-slot per reflection in model
    pos_scale: float = 100.0,       # internal rescale: pos_um = pos_scale * pos_param
    precomputed_match: MatchResult | None = None,
) -> GrainFitResult:
    """Refine one grain.

    The caller provides ``model`` and ``pred_ring_slot`` (built once for the
    full block) so per-grain work is minimal.

    Initial guess in radians for euler, micrometers for position, refined
    lattice constants for lattice.
    """
    device = init_position.device
    dtype = init_position.dtype

    # The optimizer parameters (`pos_scaled`, euler, lattice) are crafted so
    # all three have comparable gradient magnitudes. ``pos_scaled`` is in
    # units of ``pos_scale`` micrometers per unit; the closure converts back
    # via ``pos = pos_scaled * pos_scale``.
    pos_scaled = (init_position.clone().to(device=device, dtype=dtype) / pos_scale)
    euler = init_euler.clone().to(device=device, dtype=dtype)
    lattice = init_lattice.clone().to(device=device, dtype=dtype)
    pos_scaled.requires_grad_(False)
    euler.requires_grad_(False)
    lattice.requires_grad_(False)

    # --- Sigmoid box-bound reparameterization (torch-native, autograd) ---
    # When cfg.use_bounds is True we reparameterize each bounded variable as
    #
    #     x = lb + (ub - lb) * sigmoid(4 * theta / (ub - lb))
    #
    # The 4/(ub-lb) scaling cancels sigmoid's 1/4 Jacobian at theta=0, so
    # dx/dtheta = 1 at the seed. Without it, gradients w.r.t. theta are
    # (ub-lb)/4 × gradients w.r.t. x (typically 20-30× smaller); LBFGS line
    # search still works but the Hessian approximation and ftol/xtol
    # criteria operate on a poorly-scaled variable and converge to a
    # different (sometimes worse) point. With the scaling, theta has the
    # same gradient magnitude as x near the seed, so the optimizer
    # behaves like the unbounded one until you approach the bound.
    #
    # Chain rule still flows back through sigmoid to theta — no scipy /
    # numpy round-trip, runs on CPU/CUDA/MPS identically.
    _use_bounds = bool(getattr(cfg, "use_bounds", False))
    if _use_bounds:
        _euler_half = float(cfg.bound_euler_deg) * DEG2RAD
        _abc_pct = float(cfg.bound_lat_abc_pct)
        _ang_half = float(cfg.bound_lat_angle_deg)
        euler_lb = init_euler.clone().to(device=device, dtype=dtype) - _euler_half
        euler_ub = init_euler.clone().to(device=device, dtype=dtype) + _euler_half
        init_abc = init_lattice[:3].to(device=device, dtype=dtype)
        init_ang = init_lattice[3:].to(device=device, dtype=dtype)
        lat_lb = torch.cat([init_abc * (1.0 - _abc_pct), init_ang - _ang_half])
        lat_ub = torch.cat([init_abc * (1.0 + _abc_pct), init_ang + _ang_half])
        euler_scale = 4.0 / (euler_ub - euler_lb)        # (3,)
        lat_scale = 4.0 / (lat_ub - lat_lb)              # (6,)
        # theta=0 → sigmoid=0.5 → x = (lb+ub)/2 = seed.
        theta_euler = torch.zeros_like(euler)
        theta_lattice = torch.zeros_like(lattice)
        theta_euler.requires_grad_(False)
        theta_lattice.requires_grad_(False)
    else:
        euler_lb = euler_ub = lat_lb = lat_ub = None
        euler_scale = lat_scale = None
        theta_euler = theta_lattice = None

    def _euler_view() -> torch.Tensor:
        """Current euler tensor (bounded if use_bounds, else raw)."""
        if _use_bounds:
            return euler_lb + (euler_ub - euler_lb) * torch.sigmoid(theta_euler * euler_scale)
        return euler

    def _lattice_view() -> torch.Tensor:
        if _use_bounds:
            return lat_lb + (lat_ub - lat_lb) * torch.sigmoid(theta_lattice * lat_scale)
        return lattice

    def _state_fn():
        """Returns (pos, euler, lattice) with reparameterization applied."""
        return pos_scaled * pos_scale, _euler_view(), _lattice_view()

    # Pre-compute the ring slot per observed spot (does not depend on state).
    obs_ring_slot = ring_slot_lookup(cfg.RingNumbers, obs.ring_nr)

    # Tolerances for re-association (radians).
    #
    # ``cfg.MarginOme`` is in degrees → convert with DEG2RAD.
    # ``cfg.MarginEta`` is in MICROMETERS (arc distance on the detector,
    # matches IndexerOMP.c:1773-1779 convention). Convert to an angular
    # tolerance via ``atan(MarginEta / ring_radius)``. Use the median
    # ring radius as a representative single-tol value — per-ring tols
    # would be more correct but require associate() to accept a vector.
    #
    # No floor: paramstest values like MarginOme=0.5°, MarginEta=500µm
    # were silently inflated to 2°/5° before this fix (the 2°/5° were a
    # placeholder bound that turned out to defeat user-set tolerances).
    omega_tol = float(cfg.MarginOme) * DEG2RAD
    if cfg.RingRadii:
        import math
        # Use the median positive ring radius — guards against any
        # zero-padded entries from the param-parser.
        radii_pos = [r for r in cfg.RingRadii if r > 0]
        if radii_pos:
            rep_radius = float(np.median(radii_pos))
            eta_tol = float(math.atan(cfg.MarginEta / rep_radius))
        else:
            eta_tol = float(cfg.MarginEta) * DEG2RAD   # fallback: treat as deg
    else:
        eta_tol = float(cfg.MarginEta) * DEG2RAD       # fallback: treat as deg

    # Initial association.
    if precomputed_match is not None:
        match = precomputed_match
    else:
        with torch.no_grad():
            match = _match_with_state(
                model, pos=pos_scaled * pos_scale,
                euler=_euler_view(), lattice=_lattice_view(),
                obs=obs, obs_ring_slot=obs_ring_slot, pred_ring_slot=pred_ring_slot,
                omega_tolerance=omega_tol, eta_tolerance=eta_tol,
            )

    solver_fn = get_solver(cfg.solver)

    # Helper: run one solver phase with a given active parameter set.
    histories: list[float] = []
    converged_phases: list[bool] = []

    kind = closure_kind(cfg.solver)

    # Translate "user-facing variable" → underlying optimizer variable.
    # When bounds are active, euler/lattice are NOT the optimizer's leaf
    # tensors (theta_euler / theta_lattice are). Position is always the
    # raw pos_scaled regardless of bounds.
    def _opt_var(v: torch.Tensor) -> torch.Tensor:
        if _use_bounds:
            if v is euler:
                return theta_euler
            if v is lattice:
                return theta_lattice
        return v

    def _run_phase(active: list[torch.Tensor], **solver_opts):
        opt_active = [_opt_var(p) for p in active]
        for p in opt_active:
            p.requires_grad_(True)
        closures = _make_closures(
            model=model, obs=obs, match=match,
            state_fn=_state_fn,
            px=cfg.px, y_BC=model.y_BC, z_BC=model.z_BC,
            loss_kind=cfg.loss,
            active_params=opt_active,
        )
        opts = {"max_iter": cfg.max_iter, "ftol": cfg.ftol, "xtol": cfg.xtol}
        opts.update(solver_opts)        # caller wins
        result = solver_fn(closures[kind], opt_active, **opts)
        for p in opt_active:
            p.requires_grad_(False)
        histories.extend(result["history"])
        converged_phases.append(result["converged"])
        return result

    def _rematch():
        nonlocal match
        with torch.no_grad():
            match = _match_with_state(
                model, pos=pos_scaled * pos_scale,
                euler=_euler_view(), lattice=_lattice_view(),
                obs=obs, obs_ring_slot=obs_ring_slot, pred_ring_slot=pred_ring_slot,
                omega_tolerance=omega_tol, eta_tolerance=eta_tol,
            )

    # --- Scan-aware position-mode handling (pf-HEDM) -----------------
    # When scan mode is active (cfg.scan_pos_tol_um > 0) the position
    # has two refinement options per plan §1e:
    #   "fixed"          → position is locked to the voxel center
    #                      (matches C IndexerScanningOMP behavior).
    #   "voxel_bounded"  → position refines inside
    #                      [init_y − beam_size/2, init_y + beam_size/2]
    #                      along Y; clamp post-phase. New v1 feature.
    # FF runs leave scan_pos_tol_um == 0 ⇒ position_mode ignored,
    # legacy behavior preserved.
    _scan_mode = float(getattr(cfg, "scan_pos_tol_um", 0.0)) > 0.0
    _pos_mode = getattr(cfg, "position_mode", "fixed") if _scan_mode else "refine"
    _beam_size = float(getattr(cfg, "beam_size_um", 0.0))

    def _maybe_clamp_pos_to_voxel_bound():
        """Project pos_scaled.data into the voxel-bounded box along Y.

        No-op outside voxel_bounded mode. Y is dim 1 of pos (PF scans
        run along Y per P0 audit §1a).
        """
        if _pos_mode != "voxel_bounded" or _beam_size <= 0:
            return
        # Convert bounds (in µm) into pos_scaled units.
        half = (_beam_size / 2.0) / pos_scale
        init_y_scaled = float(init_position[1].item()) / pos_scale
        lb = init_y_scaled - half
        ub = init_y_scaled + half
        with torch.no_grad():
            pos_scaled[1].clamp_(lb, ub)

    if cfg.mode == "all_at_once":
        if _pos_mode == "fixed":
            _run_phase([euler, lattice])
        else:
            _run_phase([pos_scaled, euler, lattice])
            _maybe_clamp_pos_to_voxel_bound()
    elif cfg.mode == "iterative":
        ph_pos, ph_or, ph_lat, ph_joint = cfg.phase_steps
        if _pos_mode != "fixed":
            _run_phase([pos_scaled], max_iter=ph_pos * 5 + 5)
            _maybe_clamp_pos_to_voxel_bound()
            _rematch()
        _run_phase([euler], max_iter=ph_or * 5 + 5)
        _rematch()
        _run_phase([lattice], max_iter=ph_lat * 5 + 5)
        _rematch()
        # Final joint polish — no further re-match, per spec.
        if _pos_mode == "fixed":
            _run_phase([euler, lattice], max_iter=ph_joint * 5 + 5)
        else:
            _run_phase([pos_scaled, euler, lattice], max_iter=ph_joint * 5 + 5)
            _maybe_clamp_pos_to_voxel_bound()
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    pos_final = (pos_scaled * pos_scale).detach()
    with torch.no_grad():
        euler_final = _euler_view().detach() if _use_bounds else euler.detach()
        lattice_final = _lattice_view().detach() if _use_bounds else lattice.detach()

    # Final residuals at converged state.
    with torch.no_grad():
        res = grain_residuals(
            model,
            grain_euler=euler_final, grain_position=pos_final, grain_lattice=lattice_final,
            obs=obs, match=match, kind=cfg.loss,
            px=cfg.px, y_BC=model.y_BC, z_BC=model.z_BC,
        )
        loss_final = float((res * res).sum().item()) if res.numel() else float("inf")

    return GrainFitResult(
        position=pos_final,
        euler=euler_final,
        lattice=lattice_final,
        final_loss=loss_final,
        n_matched=int(match.mask.sum().item()),
        history=histories,
        converged=any(converged_phases),
        match=match,
        per_spot_residuals=res.detach(),
    )
