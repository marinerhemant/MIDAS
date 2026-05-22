"""Batched multi-grain refinement: ``refine_block``.

Processes ``B`` grains in one batched forward + backward per optimizer step.
This is what enables CPU/GPU performance: per-grain Python overhead would
defeat the purpose of using PyTorch.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from midas_diffract import HEDMForwardModel  # type: ignore

from .batch import MatchBatch, ObservedBatch, batch_residuals
from .config import FitConfig, LossKind
from .matching import MatchResult, associate, ring_slot_lookup
from .observations import ObservedSpots
from .refine import GrainFitResult
from .solvers import closure_kind, get_solver
from .solvers.lm_batched import minimize_lm_batched

DEG2RAD = math.pi / 180.0


@dataclass
class BlockFitResult:
    """Aggregate result of :func:`refine_block`."""
    grains: List[GrainFitResult]
    final_total_loss: float
    n_iter: int
    converged: bool


def _rematch_batch(
    *,
    model: HEDMForwardModel,
    pos: torch.Tensor,         # (B, 3)
    euler: torch.Tensor,       # (B, 3)
    lattice: torch.Tensor,     # (B, 6)
    obs: ObservedBatch,
    obs_ring_slot: torch.Tensor,    # (B, S_max) ring slot per obs spot
    pred_ring_slot: torch.Tensor,   # (M,) ring slot per reflection
    omega_tolerance: float,
    eta_tolerance: float,
) -> MatchBatch:
    """Re-associate observed↔predicted on every grain in the batch."""
    B = obs.n_grains
    with torch.no_grad():
        spots = model(euler.view(B, 1, 3), pos.view(B, 1, 3),
                      lattice_params=lattice.view(B, 6))
    pred_omega = spots.omega.detach()       # (B, K, M)
    pred_eta = spots.eta.detach()
    pred_valid = spots.valid.detach()
    K = pred_omega.shape[1]
    M = pred_omega.shape[2]

    # Wrap angular differences into [-π, π]; cost = |Δω| + 1e-3·|Δη|.
    BIG = torch.tensor(1e9, dtype=pred_omega.dtype, device=pred_omega.device)

    obs_om = obs.omega.unsqueeze(-1).unsqueeze(-1)            # (B, S, 1, 1)
    obs_et = obs.eta.unsqueeze(-1).unsqueeze(-1)
    pre_om = pred_omega.unsqueeze(1)                          # (B, 1, K, M)
    pre_et = pred_eta.unsqueeze(1)

    d_om = ((pre_om - obs_om + math.pi) % (2 * math.pi)) - math.pi  # (B, S, K, M)
    d_et = ((pre_et - obs_et + math.pi) % (2 * math.pi)) - math.pi

    # Equal-weight √(Δω² + Δη²): within-ring Laue multiplicity means many
    # reflections share |G| and so cluster in ω; η disambiguates them.
    cost = torch.sqrt(d_om * d_om + d_et * d_et)

    # Disqualify ring-mismatched and invalid-pred entries.
    obs_ring = obs_ring_slot.unsqueeze(-1).unsqueeze(-1)      # (B, S, 1, 1)
    pre_ring = pred_ring_slot.view(1, 1, 1, M)                # (1, 1, 1, M)
    ring_match = obs_ring == pre_ring                         # (B, S, 1, M)
    valid_pred = pred_valid.unsqueeze(1).bool()               # (B, 1, K, M)
    disq = ~(ring_match & valid_pred)                         # (B, S, K, M)
    cost = torch.where(disq, BIG, cost)

    flat = cost.reshape(*cost.shape[:2], K * M)               # (B, S, K*M)
    best_idx = flat.argmin(dim=-1)                            # (B, S)
    best_cost = flat.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)

    k_idx = best_idx // M
    m_idx = best_idx %  M

    # Recover signed Δω, Δη at the chosen pair to apply tolerances.
    bs_b, bs_s = torch.meshgrid(
        torch.arange(obs.n_grains, device=obs.omega.device),
        torch.arange(obs.s_max, device=obs.omega.device),
        indexing="ij",
    )
    chosen_d_om = d_om[bs_b, bs_s, k_idx, m_idx]
    chosen_d_et = d_et[bs_b, bs_s, k_idx, m_idx]

    mask = (
        (best_cost < BIG / 2.0)
        & (chosen_d_om.abs() <= omega_tolerance)
        & (chosen_d_et.abs() <= eta_tolerance)
        & obs.valid
    )

    return MatchBatch(k_idx=k_idx, m_idx=m_idx, mask=mask)


def _compile_enabled() -> bool:
    """Whether to apply ``torch.compile`` to the per-call residual.

    Off by default. Set ``MIDAS_FIT_GRAIN_COMPILE=1`` to enable. Useful on
    CUDA where the per-iter Python coordination of LBFGS line search dominates
    wall-clock at park22 scale (~4k grains, ~30 spots each, pixel loss).
    """
    return os.environ.get("MIDAS_FIT_GRAIN_COMPILE", "0") in ("1", "true", "yes")


def _make_block_closures(
    *,
    model: HEDMForwardModel,
    obs: ObservedBatch,
    match: MatchBatch,
    pos_scaled: torch.Tensor, pos_scale: float,
    euler: torch.Tensor, lattice: torch.Tensor,
    px: float, y_BC: float, z_BC: float,
    loss_kind: LossKind,
    active_params: list[torch.Tensor],
):
    """Build closure variants for the four solver protocols (batched).

    When ``MIDAS_FIT_GRAIN_COMPILE=1`` and CUDA is in use, the residual
    forward is wrapped in ``torch.compile(mode="reduce-overhead")`` so the
    per-line-search-probe forward + reduction is fused into a CUDA graph
    that LBFGS can replay at kernel-launch latency. Backward stays
    autograd-driven (compile doesn't trace .backward()), but the captured
    forward graph still helps because each closure call reuses its op
    sequence rather than re-tracing through Python.
    """

    def _residual_uncompiled() -> torch.Tensor:
        pos = pos_scaled * pos_scale
        return batch_residuals(
            model,
            grain_position=pos, grain_euler=euler, grain_lattice=lattice,
            obs=obs, match=match, kind=loss_kind,
            px=px, y_BC=y_BC, z_BC=z_BC,
        )

    if _compile_enabled() and pos_scaled.is_cuda:
        try:
            _residual = torch.compile(
                _residual_uncompiled,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )
        except Exception:
            _residual = _residual_uncompiled
    else:
        _residual = _residual_uncompiled

    def _scalar_loss(res: torch.Tensor) -> torch.Tensor:
        if res.numel() == 0:
            loss = torch.tensor(1e10, dtype=pos_scaled.dtype, device=pos_scaled.device)
        else:
            # Reduce per grain, then zero out any grain whose forward produced
            # a non-finite residual. The whole batch shares one scalar loss +
            # one LBFGS line search/history, so without this a single
            # degenerate grain's NaN would poison the step for EVERY grain in
            # the block. A neutralised grain contributes 0 (and 0 gradient via
            # nan_to_num's backward), so it stays frozen at its seed and is
            # filtered downstream by completeness — the other grains refine
            # normally.
            sq = res * res
            per_grain = sq.reshape(sq.shape[0], -1).sum(dim=1)   # (B,)
            per_grain = torch.nan_to_num(per_grain, nan=0.0,
                                         posinf=0.0, neginf=0.0)
            loss = per_grain.sum()
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
        # Hard backstop: a degenerate grain can still emit NaN/inf gradients
        # through the forward graph even after its loss term is masked
        # (0 * inf = NaN). Sanitise so the batched LBFGS step direction stays
        # finite for every grain; the bad grain just gets a zero update.
        for p in active_params:
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
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


def refine_block(
    cfg: FitConfig,
    *,
    model: HEDMForwardModel,
    grains_obs: Sequence[ObservedSpots],
    init_positions: torch.Tensor,    # (B, 3) um
    init_eulers:    torch.Tensor,    # (B, 3) rad
    init_lattices:  torch.Tensor,    # (B, 6)
    pred_ring_slot: torch.Tensor,    # (M,)
    pos_scale: float = 100.0,
    precomputed_matches: Optional[Sequence[MatchResult]] = None,
) -> BlockFitResult:
    """Refine ``B`` grains in one batched call.

    Parameter and output conventions mirror :func:`refine_grain`.
    """
    if not grains_obs:
        return BlockFitResult(grains=[], final_total_loss=0.0,
                              n_iter=0, converged=True)
    B = len(grains_obs)
    device = init_positions.device
    dtype = init_positions.dtype

    obs = ObservedBatch.pack(grains_obs, device=device, dtype=dtype)
    obs_ring_slot = ring_slot_lookup(cfg.RingNumbers, obs.ring_nr)

    # Match seed: either provided per-grain, or computed at init state.
    if precomputed_matches is not None:
        match = MatchBatch.pack(precomputed_matches, s_max=obs.s_max, device=device)
    else:
        match = _rematch_batch(
            model=model,
            pos=init_positions, euler=init_eulers, lattice=init_lattices,
            obs=obs, obs_ring_slot=obs_ring_slot, pred_ring_slot=pred_ring_slot,
            omega_tolerance=max(cfg.MarginOme, 2.0) * DEG2RAD,
            eta_tolerance=max(cfg.MarginEta, 5.0) * DEG2RAD,
        )

    omega_tol = max(cfg.MarginOme, 2.0) * DEG2RAD
    eta_tol = max(cfg.MarginEta, 5.0) * DEG2RAD

    pos_scaled = (init_positions / pos_scale).clone()
    euler = init_eulers.clone()
    lattice = init_lattices.clone()
    pos_scaled.requires_grad_(False)
    euler.requires_grad_(False)
    lattice.requires_grad_(False)

    # FF grain-position bound: the grain centre must lie inside the illuminated
    # sample cylinder — |X|,|Y| <= Rsample, |Z| <= Hbeam/2. Without this the
    # weakly-constrained X-along-beam coordinate drifts to unphysical values
    # (seeds from the indexer can carry placeholder positions far outside the
    # sample). NOTE: this is the correct position bound — ``BoxSize`` is the
    # detector active-area, NOT a grain-position bound. No-op for PF scanning
    # (scan_pos_tol_um > 0), which bounds position to the scan grid instead.
    _ff_pos_bound = float(getattr(cfg, "scan_pos_tol_um", 0.0)) <= 0.0
    _Rs = float(getattr(cfg, "Rsample", 0.0))
    _Hb = float(getattr(cfg, "Hbeam", 0.0))

    def _clamp_pos_to_sample():
        if not _ff_pos_bound:
            return
        with torch.no_grad():
            if _Rs > 0.0:
                r = _Rs / pos_scale
                pos_scaled[:, 0].clamp_(-r, r)
                pos_scaled[:, 1].clamp_(-r, r)
            if _Hb > 0.0:
                h = (_Hb / 2.0) / pos_scale
                pos_scaled[:, 2].clamp_(-h, h)

    _clamp_pos_to_sample()   # start refinement from inside the sample volume

    # The lm_batched solver bypasses the closure-based registry — it
    # works directly on the (B, P) packed param tensors. Skip the
    # registry lookup when it's selected.
    if cfg.solver == "lm_batched":
        solver_fn = None
        kind = None
    else:
        solver_fn = get_solver(cfg.solver)
        kind = closure_kind(cfg.solver)
    histories: list[float] = []
    converged_phases: list[bool] = []
    total_iter = 0

    def _run_phase(active: list[torch.Tensor], loss_kind: str = None,
                   **solver_opts):
        nonlocal total_iter
        for p in active:
            p.requires_grad_(True)
        closures = _make_block_closures(
            model=model, obs=obs, match=match,
            pos_scaled=pos_scaled, pos_scale=pos_scale,
            euler=euler, lattice=lattice,
            px=cfg.px, y_BC=model.y_BC, z_BC=model.z_BC,
            loss_kind=loss_kind or cfg.loss,
            active_params=active,
        )
        opts = {"max_iter": cfg.max_iter, "ftol": cfg.ftol, "xtol": cfg.xtol}
        opts.update(solver_opts)
        result = solver_fn(closures[kind], active, **opts)
        for p in active:
            p.requires_grad_(False)
        histories.extend(result["history"])
        converged_phases.append(result["converged"])
        total_iter += result["n_iter"]
        return result

    def _rematch():
        nonlocal match
        match = _rematch_batch(
            model=model,
            pos=pos_scaled * pos_scale, euler=euler, lattice=lattice,
            obs=obs, obs_ring_slot=obs_ring_slot, pred_ring_slot=pred_ring_slot,
            omega_tolerance=omega_tol, eta_tolerance=eta_tol,
        )

    use_batched_lm = cfg.solver == "lm_batched"

    def _batched_lm_phase(active_param_indices: list[int], max_iter: int,
                          loss_kind: str = None):
        """One LM phase, batched across all grains, on the active param subset.

        ``active_param_indices`` is over the 12-component flat layout
        ``[px, py, pz, e1, e2, e3, a, b, c, alpha, beta, gamma]``.
        """
        nonlocal pos_scaled, euler, lattice
        _lk = loss_kind or cfg.loss
        active_mask = torch.zeros(12, dtype=torch.bool, device=device)
        active_mask[active_param_indices] = True

        def _residual_fn(p, e, l):
            return batch_residuals(
                model,
                grain_position=p, grain_euler=e, grain_lattice=l,
                obs=obs, match=match, kind=_lk,
                px=cfg.px, y_BC=model.y_BC, z_BC=model.z_BC,
            ).reshape(B, -1)

        result = minimize_lm_batched(
            _residual_fn,
            pos_scaled, euler, lattice,
            pos_scale=pos_scale,
            max_iter=max_iter,
            ftol=cfg.ftol, xtol=cfg.xtol,
            active_mask=active_mask,
        )
        pos_scaled = result["pos_scaled"]
        euler = result["euler"]
        lattice = result["lattice"]
        converged_phases.append(result["converged"])
        nonlocal_total_iter[0] += result["n_iter"]

    nonlocal_total_iter = [0]   # closure-shared scalar; 'total_iter' rebound below

    # NOTE: a systematic per-phase-loss + orientation-first variant
    # (internal_angle for orientation/strain, full3d for position — matching
    # C FitPosOrStrains and the midas_diffract paper) is designed and passes
    # the synthetic refine tests, but regresses real FF data to zero refined
    # grains. Kept OUT of the default path pending debugging; the original
    # single-loss order below is the validated path. The Rsample/Hbeam sample
    # bound (_clamp_pos_to_sample) is retained — it is correct and contains the
    # X-along-beam drift. See project_fitgrain_ff_position_divergence memory.

    if use_batched_lm:
        # Active-param indices for each phase.
        IDX_POS = [0, 1, 2]
        IDX_EUL = [3, 4, 5]
        IDX_LAT = [6, 7, 8, 9, 10, 11]
        IDX_ALL = list(range(12))

        if cfg.mode == "all_at_once":
            _batched_lm_phase(IDX_ALL, max_iter=cfg.max_iter)
            _clamp_pos_to_sample()
        elif cfg.mode == "iterative":
            ph_pos, ph_or, ph_lat, ph_joint = cfg.phase_steps
            _batched_lm_phase(IDX_POS, max_iter=ph_pos * 5 + 5)
            _clamp_pos_to_sample()
            _rematch()
            _batched_lm_phase(IDX_EUL, max_iter=ph_or * 5 + 5)
            _rematch()
            _batched_lm_phase(IDX_LAT, max_iter=ph_lat * 5 + 5)
            _rematch()
            _batched_lm_phase(IDX_ALL, max_iter=ph_joint * 5 + 5)
            _clamp_pos_to_sample()
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")
        total_iter = nonlocal_total_iter[0]
    elif cfg.mode == "all_at_once":
        _run_phase([pos_scaled, euler, lattice])
        _clamp_pos_to_sample()
    elif cfg.mode == "iterative":
        import os as _os
        _decouple = _os.environ.get("MIDAS_FG_DECOUPLE", "0") == "1"
        ph_pos, ph_or, ph_lat, ph_joint = cfg.phase_steps
        if _decouple:
            # Decoupled per-phase loss (experimental, env-gated): orientation &
            # strain via the smooth ``angular`` loss (2θ,η,ω — position-
            # independent; NOT internal_angle, whose acos gradient is singular
            # near a good match), position via spatial ``full3d``. Orientation
            # first so spots match before position is fit.
            _run_phase([euler], max_iter=ph_or * 5 + 5, loss_kind="angular")
            _rematch()
            _run_phase([lattice], max_iter=ph_lat * 5 + 5, loss_kind="angular")
            _rematch()
            _run_phase([pos_scaled], max_iter=ph_pos * 5 + 5, loss_kind="full3d")
            _clamp_pos_to_sample()
            _rematch()
            _run_phase([pos_scaled, euler, lattice],
                       max_iter=ph_joint * 5 + 5, loss_kind="full3d")
            _clamp_pos_to_sample()
        else:
            _run_phase([pos_scaled], max_iter=ph_pos * 5 + 5)
            _clamp_pos_to_sample()
            _rematch()
            _run_phase([euler], max_iter=ph_or * 5 + 5)
            _rematch()
            _run_phase([lattice], max_iter=ph_lat * 5 + 5)
            _rematch()
            _run_phase([pos_scaled, euler, lattice], max_iter=ph_joint * 5 + 5)
            _clamp_pos_to_sample()
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    _clamp_pos_to_sample()   # final safety net
    pos_final = (pos_scaled * pos_scale).detach()
    euler_final = euler.detach()
    lattice_final = lattice.detach()

    # Final residual per grain (for FitBest output and final-loss reporting).
    with torch.no_grad():
        res_full = batch_residuals(
            model,
            grain_position=pos_final, grain_euler=euler_final,
            grain_lattice=lattice_final,
            obs=obs, match=match, kind=cfg.loss,
            px=cfg.px, y_BC=model.y_BC, z_BC=model.z_BC,
        )
    # Per-grain loss.
    per_grain_loss = (res_full * res_full).sum(dim=(-2, -1))  # (B,)
    total_loss = float(per_grain_loss.sum().item())

    # Reconstruct per-grain GrainFitResult by slicing the batched buffers.
    out: list[GrainFitResult] = []
    for b in range(B):
        n = int(obs.n_spots[b].item())
        per_spot_res = res_full[b, :n]                # (n_spots, K_res)
        # MatchResult slice — ragged, only real spots.
        s_match = MatchResult(
            k_idx=match.k_idx[b, :n].clone(),
            m_idx=match.m_idx[b, :n].clone(),
            mask=match.mask[b, :n].clone(),
            delta_omega=torch.zeros(n, dtype=dtype, device=device),
            delta_eta=torch.zeros(n, dtype=dtype, device=device),
        )
        out.append(GrainFitResult(
            position=pos_final[b].clone(),
            euler=euler_final[b].clone(),
            lattice=lattice_final[b].clone(),
            final_loss=float(per_grain_loss[b].item()),
            n_matched=int(s_match.mask.sum().item()),
            history=[],          # global history is in BlockFitResult
            converged=any(converged_phases),
            match=s_match,
            per_spot_residuals=per_spot_res.detach(),
        ))

    return BlockFitResult(
        grains=out,
        final_total_loss=total_loss,
        n_iter=total_iter,
        converged=any(converged_phases),
    )
