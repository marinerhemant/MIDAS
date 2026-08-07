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

    # How far the optimizer actually moved each grain's position from the seed
    # it was handed, in µm. Reported because ``converged`` cannot express the
    # one failure that matters here: when torch.optim.LBFGS's strong-Wolfe
    # line search gives up and returns t = 0, the loss repeats bit-for-bit,
    # the ftol counter trips, and the solver truthfully reports "the loss
    # stopped changing" — while nothing was ever refined. fp32 does exactly
    # that to FF grain fits: every grain kept its seed position, ~158 µm off
    # the C reference FitPosOrStrainsOMP, and nothing downstream noticed
    # (1-ID GE5 Au3, 2026-07-30).
    #
    # Judge these against a scale that carries units — the detector pixel
    # (``cfg.px``). A block in which not one grain's position moved by even a
    # pixel-equivalent did no useful work, whatever ``converged`` says.
    max_position_move_um: float = 0.0
    median_position_move_um: float = 0.0
    # Spread of the SEED positions handed in. The scale that says whether the
    # movement above was meaningful: a fit that moves far less than its seeds
    # disagree has not resolved that disagreement, whatever the loss reports.
    seed_position_spread_um: float = 0.0
    # Bit-identical to the seed: the strongest form of the same statement.
    n_unmoved_position: int = 0
    n_grains: int = 0


def _match_chunk_size(obs, pred_ring_slot, dtype) -> int:
    """Grains per association chunk, from a memory budget.

    The association materialises roughly four (B, S, K, M) float tensors plus
    boolean masks at once. Size the chunk so ONE of them fits the budget, which
    leaves headroom for the rest and for the forward model's (B, K, M).
    """
    import os
    try:
        budget_gib = float(os.environ.get("MIDAS_FIT_GRAIN_MATCH_GIB", "2.0"))
    except ValueError:
        budget_gib = 2.0
    S = int(obs.s_max)
    M = int(pred_ring_slot.numel())
    K = 2                     # Friedel pair; the model's second axis
    itemsize = torch.empty(0, dtype=dtype).element_size()
    per_grain = max(1, S * K * M * itemsize)
    chunk = int((budget_gib * (1024 ** 3)) // per_grain)
    return max(1, chunk)


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
    """Re-associate observed↔predicted on every grain in the batch.

    Chunked over grains. The cost machinery below materialises several
    ``(B, S, K, M)`` tensors at once, which at full-layer FF scale is
    ``22327 x 244 x 2 x 168`` = **13.3 GiB each** in float64 — enough to OOM a
    47 GiB A6000 and make ``--refine-backend python --device cuda`` unusable on
    a whole layer, since the pipeline hands the refiner one block of every
    seed. Each grain's association is independent of every other, so slicing B
    is exact: the returned indices and mask are identical to the unchunked
    result, only peak memory changes.

    Budget via ``MIDAS_FIT_GRAIN_MATCH_GIB`` (default 2.0 GiB per intermediate).
    """
    B_total = obs.n_grains
    if B_total > 1:
        chunk = _match_chunk_size(obs, pred_ring_slot, pos.dtype)
        if chunk < B_total:
            k_parts, m_parts, mask_parts = [], [], []
            for lo in range(0, B_total, chunk):
                hi = min(lo + chunk, B_total)
                sub = _rematch_batch(
                    model=model, pos=pos[lo:hi], euler=euler[lo:hi],
                    lattice=lattice[lo:hi], obs=obs.slice_grains(lo, hi),
                    obs_ring_slot=obs_ring_slot[lo:hi],
                    pred_ring_slot=pred_ring_slot,
                    omega_tolerance=omega_tolerance,
                    eta_tolerance=eta_tolerance,
                )
                k_parts.append(sub.k_idx)
                m_parts.append(sub.m_idx)
                mask_parts.append(sub.mask)
            return MatchBatch(k_idx=torch.cat(k_parts, 0),
                              m_idx=torch.cat(m_parts, 0),
                              mask=torch.cat(mask_parts, 0))

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
    lattice_scale: float = 1.0,
    px: float, y_BC: float, z_BC: float,
    loss_kind: LossKind,
    active_params: list[torch.Tensor],
    reduction: str = "sumsq",
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
        lat = lattice * lattice_scale
        return batch_residuals(
            model,
            grain_position=pos, grain_euler=euler, grain_lattice=lat,
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
            # Reduction over SPOTS. "sumsq" is Sigma(r^2) -- least squares,
            # the historical behaviour. "sumnorm" is Sigma||r_i|| -- the sum of
            # per-spot DISTANCES, which is what the C refiner accumulates
            # (`Error += CalcNorm2(...)`). Least squares weights a spot by its
            # error SQUARED, so a few badly-matched spots dominate the fit;
            # sum-of-norms is robust to them. Measured to be the whole of the
            # ~40 um FF position deficit -- see midas_fit_grain/c_recipe.py.
            if reduction == "sumnorm":
                r = res.reshape(res.shape[0], -1, res.shape[-1])
                sq = torch.sqrt((r * r).sum(dim=-1) + 1e-30)      # (B, S)
            else:
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


# Bounds on the auto-derived position scale. The lower bound keeps the
# historical 100 µm behaviour as a floor; the upper one stops a seed that
# happens to sit at a position stationary point (|g_pos| → 0) from producing
# an absurd scale.
_POS_SCALE_MIN = 100.0
_POS_SCALE_MAX = 1.0e9


def _equilibrated_pos_scale(
    *, model, obs, match, cfg,
    init_positions: torch.Tensor,
    init_eulers: torch.Tensor,
    init_lattices: torch.Tensor,
) -> float:
    """Choose ``pos_scale`` so the position gradient block matches the
    largest other block at the seed.

    L-BFGS applies ONE step length to the concatenated
    ``(pos_scaled, euler, lattice)`` vector, so a block whose gradient is much
    smaller than the others barely moves. With the historical fixed
    ``pos_scale = 100`` the FF orientation gradient is ~1500× the position
    gradient, and position advances ~1500× less per step. fp64 has the mantissa
    headroom to keep resolving that; fp32, whose gradient carries ~1e-4
    relative rounding noise, does not — the position component of the step
    lands under the noise, the line search finds no further descent, and the
    grain keeps its seed position. Measured on the synthetic fixture:

        pos_scale   |g|pos    |g|euler   ratio   fp32 error vs truth
             1e2      95.8     1.47e5    1537            154.27 µm
             1e3       958     1.47e5     154              0.75 µm
             1e4      9581     1.47e5    15.4              0.013 µm
             1e5    9.58e4     1.47e5     1.5              0.004 µm

    Accuracy tracks the ratio monotonically, so the scale is not tuned — it is
    the value that makes the ratio 1. Since ``d/d(pos_scaled) = s · d/d(pos)``,
    that value is ``s = |g_other| / |g_pos_µm|``.

    This is a pure reparameterization: every other use of ``pos_scale``
    (``pos = pos_scaled · pos_scale`` and the sample-cylinder clamp, which
    divides the µm bounds by it) stays consistent for any ``s``.
    """
    ps = init_positions.detach().clone().requires_grad_(True)
    eu = init_eulers.detach().clone().requires_grad_(True)
    la = init_lattices.detach().clone().requires_grad_(True)
    try:
        res = batch_residuals(
            model, grain_position=ps, grain_euler=eu, grain_lattice=la,
            obs=obs, match=match, kind=cfg.loss,
            px=cfg.px, y_BC=model.y_BC, z_BC=model.z_BC,
        )
        (res * res).sum().backward()
    except Exception:                                        # noqa: BLE001
        # Never let the preconditioner break a fit that would otherwise run.
        return _POS_SCALE_MIN

    def _n(t):
        return float(t.detach().norm()) if t is not None else 0.0

    g_pos = _n(ps.grad)
    g_other = max(_n(eu.grad), _n(la.grad))
    if not (g_pos > 0.0) or not (g_other > 0.0):
        return _POS_SCALE_MIN
    s = g_other / g_pos
    if not math.isfinite(s):
        return _POS_SCALE_MIN
    return float(min(max(s, _POS_SCALE_MIN), _POS_SCALE_MAX))


# The lattice block needs the same treatment as position, and for the same
# reason. ``_equilibrated_pos_scale`` lifts position to match the LARGEST of
# the other two blocks — which leaves the SMALLEST block starved by
# construction. In the real FCC-parent geometry that smallest block is the lattice:
#
#     |g| position 2.98   |g| euler 8.51e5   |g| lattice 7.03e4
#     pos_scale = max(euler, lattice)/pos = 2.85e5
#     after rescale:  pos 8.51e5   euler 8.51e5   lattice 7.03e4   → 12.1× down
#
# One shared L-BFGS step length then advances the lattice ~12× less per step.
# Measured cost, bridge at 1.6 px against a known 1222 µε deviatoric truth:
# ``all_at_once`` recovers 16 µε (1.3 %), ``iterative`` — which gives the
# lattice its own phase and therefore its own step length — recovers 530 µε
# (43 %). On the real FCC parent both are worse still (20 µε against C's 770 µε).
#
# The package's synthetic fixture does NOT reproduce this: there |g| lattice
# is the LARGEST block (1.7e6 vs euler 9.4e5), so the joint fit is already
# well conditioned and the unit tests cannot see the defect. Any regression
# test for this must use FF-scale geometry (Lsd ~1.7e6 µm), not the fixture.
#
# ─── DEFAULT OFF. Measured, and it TRADES. ────────────────────────────────
# Switching this on confirms the mechanism by intervention — but it buys
# strain with position, so it is opt-in, not "auto". Bridge, 200 grains,
# truth deviatoric 1222 µε, ``all_at_once``:
#
#     lattice_scale     recovered strain      position median (µm) by noise px
#                       µε (err vs truth)     0.0     0.05    0.2    0.5    1.6
#     1.0 (off)          16.4  (1209.4)      38.7    26.4   42.1   21.0   55.4
#     "auto"            680.0   (664.6)     106.1    60.2   75.5   71.4   63.9
#
# Strain recovery goes 1.3 % → 56 % of truth and its error (665 µε) beats
# ``iterative`` (748) and nears c-orig (735). Position degrades at every
# noise level, badly at low noise. So the under-refinement IS block
# conditioning — rescaling the block fixes it — but a single scalar gradient
# equalizer is the wrong cure: it is not curvature, and the lattice block is
# internally heterogeneous (Å alongside degrees), so no ONE factor can
# equilibrate both halves. A per-component scale, or reparameterizing to the
# dimensionless strain tensor, is the shape a real fix would have.
#
# ``iterative`` (the default mode, FitAllAtOnce=0) is BIT-IDENTICAL with this
# on or off: its lattice phase optimizes that block alone, and a converged
# single-block L-BFGS phase is scale-invariant in its answer. So this knob
# only ever affects the joint fit.
_LAT_SCALE_MIN = 1.0          # never make conditioning worse than today
_LAT_SCALE_MAX = 1.0e6


def _equilibrated_lattice_scale(
    *, model, obs, match, cfg,
    init_positions: torch.Tensor,
    init_eulers: torch.Tensor,
    init_lattices: torch.Tensor,
) -> float:
    """Scale the lattice block so its gradient matches the EULER block.

    Same reparameterization argument as :func:`_equilibrated_pos_scale`: since
    ``d/d(lat_scaled) = s · d/d(lattice)``, the value that equalises two blocks
    is ``s = |g_target| / |g_lattice|``.

    Euler is the target rather than ``max`` of the others because euler is the
    only block that is never reparameterized — it is the fixed anchor. Using
    ``max`` here would be circular, since position is itself being rescaled to
    ``max(euler, lattice)`` at the same time.
    """
    ps = init_positions.detach().clone().requires_grad_(True)
    eu = init_eulers.detach().clone().requires_grad_(True)
    la = init_lattices.detach().clone().requires_grad_(True)
    try:
        res = batch_residuals(
            model, grain_position=ps, grain_euler=eu, grain_lattice=la,
            obs=obs, match=match, kind=cfg.loss,
            px=cfg.px, y_BC=model.y_BC, z_BC=model.z_BC,
        )
        (res * res).sum().backward()
    except Exception:                                        # noqa: BLE001
        return _LAT_SCALE_MIN

    def _n(t):
        return float(t.detach().norm()) if t is not None else 0.0

    g_lat = _n(la.grad)
    g_other = max(_n(eu.grad), _n(ps.grad))
    if not (g_lat > 0.0) or not (g_other > 0.0):
        return _LAT_SCALE_MIN
    s = g_other / g_lat
    if not math.isfinite(s):
        return _LAT_SCALE_MIN
    return float(min(max(s, _LAT_SCALE_MIN), _LAT_SCALE_MAX))


def refine_block(
    cfg: FitConfig,
    *,
    model: HEDMForwardModel,
    grains_obs: Sequence[ObservedSpots],
    init_positions: torch.Tensor,    # (B, 3) um
    init_eulers:    torch.Tensor,    # (B, 3) rad
    init_lattices:  torch.Tensor,    # (B, 6)
    pred_ring_slot: torch.Tensor,    # (M,)
    pos_scale: float | str = "auto",
    lattice_scale: float | str = 1.0,
    precomputed_matches: Optional[Sequence[MatchResult]] = None,
) -> BlockFitResult:
    """Refine ``B`` grains in one batched call.

    Parameter and output conventions mirror :func:`refine_grain`.
    """
    if not grains_obs:
        return BlockFitResult(grains=[], final_total_loss=0.0,
                              n_iter=0, converged=True)

    # The ported C recipe is per-grain and derivative-free, so it does not use
    # the batched closure/solver machinery below at all. Dispatch before any of
    # it is built. See midas_fit_grain/c_recipe.py for why it exists.
    if cfg.mode == "c_recipe":
        from .c_recipe import refine_block_c_recipe
        return refine_block_c_recipe(
            cfg, model=model, grains_obs=grains_obs,
            init_positions=init_positions, init_eulers=init_eulers,
            init_lattices=init_lattices, pred_ring_slot=pred_ring_slot,
        )

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

    if isinstance(pos_scale, str):
        pos_scale = _equilibrated_pos_scale(
            model=model, obs=obs, match=match, cfg=cfg,
            init_positions=init_positions, init_eulers=init_eulers,
            init_lattices=init_lattices,
        )

    # The lm_batched solver bypasses the closure registry and works on packed
    # (B, P) tensors, so the reparameterization below would not reach it.
    # Leave it at 1.0 there rather than silently half-applying the fix.
    if isinstance(lattice_scale, str):
        lattice_scale = (
            1.0 if cfg.solver == "lm_batched"
            else _equilibrated_lattice_scale(
                model=model, obs=obs, match=match, cfg=cfg,
                init_positions=init_positions, init_eulers=init_eulers,
                init_lattices=init_lattices,
            )
        )

    pos_scaled = (init_positions / pos_scale).clone()
    euler = init_eulers.clone()
    lattice = (init_lattices / lattice_scale).clone()   # SCALED, not raw
    pos_scaled.requires_grad_(False)
    euler.requires_grad_(False)
    lattice.requires_grad_(False)

    def _lat():
        """Physical lattice from the scaled optimization variable."""
        return lattice * lattice_scale

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
        # The 2-D 'pixel' loss omits omega, so with orientation FREE the
        # crystal can rotate about the omega direction at no cost (~20° drift
        # on real PF data, 2026-05). It is safe — and is what the C refiner
        # uses — only while orientation is held fixed. Enforce that here rather
        # than trusting call sites.
        if (loss_kind or cfg.loss) == "pixel" and any(
                p is euler for p in active):
            raise ValueError(
                "loss 'pixel' (2-D, no omega) cannot be used in a phase that "
                "fits orientation: the rotation about omega is unconstrained. "
                "Use it only for position/lattice phases, as the C does."
            )
        for p in active:
            p.requires_grad_(True)
        closures = _make_block_closures(
            model=model, obs=obs, match=match,
            pos_scaled=pos_scaled, pos_scale=pos_scale,
            euler=euler, lattice=lattice, lattice_scale=lattice_scale,
            px=cfg.px, y_BC=model.y_BC, z_BC=model.z_BC,
            loss_kind=loss_kind or cfg.loss,
            active_params=active,
            reduction=str(getattr(cfg, "reduction", "sumsq") or "sumsq"),
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
            pos=pos_scaled * pos_scale, euler=euler, lattice=_lat(),
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
        _pl = getattr(cfg, "phase_losses", None)
        if _pl:
            # Per-phase objectives, mirroring the C refiner: position and
            # lattice on the 2-D detector loss, orientation on an angular one.
            # Measured to be the whole of the ~40 µm position deficit; see
            # midas_fit_grain/c_recipe.py.
            l_pos, l_or, l_lat, l_joint = _pl
            _run_phase([euler], max_iter=ph_or * 5 + 5, loss_kind=l_or)
            _rematch()
            _run_phase([pos_scaled], max_iter=ph_pos * 5 + 5, loss_kind=l_pos)
            _clamp_pos_to_sample()
            _rematch()
            _run_phase([lattice], max_iter=ph_lat * 5 + 5, loss_kind=l_lat)
            _rematch()
            if l_joint:
                _run_phase([pos_scaled, euler, lattice],
                           max_iter=ph_joint * 5 + 5, loss_kind=l_joint)
                _clamp_pos_to_sample()
        elif _decouple:
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
    lattice_final = _lat().detach()

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

    # Did the optimizer actually move each grain's position? See the fields on
    # BlockFitResult for why this is reported.
    _seed_pos = init_positions.to(device=pos_final.device, dtype=pos_final.dtype)
    _move = (pos_final - _seed_pos).norm(dim=-1).double()
    n_unmoved = int((pos_final == _seed_pos).all(dim=-1).sum())

    # Seed spread: RMS distance of the seeds from their own centroid, a
    # scale-free measure of how much disagreement the fit was asked to resolve.
    if init_positions.numel():
        _c = init_positions.double().mean(dim=0, keepdim=True)
        _seed_spread = float(
            (init_positions.double() - _c).norm(dim=1).median()
        )
    else:
        _seed_spread = 0.0

    return BlockFitResult(
        grains=out,
        final_total_loss=total_loss,
        n_iter=total_iter,
        converged=any(converged_phases),
        max_position_move_um=float(_move.max()) if _move.numel() else 0.0,
        median_position_move_um=(
            float(_move.median()) if _move.numel() else 0.0
        ),
        seed_position_spread_um=_seed_spread,
        n_unmoved_position=n_unmoved,
        n_grains=int(B),
    )
