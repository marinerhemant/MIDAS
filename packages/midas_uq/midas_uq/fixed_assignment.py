"""Fixed-assignment UQ — the correct shape for scoring already-refined grains.

The default ``half_half_spots`` / ``jackknife_spots`` in :mod:`midas_uq.spots`
do **nearest-neighbour association at every LBFGS step**: the obs-to-pred-hkl
mapping is recomputed as the optimiser moves orientation/lattice. That's the
right behaviour for an *initial* refinement from a coarse seed (it lets the
optimiser climb out of a wrong association), but it's the **wrong** shape for a
UQ test on an already-refined grain — the optimiser can re-pair its way out of
any small data perturbation, the loss surface becomes effectively flat under
re-association, and small init perturbations either return the input unchanged
or drift to a totally different basin (we saw both on real Ni FF data: lattice
moved 0.67 Å, misori 1.14° under a 0.3° euler perturbation).

This module provides the **fixed-assignment** counterparts:

* :func:`freeze_associations`     — match obs↔pred-hkl ONCE at the input state.
* :func:`fit_grain_spots_fixed`   — three-phase LBFGS with the assignment held.
* :func:`half_half_fixed`         — random 50/50 spot splits, fixed pairing.
* :func:`jackknife_fixed`         — leave-one-out per spot, fixed pairing.
* :func:`per_spot_residuals`      — per-spot (Δ2θ, Δη, Δω) at the refined state.
* :func:`trust_score`             — convenience: run all three and return dict.

All entries take the refined :class:`GrainState` and observations in the same
angular ``(2θ, η, ω)`` radian convention as :mod:`midas_uq.spots`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from midas_diffract import HEDMForwardModel, SpotMatchingLoss

from ._common import (
    DEG2RAD, RAD2DEG, GrainState, default_loss, misori_deg, misori_deg_sym,
    lattice_max_abs,
)


# Default association cap for FIXED-assignment UQ: 0.05 rad ≈ 2.86°.
# Much tighter than the dynamic-association 0.5-rad default (≈28.6°) because at
# the refined state we expect predictions to be within a small fraction of a
# degree of observations. A loose cap on a fixed assignment isn't dangerous
# (the assignment doesn't change), but it admits a few spurious near-neighbour
# pairs; tight is safer.
_DEFAULT_MAX_DIST = 0.05


# ---------------------------------------------------------------------------
#  Association — match obs to pred ONCE at the input state.
# ---------------------------------------------------------------------------

def freeze_associations(
    model: HEDMForwardModel,
    state: GrainState,
    observations: torch.Tensor,
    max_dist: float = _DEFAULT_MAX_DIST,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve obs→pred correspondences at the given grain state, ONCE.

    Returns
    -------
    pred_idx : LongTensor (M,)
        Flat indices into the model's predicted (K, M_hkls, 3) tensor — one
        entry per surviving observed spot.
    obs_idx : LongTensor (M,)
        Indices into ``observations`` — the obs each predicted index pairs to.

    M is the number of pairs whose Euclidean distance in (2θ, η, ω) rad is
    below ``max_dist``. With both indices stored, you can pull aligned
    ``pred_match = pred_flat[pred_idx]`` and ``obs_match = observations[obs_idx]``
    inside any closure that re-evaluates the model.
    """
    pos = (state.pos if state.pos is not None
           else torch.zeros(3, dtype=state.euler_rad.dtype,
                            device=state.euler_rad.device))
    with torch.no_grad():
        spots = model(state.euler_rad.unsqueeze(0), pos.unsqueeze(0),
                      lattice_params=state.latc)
        coords, valid = HEDMForwardModel.predict_spot_coords(spots, space="angular")
        pred = coords.squeeze().reshape(-1, 3)
        vmask = valid.squeeze().reshape(-1) > 0.5
        valid_idx = torch.nonzero(vmask, as_tuple=False).squeeze(-1)
        pred_v = pred[vmask]
        dists = torch.cdist(observations, pred_v)
        min_d, nn = dists.min(dim=1)
        keep = min_d < max_dist
        return valid_idx[nn[keep]], torch.nonzero(keep, as_tuple=False).squeeze(-1)


# ---------------------------------------------------------------------------
#  Per-spot residuals at the input state (no optimisation).
# ---------------------------------------------------------------------------

@dataclass
class ResidualsResult:
    """Per-spot residuals at a fixed grain state.

    Attributes
    ----------
    delta_2theta_rad, delta_eta_rad, delta_omega_rad : Tensor (M,)
        Per-spot residual = predicted − observed in each angular coord.
    pos_err_lab_um : Tensor (M,)
        Per-spot lab-frame position error, projected to the detector plane
        from the angular residual. (Approximate; uses small-angle in 2θ.)
    rmse_rad : float
        Root-mean-square 3-D angular residual.
    n_spots, n_obs, frac_matched : int / float
    """
    delta_2theta_rad: torch.Tensor
    delta_eta_rad: torch.Tensor
    delta_omega_rad: torch.Tensor
    pos_err_lab_um: torch.Tensor
    rmse_rad: float
    n_spots: int
    n_obs: int
    frac_matched: float


def per_spot_residuals(
    model: HEDMForwardModel,
    state: GrainState,
    observations: torch.Tensor,
    max_dist: float = _DEFAULT_MAX_DIST,
) -> ResidualsResult:
    """Per-spot residuals at the (fixed) refined state — no optimisation.

    Use this when you want a quick, deterministic "how well does this grain
    fit its spots" metric. No optimiser is invoked; the residuals are read
    directly from the forward model at ``state``.

    The lab-frame position error is a small-angle projection of the angular
    residual onto the detector at ``Lsd`` (so ``pos_err_lab_um ≈ Lsd·Δ2θ``).
    """
    pred_idx, obs_idx = freeze_associations(model, state, observations, max_dist)
    pos = (state.pos if state.pos is not None
           else torch.zeros(3, dtype=state.euler_rad.dtype,
                            device=state.euler_rad.device))
    with torch.no_grad():
        spots = model(state.euler_rad.unsqueeze(0), pos.unsqueeze(0),
                      lattice_params=state.latc)
        coords, _ = HEDMForwardModel.predict_spot_coords(spots, space="angular")
        pred_flat = coords.squeeze().reshape(-1, 3)
    pred_match = pred_flat[pred_idx]
    obs_match = observations[obs_idx]
    delta = pred_match - obs_match
    rmse = torch.sqrt((delta ** 2).sum(dim=1).mean()).item() if delta.numel() else float("nan")
    Lsd = float(getattr(model, "Lsd", 0.0))
    pos_err = (Lsd * delta[:, 0]).abs() if Lsd > 0 else delta[:, 0].abs() * 0.0
    return ResidualsResult(
        delta_2theta_rad=delta[:, 0],
        delta_eta_rad=delta[:, 1],
        delta_omega_rad=delta[:, 2],
        pos_err_lab_um=pos_err,
        rmse_rad=rmse,
        n_spots=int(pred_match.shape[0]),
        n_obs=int(observations.shape[0]),
        frac_matched=(float(pred_match.shape[0]) / max(int(observations.shape[0]), 1)),
    )


# ---------------------------------------------------------------------------
#  Fixed-assignment refit (the optimiser the half-half / jackknife wrap).
# ---------------------------------------------------------------------------

def fit_grain_spots_fixed(
    model: HEDMForwardModel,
    init: GrainState,
    observations: torch.Tensor,
    pred_idx: torch.Tensor,
    obs_idx: torch.Tensor,
    loss_fn: Optional[SpotMatchingLoss] = None,
    *,
    phase_steps: Tuple[int, int, int] = (10, 10, 8),
    lbfgs_max_iter: int = 20,
) -> GrainState:
    """Three-phase L-BFGS refit (orientation → lattice → joint) with the
    obs↔pred mapping HELD FIXED at the (pred_idx, obs_idx) passed in.

    The closure uses ``pred_flat[pred_idx]`` and ``observations[obs_idx]`` —
    so as orientation/lattice move during optimisation, each observed spot
    stays paired with the SAME predicted hkl. The optimiser can't re-pair its
    way out of a small init perturbation; small perturbations produce smoothly
    varying loss; the converged state genuinely measures local reproducibility.
    """
    if loss_fn is None:
        loss_fn = default_loss()
    opt_euler = init.euler_rad.clone()
    opt_latc = init.latc.clone()
    pos = (init.pos if init.pos is not None
           else torch.zeros(3, dtype=opt_euler.dtype, device=opt_euler.device))

    def closure_factory(params: Sequence[torch.Tensor]):
        def closure():
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            spots = model(opt_euler.unsqueeze(0), pos.unsqueeze(0),
                          lattice_params=opt_latc)
            coords, _ = HEDMForwardModel.predict_spot_coords(spots, space="angular")
            pred_flat = coords.squeeze().reshape(-1, 3)
            pred_match = pred_flat[pred_idx]
            obs_match = observations[obs_idx]
            l = loss_fn(pred_match, obs_match)
            l.backward()
            return l
        return closure

    p1, p2, p3 = phase_steps
    # Phase 1: orientation
    opt_euler.requires_grad_(True); opt_latc.requires_grad_(False)
    o = torch.optim.LBFGS([opt_euler], lr=1.0, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(p1): o.step(closure_factory([opt_euler]))
    # Phase 2: lattice
    opt_euler.requires_grad_(False); opt_latc.requires_grad_(True)
    o = torch.optim.LBFGS([opt_latc], lr=1.0, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(p2): o.step(closure_factory([opt_latc]))
    # Phase 3: joint
    opt_euler.requires_grad_(True); opt_latc.requires_grad_(True)
    o = torch.optim.LBFGS([opt_euler, opt_latc], lr=0.5, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(p3): o.step(closure_factory([opt_euler, opt_latc]))

    return GrainState(opt_euler.detach().clone(),
                      opt_latc.detach().clone(),
                      pos.detach().clone())


# ---------------------------------------------------------------------------
#  Half-half UQ (fixed-assignment).
# ---------------------------------------------------------------------------

@dataclass
class HalfHalfFixedResult:
    misori_deg: np.ndarray
    lattice_max_abs_A: np.ndarray
    misori_median_deg: float
    misori_p90_deg: float
    lattice_median_A: float
    lattice_p90_A: float
    per_half_states: list
    n_obs: int
    n_pairs: int
    n_splits: int


def half_half_fixed(
    model: HEDMForwardModel,
    state: GrainState,
    observations: torch.Tensor,
    *,
    max_dist: float = _DEFAULT_MAX_DIST,
    n_splits: int = 5,
    seed: int = 0,
    loss: Optional[SpotMatchingLoss] = None,
    phase_steps: Tuple[int, int, int] = (10, 10, 8),
    space_group: Optional[int] = None,
) -> HalfHalfFixedResult:
    """K random 50/50 splits over the FIXED-PAIRING (obs, pred-hkl) set, fit
    each half independently with the assignment held, compare the two refined
    states.

    A trustworthy grain has small misori between the two halves across all K
    splits (the refined state is reproducible under data resampling). The
    fixed assignment guarantees that the K runs are measuring genuine local
    reproducibility, not the freedom of a dynamic re-association.
    """
    pred_idx, obs_idx = freeze_associations(model, state, observations, max_dist)
    n_pairs = int(pred_idx.shape[0])
    if n_pairs < 10:
        # Too few pairs for a meaningful half-half.
        empty = np.full(n_splits, np.nan)
        return HalfHalfFixedResult(
            misori_deg=empty, lattice_max_abs_A=empty,
            misori_median_deg=float("nan"), misori_p90_deg=float("nan"),
            lattice_median_A=float("nan"), lattice_p90_A=float("nan"),
            per_half_states=[], n_obs=int(observations.shape[0]),
            n_pairs=n_pairs, n_splits=n_splits)

    gen = torch.Generator().manual_seed(seed)
    miso = np.zeros(n_splits)
    latd = np.zeros(n_splits)
    halves = []
    for k in range(n_splits):
        perm = torch.randperm(n_pairs, generator=gen)
        half = n_pairs // 2
        a, b = perm[:half], perm[half:]
        sa = fit_grain_spots_fixed(model, state, observations,
                                    pred_idx[a], obs_idx[a], loss,
                                    phase_steps=phase_steps)
        sb = fit_grain_spots_fixed(model, state, observations,
                                    pred_idx[b], obs_idx[b], loss,
                                    phase_steps=phase_steps)
        from ._common import euler2mat_safe
        Ra = euler2mat_safe(sa.euler_rad); Rb = euler2mat_safe(sb.euler_rad)
        miso[k] = (misori_deg_sym(Ra, Rb, space_group)
                   if space_group is not None else misori_deg(Ra, Rb))
        latd[k] = lattice_max_abs(sa.latc, sb.latc)
        halves.append((sa, sb))
    return HalfHalfFixedResult(
        misori_deg=miso, lattice_max_abs_A=latd,
        misori_median_deg=float(np.median(miso)),
        misori_p90_deg=float(np.percentile(miso, 90)),
        lattice_median_A=float(np.median(latd)),
        lattice_p90_A=float(np.percentile(latd, 90)),
        per_half_states=halves,
        n_obs=int(observations.shape[0]), n_pairs=n_pairs, n_splits=n_splits)


# ---------------------------------------------------------------------------
#  Jackknife (leave-one-out per spot, fixed assignment).
# ---------------------------------------------------------------------------

@dataclass
class JackknifeFixedResult:
    influence_misori_deg: np.ndarray   # (n_pairs,) misori vs full-fit per LOO
    influence_lattice_A: np.ndarray    # (n_pairs,) lattice deviation per LOO
    max_influence_misori: float
    max_influence_lattice: float
    median_influence_misori: float
    n_pairs: int


def jackknife_fixed(
    model: HEDMForwardModel,
    state: GrainState,
    observations: torch.Tensor,
    *,
    max_dist: float = _DEFAULT_MAX_DIST,
    loss: Optional[SpotMatchingLoss] = None,
    phase_steps: Tuple[int, int, int] = (5, 5, 3),
    space_group: Optional[int] = None,
    max_spots: Optional[int] = None,
) -> JackknifeFixedResult:
    """Leave-one-spot-out per spot, with the FIXED obs↔pred mapping.

    Returns per-spot "influence": how much the refined state moves when that
    spot is dropped. Spots with high influence are driving the fit (possibly
    an outlier). A grain with all-low influences is well-supported by every
    spot; a grain with one or two dominant spots is fragile.

    For large grains (n_pairs > 100), set ``max_spots`` to subsample the LOO
    universe (random pick); the full N×fit can be slow.
    """
    pred_idx, obs_idx = freeze_associations(model, state, observations, max_dist)
    n_pairs = int(pred_idx.shape[0])
    if n_pairs < 10:
        empty = np.full(0, np.nan)
        return JackknifeFixedResult(
            influence_misori_deg=empty, influence_lattice_A=empty,
            max_influence_misori=float("nan"), max_influence_lattice=float("nan"),
            median_influence_misori=float("nan"), n_pairs=n_pairs)
    # Full fit (reference state we measure influence against).
    full = fit_grain_spots_fixed(model, state, observations,
                                  pred_idx, obs_idx, loss,
                                  phase_steps=phase_steps)
    idx_iter = list(range(n_pairs))
    if max_spots is not None and n_pairs > max_spots:
        rng = np.random.default_rng(0)
        idx_iter = rng.choice(n_pairs, size=max_spots, replace=False).tolist()
    miso = np.zeros(len(idx_iter)); latd = np.zeros(len(idx_iter))
    from ._common import euler2mat_safe
    R_full = euler2mat_safe(full.euler_rad)
    for k, drop in enumerate(idx_iter):
        keep = torch.cat([torch.arange(drop), torch.arange(drop + 1, n_pairs)])
        s = fit_grain_spots_fixed(model, state, observations,
                                   pred_idx[keep], obs_idx[keep], loss,
                                   phase_steps=phase_steps)
        R = euler2mat_safe(s.euler_rad)
        miso[k] = (misori_deg_sym(R, R_full, space_group)
                   if space_group is not None else misori_deg(R, R_full))
        latd[k] = lattice_max_abs(s.latc, full.latc)
    return JackknifeFixedResult(
        influence_misori_deg=miso, influence_lattice_A=latd,
        max_influence_misori=float(miso.max()),
        max_influence_lattice=float(latd.max()),
        median_influence_misori=float(np.median(miso)),
        n_pairs=n_pairs)


# ---------------------------------------------------------------------------
#  Convenience: one-shot grain trust score.
# ---------------------------------------------------------------------------

@dataclass
class TrustScore:
    """One-shot grain trust summary combining the three fixed-assignment
    diagnostics. Smaller values across the board => more trustworthy."""
    n_spots: int
    frac_matched: float
    rmse_rad: float
    per_spot_pos_um_p95: float
    half_half_misori_med_deg: float
    half_half_lat_med_A: float
    jackknife_max_misori_deg: float


def trust_score(
    model: HEDMForwardModel,
    state: GrainState,
    observations: torch.Tensor,
    *,
    max_dist: float = _DEFAULT_MAX_DIST,
    n_splits: int = 3,
    phase_steps: Tuple[int, int, int] = (10, 10, 8),
    space_group: Optional[int] = None,
    do_jackknife: bool = True,
    jackknife_max_spots: int = 50,
) -> TrustScore:
    """Run all three fixed-assignment diagnostics and pack into a TrustScore.

    Suggested filter for "trustworthy": rmse < ~few-mrad,
    per-spot p95 pos < ~200 µm, half_half misori med < ~0.05°, half_half
    lattice < ~0.005 Å, jackknife max misori < ~0.1°. Tune to the dataset.
    """
    res = per_spot_residuals(model, state, observations, max_dist)
    hh = half_half_fixed(model, state, observations,
                         max_dist=max_dist, n_splits=n_splits,
                         phase_steps=phase_steps, space_group=space_group)
    if do_jackknife and res.n_spots >= 10:
        jk = jackknife_fixed(model, state, observations,
                             max_dist=max_dist, phase_steps=phase_steps,
                             space_group=space_group, max_spots=jackknife_max_spots)
        jk_max = jk.max_influence_misori
    else:
        jk_max = float("nan")
    return TrustScore(
        n_spots=res.n_spots, frac_matched=res.frac_matched, rmse_rad=res.rmse_rad,
        per_spot_pos_um_p95=float(np.percentile(res.pos_err_lab_um.numpy(), 95))
                              if res.pos_err_lab_um.numel() else float("nan"),
        half_half_misori_med_deg=hh.misori_median_deg,
        half_half_lat_med_A=hh.lattice_median_A,
        jackknife_max_misori_deg=jk_max,
    )
