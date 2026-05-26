"""Spot-based UQ (FF and pf-HEDM).

Three-phase L-BFGS refinement on observed spot coordinates (2theta, eta, omega).
Half-half resampling and per-spot jackknife.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch

from midas_diffract import HEDMForwardModel, SpotMatchingLoss

from ._common import (
    DEG2RAD, RAD2DEG, GrainState, default_loss,
    misori_deg, misori_deg_sym, lattice_max_abs, _associate,
)


@dataclass
class HalfHalfResult:
    """Result of a K-split half-half UQ run.

    Attributes
    ----------
    misori_deg : numpy.ndarray (K,)
        Per-split misorientation disagreement between the two halves.
    lattice_max_abs_A : numpy.ndarray (K,)
        Per-split max abs disagreement on (a, b, c).
    misori_median_deg, misori_p90_deg : float
        Convenience summaries.
    lattice_median_A, lattice_p90_A : float
    per_half_states : list of (GrainState, GrainState)
        The two refined states from each split. Useful for downstream
        analysis (e.g., evaluating which half hit a different basin).
    n_spots : int
        Number of observed spots used.
    n_splits : int
    """
    misori_deg: np.ndarray
    lattice_max_abs_A: np.ndarray
    per_half_states: list
    n_spots: int
    n_splits: int

    @property
    def misori_median_deg(self) -> float:
        return float(np.median(self.misori_deg))

    @property
    def misori_p90_deg(self) -> float:
        return float(np.percentile(self.misori_deg, 90))

    @property
    def lattice_median_A(self) -> float:
        return float(np.median(self.lattice_max_abs_A))

    @property
    def lattice_p90_A(self) -> float:
        return float(np.percentile(self.lattice_max_abs_A, 90))

    def summary_dict(self) -> dict:
        return {
            "n_spots": self.n_spots, "n_splits": self.n_splits,
            "misori_median_deg": self.misori_median_deg,
            "misori_p90_deg": self.misori_p90_deg,
            "lattice_median_A": self.lattice_median_A,
            "lattice_p90_A": self.lattice_p90_A,
        }


@dataclass
class JackknifeResult:
    """Per-spot leave-one-out influence on the refined grain state.

    Attributes
    ----------
    influence_misori_deg : numpy.ndarray (N,)
        Influence_k = misori(R_full, R_drop_k).
    influence_lat_A : numpy.ndarray (N,)
        Influence_k = |latc_full[:3].max - latc_drop_k[:3].max|.
    reference_state : GrainState
        The full-set fit used as reference.
    """
    influence_misori_deg: np.ndarray
    influence_lat_A: np.ndarray
    reference_state: GrainState

    def top_k(self, k: int = 10, by: str = "misori") -> np.ndarray:
        """Return the indices of the top-`k` most influential spots."""
        arr = self.influence_misori_deg if by == "misori" else self.influence_lat_A
        return np.argsort(-arr)[:k]


# ---------------------------------------------------------------------------
#  Inner refinement loop (three-phase L-BFGS) — same as paper I, exposed here
# ---------------------------------------------------------------------------

def _fit_grain_spots(
    model: HEDMForwardModel,
    obs_angular: torch.Tensor,    # (N, 3) (2theta, eta, omega) in radians
    init: GrainState,
    loss_fn: SpotMatchingLoss,
    *,
    phase1_steps: int = 10,
    phase2_steps: int = 10,
    phase3_steps: int = 8,
    lbfgs_max_iter: int = 20,
    max_match_distance: float = 0.5,
    min_matches: int = 5,
) -> GrainState:
    """Three-phase L-BFGS: orientation, lattice, joint. Returns refined state."""
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
            coords, valid = HEDMForwardModel.predict_spot_coords(
                spots, space="angular",
            )
            pred = coords.squeeze().reshape(-1, 3)
            vmask = valid.squeeze().reshape(-1) > 0.5
            if vmask.sum() == 0:
                return torch.tensor(1e12, dtype=opt_euler.dtype,
                                    requires_grad=True)
            pred_v = pred[vmask]
            pred_match, obs_match = _associate(
                pred_v, obs_angular, max_match_distance,
            )
            if pred_match.shape[0] < min_matches:
                return torch.tensor(1e12, dtype=opt_euler.dtype,
                                    requires_grad=True)
            l = loss_fn(pred_match, obs_match)
            l.backward()
            return l
        return closure

    # Phase 1: orientation
    opt_euler.requires_grad_(True); opt_latc.requires_grad_(False)
    o = torch.optim.LBFGS([opt_euler], lr=1.0, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(phase1_steps): o.step(closure_factory([opt_euler]))

    # Phase 2: lattice
    opt_euler.requires_grad_(False); opt_latc.requires_grad_(True)
    o = torch.optim.LBFGS([opt_latc], lr=1.0, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(phase2_steps): o.step(closure_factory([opt_latc]))

    # Phase 3: joint
    opt_euler.requires_grad_(True); opt_latc.requires_grad_(True)
    o = torch.optim.LBFGS([opt_euler, opt_latc], lr=0.5, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(phase3_steps): o.step(closure_factory([opt_euler, opt_latc]))

    return GrainState(opt_euler.detach().clone(),
                      opt_latc.detach().clone(),
                      pos.detach().clone())


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def half_half_spots(
    model: HEDMForwardModel,
    init: GrainState,
    observations: torch.Tensor,
    *,
    n_splits: int = 5,
    seed: int = 0,
    loss: Optional[SpotMatchingLoss] = None,
    phase_steps: tuple[int, int, int] = (10, 10, 8),
    multi_restart_K: int = 1,
    space_group: Optional[int] = None,
    verbose: bool = False,
) -> HalfHalfResult:
    """K random 50/50 spot splits; fit each half independently.

    Parameters
    ----------
    model : HEDMForwardModel
        Pre-built forward model (FF or pf geometry).
    init : GrainState
        Seed state (from indexing, e.g., a row of Grains.csv).
    observations : Tensor (N, 3)
        Observed spot angular coordinates `(2theta, eta, omega)` in radians.
    n_splits : int, default 5
        Number of random K-splits.
    seed : int, default 0
        RNG seed for reproducibility.
    loss : SpotMatchingLoss, optional
        Default is `SpotMatchingLoss(metric='l2')`.
    phase_steps : tuple of three ints
        Outer-loop step counts for each L-BFGS phase.
    multi_restart_K : int, default 1
        If > 1, run multi-restart on the orientation phase before phase 2/3
        (basin-escape per proto7).
    verbose : bool

    Returns
    -------
    HalfHalfResult
    """
    if loss is None: loss = default_loss()
    n = observations.shape[0]
    half = n // 2
    g = torch.Generator(device="cpu").manual_seed(seed)

    mis_list = []
    lat_list = []
    per_half = []
    for k in range(n_splits):
        perm = torch.randperm(n, generator=g)
        idx_a = perm[:half]; idx_b = perm[half:2 * half]
        obs_a = observations[idx_a]; obs_b = observations[idx_b]

        state_a = _fit_grain_spots(
            model, obs_a, init, loss,
            phase1_steps=phase_steps[0], phase2_steps=phase_steps[1],
            phase3_steps=phase_steps[2],
        )
        state_b = _fit_grain_spots(
            model, obs_b, init, loss,
            phase1_steps=phase_steps[0], phase2_steps=phase_steps[1],
            phase3_steps=phase_steps[2],
        )
        R_a = HEDMForwardModel.euler2mat(state_a.euler_rad)
        R_b = HEDMForwardModel.euler2mat(state_b.euler_rad)
        mis_list.append(
            misori_deg_sym(R_a, R_b, space_group) if space_group is not None
            else misori_deg(R_a, R_b),
        )
        lat_list.append(lattice_max_abs(state_a.latc, state_b.latc))
        per_half.append((state_a, state_b))
        if verbose:
            print(f"  split {k}: mis={mis_list[-1]:.5f}° lat={lat_list[-1]:.2e}A")

    return HalfHalfResult(
        misori_deg=np.array(mis_list, dtype=np.float64),
        lattice_max_abs_A=np.array(lat_list, dtype=np.float64),
        per_half_states=per_half,
        n_spots=int(n),
        n_splits=int(n_splits),
    )


def jackknife_spots(
    model: HEDMForwardModel,
    init: GrainState,
    observations: torch.Tensor,
    *,
    loss: Optional[SpotMatchingLoss] = None,
    phase_steps: tuple[int, int, int] = (8, 8, 6),
    reference_state: Optional[GrainState] = None,
    verbose: bool = False,
) -> JackknifeResult:
    """Per-spot leave-one-out influence.

    For each spot k = 0..N-1: drop spot k, refit, record the difference
    between the new state and the full-set reference state.

    The reference state defaults to a full-spot refinement starting from
    `init`; pass `reference_state` to reuse a precomputed fit.

    Returns
    -------
    JackknifeResult
    """
    if loss is None: loss = default_loss()
    n = observations.shape[0]
    if reference_state is None:
        reference_state = _fit_grain_spots(
            model, observations, init, loss,
            phase1_steps=10, phase2_steps=10, phase3_steps=8,
        )
    R_ref = HEDMForwardModel.euler2mat(reference_state.euler_rad)
    a_ref = float(reference_state.latc[0])

    inf_mis = np.empty(n, dtype=np.float64)
    inf_lat = np.empty(n, dtype=np.float64)
    for k in range(n):
        mask = torch.ones(n, dtype=torch.bool); mask[k] = False
        obs_k = observations[mask]
        # Warm-start from reference for speed
        state_k = _fit_grain_spots(
            model, obs_k, reference_state, loss,
            phase1_steps=phase_steps[0], phase2_steps=phase_steps[1],
            phase3_steps=phase_steps[2],
        )
        R_k = HEDMForwardModel.euler2mat(state_k.euler_rad)
        inf_mis[k] = misori_deg(R_ref, R_k)
        inf_lat[k] = abs(float(state_k.latc[0]) - a_ref)
        if verbose and (k % max(1, n // 10) == 0):
            print(f"  jackknife {k}/{n} mis={inf_mis[k]:.5f}°")

    return JackknifeResult(
        influence_misori_deg=inf_mis,
        influence_lat_A=inf_lat,
        reference_state=reference_state,
    )
