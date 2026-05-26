"""Frame-based UQ for near-field HEDM (NF).

The NF forward model produces a 3D detector volume (frames * y * z). A
spot-style 50/50 random split is replaced with a frame split: even/odd
or random frame indices form the train and holdout subsets, each grain
is refined against its training frames, and the disagreement between
the two refined states is the half-half disagreement.

This is the NF analog of `midas_uq.spots.half_half_spots`. The
underlying differentiable forward model is the same one from paper I
(`midas_diffract.HEDMForwardModel`), used in image-output mode
(`predict_images`) instead of spot-output mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from midas_diffract import HEDMForwardModel
from midas_diffract.losses import ImageComparisonLoss

from ._common import GrainState, misori_deg, lattice_max_abs
from .spots import HalfHalfResult, JackknifeResult


def _fit_grain_nf(
    model: HEDMForwardModel,
    obs_volume: torch.Tensor,             # (F, H, W)
    init: GrainState,
    *,
    frame_indices: Optional[torch.Tensor] = None,
    loss_mode: str = "ncc",
    phase_steps: tuple[int, int, int] = (10, 10, 8),
    lbfgs_max_iter: int = 20,
) -> GrainState:
    """Three-phase L-BFGS NF refinement against a subset of frames.

    `frame_indices` selects which slices of `obs_volume` participate in
    the loss; predicted images are likewise computed for those frames
    only. If None, all frames are used.
    """
    loss_fn = ImageComparisonLoss(mode=loss_mode)
    opt_euler = init.euler_rad.clone()
    opt_latc = init.latc.clone()
    pos = init.pos if init.pos is not None else torch.zeros(
        3, dtype=opt_euler.dtype, device=opt_euler.device,
    )

    if frame_indices is None:
        frame_indices = torch.arange(obs_volume.shape[0])
    target = obs_volume[frame_indices]

    n_frames = model.n_frames
    n_pix_y = model.n_pixels_y
    n_pix_z = model.n_pixels_z

    def closure_factory(params):
        def closure():
            for p in params:
                if p.grad is not None: p.grad.zero_()
            spots = model(
                opt_euler.unsqueeze(0), pos.unsqueeze(0),
                lattice_params=opt_latc,
            )
            pred_volume = HEDMForwardModel.predict_images(
                spots, n_frames=n_frames,
                n_pixels_y=n_pix_y, n_pixels_z=n_pix_z,
            ).squeeze(0)
            pred_sub = pred_volume[frame_indices]
            l = loss_fn(pred_sub, target)
            l.backward()
            return l
        return closure

    # Phase 1: orientation
    opt_euler.requires_grad_(True); opt_latc.requires_grad_(False)
    o = torch.optim.LBFGS([opt_euler], lr=1.0, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(phase_steps[0]): o.step(closure_factory([opt_euler]))

    # Phase 2: lattice
    opt_euler.requires_grad_(False); opt_latc.requires_grad_(True)
    o = torch.optim.LBFGS([opt_latc], lr=1.0, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(phase_steps[1]): o.step(closure_factory([opt_latc]))

    # Phase 3: joint
    opt_euler.requires_grad_(True); opt_latc.requires_grad_(True)
    o = torch.optim.LBFGS([opt_euler, opt_latc], lr=0.5, max_iter=lbfgs_max_iter,
                          line_search_fn="strong_wolfe")
    for _ in range(phase_steps[2]): o.step(closure_factory([opt_euler, opt_latc]))

    return GrainState(opt_euler.detach().clone(), opt_latc.detach().clone(),
                      pos.detach().clone())


def half_half_frames(
    model: HEDMForwardModel,
    init: GrainState,
    observations: torch.Tensor,          # (F, H, W) NF image stack
    *,
    n_splits: int = 5,
    seed: int = 0,
    loss_mode: str = "ncc",
    phase_steps: tuple[int, int, int] = (10, 10, 8),
    verbose: bool = False,
) -> HalfHalfResult:
    """K random 50/50 omega-frame splits; fit each half independently.

    Frame-based analog of `half_half_spots`. The NF forward model produces
    a 3D detector volume; we split which frames participate in the loss
    and refine each half separately. Half-half disagreement is computed in
    the same (orientation, lattice) space as the spot version.

    Parameters
    ----------
    model : HEDMForwardModel
        Configured for NF (image output).
    init : GrainState
        Seed orientation/lattice.
    observations : Tensor (F, H, W)
        Observed NF image stack across F omega frames.
    n_splits, seed, loss_mode, phase_steps : as in `half_half_spots`.

    Returns
    -------
    HalfHalfResult — same shape as the FF/pf version.
    """
    F = observations.shape[0]
    half = F // 2
    g = torch.Generator(device="cpu").manual_seed(seed)

    mis_list = []; lat_list = []; per_half = []
    for k in range(n_splits):
        perm = torch.randperm(F, generator=g)
        idx_a = perm[:half]; idx_b = perm[half:2 * half]

        state_a = _fit_grain_nf(
            model, observations, init, frame_indices=idx_a,
            loss_mode=loss_mode, phase_steps=phase_steps,
        )
        state_b = _fit_grain_nf(
            model, observations, init, frame_indices=idx_b,
            loss_mode=loss_mode, phase_steps=phase_steps,
        )
        R_a = HEDMForwardModel.euler2mat(state_a.euler_rad)
        R_b = HEDMForwardModel.euler2mat(state_b.euler_rad)
        mis_list.append(misori_deg(R_a, R_b))
        lat_list.append(lattice_max_abs(state_a.latc, state_b.latc))
        per_half.append((state_a, state_b))
        if verbose:
            print(f"  NF split {k}: mis={mis_list[-1]:.5f}° "
                  f"lat={lat_list[-1]:.2e}A")

    return HalfHalfResult(
        misori_deg=np.array(mis_list, dtype=np.float64),
        lattice_max_abs_A=np.array(lat_list, dtype=np.float64),
        per_half_states=per_half,
        n_spots=int(F),    # field name is `n_spots` for symmetry; here = n_frames
        n_splits=int(n_splits),
    )


def jackknife_frames(
    model: HEDMForwardModel,
    init: GrainState,
    observations: torch.Tensor,
    *,
    loss_mode: str = "ncc",
    phase_steps: tuple[int, int, int] = (8, 8, 6),
    reference_state: Optional[GrainState] = None,
    verbose: bool = False,
) -> JackknifeResult:
    """Per-frame leave-one-out NF influence.

    NF analog of `jackknife_spots`. For each frame k in 0..F-1:
    drop frame k from the observation stack, refit, record the
    misorientation and lattice influence vs the full-stack fit.

    For grains with many frames this is expensive; we recommend
    running only on grains flagged by a half-half pass first.

    Returns
    -------
    JackknifeResult
    """
    F = observations.shape[0]
    if reference_state is None:
        reference_state = _fit_grain_nf(
            model, observations, init, frame_indices=None,
            loss_mode=loss_mode, phase_steps=(10, 10, 8),
        )
    R_ref = HEDMForwardModel.euler2mat(reference_state.euler_rad)
    a_ref = float(reference_state.latc[0])

    inf_mis = np.empty(F, dtype=np.float64)
    inf_lat = np.empty(F, dtype=np.float64)
    keep_all = torch.arange(F)
    for k in range(F):
        mask = (keep_all != k)
        idx = keep_all[mask]
        state_k = _fit_grain_nf(
            model, observations, reference_state, frame_indices=idx,
            loss_mode=loss_mode, phase_steps=phase_steps,
        )
        R_k = HEDMForwardModel.euler2mat(state_k.euler_rad)
        inf_mis[k] = misori_deg(R_ref, R_k)
        inf_lat[k] = abs(float(state_k.latc[0]) - a_ref)
        if verbose and (k % max(1, F // 10) == 0):
            print(f"  NF jackknife {k}/{F}: mis={inf_mis[k]:.5f}°")

    return JackknifeResult(
        influence_misori_deg=inf_mis,
        influence_lat_A=inf_lat,
        reference_state=reference_state,
    )
