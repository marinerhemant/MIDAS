"""Modality-dispatching public API.

Provides `half_half(...)` and `jackknife(...)` entry points that accept
the same `mode` parameter ('ff', 'pf', 'nf') and dispatch to the
spot-based or frame-based implementation. The spot path covers FF and
pf-HEDM (both produce a per-grain spot list); the frame path covers NF.
"""
from __future__ import annotations

from typing import Optional

import torch

from midas_diffract import HEDMForwardModel, SpotMatchingLoss

from ._common import GrainState
from .spots import (
    half_half_spots, jackknife_spots,
    HalfHalfResult, JackknifeResult,
)
from .images import half_half_frames, jackknife_frames


_SPOT_MODES = {"ff", "pf"}
_IMAGE_MODES = {"nf"}


def half_half(
    model: HEDMForwardModel,
    init: GrainState,
    observations: torch.Tensor,
    *,
    mode: str = "ff",
    **kwargs,
) -> HalfHalfResult:
    """Modality-aware half-half UQ.

    Parameters
    ----------
    model : HEDMForwardModel
    init : GrainState
        Seed grain state (orientation, lattice, position).
    observations : Tensor
        For mode in {'ff', 'pf'}: shape (N, 3), angular spot coords
        (2theta, eta, omega) in radians.
        For mode='nf': shape (F, H, W), the observed NF image stack.
    mode : {'ff', 'pf', 'nf'}, default 'ff'
        Output / observation modality. 'ff' and 'pf' share the spot-based
        implementation; 'nf' uses the frame-based implementation.
    **kwargs
        Forwarded to the underlying modality-specific function:
        - For 'ff'/'pf': `n_splits, seed, loss, phase_steps,
          multi_restart_K, verbose`.
        - For 'nf': `n_splits, seed, loss_mode, phase_steps, verbose`.

    Returns
    -------
    HalfHalfResult
    """
    mode = mode.lower()
    if mode in _SPOT_MODES:
        return half_half_spots(model, init, observations, **kwargs)
    if mode in _IMAGE_MODES:
        return half_half_frames(model, init, observations, **kwargs)
    raise ValueError(f"Unknown mode: {mode!r}. Use one of "
                     f"{_SPOT_MODES | _IMAGE_MODES}.")


def jackknife(
    model: HEDMForwardModel,
    init: GrainState,
    observations: torch.Tensor,
    *,
    mode: str = "ff",
    **kwargs,
) -> JackknifeResult:
    """Modality-aware jackknife per-observation influence.

    `observations` is interpreted as per `half_half`; the modality
    determines whether 'leave-one-out' is across spots or frames.

    Returns
    -------
    JackknifeResult
    """
    mode = mode.lower()
    if mode in _SPOT_MODES:
        return jackknife_spots(model, init, observations, **kwargs)
    if mode in _IMAGE_MODES:
        return jackknife_frames(model, init, observations, **kwargs)
    raise ValueError(f"Unknown mode: {mode!r}. Use one of "
                     f"{_SPOT_MODES | _IMAGE_MODES}.")
