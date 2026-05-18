"""Downstream HEDM strain coupling — feeds per-grain strain residual back
into the calibration objective.

This is the "calibration tested by science output" milestone (M6).  It
requires a differentiable HEDM grain refinement module — work in flight in
``midas_diffract`` / ``midas_grain_odf``.  The interface here uses a
user-supplied callable so the package can ship before the differentiable
HEDM forward model lands.

API:

    downstream_loss(calibration_unpacked, hedm_evaluator) -> scalar

where ``hedm_evaluator(unpacked) -> tensor`` is the user-supplied
differentiable HEDM forward model (typically returning per-grain strain
tensors or a scalar fitness metric).
"""
from __future__ import annotations

from typing import Callable, Dict

import torch


def downstream_strain_loss(
    unpacked: Dict[str, torch.Tensor],
    hedm_evaluator: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
) -> torch.Tensor:
    """Wrap a downstream HEDM evaluator as a calibration loss term."""
    return hedm_evaluator(unpacked)


__all__ = ["downstream_strain_loss"]
