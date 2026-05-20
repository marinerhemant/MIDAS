"""Soft beam-attribution weight constructors for ``compare_spots``.

A soft beam-weight function maps the absolute distance ``|s_proj − scan_pos|``
(µm) between a voxel's projected scan position and an observed spot's
scan-position-of-origin to a continuous weight in ``[0, 1]``.

The binary back-compat is the ``hard_window_fn``::

    hard_window_fn(tol_um)(d) = (d < tol_um).to(float)

which reproduces the existing ``scan_pos_tol_um`` filter exactly.  The
``soft_top_hat_fn`` and ``soft_gaussian_fn`` constructors provide
smoothly differentiable alternatives that produce fractional matches at
beam edges — this directly suppresses pf-HEDM salt-pepper artifacts at
grain boundaries by attributing each spot to multiple neighbouring
voxels with appropriate weight.

These callables are pure torch ops — autograd-differentiable in the
distance argument (for downstream beam-width refinement) and
device-portable (CPU / CUDA / MPS).
"""
from __future__ import annotations

import math
from typing import Callable

import torch


__all__ = [
    "hard_window_fn",
    "soft_top_hat_fn",
    "soft_gaussian_fn",
]


def hard_window_fn(tol_um: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Reproduce the legacy binary ``scan_pos_tol_um`` filter as a soft fn.

    Useful for back-compat testing and for opting into the soft-mode code
    path without changing semantics.
    """
    tol = float(tol_um)

    def f(d: torch.Tensor) -> torch.Tensor:
        return (d < tol).to(d.dtype)
    return f


def soft_top_hat_fn(
    beam_width_um: float, *, fall_off_um: float = 0.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Trapezoidal beam profile: weight 1 inside ``[0, beam/2)``, linearly
    decays to 0 over ``fall_off_um`` past the edge.

    ``fall_off_um=0`` recovers the strict top-hat (= binary mask of half-width
    ``beam/2``); positive ``fall_off_um`` introduces a smooth ramp at the
    edge for stable gradients during beam-width refinement.
    """
    half_w = 0.5 * float(beam_width_um)
    fo = float(fall_off_um)

    def f(d: torch.Tensor) -> torch.Tensor:
        if fo <= 0.0:
            return (d < half_w).to(d.dtype)
        # 1.0 inside [0, half_w]; linear ramp 1->0 over [half_w, half_w+fo]
        ramp = torch.clamp((half_w + fo - d) / fo, min=0.0, max=1.0)
        inside = (d <= half_w).to(d.dtype)
        return inside + (1.0 - inside) * ramp
    return f


def soft_gaussian_fn(
    fwhm_um: float,
    *,
    truncate_at: float = 0.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Gaussian beam profile (peak normalized to 1): ``exp(-d² / (2σ²))``
    with ``σ = fwhm / (2 √(2 ln 2))``.

    ``truncate_at`` (µm) optionally zeros out the tails beyond that
    distance — keeps the candidate set finite without further changes.
    """
    sigma = float(fwhm_um) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    cut = float(truncate_at)

    def f(d: torch.Tensor) -> torch.Tensor:
        w = torch.exp(-(d * d) / (2.0 * sigma * sigma))
        if cut > 0.0:
            w = torch.where(d <= cut, w, torch.zeros_like(w))
        return w
    return f
