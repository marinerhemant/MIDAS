"""Shared utilities: misorientation, fitter wrapper, type aliases."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from midas_diffract import HEDMForwardModel, SpotMatchingLoss

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def misori_deg(R_a: torch.Tensor, R_b: torch.Tensor) -> float:
    """Misorientation between two rotation matrices, in degrees.

    No symmetry reduction. For small misorientations between bit-near
    rotations this is monotone in the underlying so(3) distance and is
    fine for use as a half-half disagreement metric in noisy regimes.
    For cubic (sg 195-230) and hexagonal (sg 168-194) crystals where
    the 24 / 12 symmetry-equivalent orientations can be reached by
    different half-fits, use :func:`misori_deg_sym` instead.
    """
    trace = float((R_a.T @ R_b).diagonal().sum())
    cos_t = max(-1.0, min(1.0, (trace - 1) / 2))
    return math.degrees(math.acos(cos_t))


def misori_deg_sym(R_a: torch.Tensor, R_b: torch.Tensor,
                   space_group: int) -> float:
    """Symmetry-reduced misorientation in degrees.

    Wraps :func:`midas_stress.orientation.misorientation_om`, which
    enumerates the proper rotation symmetries of the given space group
    and returns the *minimum* misorientation across symmetry operators
    on the disorientation set. Use this for the half-half disagreement
    metric in cubic, hexagonal, or tetragonal crystals.

    Falls back to plain :func:`misori_deg` if midas-stress is not
    available, with a warning.
    """
    try:
        from midas_stress.orientation import misorientation_om
        # midas_stress returns radians; expects 3x3 or flat-9 OMs.
        angle_rad, _ = misorientation_om(R_a, R_b, int(space_group))
        if hasattr(angle_rad, "item"):
            angle_rad = float(angle_rad.item())
        return math.degrees(float(angle_rad))
    except ImportError:
        import warnings
        warnings.warn(
            "midas-stress not installed; falling back to non-symmetric "
            "misorientation. Install midas-stress for cubic / hexagonal "
            "symmetry-aware UQ.",
        )
        return misori_deg(R_a, R_b)


def lattice_max_abs(latc_a: torch.Tensor, latc_b: torch.Tensor) -> float:
    """Max absolute disagreement in the three lattice axes (A)."""
    return float((latc_a - latc_b)[:3].abs().max())


def euler2mat_safe(euler: torch.Tensor) -> torch.Tensor:
    """Bunge Euler -> rotation matrix, autograd-safe wrapper."""
    return HEDMForwardModel.euler2mat(euler)


def default_loss(wavelength_aware: bool = False) -> SpotMatchingLoss:
    """Default spot-matching loss (L2, equal coordinate weights)."""
    return SpotMatchingLoss(metric="l2")


@dataclass
class GrainState:
    """Minimal grain state needed by a refinement run.

    `euler_rad` and `latc` are torch tensors (float64 recommended).
    `pos` is in lab-frame microns; defaults to origin if not provided.
    """
    euler_rad: torch.Tensor
    latc: torch.Tensor
    pos: Optional[torch.Tensor] = None

    def clone(self) -> "GrainState":
        return GrainState(
            self.euler_rad.detach().clone(),
            self.latc.detach().clone(),
            None if self.pos is None else self.pos.detach().clone(),
        )

    def to(self, *, dtype=None, device=None) -> "GrainState":
        def conv(t):
            if t is None: return None
            return t.to(dtype=dtype) if device is None else t.to(dtype=dtype, device=device)
        return GrainState(conv(self.euler_rad), conv(self.latc), conv(self.pos))


def _associate(pred_valid: torch.Tensor, observed: torch.Tensor,
               max_dist: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-neighbour observed -> predicted association, threshold-filtered.

    Returns (pred_matched, obs_matched) — both ordered so that index k of
    each refers to the same observed->predicted pair, with shape (M, 3)
    where M is the number of pairs surviving `max_dist`.
    """
    dists = torch.cdist(observed, pred_valid)
    min_d, nn = dists.min(dim=1)
    keep = min_d < max_dist
    return pred_valid[nn[keep]], observed[keep]
