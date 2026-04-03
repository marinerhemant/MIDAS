"""Loss functions and spot matching utilities for HEDM optimization.

Two output modes:
    NF-HEDM: Image comparison losses (NCC, L2, log-ratio, SSIM)
    FF/pf-HEDM: Spot coordinate matching losses (L2, angular, Huber)

Also provides SpotAssigner for non-differentiable spot-to-spot matching
used in the FF/pf optimization loop.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  Image comparison losses (NF-HEDM)
# ---------------------------------------------------------------------------

class ImageComparisonLoss(nn.Module):
    """Loss for comparing predicted vs observed detector images.

    Used in NF-HEDM where the forward model produces full predicted images
    via Gaussian splatting and we compare to observed detector images.

    Parameters
    ----------
    mode : str
        ``"ncc"`` : Normalized Cross-Correlation (scale-invariant, recommended).
        ``"l2"``  : Mean Squared Error.
        ``"log_ratio"`` : Log-ratio loss (marginalizes unknown scale factor).
    """

    def __init__(self, mode: str = "ncc"):
        super().__init__()
        if mode not in ("ncc", "l2", "log_ratio"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode

    def forward(
        self,
        pred: torch.Tensor,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute image comparison loss.

        Parameters
        ----------
        pred : Tensor (..., H, W) or (..., F, H, W)
            Predicted images.
        obs : Tensor (same shape as pred)
            Observed images.
        mask : Tensor (same shape), optional
            Binary mask. 1 = include pixel, 0 = ignore.

        Returns
        -------
        Scalar loss tensor.
        """
        if mask is not None:
            pred = pred * mask
            obs = obs * mask

        if self.mode == "ncc":
            return self._ncc_loss(pred, obs)
        elif self.mode == "l2":
            return self._l2_loss(pred, obs)
        elif self.mode == "log_ratio":
            return self._log_ratio_loss(pred, obs)

    @staticmethod
    def _ncc_loss(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Normalized Cross-Correlation loss (1 - NCC).

        NCC = sum(pred * obs) / (||pred|| * ||obs||)
        Loss = 1 - NCC  (so 0 = perfect match)
        """
        # Flatten spatial dims for dot product
        p = pred.reshape(pred.shape[0], -1) if pred.ndim > 1 else pred.unsqueeze(0)
        o = obs.reshape(obs.shape[0], -1) if obs.ndim > 1 else obs.unsqueeze(0)

        # General flatten: merge all but keep at least 1 batch dim
        p_flat = pred.flatten()
        o_flat = obs.flatten()

        dot = torch.sum(p_flat * o_flat)
        norm_p = torch.norm(p_flat).clamp(min=1e-12)
        norm_o = torch.norm(o_flat).clamp(min=1e-12)
        ncc = dot / (norm_p * norm_o)
        return 1.0 - ncc

    @staticmethod
    def _l2_loss(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Mean Squared Error loss."""
        return torch.mean((pred - obs) ** 2)

    @staticmethod
    def _log_ratio_loss(
        pred: torch.Tensor, obs: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Log-ratio loss: ||log(pred+eps) - log(obs+eps) - mu||^2.

        Analytically marginalizes out the unknown global scaling factor
        by subtracting the mean log-ratio (mu).
        """
        log_pred = torch.log(pred + eps)
        log_obs = torch.log(obs + eps)
        diff = log_pred - log_obs
        mu = torch.mean(diff)
        return torch.mean((diff - mu) ** 2)


# ---------------------------------------------------------------------------
#  Spot coordinate matching losses (FF/pf-HEDM)
# ---------------------------------------------------------------------------

class SpotMatchingLoss(nn.Module):
    """Loss for matching predicted spot coordinates to observed spot COMs.

    Used in FF/pf-HEDM where the forward model predicts spot coordinates
    (2theta, eta, omega) and we compare to observed center-of-mass positions.

    The assignment of predicted-to-observed spots is done externally by
    ``SpotAssigner`` (non-differentiable). Given fixed assignments, this
    loss is fully differentiable w.r.t. predicted coordinates.

    Parameters
    ----------
    metric : str
        ``"l2"``    : Euclidean distance (sum of squared differences).
        ``"huber"`` : Smooth L1 (robust to outliers).
        ``"angular"``: Weighted angular distance with per-coordinate weights.
    weights : Tensor (3,), optional
        Per-coordinate weights for [2theta, eta, omega].
        Default: equal weights [1, 1, 1].
    """

    def __init__(
        self,
        metric: str = "l2",
        weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if metric not in ("l2", "huber", "angular"):
            raise ValueError(f"Unknown metric: {metric!r}")
        self.metric = metric
        if weights is not None:
            self.register_buffer("weights", weights.float())
        else:
            self.weights = None

    def forward(
        self,
        pred_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        spot_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute spot matching loss.

        Parameters
        ----------
        pred_coords : Tensor (N_matched, 3)
            Predicted spot coordinates.
        obs_coords : Tensor (N_matched, 3)
            Observed spot coordinates (same order as pred).
        spot_weights : Tensor (N_matched,), optional
            Per-spot weights (e.g., intensity-based).

        Returns
        -------
        Scalar loss tensor.
        """
        diff = pred_coords - obs_coords

        if self.weights is not None:
            diff = diff * self.weights.unsqueeze(0)

        if self.metric == "l2":
            per_spot = torch.sum(diff ** 2, dim=-1)
        elif self.metric == "huber":
            per_spot = torch.sum(
                torch.nn.functional.smooth_l1_loss(
                    diff, torch.zeros_like(diff), reduction="none"
                ),
                dim=-1,
            )
        elif self.metric == "angular":
            per_spot = torch.sum(diff ** 2, dim=-1)

        if spot_weights is not None:
            per_spot = per_spot * spot_weights

        return torch.mean(per_spot)


# ---------------------------------------------------------------------------
#  Spot assignment (non-differentiable)
# ---------------------------------------------------------------------------

class SpotAssigner:
    """Assign predicted spots to nearest observed spots.

    This is a non-differentiable operation used in the FF/pf-HEDM
    optimization loop: run periodically to update assignments, then
    use ``SpotMatchingLoss`` with fixed assignments for gradient steps.

    Matches by nearest neighbor in (2theta, eta, omega) space, optionally
    restricted to the same ring number (HKL family).

    Parameters
    ----------
    obs_coords : Tensor (N_obs, 3)
        Observed spot coordinates (2theta, eta, omega) in radians.
    obs_ring_numbers : Tensor (N_obs,), optional
        Ring number for each observed spot. If provided, matching is
        restricted to same-ring spots only.
    """

    def __init__(
        self,
        obs_coords: torch.Tensor,
        obs_ring_numbers: Optional[torch.Tensor] = None,
    ):
        self.obs_coords = obs_coords
        self.obs_ring_numbers = obs_ring_numbers

    @torch.no_grad()
    def assign(
        self,
        pred_coords: torch.Tensor,
        pred_valid: torch.Tensor,
        pred_ring_numbers: Optional[torch.Tensor] = None,
        max_distance: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find nearest observed spot for each valid predicted spot.

        Parameters
        ----------
        pred_coords : Tensor (..., K, M, 3)
            Predicted spot coordinates from ``predict_spot_coords``.
        pred_valid : Tensor (..., K, M)
            Validity mask.
        pred_ring_numbers : Tensor (M,), optional
            Ring number per HKL. If provided, matching restricted to same ring.
        max_distance : float
            Maximum matching distance in radians. Pairs beyond this are rejected.

        Returns
        -------
        pred_matched : Tensor (N_matched, 3)
            Matched predicted coordinates (detached, but index-aligned with obs_matched).
        obs_matched : Tensor (N_matched, 3)
            Matched observed coordinates.
        pred_indices : Tensor (N_matched,) of long
            Flat indices into the valid predicted spots (for gradient routing).
        """
        # Flatten predicted spots
        flat_coords = pred_coords.reshape(-1, 3)
        flat_valid = pred_valid.reshape(-1)

        # Get valid indices
        valid_idx = torch.nonzero(flat_valid > 0.5, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            empty = torch.zeros(0, 3, device=flat_coords.device)
            return empty, empty, torch.zeros(0, dtype=torch.long, device=flat_coords.device)

        valid_coords = flat_coords[valid_idx]  # (V, 3)

        # Compute distances to all observed spots
        # valid_coords: (V, 3), obs_coords: (N_obs, 3)
        # Use cdist for efficiency
        dists = torch.cdist(valid_coords, self.obs_coords)  # (V, N_obs)

        # If ring numbers provided, mask cross-ring matches
        if (pred_ring_numbers is not None and
                self.obs_ring_numbers is not None):
            # Expand ring numbers for valid spots
            # pred_ring_numbers: (M,), repeat for K*M pattern
            M = pred_ring_numbers.shape[0]
            K_total = flat_valid.shape[0] // M if M > 0 else 0
            if K_total > 0:
                flat_rings = pred_ring_numbers.repeat(K_total)
                valid_rings = flat_rings[valid_idx]  # (V,)
                ring_mismatch = (
                    valid_rings.unsqueeze(1) != self.obs_ring_numbers.unsqueeze(0)
                )
                dists = dists + ring_mismatch.float() * 1e6

        # Nearest neighbor
        min_dists, nn_idx = dists.min(dim=1)  # (V,), (V,)

        # Filter by max distance
        keep = min_dists < max_distance
        if not keep.any():
            empty = torch.zeros(0, 3, device=flat_coords.device)
            return empty, empty, torch.zeros(0, dtype=torch.long, device=flat_coords.device)

        pred_matched = valid_coords[keep]
        obs_matched = self.obs_coords[nn_idx[keep]]
        pred_indices = valid_idx[keep]

        return pred_matched, obs_matched, pred_indices
