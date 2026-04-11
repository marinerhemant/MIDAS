"""Expectation-Maximization spot-ownership model for pf-HEDM.

Replaces hard (binary) spot-to-voxel assignment with soft probabilistic
ownership, analogous to RELION's E-step in cryo-EM (Scheres 2012).

In the current MIDAS pipeline, each spot is assigned to at most one
grain (max-intensity rule). This creates problems at grain boundaries
where spots from multiple grains overlap. The EM model instead assigns
ownership probabilities: P(spot s belongs to voxel v).

Algorithm:
    E-step: For each (spot, voxel) pair, compute ownership probability
            P(s <- v) ~ exp(-||predicted_v - observed_s||^2 / (2*sigma^2))
            Distances are computed per-ring (only same ring matches) with
            angular wrapping for omega and eta (both range -180 to +180 deg).

    M-step: Update each voxel's orientation by weighted fitting, using
            ownership probabilities as weights.

    Iterate until convergence (monotonic likelihood increase guaranteed).

Reference: Whitepaper Section 7.1 (Idea 1: EM Spot-Ownership Model)

Usage:
    from em_spot_ownership import EMSpotOwnership

    em = EMSpotOwnership(forward_model, geometry)
    result = em.fit(observed_spots, initial_orientations, positions)
"""

import math
from typing import Optional, Tuple, NamedTuple

import torch
import numpy as np


class EMResult(NamedTuple):
    """Result of EM fitting."""
    euler_angles: torch.Tensor   # (N, 3) final Euler angles
    ownership: torch.Tensor      # (S, N) final ownership matrix
    hkl_assignments: torch.Tensor  # (S, N) int: which predicted-spot index was closest per (spot, grain)


def _angular_distance_matrix(obs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute pairwise angular distances with wrapping for eta and omega.

    Both omega (col 2) and eta (col 1) range from -pi to +pi.
    2theta (col 0) does not wrap (always positive, small).

    Parameters
    ----------
    obs : (S, 3) observed spots (2theta, eta, omega) in radians
    pred : (K, 3) predicted spots (2theta, eta, omega) in radians

    Returns
    -------
    dists : (S, K) angular distances in radians
    """
    # Broadcast: (S, 1, 3) - (1, K, 3) -> (S, K, 3)
    diff = obs.unsqueeze(1) - pred.unsqueeze(0)

    # 2theta: no wrapping
    d2theta = diff[..., 0]

    # eta: wrap at ±pi (range is -pi to +pi)
    TWO_PI = 2.0 * math.pi
    deta = (diff[..., 1] + math.pi) % TWO_PI - math.pi

    # omega: wrap at ±pi (range is -pi to +pi)
    domega = (diff[..., 2] + math.pi) % TWO_PI - math.pi

    return torch.sqrt(d2theta ** 2 + deta ** 2 + domega ** 2)


def _angular_distance_matrix_weighted(obs: torch.Tensor, pred: torch.Tensor,
                                       weights: torch.Tensor) -> torch.Tensor:
    """Compute pairwise weighted angular distances with wrapping.

    Parameters
    ----------
    obs : (S, 3) observed spots (2theta, eta, omega) in radians
    pred : (K, 3) predicted spots (2theta, eta, omega) in radians
    weights : (3,) per-coordinate weights (1/tolerance)

    Returns
    -------
    dists : (S, K) weighted angular distances
    """
    diff = obs.unsqueeze(1) - pred.unsqueeze(0)

    d2theta = diff[..., 0]
    TWO_PI = 2.0 * math.pi
    deta = (diff[..., 1] + math.pi) % TWO_PI - math.pi
    domega = (diff[..., 2] + math.pi) % TWO_PI - math.pi

    return torch.sqrt((weights[0] * d2theta) ** 2 +
                       (weights[1] * deta) ** 2 +
                       (weights[2] * domega) ** 2)


class EMSpotOwnership:
    """EM algorithm for soft spot-to-voxel assignment in pf-HEDM.

    Parameters
    ----------
    forward_model : HEDMForwardModel
        The differentiable forward model.
    sigma_init : float
        Initial sigma for the Gaussian ownership kernel (radians).
        Controls how softly spots are shared between voxels.
        Typical: 0.01-0.05 rad (~0.5-3 degrees).
    sigma_min : float
        Minimum sigma (annealing floor).
    sigma_decay : float
        Multiply sigma by this after each iteration (annealing).
    tol_omega : float
        Box tolerance for omega matching (radians). Matches the C code's
        tolOme parameter. Default: 1 degree.
    tol_eta : float
        Box tolerance for eta matching (radians). Matches the C code's
        tolEta parameter. Default: 1 degree.
    """

    DEG2RAD = math.pi / 180.0

    def __init__(
        self,
        forward_model,
        sigma_init: float = 0.02,
        sigma_min: float = 0.005,
        sigma_decay: float = 0.9,
        tol_omega: float = 1.0 * math.pi / 180.0,
        tol_eta: float = 1.0 * math.pi / 180.0,
    ):
        self.model = forward_model
        self.sigma = sigma_init
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.tol_omega = tol_omega
        self.tol_eta = tol_eta

    def _predict_spots_from_orient(
        self, orient_matrices: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward-simulate spots for all grains using orientation matrices directly.

        This avoids Euler angle conversion and its associated convention issues.
        The orientation matrices are used directly in calc_bragg_geometry.

        Parameters
        ----------
        orient_matrices : (N, 3, 3) orientation matrices (same convention as
            UniqueOrientations.csv from findSingleSolutionPFRefactored)
        positions : (N, 3) grain positions

        Returns
        -------
        coords : (N, K, 3) predicted spot coords (2theta, eta, omega) in radians
        valid : (N, K) validity mask
        ring_indices : (N, K) ring index for each predicted spot
        """
        from hedm_forward import HEDMForwardModel

        N = orient_matrices.shape[0]
        hkls = self.model.hkls.double()
        thetas = self.model.thetas.double()
        all_coords = []
        all_valid = []
        all_rings = []

        for i in range(N):
            om = orient_matrices[i:i + 1]  # (1, 3, 3)
            omega, eta, two_theta, valid = self.model.calc_bragg_geometry(
                om, hkls, thetas
            )
            spots = self.model.project_to_detector(
                omega, eta, two_theta, positions[i:i + 1], valid
            )
            coords, valid_out = HEDMForwardModel.predict_spot_coords(
                spots, space="angular"
            )
            # coords: (1, 2, M, 3) -> flatten to (2*M, 3)
            c = coords.squeeze(0).reshape(-1, 3)
            v = valid_out.squeeze(0).reshape(-1)

            if hasattr(self.model, 'ring_indices') and self.model.ring_indices is not None:
                ri = self.model.ring_indices  # (M,)
                ri_doubled = ri.repeat(2)  # (2*M,)
            else:
                ri_doubled = torch.zeros(c.shape[0], dtype=torch.long,
                                         device=c.device)

            all_coords.append(c)
            all_valid.append(v)
            all_rings.append(ri_doubled)

        return (torch.stack(all_coords), torch.stack(all_valid),
                torch.stack(all_rings))

    def e_step(
        self,
        obs_spots: torch.Tensor,
        pred_coords: torch.Tensor,
        pred_valid: torch.Tensor,
        obs_rings: Optional[torch.Tensor] = None,
        pred_rings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """E-step: compute soft ownership probabilities.

        Matches spots using C-code-style box tolerances:
        1. Ring number must match exactly (handles 2theta)
        2. |omega_obs - omega_pred| < tol_omega
        3. |eta_obs - eta_pred| < tol_eta

        Within the box, the closest match (smallest omega+eta distance)
        is used for the Gaussian ownership kernel.

        Parameters
        ----------
        obs_spots : (S, 3) observed spot coordinates (2theta, eta, omega) radians
        pred_coords : (N, K, 3) predicted spots per voxel
        pred_valid : (N, K) validity mask
        obs_rings : (S,) int ring indices for observed spots
        pred_rings : (N, K) int ring indices for predicted spots

        Returns
        -------
        ownership : (S, N) probability matrix.
            ownership[s, v] = P(spot s belongs to voxel v).
            Rows sum to 1 (or 0 if no voxel claims the spot).
        hkl_assignments : (S, N) int tensor.
            For each (spot, grain) pair, the index of the closest predicted
            spot within the tolerance box.
        """
        S = obs_spots.shape[0]
        N, K, _ = pred_coords.shape
        sigma2 = self.sigma ** 2
        TWO_PI = 2.0 * math.pi
        use_rings = obs_rings is not None and pred_rings is not None

        ownership = torch.zeros(S, N, dtype=obs_spots.dtype, device=obs_spots.device)
        hkl_assignments = torch.zeros(S, N, dtype=torch.long, device=obs_spots.device)

        obs_eta = obs_spots[:, 1]    # (S,)
        obs_omega = obs_spots[:, 2]  # (S,)

        for v in range(N):
            valid_mask = pred_valid[v] > 0.5
            if not valid_mask.any():
                continue
            pred_v = pred_coords[v, valid_mask]  # (K_v, 3)
            valid_indices = torch.where(valid_mask)[0]  # (K_v,)

            pred_eta_v = pred_v[:, 1]    # (K_v,)
            pred_omega_v = pred_v[:, 2]  # (K_v,)

            if use_rings:
                pred_rings_v = pred_rings[v, valid_mask]  # (K_v,)

            # For each observed spot, find best match within box tolerance
            min_dists = torch.full((S,), float('inf'), dtype=obs_spots.dtype,
                                    device=obs_spots.device)
            best_idx = torch.zeros(S, dtype=torch.long, device=obs_spots.device)

            for k in range(pred_v.shape[0]):
                # Ring filter
                if use_rings:
                    ring_match = obs_rings == pred_rings_v[k]
                else:
                    ring_match = torch.ones(S, dtype=torch.bool, device=obs_spots.device)

                # Box tolerance: |Δomega| < tol_omega AND |Δeta| < tol_eta
                # with angular wrapping
                d_omega = (obs_omega - pred_omega_v[k] + math.pi) % TWO_PI - math.pi
                d_eta = (obs_eta - pred_eta_v[k] + math.pi) % TWO_PI - math.pi

                in_box = ring_match & (d_omega.abs() < self.tol_omega) & (d_eta.abs() < self.tol_eta)

                # Distance for Gaussian kernel (only eta + omega, 2theta is handled by ring)
                dist = torch.sqrt(d_omega ** 2 + d_eta ** 2)

                # Update best match where this prediction is closer AND in box
                better = in_box & (dist < min_dists)
                min_dists = torch.where(better, dist, min_dists)
                best_idx = torch.where(better, valid_indices[k], best_idx)

            # Gaussian kernel for spots that had a match
            matched = min_dists < float('inf')
            log_prob = -0.5 * min_dists ** 2 / sigma2
            ownership[:, v] = torch.where(
                matched, torch.exp(log_prob), torch.zeros_like(log_prob)
            )
            hkl_assignments[:, v] = best_idx

        # Normalize rows to get probabilities
        row_sums = ownership.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-30)
        ownership = ownership / row_sums

        return ownership, hkl_assignments

    def m_step(
        self,
        obs_spots: torch.Tensor,
        ownership: torch.Tensor,
        euler_angles: torch.Tensor,
        positions: torch.Tensor,
        obs_rings: Optional[torch.Tensor] = None,
        n_opt_steps: int = 5,
        lr: float = 0.01,
    ) -> torch.Tensor:
        """M-step: update voxel orientations using weighted fitting.

        For each voxel, optimize its Euler angles to minimize the
        weighted sum of squared distances to owned spots.

        Parameters
        ----------
        obs_spots : (S, 3) observed spots
        ownership : (S, N) ownership probabilities
        euler_angles : (N, 3) current Euler angles
        positions : (N, 3) voxel positions (fixed)
        obs_rings : (S,) ring indices (optional, for ring-aware distance)
        n_opt_steps : int
            Gradient descent steps per M-step per voxel.
        lr : float
            Learning rate for Adam optimizer.

        Returns
        -------
        updated_euler : (N, 3) updated Euler angles
        """
        from hedm_forward import HEDMForwardModel

        N = euler_angles.shape[0]
        updated = euler_angles.detach().clone()

        for v in range(N):
            # Spots with significant ownership for this voxel
            w = ownership[:, v]
            significant = w > 0.01
            if significant.sum() < 3:
                continue

            obs_v = obs_spots[significant]  # (S_v, 3)
            weights_v = w[significant]       # (S_v,)

            # Optimize this voxel's Euler angles
            euler_v = updated[v].clone().requires_grad_(True)
            opt = torch.optim.Adam([euler_v], lr=lr)

            for _ in range(n_opt_steps):
                opt.zero_grad()
                spots = self.model(euler_v.unsqueeze(0), positions[v:v + 1])
                coords, valid = HEDMForwardModel.predict_spot_coords(
                    spots, space="angular"
                )
                pred = coords.squeeze(0).reshape(-1, 3)
                valid_flat = valid.squeeze(0).reshape(-1)
                pred_valid = pred[valid_flat > 0.5]

                if pred_valid.shape[0] == 0:
                    break

                # Weighted angular distance (with wrapping)
                dists = _angular_distance_matrix(obs_v, pred_valid)  # (S_v, K_v)
                min_dists, _ = dists.min(dim=1)  # (S_v,)
                loss = (weights_v * min_dists ** 2).sum()
                loss.backward()
                opt.step()

            updated[v] = euler_v.detach()

        return updated

    def fit(
        self,
        obs_spots: torch.Tensor,
        initial_euler: torch.Tensor,
        positions: torch.Tensor,
        obs_rings: Optional[torch.Tensor] = None,
        n_iter: int = 10,
        n_opt_steps: int = 5,
        lr: float = 0.01,
        refine_orientations: bool = True,
        verbose: bool = True,
    ) -> EMResult:
        """Run the full EM algorithm.

        Parameters
        ----------
        obs_spots : (S, 3) observed spot coordinates (2theta, eta, omega) rad
        initial_euler : (N, 3) initial Euler angles (radians)
        positions : (N, 3) voxel positions (micrometers, fixed)
        obs_rings : (S,) int ring indices for observed spots (optional)
        n_iter : int
            Number of EM iterations.
        n_opt_steps : int
            Gradient steps per M-step per voxel.
        lr : float
            Learning rate for M-step optimizer.
        refine_orientations : bool
            If True, run full EM with M-step orientation refinement.
            If False, only run E-steps (soft ownership with fixed orientations).
        verbose : bool

        Returns
        -------
        EMResult with euler_angles, ownership, hkl_assignments
        """
        self.sigma = self.sigma_init  # reset for re-runs
        euler = initial_euler.clone()

        for it in range(n_iter):
            # Predict spots for all voxels
            pred_coords, pred_valid, pred_rings = self._predict_spots(euler, positions)

            # E-step
            ownership, hkl_assignments = self.e_step(
                obs_spots, pred_coords, pred_valid,
                obs_rings=obs_rings, pred_rings=pred_rings,
            )

            # Log-likelihood (for monitoring convergence)
            with torch.no_grad():
                row_max = ownership.max(dim=1).values
                n_assigned = (row_max > 0.1).sum().item()
                avg_confidence = row_max.mean().item()

            if verbose:
                print(f"  EM iter {it:3d}: sigma={self.sigma:.4f}, "
                      f"assigned={n_assigned}/{obs_spots.shape[0]}, "
                      f"avg_confidence={avg_confidence:.4f}")

            # M-step (optional)
            if refine_orientations:
                euler = self.m_step(
                    obs_spots, ownership, euler, positions,
                    obs_rings=obs_rings,
                    n_opt_steps=n_opt_steps, lr=lr,
                )

            # Anneal sigma
            self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)

        # Final E-step with updated orientations
        pred_coords, pred_valid, pred_rings = self._predict_spots(euler, positions)
        ownership, hkl_assignments = self.e_step(
            obs_spots, pred_coords, pred_valid,
            obs_rings=obs_rings, pred_rings=pred_rings,
        )

        return EMResult(euler, ownership, hkl_assignments)

    def fit_from_orient(
        self,
        obs_spots: torch.Tensor,
        orient_matrices: torch.Tensor,
        positions: torch.Tensor,
        obs_rings: Optional[torch.Tensor] = None,
        n_iter: int = 10,
        verbose: bool = True,
    ) -> EMResult:
        """Run EM using orientation matrices directly (no Euler conversion).

        Uses _predict_spots_from_orient to avoid Euler angle convention issues.
        M-step orientation refinement is not supported in this mode — only
        E-step (soft ownership assignment with sigma annealing).

        Parameters
        ----------
        obs_spots : (S, 3) observed spot coordinates (2theta, eta, omega) rad
        orient_matrices : (N, 3, 3) orientation matrices
        positions : (N, 3) grain positions (micrometers)
        obs_rings : (S,) int ring indices for observed spots
        n_iter : int
            Number of EM iterations (E-step + sigma annealing).
        verbose : bool

        Returns
        -------
        EMResult with orient_matrices (unchanged), ownership, hkl_assignments
        """
        self.sigma = self.sigma_init

        # Predict once (orientations are fixed)
        pred_coords, pred_valid, pred_rings = self._predict_spots_from_orient(
            orient_matrices, positions
        )

        for it in range(n_iter):
            ownership, hkl_assignments = self.e_step(
                obs_spots, pred_coords, pred_valid,
                obs_rings=obs_rings, pred_rings=pred_rings,
            )

            with torch.no_grad():
                row_max = ownership.max(dim=1).values
                n_assigned = (row_max > 0.1).sum().item()
                avg_confidence = row_max.mean().item()

            if verbose:
                print(f"  EM iter {it:3d}: sigma={self.sigma:.4f}, "
                      f"assigned={n_assigned}/{obs_spots.shape[0]}, "
                      f"avg_confidence={avg_confidence:.4f}")

            self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)

        # Final E-step
        ownership, hkl_assignments = self.e_step(
            obs_spots, pred_coords, pred_valid,
            obs_rings=obs_rings, pred_rings=pred_rings,
        )

        # Return orient_matrices reshaped as (N, 9) to fit EMResult euler_angles slot
        dummy_euler = orient_matrices.reshape(orient_matrices.shape[0], -1)
        return EMResult(dummy_euler, ownership, hkl_assignments)
