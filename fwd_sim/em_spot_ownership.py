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
    max_distance : float
        Only compute ownership for (spot, voxel) pairs within this
        distance (radians). Prunes distant pairs for efficiency.
    coord_weights : tuple of 3 floats or None
        Per-coordinate weights (w_2theta, w_eta, w_omega) for distance.
        If None, uniform weighting (1, 1, 1). Typical: (1/tol_2theta,
        1/tol_eta, 1/tol_omega) to normalize by tolerance.
    """

    def __init__(
        self,
        forward_model,
        sigma_init: float = 0.02,
        sigma_min: float = 0.005,
        sigma_decay: float = 0.9,
        max_distance: float = 0.1,
        coord_weights: Optional[Tuple[float, float, float]] = None,
    ):
        self.model = forward_model
        self.sigma = sigma_init
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.max_distance = max_distance
        self.coord_weights = coord_weights

    def _predict_spots(
        self, euler_angles: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward-simulate spots for all voxels/grains.

        Returns
        -------
        coords : (N, K, 3) predicted spot coords (2theta, eta, omega) in radians
        valid : (N, K) validity mask
        ring_indices : (N, K) ring index for each predicted spot
        """
        from hedm_forward import HEDMForwardModel

        N = euler_angles.shape[0]
        all_coords = []
        all_valid = []
        all_rings = []

        for i in range(N):
            spots = self.model(
                euler_angles[i:i + 1], positions[i:i + 1]
            )
            coords, valid = HEDMForwardModel.predict_spot_coords(
                spots, space="angular"
            )
            # coords: (1, 2, M, 3) -> flatten to (2*M, 3)
            c = coords.squeeze(0).reshape(-1, 3)
            v = valid.squeeze(0).reshape(-1)

            # Extract ring indices from the model's HKL metadata
            # Each HKL has a ring index; the forward model produces 2 solutions
            # (±omega) per HKL, so ring indices are doubled
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

        For each observed spot, compute the probability that it came
        from each voxel based on the angular distance between the
        observed position and the voxel's predicted spot positions.
        Matching is constrained to spots on the same diffraction ring.

        Parameters
        ----------
        obs_spots : (S, 3) observed spot coordinates (2theta, eta, omega)
        pred_coords : (N, K, 3) predicted spots per voxel
        pred_valid : (N, K) validity mask
        obs_rings : (S,) int ring indices for observed spots (optional)
        pred_rings : (N, K) int ring indices for predicted spots (optional)

        Returns
        -------
        ownership : (S, N) probability matrix.
            ownership[s, v] = P(spot s belongs to voxel v).
            Rows sum to 1 (or 0 if no voxel claims the spot).
        hkl_assignments : (S, N) int tensor.
            For each (spot, grain) pair, the index of the closest predicted
            spot. Used for sinogram HKL-slot mapping.
        """
        S = obs_spots.shape[0]
        N, K, _ = pred_coords.shape
        sigma2 = self.sigma ** 2
        use_rings = obs_rings is not None and pred_rings is not None

        ownership = torch.zeros(S, N, dtype=obs_spots.dtype, device=obs_spots.device)
        hkl_assignments = torch.zeros(S, N, dtype=torch.long, device=obs_spots.device)

        # Prepare coordinate weights
        if self.coord_weights is not None:
            w = torch.tensor(self.coord_weights, dtype=obs_spots.dtype,
                             device=obs_spots.device)
        else:
            w = None

        for v in range(N):
            valid_mask = pred_valid[v] > 0.5
            if not valid_mask.any():
                continue
            pred_v = pred_coords[v, valid_mask]  # (K_v, 3)
            # Map from filtered index back to original K-index
            valid_indices = torch.where(valid_mask)[0]  # (K_v,)

            if use_rings:
                pred_rings_v = pred_rings[v, valid_mask]  # (K_v,)
                # Process per-ring to avoid cross-ring matching
                unique_rings = obs_rings.unique()
                min_dists_all = torch.full((S,), float('inf'),
                                           dtype=obs_spots.dtype,
                                           device=obs_spots.device)
                best_idx_all = torch.zeros(S, dtype=torch.long,
                                           device=obs_spots.device)

                for ring in unique_rings:
                    obs_mask = obs_rings == ring
                    pred_ring_mask = pred_rings_v == ring
                    if not obs_mask.any() or not pred_ring_mask.any():
                        continue

                    obs_ring = obs_spots[obs_mask]  # (S_r, 3)
                    pred_ring = pred_v[pred_ring_mask]  # (K_r, 3)
                    pred_ring_orig_idx = valid_indices[pred_ring_mask]

                    if w is not None:
                        dists = _angular_distance_matrix_weighted(obs_ring, pred_ring, w)
                    else:
                        dists = _angular_distance_matrix(obs_ring, pred_ring)

                    ring_min_dists, ring_min_idx = dists.min(dim=1)

                    # Update global min for spots on this ring
                    better = ring_min_dists < min_dists_all[obs_mask]
                    obs_indices = torch.where(obs_mask)[0]
                    for local_i, global_i in enumerate(obs_indices):
                        if better[local_i]:
                            min_dists_all[global_i] = ring_min_dists[local_i]
                            best_idx_all[global_i] = pred_ring_orig_idx[ring_min_idx[local_i]]
            else:
                # No ring filtering — original behavior with wrapping fix
                if w is not None:
                    dists = _angular_distance_matrix_weighted(obs_spots, pred_v, w)
                else:
                    dists = _angular_distance_matrix(obs_spots, pred_v)
                min_dists_all, min_idx_local = dists.min(dim=1)
                best_idx_all = valid_indices[min_idx_local]

            # Gaussian kernel (only for nearby spots)
            within_range = min_dists_all < self.max_distance
            log_prob = -0.5 * min_dists_all ** 2 / sigma2
            ownership[:, v] = torch.where(
                within_range, torch.exp(log_prob), torch.zeros_like(log_prob)
            )
            hkl_assignments[:, v] = best_idx_all

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
