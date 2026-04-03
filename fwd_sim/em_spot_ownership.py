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

    M-step: Update each voxel's orientation by weighted fitting, using
            ownership probabilities as weights.

    Iterate until convergence (monotonic likelihood increase guaranteed).

Reference: Whitepaper Section 7.1 (Idea 1: EM Spot-Ownership Model)

Usage:
    from em_spot_ownership import EMSpotOwnership

    em = EMSpotOwnership(forward_model, geometry)
    orientations = em.fit(observed_spots, initial_orientations, positions)
"""

import math
from typing import Optional, Tuple

import torch
import numpy as np


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
    """

    def __init__(
        self,
        forward_model,
        sigma_init: float = 0.02,
        sigma_min: float = 0.005,
        sigma_decay: float = 0.9,
        max_distance: float = 0.1,
    ):
        self.model = forward_model
        self.sigma = sigma_init
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.max_distance = max_distance

    def _predict_spots(
        self, euler_angles: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward-simulate spots for all voxels.

        Returns
        -------
        coords : (N, K, 3) predicted spot coords (2theta, eta, omega) in radians
        valid : (N, K) validity mask
        """
        from hedm_forward import HEDMForwardModel

        N = euler_angles.shape[0]
        all_coords = []
        all_valid = []

        # Process per-voxel to avoid huge batch memory
        for i in range(N):
            spots = self.model(
                euler_angles[i:i+1], positions[i:i+1]
            )
            coords, valid = HEDMForwardModel.predict_spot_coords(
                spots, space="angular"
            )
            # coords: (1, 2, M, 3) -> flatten to (2*M, 3)
            c = coords.squeeze(0).reshape(-1, 3)
            v = valid.squeeze(0).reshape(-1)
            all_coords.append(c)
            all_valid.append(v)

        return torch.stack(all_coords), torch.stack(all_valid)

    def e_step(
        self,
        obs_spots: torch.Tensor,
        pred_coords: torch.Tensor,
        pred_valid: torch.Tensor,
    ) -> torch.Tensor:
        """E-step: compute soft ownership probabilities.

        For each observed spot, compute the probability that it came
        from each voxel based on the angular distance between the
        observed position and the voxel's predicted spot positions.

        Parameters
        ----------
        obs_spots : (S, 3) observed spot coordinates
        pred_coords : (N, K, 3) predicted spots per voxel
        pred_valid : (N, K) validity mask

        Returns
        -------
        ownership : (S, N) probability matrix.
            ownership[s, v] = P(spot s belongs to voxel v).
            Rows sum to 1 (or 0 if no voxel claims the spot).
        """
        S = obs_spots.shape[0]
        N, K, _ = pred_coords.shape
        sigma2 = self.sigma ** 2

        # For each voxel, find the closest predicted spot to each observed spot
        # ownership[s, v] = max_k exp(-d(obs_s, pred_v_k)^2 / (2*sigma^2))
        ownership = torch.zeros(S, N, dtype=obs_spots.dtype, device=obs_spots.device)

        for v in range(N):
            # Valid predicted spots for this voxel
            valid_mask = pred_valid[v] > 0.5
            if not valid_mask.any():
                continue
            pred_v = pred_coords[v, valid_mask]  # (K_v, 3)

            # Pairwise distances: (S,) = min over K_v for each obs spot
            dists = torch.cdist(obs_spots, pred_v)  # (S, K_v)
            min_dists, _ = dists.min(dim=1)  # (S,)

            # Gaussian kernel (only for nearby spots)
            within_range = min_dists < self.max_distance
            log_prob = -0.5 * min_dists ** 2 / sigma2
            ownership[:, v] = torch.where(
                within_range, torch.exp(log_prob), torch.zeros_like(log_prob)
            )

        # Normalize rows to get probabilities
        row_sums = ownership.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-30)
        ownership = ownership / row_sums

        return ownership

    def m_step(
        self,
        obs_spots: torch.Tensor,
        ownership: torch.Tensor,
        euler_angles: torch.Tensor,
        positions: torch.Tensor,
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
        n_opt_steps : int
            Gradient descent steps per M-step.
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
                spots = self.model(euler_v.unsqueeze(0), positions[v:v+1])
                coords, valid = HEDMForwardModel.predict_spot_coords(
                    spots, space="angular"
                )
                pred = coords.squeeze(0).reshape(-1, 3)
                valid_flat = valid.squeeze(0).reshape(-1)
                pred_valid = pred[valid_flat > 0.5]

                if pred_valid.shape[0] == 0:
                    break

                # Weighted nearest-neighbor distance
                dists = torch.cdist(obs_v, pred_valid)  # (S_v, K_v)
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
        n_iter: int = 10,
        n_opt_steps: int = 5,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the full EM algorithm.

        Parameters
        ----------
        obs_spots : (S, 3) observed spot coordinates (2theta, eta, omega) rad
        initial_euler : (N, 3) initial Euler angles (radians)
        positions : (N, 3) voxel positions (micrometers, fixed)
        n_iter : int
            Number of EM iterations.
        n_opt_steps : int
            Gradient steps per M-step per voxel.
        lr : float
            Learning rate for M-step optimizer.
        verbose : bool

        Returns
        -------
        euler_angles : (N, 3) final Euler angles
        ownership : (S, N) final ownership matrix
        """
        euler = initial_euler.clone()

        for it in range(n_iter):
            # Predict spots for all voxels
            pred_coords, pred_valid = self._predict_spots(euler, positions)

            # E-step
            ownership = self.e_step(obs_spots, pred_coords, pred_valid)

            # Log-likelihood (for monitoring convergence)
            with torch.no_grad():
                # Approximate: sum of log(sum_v P(s|v))
                row_max = ownership.max(dim=1).values
                n_assigned = (row_max > 0.1).sum().item()
                avg_confidence = row_max.mean().item()

            if verbose:
                print(f"  EM iter {it:3d}: sigma={self.sigma:.4f}, "
                      f"assigned={n_assigned}/{obs_spots.shape[0]}, "
                      f"avg_confidence={avg_confidence:.4f}")

            # M-step
            euler = self.m_step(
                obs_spots, ownership, euler, positions,
                n_opt_steps=n_opt_steps, lr=lr,
            )

            # Anneal sigma
            self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)

        # Final E-step with updated orientations
        pred_coords, pred_valid = self._predict_spots(euler, positions)
        ownership = self.e_step(obs_spots, pred_coords, pred_valid)

        return euler, ownership
