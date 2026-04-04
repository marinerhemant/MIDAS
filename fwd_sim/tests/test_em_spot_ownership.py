"""Tests for EM spot-ownership model.

Run with:
    cd /Users/hsharma/opt/MIDAS/fwd_sim
    python -m pytest tests/test_em_spot_ownership.py -v
"""

import math
import sys
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hedm_forward import HEDMForwardModel, HEDMGeometry
from em_spot_ownership import (
    EMSpotOwnership, EMResult,
    _angular_distance_matrix, _angular_distance_matrix_weighted,
)

MIDAS_HOME = Path("/Users/hsharma/opt/MIDAS")
BUILD_BIN = MIDAS_HOME / "build" / "bin"
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


@pytest.fixture
def ff_model():
    """Create a simple FF model with cubic Au HKLs."""
    latc = [4.08, 4.08, 4.08, 90.0, 90.0, 90.0]
    wl = 0.172979
    with tempfile.TemporaryDirectory() as tmpdir:
        work = Path(tmpdir)
        param_file = work / "params.txt"
        with open(param_file, "w") as f:
            f.write(f"LatticeParameter {' '.join(str(v) for v in latc)}\n")
            f.write(f"Wavelength {wl}\nSpaceGroup 225\nLsd 1000000\n")
            f.write(f"MaxRingRad 500000\n")
        result = subprocess.run(
            [str(BUILD_BIN / "GetHKLList"), str(param_file)],
            cwd=str(work), capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        data = np.loadtxt(work / "hkls.csv", skiprows=1)

    hkls_cart = torch.tensor(data[:, 5:8], dtype=torch.float64)
    thetas = torch.tensor(data[:, 8] * DEG2RAD, dtype=torch.float64)
    ring_indices = torch.tensor(data[:, 4].astype(int), dtype=torch.long)

    geometry = HEDMGeometry(
        Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=0.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0, wavelength=wl,
    )
    model = HEDMForwardModel(hkls=hkls_cart, thetas=thetas, geometry=geometry)
    model.ring_indices = ring_indices
    return model


# --- Angular distance tests ---

class TestAngularDistance:
    def test_no_wrapping_needed(self):
        """Points close together, no wrapping."""
        obs = torch.tensor([[0.1, 0.5, 1.0]], dtype=torch.float64)
        pred = torch.tensor([[0.1, 0.6, 1.1]], dtype=torch.float64)
        d = _angular_distance_matrix(obs, pred)
        expected = math.sqrt(0.0 + 0.1**2 + 0.1**2)
        torch.testing.assert_close(d[0, 0], torch.tensor(expected, dtype=torch.float64),
                                   atol=1e-12, rtol=0)

    def test_omega_wrapping(self):
        """Omega near ±pi boundary should wrap correctly."""
        obs = torch.tensor([[0.1, 0.5, math.pi - 0.05]], dtype=torch.float64)
        pred = torch.tensor([[0.1, 0.5, -math.pi + 0.05]], dtype=torch.float64)
        d = _angular_distance_matrix(obs, pred)
        # Should be 0.1, not 2*pi - 0.1
        torch.testing.assert_close(d[0, 0], torch.tensor(0.1, dtype=torch.float64),
                                   atol=1e-12, rtol=0)

    def test_eta_wrapping(self):
        """Eta near ±pi boundary should wrap correctly."""
        obs = torch.tensor([[0.1, math.pi - 0.02, 1.0]], dtype=torch.float64)
        pred = torch.tensor([[0.1, -math.pi + 0.03, 1.0]], dtype=torch.float64)
        d = _angular_distance_matrix(obs, pred)
        expected = 0.05  # wrapped eta distance
        torch.testing.assert_close(d[0, 0], torch.tensor(expected, dtype=torch.float64),
                                   atol=1e-12, rtol=0)

    def test_weighted_distance(self):
        """Weighted distance scales coordinates correctly."""
        obs = torch.tensor([[0.1, 0.5, 1.0]], dtype=torch.float64)
        pred = torch.tensor([[0.2, 0.5, 1.0]], dtype=torch.float64)
        w = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64)
        d = _angular_distance_matrix_weighted(obs, pred, w)
        # d2theta = 0.1, weighted by 2.0 → 0.2
        torch.testing.assert_close(d[0, 0], torch.tensor(0.2, dtype=torch.float64),
                                   atol=1e-12, rtol=0)


# --- E-step tests ---

class TestEStep:
    def test_single_voxel_high_confidence(self, ff_model):
        """A voxel generating the spots should have high self-ownership."""
        euler = torch.tensor([[0.5, 0.3, 0.7]], dtype=torch.float64)
        pos = torch.zeros(1, 3, dtype=torch.float64)

        # Generate "observed" spots from this voxel
        spots = ff_model(euler, pos)
        coords, valid = HEDMForwardModel.predict_spot_coords(spots, "angular")
        obs = coords.squeeze()[valid.squeeze() > 0.5]

        em = EMSpotOwnership(ff_model, sigma_init=0.01)
        pred_coords, pred_valid, pred_rings = em._predict_spots(euler, pos)
        ownership, hkl_asgn = em.e_step(obs, pred_coords, pred_valid)

        # Single voxel: all spots should be owned by voxel 0
        assert ownership.shape == (obs.shape[0], 1)
        assert (ownership[:, 0] > 0.99).all()

    def test_two_voxels_separate_ownership(self, ff_model):
        """Two voxels with different orientations: spots should be owned
        by their respective voxels."""
        euler = torch.tensor([
            [0.5, 0.3, 0.7],
            [1.5, 0.8, 2.1],
        ], dtype=torch.float64)
        pos = torch.zeros(2, 3, dtype=torch.float64)

        # Generate spots from voxel 0 only
        spots_0 = ff_model(euler[0:1], pos[0:1])
        coords_0, valid_0 = HEDMForwardModel.predict_spot_coords(spots_0, "angular")
        obs_0 = coords_0.squeeze()[valid_0.squeeze() > 0.5]

        em = EMSpotOwnership(ff_model, sigma_init=0.01)
        pred_coords, pred_valid, pred_rings = em._predict_spots(euler, pos)
        ownership, _ = em.e_step(obs_0, pred_coords, pred_valid)

        assert ownership.shape[1] == 2
        # Voxel 0 should own most spots
        mean_own_0 = ownership[:, 0].mean().item()
        mean_own_1 = ownership[:, 1].mean().item()
        assert mean_own_0 > mean_own_1

    def test_ownership_sums_to_one(self, ff_model):
        """Each spot's ownership should sum to 1 across voxels."""
        euler = torch.tensor([
            [0.5, 0.3, 0.7],
            [0.6, 0.3, 0.7],  # nearby orientation
        ], dtype=torch.float64)
        pos = torch.zeros(2, 3, dtype=torch.float64)

        spots = ff_model(euler[0:1], pos[0:1])
        coords, valid = HEDMForwardModel.predict_spot_coords(spots, "angular")
        obs = coords.squeeze()[valid.squeeze() > 0.5]

        em = EMSpotOwnership(ff_model, sigma_init=0.05)
        pred_coords, pred_valid, pred_rings = em._predict_spots(euler, pos)
        ownership, _ = em.e_step(obs, pred_coords, pred_valid)

        row_sums = ownership.sum(dim=1)
        claimed = row_sums > 0.01
        torch.testing.assert_close(
            row_sums[claimed],
            torch.ones(claimed.sum(), dtype=torch.float64),
            atol=1e-10, rtol=0,
        )

    def test_ring_filtering_prevents_cross_ring(self, ff_model):
        """E-step with ring filtering should not match spots across rings."""
        euler = torch.tensor([[0.5, 0.3, 0.7]], dtype=torch.float64)
        pos = torch.zeros(1, 3, dtype=torch.float64)

        spots = ff_model(euler, pos)
        coords, valid = HEDMForwardModel.predict_spot_coords(spots, "angular")
        obs = coords.squeeze()[valid.squeeze() > 0.5]

        em = EMSpotOwnership(ff_model, sigma_init=0.05)
        pred_coords, pred_valid, pred_rings = em._predict_spots(euler, pos)

        # Create fake obs_rings: assign wrong ring to first spot
        obs_rings = pred_rings[0, :obs.shape[0]].clone()
        if obs_rings.shape[0] > 0:
            # Change ring of first spot to a non-existent ring
            obs_rings_wrong = obs_rings.clone()
            obs_rings_wrong[0] = 9999

            ownership_wrong, _ = em.e_step(obs, pred_coords, pred_valid,
                                            obs_rings=obs_rings_wrong,
                                            pred_rings=pred_rings)
            # First spot should get zero ownership (no matching ring)
            assert ownership_wrong[0, 0].item() < 0.01

    def test_returns_hkl_assignments(self, ff_model):
        """E-step should return HKL assignment indices."""
        euler = torch.tensor([[0.5, 0.3, 0.7]], dtype=torch.float64)
        pos = torch.zeros(1, 3, dtype=torch.float64)

        spots = ff_model(euler, pos)
        coords, valid = HEDMForwardModel.predict_spot_coords(spots, "angular")
        obs = coords.squeeze()[valid.squeeze() > 0.5]

        em = EMSpotOwnership(ff_model, sigma_init=0.01)
        pred_coords, pred_valid, pred_rings = em._predict_spots(euler, pos)
        ownership, hkl_asgn = em.e_step(obs, pred_coords, pred_valid)

        assert hkl_asgn.shape == (obs.shape[0], 1)
        # Assignments should be valid indices
        assert (hkl_asgn >= 0).all()
        K = pred_coords.shape[1]
        assert (hkl_asgn < K).all()


# --- M-step tests ---

class TestMStep:
    def test_improves_orientation(self, ff_model):
        """M-step should move orientation toward the data."""
        gt_euler = torch.tensor([0.5, 0.3, 0.7], dtype=torch.float64)
        pos = torch.zeros(1, 3, dtype=torch.float64)

        # Generate ground truth spots
        spots = ff_model(gt_euler.unsqueeze(0), pos)
        coords, valid = HEDMForwardModel.predict_spot_coords(spots, "angular")
        obs = coords.squeeze()[valid.squeeze() > 0.5]

        # Start with perturbed orientation
        init_euler = gt_euler + 0.02 * torch.randn(3, dtype=torch.float64)

        em = EMSpotOwnership(ff_model, sigma_init=0.05)

        # E-step
        pred_coords, pred_valid, pred_rings = em._predict_spots(init_euler.unsqueeze(0), pos)
        ownership, _ = em.e_step(obs, pred_coords, pred_valid)

        # M-step
        updated = em.m_step(obs, ownership, init_euler.unsqueeze(0), pos,
                            n_opt_steps=10, lr=0.005)

        # Compute misorientation before and after
        R_gt = HEDMForwardModel.euler2mat(gt_euler)
        R_init = HEDMForwardModel.euler2mat(init_euler)
        R_updated = HEDMForwardModel.euler2mat(updated[0])

        misori_before = torch.acos(
            torch.clamp((torch.trace(R_gt.T @ R_init) - 1) / 2, -1, 1)
        ) * RAD2DEG
        misori_after = torch.acos(
            torch.clamp((torch.trace(R_gt.T @ R_updated) - 1) / 2, -1, 1)
        ) * RAD2DEG

        assert misori_after < misori_before, (
            f"M-step should improve: before={misori_before:.4f} "
            f"after={misori_after:.4f}"
        )


# --- Full EM tests ---

class TestFullEM:
    def test_single_grain_recovery(self, ff_model):
        """Full EM should recover a single grain's orientation."""
        gt_euler = torch.tensor([[0.5, 0.3, 0.7]], dtype=torch.float64)
        pos = torch.zeros(1, 3, dtype=torch.float64)

        # Generate observed spots
        spots = ff_model(gt_euler, pos)
        coords, valid = HEDMForwardModel.predict_spot_coords(spots, "angular")
        obs = coords.squeeze()[valid.squeeze() > 0.5]

        # Perturb
        init_euler = gt_euler + 0.03 * torch.randn(1, 3, dtype=torch.float64)

        em = EMSpotOwnership(ff_model, sigma_init=0.03, sigma_decay=0.85)
        result = em.fit(
            obs, init_euler, pos, n_iter=5, n_opt_steps=10, lr=0.005,
            verbose=True,
        )

        assert isinstance(result, EMResult)

        # Check misorientation improved
        R_gt = HEDMForwardModel.euler2mat(gt_euler[0])
        R_final = HEDMForwardModel.euler2mat(result.euler_angles[0])
        misori = torch.acos(
            torch.clamp((torch.trace(R_gt.T @ R_final) - 1) / 2, -1, 1)
        ) * RAD2DEG

        R_init = HEDMForwardModel.euler2mat(init_euler[0])
        misori_init = torch.acos(
            torch.clamp((torch.trace(R_gt.T @ R_init) - 1) / 2, -1, 1)
        ) * RAD2DEG

        print(f"  Initial misori: {misori_init.item():.4f} deg")
        print(f"  Final misori:   {misori.item():.4f} deg")
        assert misori < misori_init

    def test_e_step_only_mode(self, ff_model):
        """With refine_orientations=False, orientations should stay fixed."""
        gt_euler = torch.tensor([[0.5, 0.3, 0.7]], dtype=torch.float64)
        pos = torch.zeros(1, 3, dtype=torch.float64)

        spots = ff_model(gt_euler, pos)
        coords, valid = HEDMForwardModel.predict_spot_coords(spots, "angular")
        obs = coords.squeeze()[valid.squeeze() > 0.5]

        em = EMSpotOwnership(ff_model, sigma_init=0.02)
        result = em.fit(
            obs, gt_euler, pos, n_iter=3,
            refine_orientations=False, verbose=False,
        )

        # Orientations should be unchanged
        torch.testing.assert_close(result.euler_angles, gt_euler, atol=1e-14, rtol=0)
        # Ownership should still be computed
        assert result.ownership.shape == (obs.shape[0], 1)
        assert (result.ownership[:, 0] > 0.99).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
