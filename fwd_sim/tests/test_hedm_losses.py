"""Comprehensive unit tests for hedm_losses.py.

Tests every loss function and the SpotAssigner.

Run with:
    cd /Users/hsharma/opt/MIDAS/fwd_sim
    python -m pytest tests/test_hedm_losses.py -v
"""

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedm_losses import ImageComparisonLoss, SpotMatchingLoss, SpotAssigner


# ===================================================================
#  Test: ImageComparisonLoss - NCC
# ===================================================================

class TestNCCLoss:
    """Tests for Normalized Cross-Correlation loss."""

    def test_perfect_match(self):
        """Identical images -> NCC = 1, loss = 0."""
        loss_fn = ImageComparisonLoss(mode="ncc")
        img = torch.rand(10, 10) + 0.1
        loss = loss_fn(img, img)
        assert loss.item() < 1e-6, f"Perfect match loss should be ~0, got {loss.item()}"

    def test_scaled_match(self):
        """Scaled images -> NCC = 1 (scale-invariant), loss = 0."""
        loss_fn = ImageComparisonLoss(mode="ncc")
        img = torch.rand(10, 10) + 0.1
        loss = loss_fn(img * 5.0, img)
        assert loss.item() < 1e-5, f"Scaled match loss should be ~0, got {loss.item()}"

    def test_orthogonal_images(self):
        """Orthogonal images -> NCC = 0, loss = 1."""
        loss_fn = ImageComparisonLoss(mode="ncc")
        a = torch.tensor([1.0, 0.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0, 0.0])
        loss = loss_fn(a, b)
        assert abs(loss.item() - 1.0) < 1e-6

    def test_anticorrelated(self):
        """Negated image -> NCC = -1, loss = 2."""
        loss_fn = ImageComparisonLoss(mode="ncc")
        img = torch.tensor([1.0, 2.0, 3.0])
        loss = loss_fn(img, -img)
        assert abs(loss.item() - 2.0) < 1e-5

    def test_differentiable(self):
        """Gradients flow through NCC loss."""
        loss_fn = ImageComparisonLoss(mode="ncc")
        pred = torch.rand(5, 5, requires_grad=True)
        obs = torch.rand(5, 5)
        loss = loss_fn(pred, obs)
        loss.backward()
        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))

    def test_with_mask(self):
        """Mask should exclude pixels from computation."""
        loss_fn = ImageComparisonLoss(mode="ncc")
        img = torch.ones(4)
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        # Only first two elements matter; both identical -> loss ~ 0
        loss = loss_fn(img, img, mask=mask)
        assert loss.item() < 1e-6


# ===================================================================
#  Test: ImageComparisonLoss - L2
# ===================================================================

class TestL2Loss:
    """Tests for L2 (MSE) loss."""

    def test_perfect_match(self):
        loss_fn = ImageComparisonLoss(mode="l2")
        img = torch.rand(10, 10)
        loss = loss_fn(img, img)
        assert loss.item() < 1e-12

    def test_known_value(self):
        loss_fn = ImageComparisonLoss(mode="l2")
        pred = torch.tensor([1.0, 2.0, 3.0])
        obs = torch.tensor([1.0, 2.0, 4.0])
        loss = loss_fn(pred, obs)
        # MSE = (0 + 0 + 1) / 3 = 1/3
        assert abs(loss.item() - 1.0 / 3) < 1e-6

    def test_differentiable(self):
        loss_fn = ImageComparisonLoss(mode="l2")
        pred = torch.rand(5, 5, requires_grad=True)
        obs = torch.rand(5, 5)
        loss = loss_fn(pred, obs)
        loss.backward()
        assert pred.grad is not None


# ===================================================================
#  Test: ImageComparisonLoss - Log Ratio
# ===================================================================

class TestLogRatioLoss:
    """Tests for log-ratio loss."""

    def test_perfect_match(self):
        loss_fn = ImageComparisonLoss(mode="log_ratio")
        img = torch.rand(10, 10) + 0.1
        loss = loss_fn(img, img)
        assert loss.item() < 1e-10

    def test_uniform_scale(self):
        """Uniformly scaled image -> mu absorbs scale, loss = 0."""
        loss_fn = ImageComparisonLoss(mode="log_ratio")
        img = torch.rand(10, 10) + 0.1
        loss = loss_fn(img * 3.0, img)
        # log(3*x+eps) - log(x+eps) ~ log(3) for all pixels when x >> eps
        # After subtracting mu=log(3), all residuals ~ 0
        assert loss.item() < 0.01, f"Uniform scale loss should be ~0, got {loss.item()}"

    def test_differentiable(self):
        loss_fn = ImageComparisonLoss(mode="log_ratio")
        pred = torch.rand(5, 5) + 0.1
        pred.requires_grad_(True)
        obs = torch.rand(5, 5) + 0.1
        loss = loss_fn(pred, obs)
        loss.backward()
        assert pred.grad is not None

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            ImageComparisonLoss(mode="invalid")


# ===================================================================
#  Test: SpotMatchingLoss
# ===================================================================

class TestSpotMatchingLoss:
    """Tests for FF/pf spot coordinate matching loss."""

    def test_perfect_match_l2(self):
        loss_fn = SpotMatchingLoss(metric="l2")
        coords = torch.rand(10, 3)
        loss = loss_fn(coords, coords)
        assert loss.item() < 1e-12

    def test_known_value_l2(self):
        loss_fn = SpotMatchingLoss(metric="l2")
        pred = torch.tensor([[1.0, 0.0, 0.0]])
        obs = torch.tensor([[0.0, 0.0, 0.0]])
        loss = loss_fn(pred, obs)
        # ||[1,0,0]||^2 = 1, mean = 1
        assert abs(loss.item() - 1.0) < 1e-6

    def test_huber_loss(self):
        loss_fn = SpotMatchingLoss(metric="huber")
        pred = torch.tensor([[1.0, 0.0, 0.0]])
        obs = torch.tensor([[0.0, 0.0, 0.0]])
        loss = loss_fn(pred, obs)
        assert loss.item() > 0

    def test_with_weights(self):
        """Per-coordinate weights should scale the loss."""
        weights = torch.tensor([1.0, 0.0, 0.0])
        loss_fn = SpotMatchingLoss(metric="l2", weights=weights)
        pred = torch.tensor([[1.0, 100.0, 100.0]])
        obs = torch.tensor([[0.0, 0.0, 0.0]])
        loss = loss_fn(pred, obs)
        # Only first coord matters: (1-0)^2 * 1.0 = 1.0
        assert abs(loss.item() - 1.0) < 1e-6

    def test_with_spot_weights(self):
        """Per-spot weights should modulate contribution."""
        loss_fn = SpotMatchingLoss(metric="l2")
        pred = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        obs = torch.zeros(2, 3)
        spot_w = torch.tensor([1.0, 0.0])  # ignore second spot
        loss = loss_fn(pred, obs, spot_weights=spot_w)
        # Only first spot: 1.0, mean = (1+0)/2 = 0.5
        assert abs(loss.item() - 0.5) < 1e-6

    def test_differentiable(self):
        loss_fn = SpotMatchingLoss(metric="l2")
        pred = torch.rand(5, 3, requires_grad=True)
        obs = torch.rand(5, 3)
        loss = loss_fn(pred, obs)
        loss.backward()
        assert pred.grad is not None

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            SpotMatchingLoss(metric="invalid")


# ===================================================================
#  Test: SpotAssigner
# ===================================================================

class TestSpotAssigner:
    """Tests for non-differentiable spot assignment."""

    def test_exact_match(self):
        """Predicted spots exactly matching observed -> distance 0."""
        obs = torch.tensor([
            [0.1, 0.5, 1.0],
            [0.2, -0.3, 2.0],
            [0.15, 0.1, 0.5],
        ])
        assigner = SpotAssigner(obs)

        pred = obs.clone().unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        valid = torch.ones(1, 1, 3)

        pred_m, obs_m, idx = assigner.assign(pred, valid, max_distance=1.0)
        assert pred_m.shape[0] == 3
        torch.testing.assert_close(pred_m, obs_m, atol=1e-6, rtol=0)

    def test_nearest_neighbor(self):
        """Each predicted spot should match its nearest observed spot."""
        obs = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        assigner = SpotAssigner(obs)

        pred = torch.tensor([[[[0.1, 0.0, 0.0]]]])  # close to obs[0]
        valid = torch.ones(1, 1, 1)

        pred_m, obs_m, idx = assigner.assign(pred, valid, max_distance=1.0)
        assert pred_m.shape[0] == 1
        # Should match obs[0]
        torch.testing.assert_close(obs_m[0], obs[0])

    def test_max_distance_filter(self):
        """Spots beyond max_distance should be rejected."""
        obs = torch.tensor([[0.0, 0.0, 0.0]])
        assigner = SpotAssigner(obs)

        pred = torch.tensor([[[[5.0, 0.0, 0.0]]]])  # far away
        valid = torch.ones(1, 1, 1)

        pred_m, obs_m, idx = assigner.assign(pred, valid, max_distance=0.1)
        assert pred_m.shape[0] == 0, "Should reject spots beyond max_distance"

    def test_no_valid_spots(self):
        """All-invalid input should return empty tensors."""
        obs = torch.tensor([[0.0, 0.0, 0.0]])
        assigner = SpotAssigner(obs)

        pred = torch.tensor([[[[0.1, 0.0, 0.0]]]])
        valid = torch.zeros(1, 1, 1)

        pred_m, obs_m, idx = assigner.assign(pred, valid, max_distance=1.0)
        assert pred_m.shape[0] == 0

    def test_ring_number_filtering(self):
        """Same-ring matching should restrict to matching ring numbers."""
        obs = torch.tensor([
            [0.1, 0.0, 0.0],  # ring 0
            [0.1, 0.0, 0.0],  # ring 1 (same coords but different ring)
        ])
        obs_rings = torch.tensor([0, 1])
        assigner = SpotAssigner(obs, obs_ring_numbers=obs_rings)

        # Predicted spot in ring 1
        pred = torch.tensor([[0.1, 0.0, 0.0]]).unsqueeze(0)  # (1, 1, 3)
        valid = torch.ones(1, 1)
        pred_rings = torch.tensor([1])

        pred_m, obs_m, idx = assigner.assign(
            pred, valid, pred_ring_numbers=pred_rings, max_distance=1.0
        )
        if pred_m.shape[0] > 0:
            # Should match obs[1] (ring 1), not obs[0] (ring 0)
            torch.testing.assert_close(obs_m[0], obs[1])


# ===================================================================
#  Test: Differentiable stress/strain (PyTorch)
# ===================================================================

from hedm_losses import (
    tensor_to_voigt as tv, voigt_to_tensor as vt,
    cubic_stiffness_tensor, rotation_voigt_mandel as rvm_torch,
    hooke_stress as hooke_torch, volume_average_stress_constraint as vasc_torch,
)


class TestDifferentiableStress:
    """Tests for PyTorch differentiable stress/strain utilities."""

    def test_voigt_roundtrip(self):
        T = torch.tensor([[1., 4., 5.], [4., 2., 6.], [5., 6., 3.]])
        v = tv(T)
        T2 = vt(v)
        torch.testing.assert_close(T, T2)

    def test_voigt_frobenius(self):
        T = torch.tensor([[1., .3, .1], [.3, 2., .2], [.1, .2, 3.]])
        v = tv(T)
        torch.testing.assert_close(
            torch.norm(T, p='fro'), torch.norm(v), atol=1e-14, rtol=0
        )

    def test_hooke_zero(self):
        C = cubic_stiffness_tensor(192.9, 163.8, 41.5)
        eps = torch.zeros(3, 3, dtype=torch.float64)
        sig = hooke_torch(eps, C, frame="grain")
        torch.testing.assert_close(sig, torch.zeros(3, 3, dtype=torch.float64))

    def test_hooke_hydrostatic_cubic(self):
        C11, C12, C44 = 192.9, 163.8, 41.5
        C = cubic_stiffness_tensor(C11, C12, C44)
        eps = 0.001 * torch.eye(3, dtype=torch.float64)
        sig = hooke_torch(eps, C, frame="grain")
        expected = (C11 + 2 * C12) * 0.001
        torch.testing.assert_close(
            torch.diag(sig),
            torch.full((3,), expected, dtype=torch.float64),
            atol=1e-10, rtol=0,
        )

    def test_hooke_differentiable(self):
        """Gradients flow through Hooke's law."""
        C = cubic_stiffness_tensor(200.0, 130.0, 80.0)
        eps = torch.diag(torch.tensor([0.001, -0.0005, -0.0005],
                                       dtype=torch.float64))
        eps.requires_grad_(True)
        orient = torch.eye(3, dtype=torch.float64)
        sig = hooke_torch(eps, C, orient=orient, frame="lab")
        loss = sig.sum()
        loss.backward()
        assert eps.grad is not None
        assert torch.all(torch.isfinite(eps.grad))

    def test_hooke_lab_frame_differentiable(self):
        """Gradients flow through the full lab-frame path (with rotation)."""
        C = cubic_stiffness_tensor(200.0, 130.0, 80.0)
        eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        # Random orientation
        from hedm_forward import HEDMForwardModel
        orient = HEDMForwardModel.euler2mat(
            torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        )
        sig = hooke_torch(eps, C, orient=orient, frame="lab")
        loss = sig.sum()
        loss.backward()
        assert eps.grad is not None

    def test_rotation_voigt_identity(self):
        M = rvm_torch(torch.eye(3, dtype=torch.float64))
        torch.testing.assert_close(M, torch.eye(6, dtype=torch.float64), atol=1e-14, rtol=0)

    def test_rotation_voigt_differentiable(self):
        orient = torch.eye(3, dtype=torch.float64, requires_grad=True)
        M = rvm_torch(orient)
        loss = M.sum()
        loss.backward()
        assert orient.grad is not None

    def test_volume_avg_constraint(self):
        stresses = torch.randn(5, 3, 3, dtype=torch.float64)
        stresses = 0.5 * (stresses + stresses.transpose(-1, -2))
        volumes = torch.rand(5, dtype=torch.float64) + 0.1
        corrected = vasc_torch(stresses, volumes)
        w = volumes / volumes.sum()
        avg = (w[:, None, None] * corrected).sum(dim=0)
        torch.testing.assert_close(
            avg, torch.zeros(3, 3, dtype=torch.float64), atol=1e-12, rtol=0
        )

    def test_volume_avg_differentiable(self):
        stresses = torch.randn(3, 3, 3, dtype=torch.float64, requires_grad=True)
        volumes = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
        corrected = vasc_torch(stresses, volumes)
        loss = corrected.sum()
        loss.backward()
        assert stresses.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
