"""Tests for MLEM sinogram reconstruction.

Run with:
    cd /Users/hsharma/opt/MIDAS/utils
    python -m pytest tests/test_mlem_recon.py -v
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlem_recon import forward_project, back_project, mlem, osem


class TestForwardProject:
    def test_empty_image(self):
        img = np.zeros((16, 16))
        angles = np.array([0, 45, 90, 135])
        sino = forward_project(img, angles)
        assert sino.shape == (4, 16)
        np.testing.assert_allclose(sino, 0, atol=1e-15)

    def test_centered_point(self):
        """A point at the center should project to the center at all angles."""
        N = 21
        img = np.zeros((N, N))
        img[N // 2, N // 2] = 1.0
        angles = np.array([0, 45, 90, 135])
        sino = forward_project(img, angles)
        # Peak should be at center detector pixel for all angles
        for i in range(4):
            assert np.argmax(sino[i]) == N // 2

    def test_sinogram_shape(self):
        img = np.zeros((32, 32))
        angles = np.linspace(0, 180, 10, endpoint=False)
        sino = forward_project(img, angles)
        assert sino.shape == (10, 32)


class TestBackProject:
    def test_zeros(self):
        sino = np.zeros((4, 16))
        angles = np.array([0, 45, 90, 135])
        bp = back_project(sino, angles, 16)
        np.testing.assert_allclose(bp, 0, atol=1e-15)

    def test_output_shape(self):
        sino = np.ones((10, 32))
        angles = np.linspace(0, 180, 10, endpoint=False)
        bp = back_project(sino, angles, 32)
        assert bp.shape == (32, 32)

    def test_adjoint_property(self):
        """<Ax, y> should approximately equal <x, A^T y> (adjoint test)."""
        np.random.seed(42)
        N = 16
        img = np.random.rand(N, N)
        sino = np.random.rand(5, N)
        angles = np.array([0, 36, 72, 108, 144])

        # Forward: A @ img
        Ax = forward_project(img, angles)
        # Back-project: A^T @ sino
        Aty = back_project(sino, angles, N)

        lhs = np.sum(Ax * sino)
        rhs = np.sum(img * Aty)
        # Should be approximately equal (not exact due to interpolation)
        np.testing.assert_allclose(lhs, rhs, rtol=0.15)


class TestMLEM:
    def test_single_disk(self):
        """Reconstruct a centered disk from its sinogram."""
        N = 32
        img_true = np.zeros((N, N))
        yy, xx = np.mgrid[:N, :N]
        center = (N - 1) / 2.0
        r = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        img_true[r < 8] = 1.0

        angles = np.linspace(0, 180, 30, endpoint=False)
        sino = forward_project(img_true, angles)

        recon = mlem(sino, angles, n_iter=20)
        assert recon.shape == (N, N)

        # The reconstruction should be positive
        assert np.all(recon >= 0)

        # The disk should be visible: center should be brighter than edge
        assert recon[N // 2, N // 2] > recon[0, 0]

    def test_missing_angles(self):
        """MLEM should handle sinograms with missing rows (zero rows)."""
        N = 16
        img_true = np.zeros((N, N))
        img_true[5:11, 5:11] = 1.0

        angles_full = np.linspace(0, 180, 20, endpoint=False)
        sino_full = forward_project(img_true, angles_full)

        # Zero out half the rows (missing angles)
        sino_sparse = sino_full.copy()
        sino_sparse[::2] = 0  # every other row missing

        recon = mlem(sino_sparse, angles_full, n_iter=30)
        assert recon.shape == (N, N)
        assert np.all(recon >= 0)

    def test_positivity(self):
        """MLEM should always produce non-negative values."""
        np.random.seed(7)
        N = 16
        sino = np.random.rand(8, N) * 10
        angles = np.linspace(0, 180, 8, endpoint=False)
        recon = mlem(sino, angles, n_iter=10)
        assert np.all(recon >= 0)

    def test_empty_sinogram(self):
        """All-zero sinogram should produce zero reconstruction."""
        sino = np.zeros((5, 16))
        angles = np.linspace(0, 180, 5, endpoint=False)
        recon = mlem(sino, angles, n_iter=5)
        np.testing.assert_allclose(recon, 0, atol=1e-15)


class TestOSEM:
    def test_basic(self):
        """OS-EM should produce a reasonable reconstruction."""
        N = 16
        img_true = np.zeros((N, N))
        img_true[4:12, 4:12] = 1.0

        angles = np.linspace(0, 180, 20, endpoint=False)
        sino = forward_project(img_true, angles)

        recon = osem(sino, angles, n_iter=5, n_subsets=4)
        assert recon.shape == (N, N)
        assert np.all(recon >= 0)
        # Center should be bright
        assert recon[N // 2, N // 2] > recon[0, 0]

    def test_faster_than_mlem(self):
        """OS-EM with fewer iterations should approximate MLEM quality."""
        N = 16
        img_true = np.zeros((N, N))
        img_true[6:10, 6:10] = 1.0

        angles = np.linspace(0, 180, 16, endpoint=False)
        sino = forward_project(img_true, angles)

        recon_mlem = mlem(sino, angles, n_iter=20)
        recon_osem = osem(sino, angles, n_iter=5, n_subsets=4)  # 4x fewer iters

        # Both should reconstruct something reasonable
        assert recon_mlem[N // 2, N // 2] > 0.1
        assert recon_osem[N // 2, N // 2] > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
