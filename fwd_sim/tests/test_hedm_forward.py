"""Comprehensive unit tests for hedm_forward.py.

Tests every sub-function of HEDMForwardModel with known analytical results,
symmetry checks, and cross-validation against the original nfhedm.py.

Run with:
    cd /Users/hsharma/opt/MIDAS/fwd_sim
    python -m pytest tests/test_hedm_forward.py -v
"""

import math
import sys
import os

import pytest
import torch

# Ensure fwd_sim is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedm_forward import HEDMForwardModel, HEDMGeometry, ScanConfig, SpotDescriptors


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def nf_geometry():
    """Standard NF-HEDM geometry (Lsd ~ 5mm)."""
    return HEDMGeometry(
        Lsd=5105.669354,
        y_BC=1030.793639,
        z_BC=12.861513,
        px=1.48,
        omega_start=180.0,
        omega_step=-0.25,
        n_frames=1440,
        n_pixels_y=2048,
        n_pixels_z=2048,
        min_eta=6.0,
        wavelength=0.295,
        flip_y=False,  # NF convention
    )


@pytest.fixture
def ff_geometry():
    """Standard FF-HEDM geometry (Lsd ~ 1000mm)."""
    return HEDMGeometry(
        Lsd=1000000.0,
        y_BC=1024.0,
        z_BC=1024.0,
        px=200.0,
        omega_start=0.0,
        omega_step=0.25,
        n_frames=1440,
        n_pixels_y=2048,
        n_pixels_z=2048,
        min_eta=6.0,
        wavelength=0.295,
    )


@pytest.fixture
def cubic_lattice_params():
    """Cubic iron: a=b=c=2.87 Angstroms, alpha=beta=gamma=90 degrees."""
    return torch.tensor([2.87, 2.87, 2.87, 90.0, 90.0, 90.0])


@pytest.fixture
def sample_hkls_int():
    """Common BCC iron HKL reflections (integers)."""
    return torch.tensor([
        [1, 1, 0],
        [2, 0, 0],
        [2, 1, 1],
        [2, 2, 0],
    ], dtype=torch.float32)


def make_model_with_cubic_iron(geometry, device):
    """Helper: create a model with cubic iron HKLs already transformed."""
    a = 2.87  # Angstroms
    wl = 0.295  # Angstroms

    hkls_int = torch.tensor([
        [1, 1, 0],
        [2, 0, 0],
        [2, 1, 1],
        [2, 2, 0],
    ], dtype=torch.float32)

    # B matrix for cubic: B = (1/a) * I
    B = torch.eye(3) / a
    hkls_cart = (B @ hkls_int.T).T  # (M, 3)

    # Bragg angles
    d_spacings = 1.0 / torch.norm(hkls_cart, dim=-1)
    thetas = torch.asin(wl / (2.0 * d_spacings))

    model = HEDMForwardModel(
        hkls=hkls_cart,
        thetas=thetas,
        geometry=geometry,
        hkls_int=hkls_int,
        device=device,
    )
    return model, hkls_cart, thetas


# ===================================================================
#  Test: euler2mat
# ===================================================================

class TestEuler2Mat:
    """Tests for the Euler angle to rotation matrix conversion."""

    def test_identity(self):
        """Zero Euler angles -> identity matrix."""
        angles = torch.zeros(3)
        R = HEDMForwardModel.euler2mat(angles)
        torch.testing.assert_close(R, torch.eye(3), atol=1e-7, rtol=0)

    def test_rotation_is_orthogonal(self):
        """R @ R^T = I for random Euler angles."""
        torch.manual_seed(42)
        angles = torch.rand(10, 3) * 2 * math.pi
        R = HEDMForwardModel.euler2mat(angles)
        I = torch.eye(3).expand(10, 3, 3)
        RRT = torch.bmm(R, R.transpose(-1, -2))
        torch.testing.assert_close(RRT, I, atol=1e-6, rtol=0)

    def test_determinant_is_one(self):
        """det(R) = 1 (proper rotation, not reflection)."""
        torch.manual_seed(42)
        angles = torch.rand(10, 3) * 2 * math.pi
        R = HEDMForwardModel.euler2mat(angles)
        dets = torch.det(R)
        torch.testing.assert_close(dets, torch.ones(10), atol=1e-6, rtol=0)

    def test_batch_dimensions(self):
        """Test with various batch dimensions."""
        for shape in [(3,), (5, 3), (2, 3, 3), (2, 4, 5, 3)]:
            angles = torch.zeros(*shape)
            R = HEDMForwardModel.euler2mat(angles)
            expected_shape = (*shape[:-1], 3, 3)
            assert R.shape == expected_shape, f"Shape mismatch for input {shape}"

    def test_known_rotation_90deg_phi1(self):
        """phi1=90deg, Phi=0, phi2=0: rotation around z by 90 degrees."""
        angles = torch.tensor([math.pi / 2, 0.0, 0.0])
        R = HEDMForwardModel.euler2mat(angles)
        # For ZXZ with phi1=pi/2, Phi=0, phi2=0:
        # R_z(pi/2) * I * I = [[0,-1,0],[1,0,0],[0,0,1]]
        expected = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        torch.testing.assert_close(R, expected, atol=1e-6, rtol=0)

    def test_phi2_rotation(self):
        """phi1=0, Phi=0, phi2=90deg: also rotation around z by 90 degrees.

        When Phi=0, the ZXZ convention gives Rz(phi1+phi2).
        """
        angles = torch.tensor([0.0, 0.0, math.pi / 2])
        R = HEDMForwardModel.euler2mat(angles)
        expected = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        torch.testing.assert_close(R, expected, atol=1e-6, rtol=0)

    def test_differentiable(self):
        """Gradients flow through euler2mat."""
        angles = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        R = HEDMForwardModel.euler2mat(angles)
        loss = R.sum()
        loss.backward()
        assert angles.grad is not None
        assert torch.all(torch.isfinite(angles.grad))

    def test_preserves_vector_norm(self):
        """Rotation preserves vector magnitude: |R@v| = |v|."""
        torch.manual_seed(123)
        angles = torch.rand(3) * 2 * math.pi
        R = HEDMForwardModel.euler2mat(angles)
        v = torch.tensor([1.0, 2.0, 3.0])
        Rv = R @ v
        torch.testing.assert_close(
            torch.norm(Rv), torch.norm(v), atol=1e-6, rtol=0
        )


# ===================================================================
#  Test: orthogonalize (SO(3) projection)
# ===================================================================

class TestOrthogonalize:
    """Tests for SVD-based SO(3) projection."""

    def test_already_orthogonal(self):
        """Orthogonalizing an identity matrix should return identity."""
        I = torch.eye(3, dtype=torch.float64)
        R = HEDMForwardModel.orthogonalize(I)
        torch.testing.assert_close(R, I, atol=1e-14, rtol=0)

    def test_random_orthogonal_unchanged(self):
        """Orthogonalizing a proper rotation should not change it."""
        torch.manual_seed(42)
        angles = torch.rand(3, dtype=torch.float64) * 2 * math.pi
        R = HEDMForwardModel.euler2mat(angles)
        R2 = HEDMForwardModel.orthogonalize(R)
        torch.testing.assert_close(R, R2, atol=1e-12, rtol=0)

    def test_perturbed_matrix_becomes_orthogonal(self):
        """A near-orthogonal matrix should be projected to exact SO(3)."""
        R = torch.eye(3, dtype=torch.float64)
        R[0, 1] += 1e-7  # perturb
        R[2, 2] -= 1e-8
        # Not orthogonal
        RtR = R.T @ R
        assert not torch.allclose(RtR, torch.eye(3, dtype=torch.float64), atol=1e-10)
        # After orthogonalization
        R_orth = HEDMForwardModel.orthogonalize(R)
        RtR2 = R_orth.T @ R_orth
        torch.testing.assert_close(RtR2, torch.eye(3, dtype=torch.float64), atol=1e-14, rtol=0)

    def test_det_is_one(self):
        """Orthogonalized matrix should have det = +1."""
        R = torch.eye(3, dtype=torch.float64)
        R[0, 0] = 1.0 + 1e-8  # slightly non-unit (realistic for float64 trig)
        R_orth = HEDMForwardModel.orthogonalize(R)
        det = torch.det(R_orth)
        assert abs(det.item() - 1.0) < 1e-12

    def test_reflection_corrected(self):
        """A matrix with det = -1 should be corrected to det = +1."""
        R = torch.eye(3, dtype=torch.float64)
        R[2, 2] = -1.0  # reflection
        R_orth = HEDMForwardModel.orthogonalize(R)
        det = torch.det(R_orth)
        assert abs(det.item() - 1.0) < 1e-14

    def test_batch_dimensions(self):
        """Works with batched matrices."""
        torch.manual_seed(99)
        R = torch.eye(3, dtype=torch.float64).expand(5, 3, 3).contiguous()
        R = R + torch.randn(5, 3, 3, dtype=torch.float64) * 1e-10
        R_orth = HEDMForwardModel.orthogonalize(R)
        for i in range(5):
            RtR = R_orth[i].T @ R_orth[i]
            torch.testing.assert_close(RtR, torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_differentiable(self):
        """Gradients flow through orthogonalize."""
        R = torch.eye(3, dtype=torch.float64, requires_grad=True)
        R_orth = HEDMForwardModel.orthogonalize(R)
        loss = R_orth.sum()
        loss.backward()
        assert R.grad is not None
        assert torch.all(torch.isfinite(R.grad))

    def test_euler2mat_now_orthogonal(self):
        """euler2mat should produce exactly orthogonal matrices."""
        torch.manual_seed(42)
        angles = torch.rand(100, 3, dtype=torch.float64) * 2 * math.pi
        R = HEDMForwardModel.euler2mat(angles)
        I = torch.eye(3, dtype=torch.float64).expand(100, 3, 3)
        RtR = torch.bmm(R, R.transpose(-1, -2))
        torch.testing.assert_close(RtR, I, atol=1e-13, rtol=0)
        dets = torch.det(R)
        torch.testing.assert_close(dets, torch.ones(100, dtype=torch.float64), atol=1e-13, rtol=0)


# ===================================================================
#  Test: correct_hkls_latc
# ===================================================================

class TestCorrectHKLsLatC:
    """Tests for the lattice parameter correction (B matrix construction)."""

    def test_cubic_b_matrix(self, nf_geometry, device, cubic_lattice_params):
        """Cubic lattice: B = (1/a)*I, d(hkl) = a/sqrt(h^2+k^2+l^2)."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        hkls_cart, thetas = model.correct_hkls_latc(cubic_lattice_params)

        a = 2.87
        wl = 0.295

        # Check d-spacings
        for i, hkl in enumerate(model.hkls_int):
            h, k, l = hkl.tolist()
            d_expected = a / math.sqrt(h**2 + k**2 + l**2)
            d_computed = 1.0 / torch.norm(hkls_cart[i]).item()
            assert abs(d_computed - d_expected) < 1e-6, (
                f"HKL ({h},{k},{l}): d_expected={d_expected:.6f}, d_computed={d_computed:.6f}"
            )

        # Check thetas
        for i, hkl in enumerate(model.hkls_int):
            h, k, l = hkl.tolist()
            d = a / math.sqrt(h**2 + k**2 + l**2)
            theta_expected = math.asin(wl / (2 * d))
            theta_computed = thetas[i].item()
            assert abs(theta_computed - theta_expected) < 1e-6

    def test_strained_lattice_shifts_theta(self, nf_geometry, device):
        """Tensile strain (larger a) -> smaller theta (larger d-spacing)."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)

        nominal = torch.tensor([2.87, 2.87, 2.87, 90.0, 90.0, 90.0])
        strained = torch.tensor([2.90, 2.87, 2.87, 90.0, 90.0, 90.0])

        _, thetas_nom = model.correct_hkls_latc(nominal)
        _, thetas_str = model.correct_hkls_latc(strained)

        # For (h,0,0) reflections, stretching a should decrease theta
        # (100) is not in our HKL list, but (200) is at index 1
        # d(200) = a/2, so larger a -> larger d -> smaller theta
        assert thetas_str[1] < thetas_nom[1], "Tensile strain should decrease theta"

    def test_batch_lattice_params(self, nf_geometry, device):
        """Per-voxel lattice params produce per-voxel G-vectors."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)

        # Two different lattice parameters
        lp = torch.stack([
            torch.tensor([2.87, 2.87, 2.87, 90.0, 90.0, 90.0]),
            torch.tensor([2.90, 2.90, 2.90, 90.0, 90.0, 90.0]),
        ])  # (2, 6)

        hkls_cart, thetas = model.correct_hkls_latc(lp)
        assert hkls_cart.shape == (2, 4, 3), f"Expected (2,4,3), got {hkls_cart.shape}"
        assert thetas.shape == (2, 4), f"Expected (2,4), got {thetas.shape}"

        # Different lattice params should give different thetas
        assert not torch.allclose(thetas[0], thetas[1])

    def test_differentiable(self, nf_geometry, device):
        """Gradients flow through correct_hkls_latc."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        lp = torch.tensor([2.87, 2.87, 2.87, 90.0, 90.0, 90.0], requires_grad=True)
        hkls_cart, thetas = model.correct_hkls_latc(lp)
        loss = thetas.sum() + hkls_cart.sum()
        loss.backward()
        assert lp.grad is not None
        assert torch.all(torch.isfinite(lp.grad))

    def test_raises_without_hkls_int(self, nf_geometry, device):
        """Should raise if hkls_int was not provided."""
        hkls = torch.randn(4, 3)
        thetas = torch.rand(4) * 0.1
        model = HEDMForwardModel(hkls, thetas, nf_geometry, hkls_int=None, device=device)
        with pytest.raises(RuntimeError, match="hkls_int"):
            model.correct_hkls_latc(torch.tensor([2.87, 2.87, 2.87, 90., 90., 90.]))

    def test_hexagonal_lattice(self, nf_geometry, device):
        """Hexagonal (a != c, gamma=120): verify non-orthogonal B matrix."""
        hkls_int = torch.tensor([[1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=torch.float32)
        # Use dummy nominal hkls (we'll use correct_hkls_latc)
        dummy_hkls = torch.randn(3, 3)
        dummy_thetas = torch.rand(3) * 0.1
        geometry = HEDMGeometry(
            Lsd=5000, y_BC=1024, z_BC=1024, px=1.5,
            omega_start=0, omega_step=0.25, n_frames=1440,
            n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0, wavelength=0.295
        )
        model = HEDMForwardModel(dummy_hkls, dummy_thetas, geometry,
                                 hkls_int=hkls_int, device=device)

        # Titanium HCP: a=2.95, c=4.68, alpha=beta=90, gamma=120
        lp = torch.tensor([2.95, 2.95, 4.68, 90.0, 90.0, 120.0])
        hkls_cart, thetas = model.correct_hkls_latc(lp)

        # d(001) for hexagonal = c
        d_001 = 1.0 / torch.norm(hkls_cart[1]).item()
        assert abs(d_001 - 4.68) < 0.01, f"d(001) = {d_001}, expected 4.68"

        # d(100) for hexagonal = a*sqrt(3)/2
        d_100 = 1.0 / torch.norm(hkls_cart[0]).item()
        d_100_expected = 2.95 * math.sqrt(3) / 2
        assert abs(d_100 - d_100_expected) < 0.01, f"d(100) = {d_100}, expected {d_100_expected}"


# ===================================================================
#  Test: calc_bragg_geometry
# ===================================================================

class TestCalcBraggGeometry:
    """Tests for the omega quadratic solver and eta computation."""

    def test_output_shapes(self, nf_geometry, device):
        """Verify output tensor shapes."""
        model, hkls, thetas = make_model_with_cubic_iron(nf_geometry, device)
        N, M = 5, 4  # 5 positions, 4 HKLs
        R = HEDMForwardModel.euler2mat(torch.rand(N, 3))
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)

        assert omega.shape == (2 * N, M), f"omega shape: {omega.shape}"
        assert eta.shape == (2 * N, M)
        assert two_theta.shape == (2 * N, M)
        assert valid.shape == (2 * N, M)

    def test_identity_orientation_produces_spots(self, nf_geometry, device):
        """Identity orientation should produce at least some valid spots."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        R = torch.eye(3).unsqueeze(0)  # (1, 3, 3)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)
        n_valid = valid.sum().item()
        assert n_valid > 0, "Identity orientation should produce valid spots"

    def test_omega_symmetry(self, nf_geometry, device):
        """The two omega solutions (+/-) should cover different quadrants."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        R = torch.eye(3).unsqueeze(0)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)
        # First N are positive solution, second N are negative
        omega_p = omega[:1]  # (1, M)
        omega_n = omega[1:]  # (1, M)
        # They should not be identical (in general)
        # At least for some HKLs they should differ
        assert not torch.allclose(omega_p, omega_n, atol=1e-3)

    def test_two_theta_matches_input(self, nf_geometry, device):
        """two_theta output should equal 2 * input thetas."""
        model, _, thetas = make_model_with_cubic_iron(nf_geometry, device)
        R = torch.eye(3).unsqueeze(0)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)
        expected = 2.0 * thetas.unsqueeze(0).expand(2, -1)
        torch.testing.assert_close(two_theta, expected, atol=1e-7, rtol=0)

    def test_eta_in_valid_range(self, nf_geometry, device):
        """Valid spots should have eta in [-pi, pi]."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        torch.manual_seed(42)
        R = HEDMForwardModel.euler2mat(torch.rand(3, 3) * 2 * math.pi)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)
        valid_eta = eta[valid > 0.5]
        if valid_eta.numel() > 0:
            assert valid_eta.abs().max() <= math.pi + 1e-6

    def test_batch_dimensions(self, nf_geometry, device):
        """Test with batch dimensions on orientations."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        R = HEDMForwardModel.euler2mat(torch.rand(2, 3, 3) * 0.1)  # (2, 3, 3, 3)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)
        # 2 batch, 3 positions -> 2*3=6 in K dim, 4 HKLs
        assert omega.shape == (2, 6, 4), f"Got {omega.shape}"

    def test_differentiable(self, nf_geometry, device):
        """Gradients flow through calc_bragg_geometry."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        angles = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        R = HEDMForwardModel.euler2mat(angles)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)
        loss = (omega * valid).sum() + (eta * valid).sum()
        loss.backward()
        assert angles.grad is not None
        assert torch.all(torch.isfinite(angles.grad))

    def test_strained_hkls_shift_omega(self, nf_geometry, device):
        """Strained G-vectors should produce different omega angles."""
        model, _, thetas = make_model_with_cubic_iron(nf_geometry, device)
        R = torch.eye(3).unsqueeze(0)  # (1, 3, 3)

        omega_nom, _, _, _ = model.calc_bragg_geometry(R)

        # Compute strained hkls
        strained_lp = torch.tensor([2.90, 2.87, 2.87, 90.0, 90.0, 90.0])
        hkls_s, thetas_s = model.correct_hkls_latc(strained_lp)
        omega_str, _, _, _ = model.calc_bragg_geometry(R, hkls_s, thetas_s)

        # At least some omegas should differ
        assert not torch.allclose(omega_nom, omega_str, atol=1e-5)


# ===================================================================
#  Test: project_to_detector
# ===================================================================

class TestProjectToDetector:
    """Tests for position-dependent detector projection."""

    def test_origin_position(self, nf_geometry, device):
        """Grain at origin: position should not shift spots."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        R = torch.eye(3).unsqueeze(0)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)

        pos_origin = torch.zeros(1, 3)
        spots_origin = model.project_to_detector(omega, eta, two_theta, pos_origin, valid)

        # Verify detector coordinates are computed
        assert spots_origin.y_pixel.shape == omega.shape
        assert spots_origin.z_pixel.shape == omega.shape

    def test_position_shifts_nf(self, nf_geometry, device):
        """In NF (close detector), position should significantly shift spots."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        R = torch.eye(3).unsqueeze(0)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)

        pos_origin = torch.zeros(1, 3)
        pos_shifted = torch.tensor([[100.0, 100.0, 0.0]])  # 100um shift

        spots_0 = model.project_to_detector(omega, eta, two_theta, pos_origin, valid)
        spots_s = model.project_to_detector(omega, eta, two_theta, pos_shifted, valid)

        # For NF (Lsd=5105um), a 100um shift is ~2% of Lsd -> visible effect
        diff_y = (spots_s.y_pixel - spots_0.y_pixel).abs()
        diff_z = (spots_s.z_pixel - spots_0.z_pixel).abs()
        max_diff = torch.max(diff_y.max(), diff_z.max()).item()
        assert max_diff > 1.0, f"NF position shift should be significant, got {max_diff}"

    def test_position_negligible_ff(self, ff_geometry, device):
        """In FF (far detector), position shift should be negligible."""
        model, _, _ = make_model_with_cubic_iron(ff_geometry, device)
        R = torch.eye(3).unsqueeze(0)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)

        pos_origin = torch.zeros(1, 3)
        pos_shifted = torch.tensor([[100.0, 100.0, 0.0]])  # 100um shift

        spots_0 = model.project_to_detector(omega, eta, two_theta, pos_origin, valid)
        spots_s = model.project_to_detector(omega, eta, two_theta, pos_shifted, valid)

        # For FF (Lsd=1e6 um), a 100um shift is 0.01% -> negligible
        diff_y = (spots_s.y_pixel - spots_0.y_pixel).abs()
        diff_z = (spots_s.z_pixel - spots_0.z_pixel).abs()
        max_diff = torch.max(diff_y.max(), diff_z.max()).item()
        assert max_diff < 1.0, f"FF position shift should be negligible, got {max_diff}"

    def test_3d_position(self, nf_geometry, device):
        """3D position (with z) should work and affect z_pixel."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        R = torch.eye(3).unsqueeze(0)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)

        pos_z0 = torch.tensor([[0.0, 0.0, 0.0]])
        pos_z100 = torch.tensor([[0.0, 0.0, 100.0]])

        spots_0 = model.project_to_detector(omega, eta, two_theta, pos_z0, valid)
        spots_z = model.project_to_detector(omega, eta, two_theta, pos_z100, valid)

        # z-position should shift z_pixel
        diff_z = (spots_z.z_pixel - spots_0.z_pixel).abs()
        assert diff_z.max() > 0.1, "z-position should shift z_pixel"

    def test_2d_position_backward_compat(self, nf_geometry, device):
        """(N,2) positions should work (z padded to 0)."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        R = torch.eye(3).unsqueeze(0)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)

        pos_2d = torch.zeros(1, 2)
        pos_3d = torch.zeros(1, 3)

        # Manually pad for project_to_detector (forward() does it automatically)
        pos_2d_padded = F.pad(pos_2d, (0, 1), value=0.0)
        spots_2d = model.project_to_detector(omega, eta, two_theta, pos_2d_padded, valid)
        spots_3d = model.project_to_detector(omega, eta, two_theta, pos_3d, valid)

        torch.testing.assert_close(spots_2d.y_pixel, spots_3d.y_pixel)
        torch.testing.assert_close(spots_2d.z_pixel, spots_3d.z_pixel)

    def test_frame_nr_range(self, nf_geometry, device):
        """Valid frame_nr should be in [0, n_frames)."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        torch.manual_seed(99)
        R = HEDMForwardModel.euler2mat(torch.rand(3, 3) * 0.5)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)
        pos = torch.zeros(3, 3)
        spots = model.project_to_detector(omega, eta, two_theta, pos, valid)

        valid_frames = spots.frame_nr[spots.valid > 0.5]
        if valid_frames.numel() > 0:
            assert valid_frames.min() >= 0
            assert valid_frames.max() < model.n_frames

    def test_differentiable(self, nf_geometry, device):
        """Gradients flow through project_to_detector."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        angles = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        R = HEDMForwardModel.euler2mat(angles)
        omega, eta, two_theta, valid = model.calc_bragg_geometry(R)
        pos = torch.tensor([[10.0, 20.0, 0.0]], requires_grad=True)
        spots = model.project_to_detector(omega, eta, two_theta, pos, valid)
        loss = (spots.y_pixel * spots.valid).sum()
        loss.backward()
        assert angles.grad is not None
        assert pos.grad is not None


# ===================================================================
#  Test: forward (full pipeline)
# ===================================================================

class TestForward:
    """Tests for the full forward() orchestrator."""

    def test_basic_run(self, nf_geometry, device):
        """forward() runs without error and returns SpotDescriptors."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        spots = model(euler, pos)
        assert isinstance(spots, SpotDescriptors)
        assert spots.omega.shape[-2] == 2  # 2N where N=1
        assert spots.omega.shape[-1] == 4  # M=4 HKLs

    def test_2d_positions(self, nf_geometry, device):
        """forward() accepts (N,2) positions (backward compat)."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 2)
        spots = model(euler, pos)
        assert isinstance(spots, SpotDescriptors)

    def test_with_strain(self, nf_geometry, device):
        """forward() with lattice_params produces valid output."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        lp = torch.tensor([2.87, 2.87, 2.87, 90.0, 90.0, 90.0])
        spots = model(euler, pos, lattice_params=lp)
        assert isinstance(spots, SpotDescriptors)

    def test_multiple_positions(self, nf_geometry, device):
        """forward() with N>1 positions."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        N = 5
        euler = torch.rand(N, 3) * 0.1
        pos = torch.randn(N, 3) * 10
        spots = model(euler, pos)
        assert spots.omega.shape[-2] == 2 * N

    def test_end_to_end_differentiable(self, nf_geometry, device):
        """Gradients flow from loss through full forward() pipeline."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        euler = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        pos = torch.tensor([[10.0, 20.0, 0.0]])
        spots = model(euler, pos)
        loss = (spots.y_pixel * spots.valid).sum()
        loss.backward()
        assert euler.grad is not None
        assert torch.all(torch.isfinite(euler.grad))


# ===================================================================
#  Test: filter_by_scan
# ===================================================================

# ===================================================================
#  Test: multi-distance NF-HEDM
# ===================================================================

class TestMultiDistance:
    """Tests for NF-HEDM multi-distance (multiple Lsd) projection."""

    @pytest.fixture
    def nf_multi_geometry(self):
        """NF-HEDM with 2 detector distances (like the Au example)."""
        return HEDMGeometry(
            Lsd=[8289.15, 10290.72],
            y_BC=[985.42, 985.16],
            z_BC=[17.51, 24.51],
            px=1.48,
            omega_start=180.0,
            omega_step=-0.25,
            n_frames=1440,
            n_pixels_y=2048,
            n_pixels_z=2048,
            min_eta=6.0,
            wavelength=0.295,
        )

    def test_n_distances(self, nf_multi_geometry):
        """Geometry reports correct number of distances."""
        assert nf_multi_geometry.n_distances == 2

    def test_multi_distance_output_shapes(self, nf_multi_geometry, device):
        """Multi-distance projection produces per-layer pixel coordinates."""
        model, _, _ = make_model_with_cubic_iron(nf_multi_geometry, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        spots = model(euler, pos)

        # omega, eta, two_theta, frame_nr: (..., 2N, M) -- no distance dim
        assert spots.omega.shape == (2, 4)  # 2N=2, M=4

        # y_pixel, z_pixel: (D, ..., 2N, M) where D=2
        assert spots.y_pixel.shape[0] == 2, f"Expected D=2 leading dim, got {spots.y_pixel.shape}"
        assert spots.y_pixel.shape == (2, 2, 4)  # (D, 2N, M)

        # valid: (..., 2N, M) -- combined ALL distances
        assert spots.valid.shape == (2, 4)

        # layer_valid: (D, ..., 2N, M)
        assert spots.layer_valid is not None
        assert spots.layer_valid.shape == (2, 2, 4)

    def test_single_distance_no_layer_dim(self, nf_geometry, device):
        """Single-distance geometry: y_pixel has no D dimension, layer_valid is None."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        spots = model(euler, pos)

        # For single distance, y_pixel should be (..., 2N, M) not (1, ..., 2N, M)
        assert spots.y_pixel.shape == (2, 4)
        assert spots.layer_valid is None

    def test_different_distances_different_pixels(self, nf_multi_geometry, device):
        """Spots project to different pixel coords at different distances."""
        model, _, _ = make_model_with_cubic_iron(nf_multi_geometry, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        spots = model(euler, pos)

        y0 = spots.y_pixel[0]  # distance 0
        y1 = spots.y_pixel[1]  # distance 1
        # Different Lsd -> different pixel positions
        assert not torch.allclose(y0, y1, atol=1e-3), \
            "Different distances should produce different pixel positions"

    def test_all_distances_must_be_valid(self, nf_multi_geometry, device):
        """A spot valid at distance 0 but not distance 1 should be marked invalid."""
        # Use a geometry where the second distance has a very small detector
        geom = HEDMGeometry(
            Lsd=[5000.0, 50000.0],  # second detector very far -> spots spread
            y_BC=[1024.0, 1024.0],
            z_BC=[1024.0, 1024.0],
            px=1.48,
            omega_start=180.0,
            omega_step=-0.25,
            n_frames=1440,
            n_pixels_y=2048,
            n_pixels_z=2048,
            min_eta=6.0,
            wavelength=0.295,
        )
        model, _, _ = make_model_with_cubic_iron(geom, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        spots = model(euler, pos)

        # Some spots might fall off the far detector but stay on the near one
        # The combined valid should be <= each layer's valid
        layer0_count = spots.layer_valid[0].sum()
        layer1_count = spots.layer_valid[1].sum()
        combined_count = spots.valid.sum()
        assert combined_count <= layer0_count
        assert combined_count <= layer1_count

    def test_multi_distance_differentiable(self, nf_multi_geometry, device):
        """Gradients flow through multi-distance projection."""
        model, _, _ = make_model_with_cubic_iron(nf_multi_geometry, device)
        euler = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        pos = torch.tensor([[10.0, 20.0, 0.0]])
        spots = model(euler, pos)
        loss = (spots.y_pixel[0] * spots.valid).sum()
        loss.backward()
        assert euler.grad is not None
        assert torch.all(torch.isfinite(euler.grad))


class TestFilterByScan:
    """Tests for the beam proximity filter (pf-HEDM scanning)."""

    def test_no_scan_config_passthrough(self, nf_geometry, device):
        """Without scan_config, filter_by_scan is a no-op."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        spots = model(euler, pos)
        spots2 = model.filter_by_scan(spots, pos)
        assert spots2.scan_mask is None

    def test_scan_filter_basic(self, nf_geometry, device):
        """Beam at y=0 with size 100um should include grain at y=0."""
        scan_config = ScanConfig(
            beam_positions=torch.tensor([0.0]),
            beam_size=100.0,
        )
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        model.scan_config = scan_config
        model._beam_positions = scan_config.beam_positions
        model._beam_size = scan_config.beam_size

        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)  # grain at origin
        spots = model(euler, pos)
        spots = model.filter_by_scan(spots, pos)

        assert spots.scan_mask is not None
        # Grain at y=0, beam at y=0 -> should be illuminated
        assert spots.scan_mask.sum() > 0

    def test_scan_filter_excludes_distant_grain(self, nf_geometry, device):
        """Beam at y=0 with size 100um should exclude grain at y=1000."""
        scan_config = ScanConfig(
            beam_positions=torch.tensor([0.0]),
            beam_size=100.0,
        )
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        model.scan_config = scan_config
        model._beam_positions = scan_config.beam_positions
        model._beam_size = scan_config.beam_size

        euler = torch.rand(1, 3) * 0.1
        pos = torch.tensor([[0.0, 1000.0, 0.0]])  # far from beam
        spots = model(euler, pos)
        spots = model.filter_by_scan(spots, pos)

        # At most omega angles, yRot = 0*sin(w)+1000*cos(w) = 1000*cos(w)
        # which is >> 50um for most omega values
        # Some spots near omega=pi/2 might sneak through, but most should be filtered
        total_valid = spots.valid.sum().item()
        scan_valid = spots.scan_mask.sum().item()
        if total_valid > 0:
            assert scan_valid < total_valid, "Distant grain should have fewer scan-valid spots"

    def test_multiple_scan_positions(self, nf_geometry, device):
        """Multiple beam positions create proper scan_mask shape."""
        scan_config = ScanConfig(
            beam_positions=torch.tensor([-100.0, 0.0, 100.0]),
            beam_size=100.0,
        )
        geometry = nf_geometry
        model, _, _ = make_model_with_cubic_iron(geometry, device)
        model.scan_config = scan_config
        model._beam_positions = scan_config.beam_positions
        model._beam_size = scan_config.beam_size

        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        spots = model(euler, pos)
        spots = model.filter_by_scan(spots, pos)

        # scan_mask should have S=3 as a dimension
        assert spots.scan_mask.shape[-3] == 3


# ===================================================================
#  Test: predict_images
# ===================================================================

class TestPredictImages:
    """Tests for Gaussian splatting to detector grid."""

    def test_basic_output_shape(self, nf_geometry, device):
        """predict_images returns correct shape."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        spots = model(euler, pos)

        # Use small grid for speed
        images = HEDMForwardModel.predict_images(
            spots, n_frames=100, n_pixels_y=64, n_pixels_z=64,
            sigma=0.5, radius=1
        )
        assert images.shape == (100, 64, 64)

    def test_nonzero_output(self, nf_geometry, device):
        """At least some pixels should be nonzero if there are valid spots."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        spots = model(euler, pos)

        n_valid = spots.valid.sum().item()
        if n_valid > 0:
            images = HEDMForwardModel.predict_images(
                spots, n_frames=model.n_frames, n_pixels_y=model.n_pixels_y,
                n_pixels_z=model.n_pixels_z, sigma=1.0, radius=2
            )
            assert images.sum() > 0, "Should have nonzero pixels for valid spots"

    def test_no_valid_spots(self, nf_geometry, device):
        """All-invalid spots should produce zero images."""
        spots = SpotDescriptors(
            omega=torch.zeros(2, 4),
            eta=torch.zeros(2, 4),
            two_theta=torch.zeros(2, 4),
            y_pixel=torch.zeros(2, 4),
            z_pixel=torch.zeros(2, 4),
            frame_nr=torch.zeros(2, 4),
            valid=torch.zeros(2, 4),
        )
        images = HEDMForwardModel.predict_images(spots, 10, 32, 32, sigma=1.0, radius=1)
        assert images.sum() == 0


# ===================================================================
#  Test: predict_spot_coords
# ===================================================================

class TestPredictSpotCoords:
    """Tests for FF/pf output mode."""

    def test_angular_mode(self, nf_geometry, device):
        """Angular mode returns (2theta, eta, omega)."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        spots = model(euler, pos)
        coords, valid = HEDMForwardModel.predict_spot_coords(spots, space="angular")
        assert coords.shape[-1] == 3
        # First coordinate should be 2*theta
        torch.testing.assert_close(coords[..., 0], spots.two_theta)

    def test_detector_mode(self, nf_geometry, device):
        """Detector mode returns (y_pixel, z_pixel, frame_nr)."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        euler = torch.rand(1, 3) * 0.1
        pos = torch.zeros(1, 3)
        spots = model(euler, pos)
        coords, valid = HEDMForwardModel.predict_spot_coords(spots, space="detector")
        assert coords.shape[-1] == 3
        torch.testing.assert_close(coords[..., 0], spots.y_pixel)

    def test_invalid_space(self):
        """Should raise for unknown space."""
        spots = SpotDescriptors(
            omega=torch.zeros(2, 4), eta=torch.zeros(2, 4),
            two_theta=torch.zeros(2, 4), y_pixel=torch.zeros(2, 4),
            z_pixel=torch.zeros(2, 4), frame_nr=torch.zeros(2, 4),
            valid=torch.zeros(2, 4),
        )
        with pytest.raises(ValueError, match="Unknown space"):
            HEDMForwardModel.predict_spot_coords(spots, space="invalid")


# ===================================================================
#  Test: safe_arccos
# ===================================================================

class TestSafeArccos:
    """Tests for numerically safe arccos."""

    def test_normal_range(self, nf_geometry, device):
        """Normal values: should match torch.acos."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        x = torch.tensor([-0.5, 0.0, 0.5])
        result = model.safe_arccos(x)
        expected = torch.acos(x)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)

    def test_boundary_values(self, nf_geometry, device):
        """Values at +/-1 should not produce NaN."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        x = torch.tensor([-1.0, 1.0, -1.0001, 1.0001])
        result = model.safe_arccos(x)
        assert torch.all(torch.isfinite(result))

    def test_gradient_at_boundary(self, nf_geometry, device):
        """Gradients should be finite at boundary."""
        model, _, _ = make_model_with_cubic_iron(nf_geometry, device)
        x = torch.tensor([0.9999], requires_grad=True)
        result = model.safe_arccos(x)
        result.backward()
        assert torch.all(torch.isfinite(x.grad))


# ===================================================================
#  Test: DisplacementSpots formula validation
# ===================================================================

class TestDisplacementSpotsFormula:
    """Validate detector projection formula against C code.

    C code (SharedFuncsFit.c:269-280):
        xa = a*cos(omega) - b*sin(omega)
        ya = a*sin(omega) + b*cos(omega)
        t = 1 - xa/Lsd
        Displ_y = ya + yi * t
        Displ_z = t * zi

    Where yi = -sin(eta)*Lsd*tan(2theta), zi = cos(eta)*Lsd*tan(2theta)

    Python equivalent:
        ydet = y_grain - (Lsd - x_grain) * tan(2theta) * sin(eta)
        zdet = z_grain + (Lsd - x_grain) * tan(2theta) * cos(eta)
    """

    def test_formula_equivalence(self):
        """Verify Python formula matches C formula for known values."""
        a, b = 50.0, 30.0  # grain position (x, y) in um
        Lsd = 5105.0
        omega_deg = 45.0
        eta_rad = 0.3
        theta_rad = 0.05  # Bragg angle

        omega_rad = omega_deg * math.pi / 180

        # C code path
        xa = a * math.cos(omega_rad) - b * math.sin(omega_rad)
        ya = a * math.sin(omega_rad) + b * math.cos(omega_rad)
        t = 1.0 - xa / Lsd
        yi = -math.sin(eta_rad) * Lsd * math.tan(2 * theta_rad)
        zi = math.cos(eta_rad) * Lsd * math.tan(2 * theta_rad)
        displ_y_c = ya + yi * t
        displ_z_c = t * zi  # z_grain = 0

        # Python path
        x_grain = xa  # same omega rotation
        y_grain = ya
        z_grain = 0.0
        ydet_py = y_grain - (Lsd - x_grain) * math.tan(2 * theta_rad) * math.sin(eta_rad)
        zdet_py = z_grain + (Lsd - x_grain) * math.tan(2 * theta_rad) * math.cos(eta_rad)

        assert abs(displ_y_c - ydet_py) < 1e-10, f"y: C={displ_y_c}, Py={ydet_py}"
        assert abs(displ_z_c - zdet_py) < 1e-10, f"z: C={displ_z_c}, Py={zdet_py}"

    def test_formula_with_z_position(self):
        """When z_grain != 0, verify z displacement includes it."""
        a, b, c = 50.0, 30.0, 20.0  # x, y, z position
        Lsd = 5105.0
        omega_rad = 0.7
        eta_rad = 0.3
        theta_rad = 0.05

        xa = a * math.cos(omega_rad) - b * math.sin(omega_rad)
        ya = a * math.sin(omega_rad) + b * math.cos(omega_rad)
        z_grain = c  # z unchanged by omega rotation around z-axis

        ydet = ya - (Lsd - xa) * math.tan(2 * theta_rad) * math.sin(eta_rad)
        zdet = z_grain + (Lsd - xa) * math.tan(2 * theta_rad) * math.cos(eta_rad)

        # Without z_grain
        zdet_no_z = 0.0 + (Lsd - xa) * math.tan(2 * theta_rad) * math.cos(eta_rad)

        assert abs(zdet - zdet_no_z - c) < 1e-10


# ===================================================================
#  Test: cross-validation with original nfhedm.py
# ===================================================================

class TestCrossValidation:
    """Compare HEDMForwardModel output against original NFHEDM.

    These tests import the original nfhedm.py and verify that the
    new model produces equivalent results.
    """

    def _get_original_nfhedm(self):
        """Import original NFHEDM class."""
        try:
            from nfhedm import NFHEDM
            return NFHEDM
        except ImportError:
            pytest.skip("Original nfhedm.py not importable")

    def test_euler2mat_matches(self):
        """euler2mat matches original euler2Mat for random angles."""
        NFHEDM = self._get_original_nfhedm()

        # Create dummy model for old code (needs hkls/thetas for __init__)
        hkls = torch.randn(2, 3)
        thetas = torch.rand(2) * 0.1
        args = {}
        old = NFHEDM(hkls, thetas, args, torch.device("cpu"))

        torch.manual_seed(42)
        angles = torch.rand(5, 3) * 2 * math.pi  # (5, 3)

        # Old code needs specific batch dims; test single angles
        for i in range(5):
            a = angles[i]
            R_old = old.euler2Mat(a.unsqueeze(0))  # (1, 3, 3)
            R_new = HEDMForwardModel.euler2mat(a)   # (3, 3)
            torch.testing.assert_close(
                R_new, R_old.squeeze(0), atol=1e-6, rtol=0,
                msg=f"Mismatch at angle index {i}"
            )


import torch.nn.functional as F  # for test_2d_position_backward_compat


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
