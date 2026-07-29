"""Phase 0 tests: conventions (frames/goniometer) + deformation field."""
import math

import pytest
import torch

from midas_dfxm import (
    DeformationField,
    GoniometerSetting,
    deform_reflection,
    make_uniform_field,
    polar_decomposition,
    reciprocal_basis,
    rot_x,
    rot_y,
    rot_z,
    rotation_matrix,
    with_orientation_gradient,
    with_screw_dislocation,
    with_uniform_strain,
)

DT = torch.float64


# --------------------------------------------------------------------------
# conventions
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_rotation_roundtrip_is_identity():
    for rot in (rot_x, rot_y, rot_z):
        R = rot(37.0) @ rot(-37.0)
        assert torch.allclose(R, torch.eye(3, dtype=R.dtype), atol=1e-12)


@pytest.mark.unit
def test_rotation_is_special_orthogonal():
    R = rotation_matrix((1.0, 1.0, 0.0), 53.0)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=R.dtype), atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.tensor(1.0, dtype=R.dtype), atol=1e-12)


@pytest.mark.unit
def test_rot_z_known_action():
    # 90 deg about z sends xhat -> yhat.
    R = rot_z(90.0)
    x = torch.tensor([1.0, 0.0, 0.0], dtype=R.dtype)
    assert torch.allclose(R @ x, torch.tensor([0.0, 1.0, 0.0], dtype=R.dtype), atol=1e-12)


@pytest.mark.unit
def test_goniometer_identity_at_zero():
    G = GoniometerSetting().sample_rotation(dtype=DT)
    assert torch.allclose(G, torch.eye(3, dtype=DT), atol=1e-12)


@pytest.mark.unit
def test_goniometer_is_rotation():
    G = GoniometerSetting(mu=5.0, omega=12.0, chi=-3.0, phi=7.0).sample_rotation(dtype=DT)
    assert torch.allclose(G @ G.T, torch.eye(3, dtype=DT), atol=1e-12)
    assert torch.allclose(torch.linalg.det(G), torch.tensor(1.0, dtype=DT), atol=1e-12)


@pytest.mark.autograd
def test_goniometer_compose_differentiable():
    ang = torch.tensor([5.0, 12.0, -3.0, 7.0], dtype=DT, requires_grad=True)
    G = GoniometerSetting.compose(ang[0], ang[1], ang[2], ang[3])
    G.sum().backward()
    assert ang.grad is not None and torch.isfinite(ang.grad).all()


# --------------------------------------------------------------------------
# deform operator
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_deform_identity_preserves_G():
    F = torch.eye(3, dtype=DT).expand(5, 3, 3)
    G0 = torch.tensor([1.0, 2.0, 3.0], dtype=DT)
    Q = deform_reflection(F, G0)
    assert torch.allclose(Q, G0.expand(5, 3), atol=1e-12)


@pytest.mark.unit
def test_deform_pure_rotation_rotates_G():
    # Q = F^{-T} G0; for F a pure rotation R, F^{-T} = R, so Q = R G0.
    R = rot_z(30.0).to(DT)
    G0 = torch.tensor([1.0, 0.0, 0.0], dtype=DT)
    Q = deform_reflection(R.unsqueeze(0), G0)
    assert torch.allclose(Q[0], R @ G0, atol=1e-12)


@pytest.mark.unit
def test_deform_uniaxial_strain_shifts_magnitude():
    # Stretch along x by (1+e): reciprocal component along x shrinks by 1/(1+e).
    e = 1e-3
    F = torch.diag(torch.tensor([1 + e, 1.0, 1.0], dtype=DT)).unsqueeze(0)
    G0 = torch.tensor([1.0, 0.0, 0.0], dtype=DT)
    Q = deform_reflection(F, G0)
    assert Q[0, 0].item() == pytest.approx(1.0 / (1 + e), rel=1e-9)


@pytest.mark.autograd
def test_deform_reflection_gradcheck():
    F = torch.eye(3, dtype=DT).expand(3, 3, 3).clone()
    F = F + 1e-2 * torch.randn(3, 3, 3, dtype=DT)
    F.requires_grad_(True)
    G0 = torch.tensor([1.0, 0.5, -0.3], dtype=DT)
    assert torch.autograd.gradcheck(lambda f: deform_reflection(f, G0), (F,), atol=1e-6)


# --------------------------------------------------------------------------
# polar decomposition
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_polar_reconstructs_F():
    torch.manual_seed(0)
    F = torch.eye(3, dtype=DT) + 5e-2 * torch.randn(3, 3, dtype=DT)
    R, U = polar_decomposition(F.unsqueeze(0))
    assert torch.allclose(R[0] @ U[0], F, atol=1e-10)
    # R orthogonal, U symmetric.
    assert torch.allclose(R[0] @ R[0].T, torch.eye(3, dtype=DT), atol=1e-10)
    assert torch.allclose(U[0], U[0].T, atol=1e-10)


@pytest.mark.unit
def test_polar_of_rotation_gives_identity_stretch():
    R_in = rot_y(20.0).to(DT)
    R, U = polar_decomposition(R_in.unsqueeze(0))
    assert torch.allclose(U[0], torch.eye(3, dtype=DT), atol=1e-10)
    assert torch.allclose(R[0], R_in, atol=1e-10)


# --------------------------------------------------------------------------
# reciprocal basis + field
# --------------------------------------------------------------------------
@pytest.mark.unit
def test_reciprocal_basis_cubic_magnitude():
    a = 3.6356
    latc = torch.tensor([a, a, a, 90.0, 90.0, 90.0], dtype=DT)
    B = reciprocal_basis(latc)
    G = B @ torch.tensor([1.0, 1.0, 1.0], dtype=DT)
    d111 = a / math.sqrt(3.0)
    assert torch.linalg.vector_norm(G).item() == pytest.approx(2 * math.pi / d111, rel=1e-9)


@pytest.mark.unit
def test_orientation_gradient_recovers_curvature():
    field = make_uniform_field(shape=(21, 1, 1), spacing_um=1.0, dtype=DT)
    field = with_orientation_gradient(field, axis=(0, 0, 1), deg_per_um=0.02, along=0)
    R, U = field.local_rotation_stretch()
    # Stretch stays identity; rotation angle grows linearly along x.
    assert torch.allclose(U, torch.eye(3, dtype=DT).expand_as(U), atol=1e-10)
    # angle at the two ends: total span (20 um) * 0.02 deg/um = 0.4 deg apart.
    def z_angle(Rm):
        return math.degrees(math.atan2(Rm[1, 0].item(), Rm[0, 0].item()))
    span = z_angle(R[-1]) - z_angle(R[0])
    assert span == pytest.approx(0.4, abs=1e-6)


@pytest.mark.unit
def test_screw_dislocation_local_Q_finite_and_perturbs():
    field = make_uniform_field(shape=(16, 16, 1), spacing_um=1.0, dtype=DT)
    field = with_screw_dislocation(field, burgers_A=2.556, core_radius_um=0.5)
    Q = field.local_Q((1, 1, 1))
    assert torch.isfinite(Q).all()
    Q0 = make_uniform_field(shape=(16, 16, 1), spacing_um=1.0, dtype=DT).local_Q((1, 1, 1))
    # The dislocation perturbs some voxels away from the reference Q.
    assert (torch.linalg.vector_norm(Q - Q0, dim=-1) > 1e-9).any()


@pytest.mark.autograd
def test_local_Q_differentiable_in_F():
    field = make_uniform_field(shape=(4, 4, 1), spacing_um=1.0, dtype=DT)
    field.F.requires_grad_(True)
    Q = field.local_Q((2, 0, 0))
    Q.abs().sum().backward()
    assert field.F.grad is not None and torch.isfinite(field.F.grad).all()


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_field_device_portable(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps unavailable")
    dt = torch.float32 if device == "mps" else DT
    field = make_uniform_field(shape=(8, 8, 1), spacing_um=1.0, device=device, dtype=dt)
    field = with_uniform_strain(field, 1e-3 * torch.eye(3, device=device, dtype=dt))
    Q = field.local_Q((1, 1, 1))
    assert Q.device.type == device
    assert torch.isfinite(Q).all()
