"""Torch-backend tests for midas_stress.orientation.

Verifies that:
  1. When given torch.Tensor inputs, the public functions return torch.Tensor.
  2. The torch path agrees numerically with the NumPy path to float64 tolerance.
  3. Outputs land on the same device as the input.
  4. The torch path is autograd-friendly (gradients flow).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from midas_stress.orientation import (
    axis_angle_to_orient_mat,
    euler_to_orient_mat,
    euler_to_orient_mat_batch,
    fundamental_zone,
    make_symmetries,
    matrix_mult_f33,
    misorientation_om,
    misorientation_om_batch,
    misorientation_quat_batch,
    orient_mat_to_euler,
    orient_mat_to_quat,
    quat_to_orient_mat,
    quaternion_product,
    rodrigues_to_orient_mat,
)


def _eq(a, b, atol=1e-12):
    a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    b = np.asarray(b, dtype=np.float64)
    np.testing.assert_allclose(a, b, atol=atol)


# ---------------------------------------------------------------------------
# Identity / round-trip sanity
# ---------------------------------------------------------------------------


def test_euler_to_orient_mat_torch_returns_tensor_and_matches_numpy():
    eu = [0.5, 1.0, 1.5]
    np_om = euler_to_orient_mat(eu)
    t_om = euler_to_orient_mat(torch.tensor(eu, dtype=torch.float64))
    assert isinstance(t_om, torch.Tensor)
    assert t_om.shape == (9,)
    _eq(t_om, np_om)


def test_euler_to_orient_mat_batch_torch_matches_numpy():
    eu = np.array([[0.0, 0.0, 0.0],
                   [0.5, 1.0, 1.5],
                   [1.7, -0.3, 0.9]])
    np_oms = euler_to_orient_mat_batch(eu)
    t_oms = euler_to_orient_mat_batch(torch.tensor(eu, dtype=torch.float64))
    assert isinstance(t_oms, torch.Tensor)
    assert t_oms.shape == (3, 9)
    _eq(t_oms, np_oms)


def test_axis_angle_to_orient_mat_torch_matches_numpy():
    axis = [0.0, 0.0, 1.0]
    angle_deg = 30.0
    np_R = axis_angle_to_orient_mat(axis, angle_deg)
    t_R = axis_angle_to_orient_mat(
        torch.tensor(axis, dtype=torch.float64),
        torch.tensor(angle_deg, dtype=torch.float64),
    )
    assert isinstance(t_R, torch.Tensor)
    assert t_R.shape == (3, 3)
    _eq(t_R, np_R)


def test_axis_angle_to_orient_mat_torch_batched():
    axes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        dtype=torch.float64)
    angles = torch.tensor([10.0, 25.0, 90.0], dtype=torch.float64)
    R = axis_angle_to_orient_mat(axes, angles)
    assert R.shape == (3, 3, 3)
    # Each row is a valid rotation
    for i in range(3):
        np.testing.assert_allclose(
            (R[i] @ R[i].T).numpy(), np.eye(3), atol=1e-12
        )
        np.testing.assert_allclose(torch.det(R[i]).item(), 1.0, atol=1e-12)


def test_orient_mat_to_quat_torch_matches_numpy():
    eu = [0.8, 1.2, 0.4]
    np_om = euler_to_orient_mat(eu)
    np_q = orient_mat_to_quat(np_om)
    t_om = euler_to_orient_mat(torch.tensor(eu, dtype=torch.float64))
    t_q = orient_mat_to_quat(t_om)
    assert isinstance(t_q, torch.Tensor)
    assert t_q.shape == (4,)
    _eq(t_q, np_q, atol=1e-10)


def test_orient_mat_to_quat_torch_accepts_3x3():
    eu_t = torch.tensor([0.8, 1.2, 0.4], dtype=torch.float64)
    om_flat = euler_to_orient_mat(eu_t)
    om_3x3 = om_flat.reshape(3, 3)
    q_flat = orient_mat_to_quat(om_flat)
    q_3x3 = orient_mat_to_quat(om_3x3)
    _eq(q_3x3, q_flat.numpy(), atol=1e-12)


def test_quat_to_orient_mat_torch_matches_numpy():
    q = [0.5, 0.5, 0.5, 0.5]
    np_om = quat_to_orient_mat(q)
    t_om = quat_to_orient_mat(torch.tensor(q, dtype=torch.float64))
    assert isinstance(t_om, torch.Tensor)
    assert t_om.shape == (9,)
    _eq(t_om, np_om)


def test_orient_mat_to_euler_torch_matches_numpy():
    eu = [0.7, 1.4, 2.0]
    om = euler_to_orient_mat(eu)
    np_eu = orient_mat_to_euler(om)
    t_eu = orient_mat_to_euler(torch.tensor(om, dtype=torch.float64))
    assert isinstance(t_eu, torch.Tensor)
    assert t_eu.shape == (3,)
    _eq(t_eu, np_eu, atol=1e-10)


def test_quaternion_product_torch_matches_numpy():
    q = [0.5, 0.5, 0.5, 0.5]
    r = [0.70711, 0.70711, 0.0, 0.0]
    np_qr = quaternion_product(q, r)
    t_qr = quaternion_product(
        torch.tensor(q, dtype=torch.float64),
        torch.tensor(r, dtype=torch.float64),
    )
    assert isinstance(t_qr, torch.Tensor)
    _eq(t_qr, np_qr, atol=1e-10)


def test_rodrigues_to_orient_mat_torch_matches_numpy():
    rod = [0.1, 0.2, 0.3]
    np_R = rodrigues_to_orient_mat(rod)
    t_R = rodrigues_to_orient_mat(torch.tensor(rod, dtype=torch.float64))
    assert isinstance(t_R, torch.Tensor)
    _eq(t_R, np_R, atol=1e-10)


def test_rodrigues_zero_returns_identity():
    R = rodrigues_to_orient_mat(torch.zeros(3, dtype=torch.float64))
    np.testing.assert_allclose(R.numpy(), np.eye(3), atol=1e-12)


# ---------------------------------------------------------------------------
# Symmetry / FZ / misorientation
# ---------------------------------------------------------------------------


def test_fundamental_zone_torch_matches_numpy():
    q = [0.7, 0.5, 0.3, 0.4]
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    np_qFR = fundamental_zone(q.tolist(), 225)
    t_qFR = fundamental_zone(torch.tensor(q, dtype=torch.float64), 225)
    assert isinstance(t_qFR, torch.Tensor)
    _eq(t_qFR, np_qFR, atol=1e-10)


def test_misorientation_om_torch_matches_numpy():
    eu1 = [0.1, 0.2, 0.3]
    eu2 = [0.1, 0.2, 0.4]   # small rotation around theta
    om1 = euler_to_orient_mat(eu1)
    om2 = euler_to_orient_mat(eu2)
    np_ang, _ = misorientation_om(om1, om2, 225)
    t_ang, t_axis = misorientation_om(
        torch.tensor(om1, dtype=torch.float64),
        torch.tensor(om2, dtype=torch.float64),
        225,
    )
    assert isinstance(t_ang, torch.Tensor)
    assert isinstance(t_axis, torch.Tensor)
    _eq(t_ang, np_ang, atol=1e-9)


def test_misorientation_om_batch_torch_matches_numpy():
    eulers = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [1.2, 0.0, -0.3]])
    eulers2 = eulers + np.array([[0.0, 0.0, 0.01], [0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
    oms1 = euler_to_orient_mat_batch(eulers)
    oms2 = euler_to_orient_mat_batch(eulers2)
    np_angs = misorientation_om_batch(oms1, oms2, 225)
    t_angs = misorientation_om_batch(
        torch.tensor(oms1, dtype=torch.float64),
        torch.tensor(oms2, dtype=torch.float64),
        225,
    )
    assert isinstance(t_angs, torch.Tensor)
    assert t_angs.shape == (3,)
    _eq(t_angs, np_angs, atol=1e-9)


def test_misorientation_quat_batch_torch_matches_numpy():
    qs1 = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5, 0.5]])
    qs1 = qs1 / np.linalg.norm(qs1, axis=-1, keepdims=True)
    qs2 = qs1 + 0.01
    qs2 = qs2 / np.linalg.norm(qs2, axis=-1, keepdims=True)
    np_a = misorientation_quat_batch(qs1, qs2, 225)
    t_a = misorientation_quat_batch(
        torch.tensor(qs1, dtype=torch.float64),
        torch.tensor(qs2, dtype=torch.float64),
        225,
    )
    assert isinstance(t_a, torch.Tensor)
    _eq(t_a, np_a, atol=1e-9)


# ---------------------------------------------------------------------------
# Device + dtype propagation
# ---------------------------------------------------------------------------


def test_torch_outputs_keep_input_dtype():
    eu32 = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32)
    om = euler_to_orient_mat(eu32)
    assert om.dtype == torch.float32

    eu64 = eu32.to(torch.float64)
    om2 = euler_to_orient_mat(eu64)
    assert om2.dtype == torch.float64


def test_torch_outputs_keep_input_device():
    # CPU is always available; just confirm round-trip preserves device.
    eu = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64, device="cpu")
    om = euler_to_orient_mat(eu)
    assert om.device.type == "cpu"


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


def test_axis_angle_to_orient_mat_is_differentiable():
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
    angle = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    R = axis_angle_to_orient_mat(axis, angle)
    loss = (R ** 2).sum()
    loss.backward()
    assert axis.grad is not None
    assert angle.grad is not None
    assert torch.isfinite(axis.grad).all()
    assert torch.isfinite(angle.grad).all()


def test_euler_to_orient_mat_is_differentiable():
    eu = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
    om = euler_to_orient_mat(eu)
    om.sum().backward()
    assert eu.grad is not None
    assert torch.isfinite(eu.grad).all()


# ---------------------------------------------------------------------------
# Cross-backend equivalence on a random sweep
# ---------------------------------------------------------------------------


def test_cross_backend_random_eulers():
    rng = np.random.default_rng(0)
    eulers = rng.uniform(-math.pi, math.pi, size=(20, 3))
    np_oms = euler_to_orient_mat_batch(eulers)
    t_oms = euler_to_orient_mat_batch(torch.tensor(eulers, dtype=torch.float64))
    np.testing.assert_allclose(t_oms.numpy(), np_oms, atol=1e-12)


# ---------------------------------------------------------------------------
# matrix_mult_f33 + fundamental_zone(sym=) torch parity
# ---------------------------------------------------------------------------


def test_matrix_mult_f33_torch_matches_numpy():
    rng = np.random.default_rng(123)
    A = rng.standard_normal((3, 3))
    B = rng.standard_normal((3, 3))
    np_out = matrix_mult_f33(A, B)
    t_out = matrix_mult_f33(torch.tensor(A), torch.tensor(B))
    assert isinstance(t_out, torch.Tensor)
    _eq(t_out, np_out, atol=1e-14)


def test_matrix_mult_f33_torch_batched():
    rng = np.random.default_rng(321)
    A_batch = torch.tensor(rng.standard_normal((5, 3, 3)))
    B_batch = torch.tensor(rng.standard_normal((5, 3, 3)))
    out = matrix_mult_f33(A_batch, B_batch)
    assert out.shape == (5, 3, 3)
    for i in range(5):
        _eq(out[i], A_batch[i].numpy() @ B_batch[i].numpy(), atol=1e-14)


def test_matrix_mult_f33_torch_is_differentiable():
    A = torch.eye(3, dtype=torch.float64, requires_grad=True)
    B = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                     dtype=torch.float64)
    out = matrix_mult_f33(A, B)
    out.sum().backward()
    assert A.grad is not None and A.grad.shape == (3, 3)


def test_fundamental_zone_torch_with_precomputed_sym():
    """Torch path: caller-supplied sym tensor matches the default lookup."""
    q = torch.tensor([0.1, 0.4, 0.5, 0.7], dtype=torch.float64)
    q = q / torch.linalg.norm(q)
    _, sym_list = make_symmetries(225)
    sym_tensor = torch.tensor(sym_list, dtype=torch.float64)
    q_default = fundamental_zone(q, 225)
    q_with_sym = fundamental_zone(q, sym=sym_tensor)
    _eq(q_with_sym, q_default.numpy(), atol=1e-14)


def test_fundamental_zone_torch_sym_dtype_device_follows_input():
    """The torch path should move the sym table onto the input's device/dtype."""
    q = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
    _, sym_list = make_symmetries(225)
    # Pass sym as a list (numpy/list); the torch path must coerce to fp32 / CPU.
    out = fundamental_zone(q, sym=sym_list)
    assert out.dtype == torch.float32
    assert out.device == q.device


# ---------------------------------------------------------------------------
# Device-portability gates: skip when the device isn't present
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_matrix_mult_f33_runs_on_cuda():
    A = torch.eye(3, dtype=torch.float64, device="cuda")
    B = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                     dtype=torch.float64, device="cuda")
    out = matrix_mult_f33(A, B)
    assert out.device.type == "cuda"
    _eq(out.cpu(), B.cpu().numpy(), atol=1e-14)


@pytest.mark.skipif(not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                    reason="MPS not available")
def test_matrix_mult_f33_runs_on_mps():
    A = torch.eye(3, dtype=torch.float32, device="mps")
    B = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                     dtype=torch.float32, device="mps")
    out = matrix_mult_f33(A, B)
    assert out.device.type == "mps"
    np.testing.assert_allclose(out.cpu().numpy(), B.cpu().numpy(), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fundamental_zone_with_sym_runs_on_cuda():
    q = torch.tensor([0.1, 0.4, 0.5, 0.7], dtype=torch.float64, device="cuda")
    q = q / torch.linalg.norm(q)
    _, sym_list = make_symmetries(225)
    sym = torch.tensor(sym_list, dtype=torch.float64, device="cuda")
    out = fundamental_zone(q, sym=sym)
    assert out.device.type == "cuda"
    _eq(out.cpu(), fundamental_zone(q.cpu(), sym=sym.cpu()).numpy(), atol=1e-14)


@pytest.mark.skipif(not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                    reason="MPS not available")
def test_fundamental_zone_with_sym_runs_on_mps():
    q = torch.tensor([0.1, 0.4, 0.5, 0.7], dtype=torch.float32, device="mps")
    q = q / torch.linalg.norm(q)
    _, sym_list = make_symmetries(225)
    sym = torch.tensor(sym_list, dtype=torch.float32, device="mps")
    out = fundamental_zone(q, sym=sym)
    assert out.device.type == "mps"


def test_rodrigues_torch_has_the_right_angle_not_just_numpy_parity():
    """Parity with numpy is not correctness when both backends share an error.

    test_rodrigues_to_orient_mat_torch_matches_numpy passed throughout the
    period when both returned theta/cos^2(theta/2) instead of theta. Assert the
    absolute angle on the torch path too.
    """
    import math

    for deg in (30.0, 60.0, 90.0):
        r = math.tan(math.radians(deg) / 2.0)
        R = rodrigues_to_orient_mat(torch.tensor([0.0, 0.0, r], dtype=torch.float64))
        cos_t = (torch.diagonal(R, dim1=-2, dim2=-1).sum() - 1.0) / 2.0
        got = math.degrees(math.acos(max(-1.0, min(1.0, float(cos_t)))))
        assert abs(got - deg) < 1e-9, f"{deg} deg came back as {got}"


def test_rodrigues_torch_batched_angles_are_right():
    """The batched path must not be correct only in the (1,) case."""
    import math

    degs = [10.0, 45.0, 90.0, 150.0]
    rods = torch.tensor([[0.0, 0.0, math.tan(math.radians(d) / 2.0)] for d in degs],
                        dtype=torch.float64)
    R = rodrigues_to_orient_mat(rods)
    assert R.shape == (len(degs), 3, 3)
    cos_t = (torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    for d, c in zip(degs, cos_t.tolist()):
        got = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        assert abs(got - d) < 1e-9, f"{d} deg came back as {got}"
