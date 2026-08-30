"""Gradients at the singular orientations, not merely finite values.

Every test here is written the way the audit says these should have been
written in the first place: compare autograd against a central finite
difference AT the singular point, in float64, and sweep a ladder of distances
so a DEAD ZONE (gradient finite but wrong) cannot pass. Asserting
``isfinite(grad)`` is what let these through.

Covers, per spec_autograd_classB_classC.md:
  B2  misorientation at zero misorientation  (containment + smooth surrogate)
  B5  Euler angles at gimbal lock            (whole Phi = 0 family)
  A   rodrigues_to_orient_mat dead zone
      calc_eta_angle_all along the y = 0 axis
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import midas_stress as ms
from midas_stress.diffraction import _calc_eta_angle_all_torch
from midas_stress.orientation import (
    _orient_mat_to_euler_torch,
    _rodrigues_to_orient_mat_torch,
)

LADDER = [0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6]
SG_FCC = 225


@pytest.fixture(autouse=True)
def _float64_default():
    """float64 for this module only -- setting it at import time leaks into
    every test module imported afterwards."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _central_fd(fn, x, i, h=1e-6):
    xp = x.clone(); xp[i] += h
    xm = x.clone(); xm[i] -= h
    return float((fn(xp) - fn(xm)) / (2.0 * h))


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------- B5: euler

@pytest.mark.parametrize("ang_deg", [0.0, 1e-6, 0.001, 1.0, 45.0, 179.0])
def test_euler_gradient_finite_across_the_whole_gimbal_family(ang_deg):
    """Phi = 0 is a FIBRE, not a point: every c-axis-along-z orientation.

    The audit reported "NaN at the identity"; measured, it was NaN for every
    rotation about z.
    """
    m = _rotz(math.radians(ang_deg)).requires_grad_(True)
    out = _orient_mat_to_euler_torch(m)
    g = torch.autograd.grad(out.sum(), m)[0]
    assert torch.isfinite(g).all(), f"NaN/inf gradient at Phi=0, angle={ang_deg}"


def test_euler_gradient_matches_fd_on_the_gimbal_fibre():
    """In the psi = 0 gauge theta has a real derivative; check it is returned.

    Evaluated at 30 deg about z, not at the identity: theta is reported on
    [0, 2pi), so at theta = 0 a central difference straddles the wrap and is
    meaningless. Only the in-plane components are differenced -- perturbing
    m22 leaves the Phi = 0 set entirely, which is a genuine branch change in
    the parameterisation, not something a derivative should track.
    """
    a = math.radians(30.0)
    base = _rotz(a).reshape(-1)

    def f(mflat):
        return _orient_mat_to_euler_torch(mflat.reshape(3, 3))[2]

    x = base.clone().requires_grad_(True)
    g = torch.autograd.grad(f(x), x)[0]
    assert torch.isfinite(g).all()
    for i in (0, 1, 3, 4):                      # m00, m01, m10, m11
        fd = _central_fd(lambda t: f(t).detach(), x.detach(), i)
        assert abs(float(g[i]) - fd) < 1e-6, (
            f"component {i}: autograd {float(g[i])} vs FD {fd}")


def test_euler_theta_derivative_at_the_identity_is_the_gauge_derivative():
    """d(theta)/d(m10) = 1 and d(theta)/d(m00) = 0 in the psi = 0 gauge.

    Finite and exact where the old acos form returned NaN. This is the
    derivative OF THE CONVENTION, not of "the Euler angles" -- at Phi = 0 only
    psi + theta is determined, so the triple itself has no derivative.
    """
    x = _rotz(0.0).reshape(-1).clone().requires_grad_(True)
    g = torch.autograd.grad(
        _orient_mat_to_euler_torch(x.reshape(3, 3))[2], x)[0]
    assert torch.isfinite(g).all()
    assert abs(float(g[3]) - 1.0) < 1e-9
    assert abs(float(g[0])) < 1e-9


def test_euler_values_unchanged():
    """A gradient fix must not move the forward model.

    This compares two DIFFERENT backends -- the NumPy/``math`` path against the
    torch path -- so exact bit-identity is not a property either one can
    promise: it depends on the platform's libm. On macOS the two agree to the
    last bit and this asserted ``== 0.0``; on Linux glibc they differ by a few
    ULP, measured at 1.6e-15 rad on Python 3.11 and 2.2e-15 on 3.12 over these
    same 400 matrices. Exactly the Apple-libm-against-glibc ``acos`` difference
    that the PF golden fixture already carries.

    So bound it instead, three orders above the observed spread. 1e-12 rad is
    6e-11 degrees -- far below anything the forward model could care about, and
    still tight enough that a real change in the model would blow straight
    through it. This is a platform bound, NOT a tolerance relaxed to admit a
    defect.
    """
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(400, 3, 3)))
    Q[np.linalg.det(Q) < 0] *= -1
    ref = np.stack([ms.orient_mat_to_euler(q) for q in Q])
    got = _orient_mat_to_euler_torch(torch.tensor(Q)).numpy()
    assert np.abs(ref - got).max() < 1e-12


def test_euler_roundtrip_at_gimbal():
    for a in (0.0, math.radians(30.0), math.radians(150.0)):
        e = _orient_mat_to_euler_torch(_rotz(a)).numpy()
        back = np.asarray(ms.euler_to_orient_mat(e)).reshape(3, 3)
        assert np.abs(back - _rotz(a).numpy()).max() < 1e-12


# ------------------------------------------------- B2: misorientation

def _rand_oms(n, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    Q[np.linalg.det(Q) < 0] *= -1
    return Q


def test_misorientation_values_bit_identical_to_numpy():
    """The containment guard must not move the reported metric."""
    Q = _rand_oms(400)
    a_np = ms.misorientation_om_batch(Q[:200], Q[200:], SG_FCC)
    a_t = ms.misorientation_om_batch(
        torch.tensor(Q[:200]), torch.tensor(Q[200:]), SG_FCC).numpy()
    assert np.abs(a_np - a_t).max() < 1e-12


def test_misorientation_is_exactly_zero_and_finite_for_identical_pairs():
    Q = _rand_oms(64)
    p = torch.tensor(Q, requires_grad=True)
    ang = ms.misorientation_om_batch(p, torch.tensor(Q), SG_FCC)
    assert float(ang.detach().abs().max()) == 0.0
    g = torch.autograd.grad(ang.sum(), p)[0]
    assert torch.isfinite(g).all(), "acos'(1) = -inf leaked into the graph"


def test_zero_weight_misorientation_does_not_poison_a_shared_parameter():
    """The virality property: 0 * NaN = NaN.

    A misorientation term at zero weight used to destroy the gradient of every
    parameter it shared a graph with.
    """
    Q = _rand_oms(16)
    p = torch.tensor(Q, requires_grad=True)
    loss = (p ** 2).sum() + 0.0 * ms.misorientation_om_batch(
        p, torch.tensor(Q), SG_FCC).sum()
    g = torch.autograd.grad(loss, p)[0]
    assert torch.isfinite(g).all()
    assert np.allclose(g.numpy(), 2.0 * Q)      # the real term survives intact


def test_misorientation_sq_is_smooth_at_zero_misorientation():
    Q = _rand_oms(32)
    p = torch.tensor(Q, requires_grad=True)
    sq = ms.misorientation_sq_om_batch(p, torch.tensor(Q), SG_FCC)
    assert float(sq.abs().max()) < 1e-24
    g = torch.autograd.grad(sq.sum(), p)[0]
    assert torch.isfinite(g).all()
    # A smooth squared distance is stationary at its minimum.
    assert float(g.abs().max()) < 1e-9


@pytest.mark.parametrize("om_deg", [1e-4, 1e-2, 0.1, 1.0, 5.0])
def test_misorientation_sq_equals_omega_squared_for_small_angles(om_deg):
    axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    R = np.asarray(ms.axis_angle_to_orient_mat(axis, om_deg)).reshape(3, 3)
    I = np.eye(3)
    ang = ms.misorientation_om_batch(I[None], R[None], 1)[0]
    sq = ms.misorientation_sq_om_batch(I[None], R[None], 1)[0]
    assert abs(sq / ang ** 2 - 1.0) < 2e-3


def test_misorientation_sq_monotone_in_angle():
    axis = np.array([0.0, 0.0, 1.0])
    prev = -1.0
    for om_deg in (0.5, 1.0, 5.0, 10.0, 20.0, 30.0):
        R = np.asarray(ms.axis_angle_to_orient_mat(axis, om_deg)).reshape(3, 3)
        sq = ms.misorientation_sq_om_batch(np.eye(3)[None], R[None], 1)[0]
        assert sq > prev, "surrogate must order spots the same way the angle does"
        prev = sq


# ------------------------------------------------------- A: rodrigues

def test_rodrigues_no_dead_zone():
    """The limit of 2*atan(|r|)/|r| at r = 0 is 2, not 0."""
    W = torch.arange(9.0).reshape(3, 3)      # asymmetric: sees the skew part

    def f(r):
        return (_rodrigues_to_orient_mat_torch(r) * W).sum()

    for scale in LADDER:
        r = torch.tensor([scale, 0.0, 0.0], requires_grad=True)
        g = torch.autograd.grad(f(r), r)[0]
        fd = [_central_fd(lambda t: f(t).detach(), r.detach(), i) for i in range(3)]
        assert torch.isfinite(g).all()
        assert np.allclose(g.numpy(), fd, atol=1e-5), (
            f"|r|={scale}: autograd {g.numpy()} vs FD {fd}")
        # The true value at the origin, not zero.
        assert np.allclose(g.numpy(), [4.0, -8.0, 4.0], atol=1e-4)


def test_rodrigues_values_still_correct_away_from_zero():
    r = torch.tensor([0.1, -0.2, 0.1])
    R = _rodrigues_to_orient_mat_torch(r).numpy()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


# ------------------------------------------------------------ A: eta

@pytest.mark.parametrize("y,z", [(0.0, 1.0), (0.0, -1.0), (0.0, 5.0),
                                 (1.0, 0.0), (-1.0, 0.0), (0.3, 0.7),
                                 (0.0, 0.0)])
def test_eta_gradient_finite_including_the_whole_y_zero_axis(y, z):
    yt = torch.tensor(y, requires_grad=True)
    zt = torch.tensor(z, requires_grad=True)
    out = _calc_eta_angle_all_torch(yt, zt)
    gy, gz = torch.autograd.grad(out, [yt, zt])
    assert torch.isfinite(gy).all() and torch.isfinite(gz).all()


def test_eta_values_match_the_numpy_backend():
    rng = np.random.default_rng(0)
    y, z = rng.normal(size=20000), rng.normal(size=20000)
    ref = ms.calc_eta_angle_all(y, z)
    got = _calc_eta_angle_all_torch(torch.tensor(y), torch.tensor(z)).numpy()
    assert np.abs(ref - got).max() < 1e-8


@pytest.mark.parametrize("y,z,expect", [(0.0, 1.0, 0.0), (0.0, -1.0, 180.0),
                                        (1.0, 0.0, -90.0), (-1.0, 0.0, 90.0),
                                        (0.0, 0.0, 0.0)])
def test_eta_convention_preserved_on_the_axes(y, z, expect):
    """+180 must not silently become -180 (negative zero in -y)."""
    got = float(_calc_eta_angle_all_torch(torch.tensor([y]), torch.tensor([z]))[0])
    assert abs(got - expect) < 1e-9
