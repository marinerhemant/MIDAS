"""Tests for the explicit optical axis (``ObjectiveOptics.k_out``).

``diffracted_beam_direction`` puts ``k_out`` in the vertical ``x``-``z`` plane,
which is right for DFXM (the objective sits there) and wrong for any technique
that images *one chosen grain's* reflection -- topotomography, DCT -- where the
direction is whatever crystallography gives. These tests cover the explicit-axis
path added for that case, and pin the backward compatibility of the default one.
"""
import math

import pytest
import torch

from midas_dfxm.optics import (
    ObjectiveOptics,
    detector_basis,
    diffracted_beam_direction,
    two_theta_from_k_out,
)

DT = torch.float64


# ---------------------------------------------------------------------------
# backward compatibility
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_default_axis_is_unchanged_by_the_new_field():
    """Without k_out, the projection must be bit-identical to the old behaviour."""
    opt = ObjectiveOptics(two_theta_deg=15.0, magnification=10.0, pixel_um=0.75,
                          detector_shape=(64, 64))
    pos = torch.randn(50, 3, dtype=DT)

    k_out = diffracted_beam_direction(15.0, dtype=DT)
    u, v = detector_basis(k_out)
    cu, cv = (64 - 1) / 2.0, (64 - 1) / 2.0
    want = torch.stack([10.0 * (pos @ u) / 0.75 + cu, 10.0 * (pos @ v) / 0.75 + cv], dim=-1)

    assert torch.equal(opt.project(pos), want)


@pytest.mark.unit
def test_k_out_defaults_to_none():
    assert ObjectiveOptics(two_theta_deg=10.0).k_out is None


@pytest.mark.unit
@pytest.mark.parametrize("two_theta", (0.0, 4.0, 15.0, 40.0, 90.0))
def test_optical_axis_matches_the_two_theta_form_when_unset(two_theta):
    opt = ObjectiveOptics(two_theta_deg=two_theta)
    assert torch.allclose(
        opt.optical_axis(dtype=DT), diffracted_beam_direction(two_theta, dtype=DT), atol=1e-15
    )


# ---------------------------------------------------------------------------
# two_theta_from_k_out
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("two_theta", (0.0, 4.0, 15.0, 40.0, 90.0, 150.0))
def test_two_theta_round_trips_through_the_direction(two_theta):
    k = diffracted_beam_direction(two_theta, dtype=DT)
    assert abs(two_theta_from_k_out(k) - two_theta) < 1e-9


@pytest.mark.unit
def test_two_theta_is_independent_of_azimuth():
    """2*theta is the angle from the beam, so rotating about x must not change it."""
    tt = 27.0
    s, c = math.sin(math.radians(tt)), math.cos(math.radians(tt))
    for az in (0.0, 37.0, 90.0, 213.0):
        a = math.radians(az)
        k = torch.tensor([c, s * math.cos(a), s * math.sin(a)], dtype=DT)
        assert abs(two_theta_from_k_out(k) - tt) < 1e-9


@pytest.mark.unit
def test_two_theta_ignores_the_magnitude():
    k = diffracted_beam_direction(22.0, dtype=DT)
    assert abs(two_theta_from_k_out(7.3 * k) - two_theta_from_k_out(k)) < 1e-12


@pytest.mark.unit
def test_zero_vector_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        two_theta_from_k_out(torch.zeros(3, dtype=DT))


# ---------------------------------------------------------------------------
# explicit axis
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_from_k_out_derives_a_consistent_pair():
    tt = 18.0
    a = math.radians(60.0)
    s, c = math.sin(math.radians(tt)), math.cos(math.radians(tt))
    k = torch.tensor([c, s * math.cos(a), s * math.sin(a)], dtype=DT)

    opt = ObjectiveOptics.from_k_out(k, magnification=1.0, pixel_um=1.0)
    assert abs(opt.two_theta_deg - tt) < 1e-9
    assert torch.allclose(opt.optical_axis(dtype=DT), k / torch.linalg.vector_norm(k), atol=1e-15)


@pytest.mark.unit
def test_from_k_out_normalises():
    k = torch.tensor([3.0, 0.0, 4.0], dtype=DT)      # |k| = 5
    axis = ObjectiveOptics.from_k_out(k).optical_axis(dtype=DT)
    assert abs(float(torch.linalg.vector_norm(axis)) - 1.0) < 1e-15


@pytest.mark.unit
def test_from_k_out_rejects_a_redundant_two_theta():
    with pytest.raises(TypeError, match="do not pass it as well"):
        ObjectiveOptics.from_k_out(diffracted_beam_direction(10.0, dtype=DT), two_theta_deg=10.0)


@pytest.mark.unit
def test_inconsistent_pair_is_rejected_at_construction():
    """The failure mode this guard exists for: projection and physics disagreeing."""
    k = diffracted_beam_direction(30.0, dtype=DT)
    with pytest.raises(ValueError, match="must agree"):
        ObjectiveOptics(two_theta_deg=15.0, k_out=k)


@pytest.mark.unit
def test_consistent_pair_is_accepted():
    k = diffracted_beam_direction(15.0, dtype=DT)
    opt = ObjectiveOptics(two_theta_deg=15.0, k_out=k)
    assert torch.allclose(opt.optical_axis(dtype=DT), k, atol=1e-15)


@pytest.mark.unit
def test_out_of_plane_axis_actually_changes_the_geometry():
    """An out-of-plane k_out must change what the projection is *blind to*.

    Stated as a physical claim rather than a pixel-difference one: a displacement
    along the vertical-plane beam is invisible to vertical-plane optics (motion
    along the optical axis) and clearly visible to optics looking along a
    horizontal beam at the same 2*theta. Comparing raw pixel coordinates instead
    would be weak -- at some geometries the two differ only by relabelling u and
    v, which is not a physics difference.
    """
    tt = 20.0
    s, c = math.sin(math.radians(tt)), math.cos(math.radians(tt))
    vertical = torch.tensor([c, 0.0, s], dtype=DT)       # DFXM: diffracted upward
    horizontal = torch.tensor([c, s, 0.0], dtype=DT)     # diffracted sideways

    kw = dict(magnification=1.0, pixel_um=1.0, detector_shape=(256, 256))
    origin = torch.zeros(1, 3, dtype=DT)
    shifted = origin + 12.0 * vertical                   # 12 um along the vertical beam

    blind = ObjectiveOptics(two_theta_deg=tt, **kw)
    seeing = ObjectiveOptics.from_k_out(horizontal, **kw)

    assert float(torch.abs(blind.project(shifted) - blind.project(origin)).max()) < 1e-11
    moved = float(torch.linalg.vector_norm(seeing.project(shifted) - seeing.project(origin)))
    # |component of the 12 um shift perpendicular to the horizontal axis| ~ 5.6 px
    assert moved > 3.0


@pytest.mark.unit
def test_projection_is_blind_to_displacement_along_any_axis():
    """Parallel projection: motion along the optical axis is invisible, in-plane or not."""
    for k in (
        diffracted_beam_direction(15.0, dtype=DT),
        torch.tensor([0.9, 0.3, 0.2], dtype=DT),
    ):
        opt = ObjectiveOptics.from_k_out(k, magnification=1.0, pixel_um=1.0,
                                         detector_shape=(64, 64))
        axis = opt.optical_axis(dtype=DT)
        o = torch.zeros(1, 3, dtype=DT)
        dp = opt.project(o + 12.0 * axis) - opt.project(o)
        assert float(torch.abs(dp).max()) < 1e-11


@pytest.mark.unit
def test_render_accepts_an_explicit_axis():
    k = torch.tensor([0.9, 0.3, 0.2], dtype=DT)
    opt = ObjectiveOptics.from_k_out(k, magnification=1.0, pixel_um=1.0,
                                     detector_shape=(32, 32))
    pos = torch.randn(200, 3, dtype=DT)
    img = opt.render(pos, torch.ones(200, dtype=DT))
    assert img.shape == (32, 32)
    assert float(img.sum()) > 0.0


# ---------------------------------------------------------------------------
# autograd + device
# ---------------------------------------------------------------------------
@pytest.mark.autograd
def test_gradients_flow_through_an_explicit_k_out():
    k = torch.tensor([0.9, 0.3, 0.2], dtype=DT, requires_grad=True)
    opt = ObjectiveOptics.from_k_out(k, magnification=1.0)
    opt.project(torch.randn(20, 3, dtype=DT)).sum().backward()
    assert k.grad is not None and float(k.grad.abs().sum()) > 0.0


@pytest.mark.autograd
def test_gradcheck_projection_wrt_k_out():
    pos = torch.randn(6, 3, dtype=DT)

    def f(kv):
        # Build inside the closure so the guard sees a consistent pair each time.
        return ObjectiveOptics.from_k_out(kv, magnification=1.0).project(pos)

    k = torch.tensor([0.9, 0.3, 0.2], dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(f, (k,), atol=1e-8)


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_explicit_axis_is_device_portable(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps unavailable")
    dt = torch.float32 if device == "mps" else DT   # MPS has no float64

    k = torch.tensor([0.9, 0.3, 0.2], dtype=dt, device=device)
    opt = ObjectiveOptics.from_k_out(k, magnification=1.0, detector_shape=(32, 32))
    pos = torch.randn(64, 3, dtype=dt, device=device)
    px = opt.project(pos)
    assert px.device.type == device
    assert torch.isfinite(px).all()
