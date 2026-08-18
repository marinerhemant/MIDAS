"""Phase 0 tests: TT alignment (axis parallel to G) and the DCT omega convention.

The defining TT property -- G invariant, and therefore k_h fixed in the lab,
through the entire 360 deg sweep -- is tested directly here. If it ever breaks,
every topograph in a scan is projecting along a different direction and the
reconstruction is meaningless.
"""
import math

import pytest
import torch

from midas_dct_tt import (
    DCT_OMEGA_SIGN_AERO,
    DCT_OMEGA_SIGN_CCW,
    align_vector_to,
    dct_sample_rotation,
    tt_alignment,
)

DT = torch.float64
LAMBDA_A = 0.172979          # ~71.7 keV
THETAS = (2.0, 5.0, 7.5, 10.0, 15.0, 20.0)


def _g_sample_for_theta(theta_deg, direction=(0.3, -0.5, 0.81)):
    """A sample-frame G of the right magnitude for ``theta``, pointing anywhere."""
    k_mag = 2.0 * math.pi / LAMBDA_A
    g_mag = 2.0 * k_mag * math.sin(math.radians(theta_deg))
    d = torch.as_tensor(direction, dtype=DT)
    return g_mag * d / torch.linalg.vector_norm(d)


# ---------------------------------------------------------------------------
# align_vector_to
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_align_vector_to_is_a_proper_rotation():
    s = torch.tensor([0.3, -0.5, 0.81], dtype=DT)
    t = torch.tensor([-0.2, 0.9, 0.1], dtype=DT)
    R = align_vector_to(s, t)
    eye = torch.eye(3, dtype=DT)
    assert torch.allclose(R @ R.T, eye, atol=1e-12)
    assert abs(float(torch.linalg.det(R)) - 1.0) < 1e-12


@pytest.mark.unit
def test_align_vector_to_lands_on_target():
    s = torch.tensor([0.3, -0.5, 0.81], dtype=DT)
    t = torch.tensor([-0.2, 0.9, 0.1], dtype=DT)
    R = align_vector_to(s, t)
    got = R @ (s / torch.linalg.vector_norm(s))
    want = t / torch.linalg.vector_norm(t)
    assert torch.allclose(got, want, atol=1e-12)


@pytest.mark.unit
def test_align_vector_to_parallel_is_identity():
    s = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    R = align_vector_to(s, 2.5 * s)
    assert torch.allclose(R, torch.eye(3, dtype=DT), atol=1e-14)


@pytest.mark.unit
def test_align_vector_to_antiparallel_flips_without_nan():
    for s in (
        torch.tensor([0.0, 0.0, 1.0], dtype=DT),
        torch.tensor([1.0, 0.0, 0.0], dtype=DT),
        torch.tensor([0.3, -0.5, 0.81], dtype=DT),
    ):
        s = s / torch.linalg.vector_norm(s)
        R = align_vector_to(s, -s)
        assert torch.isfinite(R).all()
        assert torch.allclose(R @ s, -s, atol=1e-12)
        assert abs(float(torch.linalg.det(R)) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# TT alignment
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
def test_alignment_puts_G_on_the_rotation_axis(theta):
    al = tt_alignment(_g_sample_for_theta(theta), LAMBDA_A)
    g_hat = al.G_lab / torch.linalg.vector_norm(al.G_lab)
    assert torch.allclose(g_hat, al.rotation_axis, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
def test_alignment_satisfies_bragg(theta):
    al = tt_alignment(_g_sample_for_theta(theta), LAMBDA_A)
    scale = float(torch.linalg.vector_norm(al.k_in)) ** 2
    assert abs(float(al.bragg_residual())) / scale < 1e-14


@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
def test_alignment_recovers_the_bragg_angle(theta):
    al = tt_alignment(_g_sample_for_theta(theta), LAMBDA_A)
    assert abs(float(al.theta_deg) - theta) < 1e-10


@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
def test_alignment_preserves_G_magnitude(theta):
    """A rotation, not a rescale: |G_lab| == |G_sample|."""
    g = _g_sample_for_theta(theta)
    al = tt_alignment(g, LAMBDA_A)
    a = float(torch.linalg.vector_norm(al.G_lab))
    b = float(torch.linalg.vector_norm(g))
    assert abs(a - b) / b < 1e-14


@pytest.mark.unit
@pytest.mark.parametrize("theta", (2.0, 7.5, 20.0))
def test_G_is_invariant_through_the_whole_psi_sweep(theta):
    """THE TT property: rotation about an axis parallel to G leaves G fixed."""
    g = _g_sample_for_theta(theta)
    al = tt_alignment(g, LAMBDA_A)
    for psi in torch.linspace(0.0, 360.0, 37, dtype=DT):
        R = al.sample_rotation(psi)
        assert torch.allclose(R @ g, al.G_lab, atol=1e-11)


@pytest.mark.unit
@pytest.mark.parametrize("theta", (2.0, 7.5, 20.0))
def test_bragg_condition_holds_through_the_whole_psi_sweep(theta):
    """The consequence: no re-alignment is needed at any scan angle."""
    from midas_dct_tt import bragg_condition_residual

    g = _g_sample_for_theta(theta)
    al = tt_alignment(g, LAMBDA_A)
    scale = float(torch.linalg.vector_norm(al.k_in)) ** 2
    for psi in torch.linspace(0.0, 360.0, 25, dtype=DT):
        G_psi = al.sample_rotation(psi) @ g
        assert abs(float(bragg_condition_residual(al.k_in, G_psi))) / scale < 1e-13


@pytest.mark.unit
def test_diffracted_beam_is_stationary_through_the_sweep():
    """k_h fixed in the lab is what makes the detector a fixed tomographic camera."""
    from midas_dct_tt import diffracted_beam_direction

    g = _g_sample_for_theta(7.5)
    al = tt_alignment(g, LAMBDA_A)
    ref = al.beam_direction()
    for psi in torch.linspace(0.0, 360.0, 25, dtype=DT):
        khat = diffracted_beam_direction(al.k_in, al.sample_rotation(psi) @ g)
        assert torch.allclose(khat, ref, atol=1e-11)


@pytest.mark.unit
def test_psi_rotation_leaves_the_axis_fixed():
    al = tt_alignment(_g_sample_for_theta(7.5), LAMBDA_A)
    for psi in (0.0, 33.0, 180.0, 359.0):
        R = al.psi_rotation(psi)
        assert torch.allclose(R @ al.rotation_axis, al.rotation_axis, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
def test_alignment_reports_the_missing_cone(theta):
    al = tt_alignment(_g_sample_for_theta(theta), LAMBDA_A)
    assert abs(float(al.axis_beam_angle_deg()) - (90.0 - theta)) < 1e-10
    assert abs(float(al.missing_cone_deg()) - theta) < 1e-10


@pytest.mark.unit
def test_alignment_rejects_inaccessible_reflection():
    """d < lambda/2 must fail loudly at alignment time, not silently at scan time."""
    k_mag = 2.0 * math.pi / LAMBDA_A
    g = torch.tensor([3.0 * k_mag, 0.0, 0.0], dtype=DT)
    with pytest.raises(ValueError, match="inaccessible"):
        tt_alignment(g, LAMBDA_A)


@pytest.mark.unit
def test_azimuth_rotates_the_diffraction_plane():
    g = _g_sample_for_theta(7.5)
    vertical = tt_alignment(g, LAMBDA_A, azimuth_deg=90.0)
    horizontal = tt_alignment(g, LAMBDA_A, azimuth_deg=0.0)
    assert abs(float(vertical.beam_direction()[1])) < 1e-13   # in x-z
    assert abs(float(horizontal.beam_direction()[2])) < 1e-13  # in x-y
    # Same Bragg angle either way -- azimuth is a mounting choice, not physics.
    assert abs(float(vertical.theta_deg) - float(horizontal.theta_deg)) < 1e-12


# ---------------------------------------------------------------------------
# DCT omega
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_dct_rotation_is_about_the_vertical_axis():
    R = dct_sample_rotation(37.0)
    z = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    assert torch.allclose(R @ z, z, atol=1e-13)


@pytest.mark.unit
def test_dct_default_sign_is_the_1id_aero_clockwise_convention():
    """Default must be the CW aero sign; a wrong sign mirrors the reconstruction."""
    assert DCT_OMEGA_SIGN_AERO == -1.0
    assert dct_sample_rotation(25.0).allclose(
        dct_sample_rotation(25.0, omega_sign=DCT_OMEGA_SIGN_AERO)
    )


@pytest.mark.unit
def test_dct_omega_signs_are_transposes_of_each_other():
    cw = dct_sample_rotation(25.0, omega_sign=DCT_OMEGA_SIGN_AERO)
    ccw = dct_sample_rotation(25.0, omega_sign=DCT_OMEGA_SIGN_CCW)
    assert torch.allclose(cw, ccw.transpose(-1, -2), atol=1e-14)
    assert not torch.allclose(cw, ccw, atol=1e-6)


@pytest.mark.unit
def test_dct_ccw_sign_matches_right_handed_rotation():
    """+90 deg CCW about +z sends xhat -> yhat (the right-handed convention)."""
    R = dct_sample_rotation(90.0, omega_sign=DCT_OMEGA_SIGN_CCW)
    x = torch.tensor([1.0, 0.0, 0.0], dtype=DT)
    y = torch.tensor([0.0, 1.0, 0.0], dtype=DT)
    assert torch.allclose(R @ x, y, atol=1e-13)


@pytest.mark.unit
def test_dct_rotation_composes_over_omega():
    a = dct_sample_rotation(20.0) @ dct_sample_rotation(15.0)
    b = dct_sample_rotation(35.0)
    assert torch.allclose(a, b, atol=1e-13)


# ---------------------------------------------------------------------------
# autograd + device
# ---------------------------------------------------------------------------
@pytest.mark.autograd
def test_gradcheck_alignment_wrt_sample_G():
    g = _g_sample_for_theta(7.5).clone().requires_grad_(True)

    def f(gv):
        return tt_alignment(gv, LAMBDA_A).G_lab

    assert torch.autograd.gradcheck(f, (g,))


@pytest.mark.autograd
def test_gradcheck_align_vector_to():
    s = torch.tensor([0.3, -0.5, 0.81], dtype=DT, requires_grad=True)
    t = torch.tensor([-0.2, 0.9, 0.1], dtype=DT)
    assert torch.autograd.gradcheck(lambda v: align_vector_to(v, t), (s,))


@pytest.mark.autograd
def test_gradient_flows_through_the_psi_scan():
    g = _g_sample_for_theta(7.5).clone().requires_grad_(True)
    al = tt_alignment(g, LAMBDA_A)
    loss = sum(al.sample_rotation(p).sum() for p in (0.0, 90.0, 180.0))
    loss.backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_device_parity_of_the_tt_invariance(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps unavailable")
    dt = torch.float32 if device == "mps" else DT   # MPS has no float64
    tol = 5e-4 if device == "mps" else 1e-11

    g = _g_sample_for_theta(7.5).to(device=device, dtype=dt)
    al = tt_alignment(g, LAMBDA_A)
    assert al.G_lab.device.type == device
    for psi in (0.0, 90.0, 217.0):
        R = al.sample_rotation(psi)
        rel = torch.linalg.vector_norm(R @ g - al.G_lab) / torch.linalg.vector_norm(al.G_lab)
        assert float(rel) < tol
