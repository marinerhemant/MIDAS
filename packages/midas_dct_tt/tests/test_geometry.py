"""Phase 0 tests: TT/DCT reciprocal-space geometry and the missing cone.

The load-bearing test in this file is
``test_diffracted_beam_sits_at_90_minus_theta_from_axis``: it is the numerical
proof of the result the whole package's sampling story rests on. It is written
so the two sides come from independent paths -- the left side from vector
algebra on ``k_h`` and ``G``, the right side from the Bragg angle -- so it
cannot pass by construction.
"""
import math

import pytest
import torch

from midas_dct_tt import (
    angle_between_deg,
    angle_kh_to_axis_deg,
    bragg_angle_deg,
    bragg_condition_residual,
    diffracted_beam_direction,
    diffracted_wavevector,
    incident_wavevector,
    missing_cone_half_angle_deg,
    rotation_axis_for_reflection,
    wavevector_magnitude,
)

DT = torch.float64

# ~71.7 keV, typical 1-ID / HEXM.
LAMBDA_A = 0.172979

# Bragg angles spanning HEXM (2-8 deg) through ESRF-typical DCT (10-20 deg).
THETAS = (2.0, 5.0, 7.5, 10.0, 15.0, 20.0)


def _G_at_bragg(theta_deg, azimuth_deg=90.0, wavelength_A=LAMBDA_A):
    """Build a reciprocal vector that exactly satisfies Bragg at ``theta``."""
    k_in = incident_wavevector(wavelength_A, dtype=DT)
    k_mag = torch.linalg.vector_norm(k_in)
    g_mag = 2.0 * k_mag * math.sin(math.radians(theta_deg))
    axis = rotation_axis_for_reflection(theta_deg, azimuth_deg=azimuth_deg, dtype=DT)
    return k_in, g_mag * axis


# ---------------------------------------------------------------------------
# Bragg basics
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_wavevector_magnitude_is_two_pi_over_lambda():
    k = wavevector_magnitude(LAMBDA_A)
    assert abs(float(k) - 2.0 * math.pi / LAMBDA_A) < 1e-12


@pytest.mark.unit
def test_incident_wavevector_is_along_lab_x():
    k_in = incident_wavevector(LAMBDA_A, dtype=DT)
    assert float(k_in[1]) == 0.0 and float(k_in[2]) == 0.0
    assert float(k_in[0]) > 0.0


@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
def test_bragg_angle_roundtrip(theta):
    k_mag = wavevector_magnitude(LAMBDA_A)
    g_mag = 2.0 * k_mag * math.sin(math.radians(theta))
    assert abs(float(bragg_angle_deg(g_mag, k_mag)) - theta) < 1e-10


@pytest.mark.unit
def test_bragg_angle_is_convention_independent():
    """The 2*pi factor cancels: same theta from (2pi/d, 2pi/lambda) and (1/d, 1/lambda).

    Passed as bare Python floats, which also exercises the float64 promotion --
    ``torch.tensor(<python float>)`` would be float32 and cost ~1e-6 deg here.
    """
    theta = 7.5
    k_2pi = 2.0 * math.pi / LAMBDA_A
    g_2pi = 2.0 * k_2pi * math.sin(math.radians(theta))
    k_inv = 1.0 / LAMBDA_A
    g_inv = 2.0 * k_inv * math.sin(math.radians(theta))
    a = float(bragg_angle_deg(g_2pi, k_2pi))
    b = float(bragg_angle_deg(g_inv, k_inv))
    assert abs(a - b) < 1e-12


@pytest.mark.unit
def test_python_float_inputs_are_not_silently_float32():
    """A bare float must not drag the whole angle chain down to float32."""
    assert wavevector_magnitude(LAMBDA_A).dtype == torch.float64
    assert bragg_angle_deg(1.0, 10.0).dtype == torch.float64
    # A float32 tensor keeps its dtype and the Python scalar goes weak -- without
    # this an MPS float32 tensor would be promoted to a dtype MPS cannot hold.
    assert bragg_angle_deg(torch.tensor(1.0, dtype=torch.float32), 10.0).dtype == torch.float32
    assert wavevector_magnitude(torch.tensor(LAMBDA_A, dtype=torch.float32)).dtype == torch.float32
    # An integer tensor is not a valid magnitude dtype; promote it rather than
    # returning an int angle.
    assert bragg_angle_deg(torch.tensor(1), torch.tensor(10)).dtype == torch.float64


@pytest.mark.unit
def test_inaccessible_reflection_raises_not_clamps():
    """|G| > 2|k| must raise -- clamping would silently invent a backscatter peak."""
    k_mag = wavevector_magnitude(LAMBDA_A)
    with pytest.raises(ValueError, match="inaccessible"):
        bragg_angle_deg(2.5 * k_mag, k_mag)


@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
def test_bragg_residual_zero_on_ewald_sphere(theta):
    k_in, G = _G_at_bragg(theta)
    resid = bragg_condition_residual(k_in, G)
    scale = float(torch.linalg.vector_norm(k_in)) ** 2
    assert abs(float(resid)) / scale < 1e-14


@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
def test_scattering_is_elastic(theta):
    k_in, G = _G_at_bragg(theta)
    kh = diffracted_wavevector(k_in, G)
    n_in = float(torch.linalg.vector_norm(k_in))
    n_out = float(torch.linalg.vector_norm(kh))
    assert abs(n_out - n_in) / n_in < 1e-14


@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
def test_deflection_from_incident_beam_is_two_theta(theta):
    k_in, G = _G_at_bragg(theta)
    ang = float(angle_between_deg(diffracted_wavevector(k_in, G), k_in))
    assert abs(ang - 2.0 * theta) < 1e-10


# ---------------------------------------------------------------------------
# THE geometric result
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("azimuth", (0.0, 45.0, 90.0, 180.0, 275.0))
def test_diffracted_beam_sits_at_90_minus_theta_from_axis(theta, azimuth):
    """angle(k_h, a_hat) == 90 - theta exactly, for every theta and every azimuth.

    Left side: vector algebra on k_h and G. Right side: the Bragg angle. The two
    share no code path, so agreement is a result, not a tautology.
    """
    k_in, G = _G_at_bragg(theta, azimuth_deg=azimuth)
    ang = float(angle_kh_to_axis_deg(k_in, G))
    assert abs(ang - (90.0 - theta)) < 1e-10


@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
def test_missing_cone_half_angle_equals_theta(theta):
    """The named quantity matches what the vectors actually do."""
    k_in, G = _G_at_bragg(theta)
    from_vectors = 90.0 - float(angle_kh_to_axis_deg(k_in, G))
    assert abs(from_vectors - missing_cone_half_angle_deg(theta)) < 1e-10


@pytest.mark.unit
def test_missing_cone_grows_with_theta():
    """The reflection-selection criterion: low theta = more complete coverage."""
    cones = [missing_cone_half_angle_deg(t) for t in THETAS]
    assert cones == sorted(cones)
    assert cones[0] < cones[-1]


@pytest.mark.unit
@pytest.mark.parametrize("theta", THETAS)
def test_rotation_axis_lies_on_the_bragg_cone(theta):
    """khat_in . a_hat == -sin(theta): the constraint that makes TT alignable."""
    k_in = incident_wavevector(LAMBDA_A, dtype=DT)
    khat = k_in / torch.linalg.vector_norm(k_in)
    for az in (0.0, 30.0, 90.0, 210.0):
        a = rotation_axis_for_reflection(theta, azimuth_deg=az, dtype=DT)
        assert abs(float(torch.linalg.vector_norm(a)) - 1.0) < 1e-14
        assert abs(float(torch.dot(khat, a)) + math.sin(math.radians(theta))) < 1e-14


@pytest.mark.unit
def test_azimuth_90_puts_diffracted_beam_in_the_vertical_plane():
    """Default azimuth reproduces the DFXM x-z diffraction plane, deflected up."""
    k_in, G = _G_at_bragg(10.0, azimuth_deg=90.0)
    khat = diffracted_beam_direction(k_in, G)
    assert abs(float(khat[1])) < 1e-14      # no horizontal component
    assert float(khat[2]) > 0.0             # deflected upward


# ---------------------------------------------------------------------------
# autograd + device
# ---------------------------------------------------------------------------
@pytest.mark.autograd
def test_gradcheck_bragg_angle_wrt_g_mag():
    k_mag = torch.tensor(2.0 * math.pi / LAMBDA_A, dtype=DT)
    g = torch.tensor(2.0 * float(k_mag) * math.sin(math.radians(7.5)), dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: bragg_angle_deg(x, k_mag), (g,))


@pytest.mark.autograd
def test_gradcheck_angle_kh_to_axis_wrt_G():
    k_in, G = _G_at_bragg(7.5)
    G = G.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda g: angle_kh_to_axis_deg(k_in, g), (G,))


@pytest.mark.autograd
def test_gradient_flows_to_wavelength():
    lam = torch.tensor(LAMBDA_A, dtype=DT, requires_grad=True)
    k_in = incident_wavevector(lam)
    torch.linalg.vector_norm(k_in).backward()
    # d|k|/dlambda = -2 pi / lambda^2
    assert abs(float(lam.grad) + 2.0 * math.pi / LAMBDA_A ** 2) < 1e-6


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_device_parity_of_the_missing_cone(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps unavailable")
    dt = torch.float32 if device == "mps" else DT   # MPS has no float64
    tol = 2e-3 if device == "mps" else 1e-10

    theta = 7.5
    k_in = incident_wavevector(LAMBDA_A, device=device, dtype=dt)
    g_mag = 2.0 * torch.linalg.vector_norm(k_in) * math.sin(math.radians(theta))
    axis = rotation_axis_for_reflection(theta, device=device, dtype=dt)
    ang = float(angle_kh_to_axis_deg(k_in, g_mag * axis))
    assert abs(ang - (90.0 - theta)) < tol
