"""Phase 1 tests: TT psi scans and DCT Bragg-flash selection.

The flash solve is closed-form, so it is checked against the condition it claims
to solve (the Bragg residual, computed independently) rather than against itself.
"""
import math

import pytest
import torch

from midas_dct_tt import (
    DCT_OMEGA_SIGN_AERO,
    DCT_OMEGA_SIGN_CCW,
    angle_between_deg,
    bragg_condition_residual,
    bragg_flashes,
    dct_omega_scan,
    dct_sample_rotation,
    incident_wavevector,
    psi_scan,
)

DT = torch.float64
LAMBDA_A = 0.172979
K_MAG = 2.0 * math.pi / LAMBDA_A


def _g_sample(theta_deg, direction=(0.3, -0.5, 0.2)):
    g_mag = 2.0 * K_MAG * math.sin(math.radians(theta_deg))
    d = torch.as_tensor(direction, dtype=DT)
    return g_mag * d / torch.linalg.vector_norm(d)


# ---------------------------------------------------------------------------
# TT psi scan
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_psi_scan_omits_the_duplicate_endpoint():
    """0 and 360 deg are the same projection; including both double-counts a view."""
    a = psi_scan(180)
    assert a.shape == (180,)
    assert float(a[0]) == 0.0
    assert abs(float(a[-1]) - 358.0) < 1e-12


@pytest.mark.unit
def test_psi_scan_endpoint_option():
    a = psi_scan(181, endpoint=True)
    assert abs(float(a[-1]) - 360.0) < 1e-12


@pytest.mark.unit
def test_psi_scan_rejects_empty():
    with pytest.raises(ValueError):
        psi_scan(0)


@pytest.mark.unit
def test_dct_omega_scan_returns_bin_centres():
    c = dct_omega_scan(720)
    assert c.shape == (720,)
    assert abs(float(c[0]) - 0.25) < 1e-12       # half of a 0.5 deg step
    assert abs(float(c[1] - c[0]) - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# DCT Bragg flashes
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("theta", (3.0, 7.5, 12.0))
def test_two_flashes_per_revolution(theta):
    assert len(bragg_flashes(_g_sample(theta), LAMBDA_A)) == 2


@pytest.mark.unit
@pytest.mark.parametrize("theta", (3.0, 7.5, 12.0))
def test_flash_satisfies_the_bragg_condition(theta):
    """Checked with the residual, which shares no code with the solve."""
    k_in = incident_wavevector(LAMBDA_A, dtype=DT)
    scale = float(torch.linalg.vector_norm(k_in)) ** 2
    for f in bragg_flashes(_g_sample(theta), LAMBDA_A):
        assert abs(float(bragg_condition_residual(k_in, f.G_lab))) / scale < 1e-14


@pytest.mark.unit
def test_flash_rotation_is_the_stated_omega():
    """G_lab must be what the stage actually produces at that omega."""
    g = _g_sample(7.5)
    for f in bragg_flashes(g, LAMBDA_A):
        R = dct_sample_rotation(f.omega_deg, dtype=DT)
        assert torch.allclose(R @ g, f.G_lab, atol=1e-10)


@pytest.mark.unit
def test_flash_k_out_is_at_two_theta_from_the_beam():
    k_in = incident_wavevector(LAMBDA_A, dtype=DT)
    for f in bragg_flashes(_g_sample(7.5), LAMBDA_A):
        assert abs(float(torch.linalg.vector_norm(f.k_out)) - 1.0) < 1e-13
        assert abs(float(angle_between_deg(f.k_out, k_in)) - 2.0 * f.theta_deg) < 1e-9


@pytest.mark.unit
def test_omega_in_range_and_sorted():
    f = bragg_flashes(_g_sample(7.5), LAMBDA_A)
    assert all(0.0 <= x.omega_deg < 360.0 for x in f)
    assert f[0].omega_deg <= f[1].omega_deg


@pytest.mark.unit
def test_blind_reflection_returns_no_flashes():
    """A G too close to the rotation axis can never reach Bragg. Empty is physics.

    With the axis vertical, omega only moves the in-plane part of G. If |G|^2/(2k)
    exceeds what that in-plane part can reach, the reflection is simply never
    excited -- one reason a DCT grain gives tens of spots, not hundreds.
    """
    g_mag = 2.0 * K_MAG * math.sin(math.radians(12.0))
    g = torch.tensor([0.02, 0.02, 1.0], dtype=DT)
    g = g_mag * g / torch.linalg.vector_norm(g)      # almost along the axis
    assert bragg_flashes(g, LAMBDA_A) == []


@pytest.mark.unit
def test_G_exactly_on_the_axis_is_rejected():
    g = torch.tensor([0.0, 0.0, 1.0], dtype=DT) * 2.0 * K_MAG * math.sin(math.radians(7.5))
    with pytest.raises(ValueError, match="along the rotation axis"):
        bragg_flashes(g, LAMBDA_A)


@pytest.mark.unit
def test_inaccessible_reflection_raises():
    with pytest.raises(ValueError, match="inaccessible"):
        bragg_flashes(torch.tensor([3.0 * K_MAG, 0.1, 0.0], dtype=DT), LAMBDA_A)


@pytest.mark.unit
def test_friedel_partner_flashes_180_degrees_away():
    """omega(-G) = omega(G) + 180 exactly -- the basis of DCT Friedel pairing.

    Rotating by a further 180 deg negates the in-plane part of G, so -G meets the
    condition exactly where G did, half a turn later. Phase 2's pairing relies on
    this, and it flips the missing cone, which is how TT recovers coverage.
    """
    g = _g_sample(7.5)
    a = sorted(f.omega_deg for f in bragg_flashes(g, LAMBDA_A))
    b = sorted(f.omega_deg for f in bragg_flashes(-g, LAMBDA_A))
    partners = sorted((x + 180.0) % 360.0 for x in a)
    for got, want in zip(b, partners):
        assert abs(got - want) < 1e-9


@pytest.mark.unit
def test_stage_sign_mirrors_the_flash_angles():
    """The aero-vs-CCW hazard, made visible: the same grain flashes at -omega."""
    g = _g_sample(7.5)
    cw = sorted(f.omega_deg for f in bragg_flashes(g, LAMBDA_A,
                                                   omega_sign=DCT_OMEGA_SIGN_AERO))
    ccw = sorted((-f.omega_deg) % 360.0 for f in bragg_flashes(
        g, LAMBDA_A, omega_sign=DCT_OMEGA_SIGN_CCW))
    for a, b in zip(cw, ccw):
        assert abs(a - b) < 1e-9


@pytest.mark.unit
def test_flash_geometry_is_independent_of_the_stage_sign():
    """Sanity: the sign changes *when* a grain flashes, not *whether* it can."""
    g = _g_sample(7.5)
    assert len(bragg_flashes(g, LAMBDA_A, omega_sign=DCT_OMEGA_SIGN_AERO)) == \
           len(bragg_flashes(g, LAMBDA_A, omega_sign=DCT_OMEGA_SIGN_CCW))
