"""Lab-frame and scattering-plane conventions.

Two things are being pinned here.

**Frame** -- which lab axis is the beam and which is up. MIDAS/ESRF (x beam, z up)
and APS/Park (Z beam, Y up) are both in use for the same experiment, and a vector
12 deg off the beam in one reads as 90 deg in the other, silently. The generic
``LabFrame`` map must agree exactly with ``midas_stress.frames``, which is the
repo-wide authority for the MIDAS<->APS pair.

**Scattering plane** -- vertical (ESRF ID06-HXM) vs horizontal (APS 6-ID-C
transmission). Not a relabelling: it changes which motor is the base tilt, and it
swaps which incident divergence limits sigma_rock vs sigma_roll.

Every default must reproduce the previous hard-coded behaviour exactly.
"""
import math

import numpy as np
import pytest
import torch

from midas_stress.frames import R_MIDAS_TO_APS

from midas_dfxm.conventions import (APS_FRAME, BEAM_X_UP_Y, GoniometerSetting,
                                    HORIZONTAL_MIDAS, LabFrame, MIDAS_FRAME,
                                    ScatteringGeometry, as_frame, convert_orientation,
                                    convert_tensor, convert_vector, frame_rotation,
                                    rot_x)
from midas_dfxm.optics import (ObjectiveOptics, detector_basis,
                               diffracted_beam_direction, two_theta_from_k_out)
from midas_dfxm.resolution import poulsen_resolution_widths
from midas_dfxm.scan import mosaicity_scan, rocking_scan

DT = torch.float64
TT = 12.0


# --------------------------------------------------------------------------
# LabFrame
# --------------------------------------------------------------------------

def test_outboard_is_derived_right_handed():
    """(beam, outboard, up) is right-handed in every preset, by construction."""
    for f in (MIDAS_FRAME, APS_FRAME, BEAM_X_UP_Y):
        b = np.array(f.beam)
        o = np.array(f.outboard)
        u = np.array(f.up)
        assert np.allclose(np.cross(b, o), u), f"{f.name} is not right-handed"
        assert np.allclose(np.linalg.norm(o), 1.0)


def test_preset_axes():
    assert MIDAS_FRAME.outboard == pytest.approx((0.0, 1.0, 0.0))   # x beam, z up
    assert APS_FRAME.outboard == pytest.approx((1.0, 0.0, 0.0))     # Z beam, Y up
    assert BEAM_X_UP_Y.outboard == pytest.approx((0.0, 0.0, -1.0))  # x beam, y up


def test_non_orthogonal_frame_rejected():
    with pytest.raises(ValueError, match="orthogonal"):
        LabFrame(beam=(1.0, 0.0, 0.0), up=(1.0, 1.0, 0.0))


def test_arbitrary_frame_accepted():
    """'As generic as possible': any orthogonal unit pair, not just +-axes."""
    s = 1.0 / math.sqrt(2.0)
    f = LabFrame(beam=(s, s, 0.0), up=(0.0, 0.0, 1.0))
    assert np.allclose(np.cross(f.beam, f.outboard), f.up)


def test_frame_rotation_matches_midas_stress_exactly():
    """The generic map must not be a second, drifting copy of the repo authority."""
    R = frame_rotation(MIDAS_FRAME, APS_FRAME).numpy()
    assert np.array_equal(R, R_MIDAS_TO_APS)


def test_frame_round_trip_identity():
    torch.manual_seed(0)
    v = torch.randn(17, 3, dtype=DT)
    for a, b in [(MIDAS_FRAME, APS_FRAME), (MIDAS_FRAME, BEAM_X_UP_Y),
                 (APS_FRAME, BEAM_X_UP_Y)]:
        back = convert_vector(convert_vector(v, a, b), b, a)
        assert torch.allclose(back, v, atol=1e-14)


def test_conversion_preserves_invariants():
    """Lengths, angles, eigenvalues: a frame change moves components, not physics."""
    torch.manual_seed(1)
    v = torch.randn(8, 3, dtype=DT)
    T = torch.randn(3, 3, dtype=DT)
    T = T + T.T
    v_aps = convert_vector(v, MIDAS_FRAME, APS_FRAME)
    T_aps = convert_tensor(T, MIDAS_FRAME, APS_FRAME)
    assert torch.allclose(v.norm(dim=-1), v_aps.norm(dim=-1), atol=1e-14)
    assert torch.allclose(torch.linalg.eigvalsh(T), torch.linalg.eigvalsh(T_aps), atol=1e-13)
    assert torch.allclose(torch.trace(T), torch.trace(T_aps), atol=1e-13)


def test_orientation_rotates_lab_side_only():
    """U maps crystal->lab, so only the lab index turns: R @ U, not R U R^T."""
    U = GoniometerSetting(mu=5.0, omega=11.0).sample_rotation()
    R = frame_rotation(MIDAS_FRAME, APS_FRAME, dtype=DT)
    assert torch.allclose(convert_orientation(U, MIDAS_FRAME, APS_FRAME), R @ U, atol=1e-15)


def test_as_frame_names_and_rejection():
    assert as_frame("aps") is APS_FRAME
    assert as_frame("MIDAS") is MIDAS_FRAME
    assert as_frame(None) is MIDAS_FRAME
    with pytest.raises(ValueError, match="unknown frame"):
        as_frame("esrf_id06_hxm")


# --------------------------------------------------------------------------
# Goniometer under a frame change
# --------------------------------------------------------------------------

def test_goniometer_default_unchanged():
    """The default must still be R_y(mu) R_z(omega) R_x(chi) R_y(phi)."""
    from midas_dfxm.conventions import rot_y, rot_z
    g = GoniometerSetting(mu=3.1, omega=-12.0, chi=0.21, phi=-0.07)
    expect = rot_y(3.1) @ rot_z(-12.0) @ rot_x(0.21) @ rot_y(-0.07)
    assert torch.allclose(g.sample_rotation(), expect, atol=1e-15)


def test_goniometer_frame_change_is_conjugation():
    """Same angles, relabelled axes: G_APS = R G_MIDAS R^T."""
    g = GoniometerSetting(mu=3.1, omega=-12.0, chi=0.21, phi=-0.07)
    R = frame_rotation(MIDAS_FRAME, APS_FRAME, dtype=DT)
    assert torch.allclose(g.in_frame(APS_FRAME).sample_rotation(),
                          R @ g.sample_rotation() @ R.T, atol=1e-14)


def test_goniometer_rotation_angle_is_frame_invariant():
    g = GoniometerSetting(mu=3.1, omega=-12.0, chi=0.21, phi=-0.07)
    ang = lambda M: math.degrees(math.acos(min(1.0, max(-1.0, (float(torch.trace(M)) - 1) / 2))))
    for f in (APS_FRAME, BEAM_X_UP_Y):
        assert ang(g.in_frame(f).sample_rotation()) == pytest.approx(
            ang(g.sample_rotation()), abs=1e-12)


def test_from_aps_lands_in_aps_frame():
    g = GoniometerSetting.from_aps(mu=2.0, omega=4.0)
    assert g.frame is APS_FRAME
    assert torch.allclose(g.in_frame(MIDAS_FRAME).sample_rotation(),
                          GoniometerSetting(mu=2.0, omega=4.0).sample_rotation(), atol=1e-14)


def test_scans_propagate_the_frame():
    """A scan built around an APS-frame centre must not silently revert to MIDAS."""
    c = GoniometerSetting(mu=1.0, frame=APS_FRAME)
    assert all(s.frame is APS_FRAME for s in mosaicity_scan(c, n_chi=3, n_phi=3))
    assert all(s.frame is APS_FRAME for s in rocking_scan(c, n=5))


# --------------------------------------------------------------------------
# Scattering geometry
# --------------------------------------------------------------------------

def test_default_geometry_reproduces_hardcoded_vertical():
    """Bit-for-bit: the old k_out was [cos, 0, sin] in the x-z plane."""
    k = diffracted_beam_direction(TT, dtype=DT)
    tt = math.radians(TT)
    assert torch.equal(k, torch.tensor([math.cos(tt), 0.0, math.sin(tt)], dtype=DT))


def test_horizontal_geometry_k_out():
    k = diffracted_beam_direction(TT, geometry="horizontal", dtype=DT)
    tt = math.radians(TT)
    assert torch.allclose(k, torch.tensor([math.cos(tt), math.sin(tt), 0.0], dtype=DT),
                          atol=1e-15)


def test_horizontal_is_vertical_rolled_90_about_the_beam():
    """The 6-ID geometry is the ESRF one rotated 90 deg about the beam. Nothing more."""
    k_v = diffracted_beam_direction(TT, dtype=DT)
    k_h = diffracted_beam_direction(TT, geometry="horizontal", dtype=DT)
    assert torch.allclose(rot_x(-90.0).to(DT) @ k_v, k_h, atol=1e-15)


def test_two_theta_recovered_in_every_geometry_and_frame():
    for geom in [None, "horizontal",
                 ScatteringGeometry(frame=APS_FRAME, plane="horizontal"),
                 ScatteringGeometry(frame=APS_FRAME, plane="vertical"),
                 ScatteringGeometry(frame=BEAM_X_UP_Y, plane="horizontal")]:
        k = diffracted_beam_direction(TT, geometry=geom, dtype=DT)
        assert two_theta_from_k_out(k, geometry=geom) == pytest.approx(TT, abs=1e-9)


def test_wrong_frame_k_out_is_the_silent_failure():
    """Regression guard: reading APS components in the MIDAS frame returns a
    plausible wrong angle, never an error.

    Horizontal APS geometry puts ``k_out = (sin 2theta, 0, cos 2theta)``, so a
    MIDAS-frame reader (beam = component 0) reports ``90 - 2theta``. The vertical
    APS case is worse still -- zero beam component, so a flat 90 deg.
    """
    k_aps_h = diffracted_beam_direction(
        TT, geometry=ScatteringGeometry(frame=APS_FRAME, plane="horizontal"), dtype=DT)
    assert two_theta_from_k_out(k_aps_h) == pytest.approx(90.0 - TT, abs=1e-9)   # misread
    assert two_theta_from_k_out(
        k_aps_h, geometry=ScatteringGeometry(frame=APS_FRAME)) == pytest.approx(TT, abs=1e-9)

    k_aps_v = diffracted_beam_direction(
        TT, geometry=ScatteringGeometry(frame=APS_FRAME, plane="vertical"), dtype=DT)
    assert two_theta_from_k_out(k_aps_v) == pytest.approx(90.0, abs=1e-9)        # misread


def test_deflection_must_be_perpendicular_to_beam():
    with pytest.raises(ValueError, match="perpendicular"):
        ScatteringGeometry(deflection=(1.0, 0.0, 0.0))


def test_explicit_deflection_gives_oblique_plane():
    s = 1.0 / math.sqrt(2.0)
    g = ScatteringGeometry(deflection=(0.0, s, s))
    assert g.plane == "custom"
    k = g.k_out(TT, dtype=DT)
    assert two_theta_from_k_out(k) == pytest.approx(TT, abs=1e-9)


def test_plane_normal_is_beam_cross_deflection():
    assert np.allclose(ScatteringGeometry().plane_normal().numpy(), [0.0, -1.0, 0.0])
    assert np.allclose(HORIZONTAL_MIDAS.plane_normal().numpy(), [0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# The consequences: base tilt, detector basis, resolution widths
# --------------------------------------------------------------------------

def test_base_tilt_motor_depends_on_the_scattering_plane():
    """Horizontal plane: omega reaches Bragg and mu is INERT for Q along outboard.

    mu rotates about outboard, which is the axis this Q lies on -- so it cannot
    move the reflection into the diffraction condition at all.
    """
    theta = TT / 2.0
    k_mag = 1.0
    q_mag = 2.0 * k_mag * math.sin(math.radians(theta))
    Q0 = torch.tensor([0.0, q_mag, 0.0], dtype=DT)          # a || outboard at mu = 0

    residual = lambda g: float(2.0 * k_mag * (g.sample_rotation() @ Q0)[0] + q_mag ** 2)
    assert residual(GoniometerSetting(omega=theta)) == pytest.approx(0.0, abs=1e-15)
    assert abs(residual(GoniometerSetting(mu=theta))) > 1e-3
    # mu leaves this Q untouched: it is the rotation axis.
    assert torch.allclose(GoniometerSetting(mu=theta).sample_rotation() @ Q0, Q0, atol=1e-15)


def test_bragg_setting_lands_on_the_horizontal_k_out():
    theta = TT / 2.0
    k_mag, q_mag = 1.0, 2.0 * math.sin(math.radians(TT / 2.0))
    Q0 = torch.tensor([0.0, q_mag, 0.0], dtype=DT)
    k_out = torch.tensor([k_mag, 0.0, 0.0], dtype=DT) + \
        GoniometerSetting(omega=theta).sample_rotation() @ Q0
    assert two_theta_from_k_out(k_out) == pytest.approx(TT, abs=1e-9)
    assert torch.allclose(k_out / k_out.norm(),
                          diffracted_beam_direction(TT, geometry="horizontal", dtype=DT),
                          atol=1e-14)


def test_detector_basis_up_axis_follows_the_frame():
    """For an in-plane k_out, v is the frame's up direction."""
    k_h = diffracted_beam_direction(TT, geometry="horizontal", dtype=DT)
    _, v = detector_basis(k_h, geometry=HORIZONTAL_MIDAS)
    assert torch.allclose(v, torch.tensor([0.0, 0.0, 1.0], dtype=DT), atol=1e-14)


def test_resolution_widths_default_unchanged():
    """Backwards compatibility: no geometry -> the published vertical-plane numbers."""
    w = poulsen_resolution_widths(3.0, two_theta_deg=20.0, div_v=0.2e-3, div_h=1.0e-3)
    q, th = 3.0, math.radians(10.0)
    na = 0.731e-3
    assert w["sigma_rock"] == pytest.approx(0.5 * q * math.hypot(0.2e-3, na))
    assert w["sigma_roll"] == pytest.approx(0.5 * q / math.sin(th) * math.hypot(1.0e-3, na))


def test_horizontal_plane_swaps_the_divergence_roles():
    """The silent one: div_v and div_h exchange roles when the plane turns."""
    kw = dict(two_theta_deg=20.0, div_v=0.2e-3, div_h=1.0e-3)
    vert = poulsen_resolution_widths(3.0, **kw)
    horz = poulsen_resolution_widths(3.0, geometry="horizontal", **kw)
    swapped = poulsen_resolution_widths(3.0, two_theta_deg=20.0,
                                        div_v=1.0e-3, div_h=0.2e-3)
    for key in ("sigma_rock", "sigma_roll", "sigma_par"):
        assert horz[key] == pytest.approx(swapped[key], rel=1e-12)
    assert horz["sigma_rock"] > 1.5 * vert["sigma_rock"]     # not a rounding-level effect
    assert horz["sigma_roll"] < 0.7 * vert["sigma_roll"]


def test_isotropic_divergence_hides_the_swap():
    """Why this needed a knob and not a comment: the defaults make it invisible."""
    kw = dict(two_theta_deg=20.0, div_v=0.53e-3, div_h=0.53e-3)
    assert (poulsen_resolution_widths(3.0, **kw)
            == pytest.approx(poulsen_resolution_widths(3.0, geometry="horizontal", **kw)))


def test_divergence_split_reduces_to_the_named_cases():
    d_v = ScatteringGeometry().divergences(0.2e-3, 1.0e-3)
    d_h = HORIZONTAL_MIDAS.divergences(0.2e-3, 1.0e-3)
    assert d_v["div_in_plane"] == pytest.approx(0.2e-3)
    assert d_v["div_out_of_plane"] == pytest.approx(1.0e-3)
    assert d_h["div_in_plane"] == pytest.approx(1.0e-3)
    assert d_h["div_out_of_plane"] == pytest.approx(0.2e-3)


def test_oblique_deflection_interpolates_divergence():
    s = 1.0 / math.sqrt(2.0)
    d = ScatteringGeometry(deflection=(0.0, s, s)).divergences(0.2e-3, 1.0e-3)
    assert d["div_in_plane"] == pytest.approx(math.hypot(s * 0.2e-3, s * 1.0e-3))


# --------------------------------------------------------------------------
# Optics integration
# --------------------------------------------------------------------------

def test_objective_optics_carries_geometry_into_projection():
    opt = ObjectiveOptics(two_theta_deg=TT, geometry="horizontal", magnification=1.0,
                          pixel_um=1.0, detector_shape=(16, 16))
    assert torch.allclose(opt.optical_axis(dtype=DT),
                          diffracted_beam_direction(TT, geometry="horizontal", dtype=DT))
    # A point displaced along up must land off-centre along the detector's v axis.
    px = opt.project(torch.tensor([[0.0, 0.0, 5.0]], dtype=DT))
    assert px[0, 1].item() == pytest.approx(7.5 + 5.0)
    assert px[0, 0].item() == pytest.approx(7.5)


def test_forward_renders_in_both_geometries():
    """Integration: the switch reaches dfxm_image, and changes only the projection.

    Total diffracted intensity is set by the resolution function and the goniometer,
    neither of which the scattering plane touches -- so it must be identical. The
    detector footprint is set by the inclined projection, which the plane *does*
    turn -- so it must differ. Both halves matter: equal sums alone would also be
    consistent with the geometry never arriving.
    """
    from midas_dfxm import (aligned_resolution, bragg_two_theta_deg, dfxm_image,
                            make_uniform_field, reference_q_nom)

    field = make_uniform_field(shape=(8, 8, 1), spacing_um=0.5)
    hkl, center = (1, 1, 1), GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q_nom)), wavelength_A=0.172979)
    res = aligned_resolution(q_nom, sigma_par=5e-3, sigma_perp=5e-3)

    def render(geom):
        opt = ObjectiveOptics(two_theta_deg=tt, magnification=10.0, pixel_um=5.0,
                              detector_shape=(32, 32), geometry=geom)
        return dfxm_image(field, hkl, center, res, opt)

    img_v, img_h = render(None), render("horizontal")
    assert float(img_v.sum()) == pytest.approx(float(img_h.sum()), rel=1e-12)
    assert float(img_v.sum()) > 0
    assert not torch.allclose(img_v, img_h)


def test_from_k_out_consistency_check_uses_the_geometry():
    """An APS-frame k_out must not be silently accepted as a MIDAS-frame one."""
    geom = ScatteringGeometry(frame=APS_FRAME, plane="horizontal")
    k = geom.k_out(TT, dtype=DT)
    opt = ObjectiveOptics.from_k_out(k, geometry=geom, magnification=1.0)
    assert opt.two_theta_deg == pytest.approx(TT, abs=1e-9)
    with pytest.raises(ValueError, match="must agree"):
        ObjectiveOptics(two_theta_deg=TT, k_out=k)     # read in the default frame
