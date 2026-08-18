"""Contract tests: the midas_dfxm surfaces this package reuses instead of porting.

``implementation_plan.md`` Section 5 claims two pieces of the TT forward model
already exist in ``midas_dfxm`` and need only re-parameterising:

* the reciprocal-space acceptance, with the objective removed -- **na = 0**;
* the projection along the diffracted beam, unmagnified -- **M = 1**.

Both are load-bearing. If a future ``midas-dfxm`` release changes either
signature or either meaning, the failure should surface *here*, as a broken
contract with a named cause, rather than as a slightly-wrong topograph three
modules downstream. These tests are the reason the reuse claim is safe to build
on; they are not testing ``midas_dfxm``'s correctness, they are pinning the
interface we consume.

Also pinned: the voxel-grid convention shared with ``midas_dfxm.io``, since a
half-voxel offset between ``chi(r)`` and ``F(r)`` would read as a small,
plausible, entirely spurious strain.
"""
import math

import pytest
import torch
from midas_dfxm.io import make_uniform_field
from midas_dfxm.optics import ObjectiveOptics
from midas_dfxm.optics import diffracted_beam_direction as dfxm_beam_direction
from midas_dfxm.resolution import poulsen_resolution_widths

from midas_dct_tt import (
    diffracted_beam_direction,
    incident_wavevector,
    regular_grid,
    rotation_axis_for_reflection,
)

DT = torch.float64
LAMBDA_A = 0.172979


# ---------------------------------------------------------------------------
# acceptance: na = 0 is the TT/DCT acceptance
# ---------------------------------------------------------------------------
@pytest.mark.contract
def test_na_zero_is_accepted_as_a_parameter():
    """TT has no objective, so the NA term is absent -- a parameter change, not a fudge."""
    w = poulsen_resolution_widths(3.0, two_theta_deg=15.0, na=0.0)
    assert set(w) == {"sigma_rock", "sigma_roll", "sigma_par"}
    assert all(v > 0 for v in w.values())


@pytest.mark.contract
def test_na_zero_leaves_divergence_and_bandwidth_only():
    """With na = 0 the widths reduce to the closed forms TT should have.

    sigma_rock = |Q|/2 * div_v ; sigma_roll = |Q|/(2 sin theta) * div_h
    """
    q, tt, div_v, div_h = 3.0, 15.0, 0.53e-3, 0.31e-3
    w = poulsen_resolution_widths(q, two_theta_deg=tt, div_v=div_v, div_h=div_h, na=0.0)
    theta = math.radians(tt / 2.0)
    # sigma_rock involves no trig and matches to the last bit.
    assert abs(w["sigma_rock"] - 0.5 * q * div_v) < 1e-18
    # sigma_roll goes through midas_dfxm's internal sin(), which is evaluated in
    # float32 (`torch.tensor(<python float>)`), costing ~6e-8 relative. Harmless
    # for a resolution width; pinned here so it is a known property and not a
    # surprise if a width ever needs more precision than that.
    want_roll = 0.5 * q / math.sin(theta) * div_h
    assert abs(w["sigma_roll"] - want_roll) / want_roll < 1e-6


@pytest.mark.contract
def test_objective_na_strictly_widens_the_acceptance():
    """Sanity on the direction: adding an objective can only add variance."""
    tt_widths = poulsen_resolution_widths(3.0, two_theta_deg=15.0, na=0.0)
    dfxm_widths = poulsen_resolution_widths(3.0, two_theta_deg=15.0, na=0.731e-3)
    for key in ("sigma_rock", "sigma_roll"):
        assert dfxm_widths[key] > tt_widths[key]


# ---------------------------------------------------------------------------
# projection: M = 1 is the TT/DCT projection
# ---------------------------------------------------------------------------
@pytest.mark.contract
def test_unit_magnification_is_a_one_to_one_projection():
    opt = ObjectiveOptics(two_theta_deg=15.0, magnification=1.0, pixel_um=1.0,
                          detector_shape=(64, 64))
    origin = torch.zeros(1, 3, dtype=DT)
    k_out = dfxm_beam_direction(15.0, dtype=DT)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    u = torch.linalg.cross(up, k_out)
    u = u / torch.linalg.vector_norm(u)

    shifted = origin + 3.0 * u
    dp = opt.project(shifted) - opt.project(origin)
    # 3 um transverse -> 3 px at M = 1, pixel = 1 um. No magnification.
    assert abs(float(dp[0, 0]) - 3.0) < 1e-12
    assert abs(float(dp[0, 1])) < 1e-12


@pytest.mark.contract
def test_projection_is_blind_to_displacement_along_the_beam():
    """The defining property of a parallel projection -- and of a topograph."""
    opt = ObjectiveOptics(two_theta_deg=15.0, magnification=1.0, pixel_um=1.0,
                          detector_shape=(64, 64))
    origin = torch.zeros(1, 3, dtype=DT)
    k_out = dfxm_beam_direction(15.0, dtype=DT)
    dp = opt.project(origin + 12.0 * k_out) - opt.project(origin)
    assert float(torch.abs(dp).max()) < 1e-11


@pytest.mark.contract
def test_magnification_ten_still_magnifies():
    """Guard against M becoming a no-op upstream: the DFXM case must still scale."""
    kw = dict(two_theta_deg=15.0, pixel_um=1.0, detector_shape=(64, 64))
    p = torch.tensor([[0.0, 2.0, 0.0]], dtype=DT)
    o = torch.zeros(1, 3, dtype=DT)
    d1 = ObjectiveOptics(magnification=1.0, **kw).project(p) - ObjectiveOptics(magnification=1.0, **kw).project(o)
    d10 = ObjectiveOptics(magnification=10.0, **kw).project(p) - ObjectiveOptics(magnification=10.0, **kw).project(o)
    assert abs(float(torch.linalg.vector_norm(d10)) - 10.0 * float(torch.linalg.vector_norm(d1))) < 1e-11


# ---------------------------------------------------------------------------
# the arbitrary-k_out path (midas-dfxm >= 0.3.0)
# ---------------------------------------------------------------------------
@pytest.mark.contract
@pytest.mark.parametrize("two_theta", (4.0, 15.0, 40.0))
def test_dfxm_beam_direction_is_confined_to_the_vertical_plane(two_theta):
    """The 2*theta-only form still puts k_out in lab x-z, always.

    Not a defect -- it is the DFXM geometry, where the objective sits in the
    vertical plane. It is simply not sufficient for TT/DCT, which is why
    geometry.diffracted_beam_direction(k_in, G) and ObjectiveOptics.from_k_out
    exist. Pinned so the assumption stays visible.
    """
    k_out = dfxm_beam_direction(two_theta, dtype=DT)
    assert abs(float(k_out[1])) < 1e-15


@pytest.mark.contract
def test_objective_optics_accepts_an_explicit_out_of_plane_axis():
    """The generalisation TT needs: an arbitrary G's beam direction as the axis.

    Added to midas-dfxm in 0.3.0 for exactly this package; the dependency floor
    is pinned accordingly. Requiring it here means a downgrade of the sibling
    fails loudly rather than silently projecting along the wrong direction.
    """
    k_in = incident_wavevector(LAMBDA_A, dtype=DT)
    g_mag = 2.0 * torch.linalg.vector_norm(k_in) * math.sin(math.radians(7.5))
    # Azimuth 35 deg: firmly out of the x-z plane, as a real grain's G would be.
    G = g_mag * rotation_axis_for_reflection(7.5, azimuth_deg=35.0, dtype=DT)
    k_out = diffracted_beam_direction(k_in, G)

    opt = ObjectiveOptics.from_k_out(k_out, magnification=1.0, pixel_um=1.0,
                                     detector_shape=(128, 128))
    assert torch.allclose(opt.optical_axis(dtype=DT), k_out, atol=1e-13)
    assert abs(float(k_out[1])) > 1e-3      # genuinely out of plane

    # Parallel projection along that axis: motion along the beam is invisible.
    o = torch.zeros(1, 3, dtype=DT)
    dp = opt.project(o + 9.0 * k_out) - opt.project(o)
    assert float(torch.abs(dp).max()) < 1e-11


@pytest.mark.contract
def test_two_theta_and_axis_cannot_disagree():
    """An inconsistent pair must raise: it would be a silent projection error."""
    with pytest.raises(ValueError, match="must agree"):
        ObjectiveOptics(two_theta_deg=15.0, k_out=dfxm_beam_direction(30.0, dtype=DT))


@pytest.mark.contract
@pytest.mark.parametrize("theta", (2.0, 7.5, 20.0))
def test_general_beam_direction_agrees_with_dfxm_in_the_vertical_plane(theta):
    """Our (k_in, G) form reduces to midas_dfxm's 2*theta form where both apply."""
    k_in = incident_wavevector(LAMBDA_A, dtype=DT)
    g_mag = 2.0 * torch.linalg.vector_norm(k_in) * math.sin(math.radians(theta))
    G = g_mag * rotation_axis_for_reflection(theta, azimuth_deg=90.0, dtype=DT)
    ours = diffracted_beam_direction(k_in, G)
    theirs = dfxm_beam_direction(2.0 * theta, dtype=DT)
    assert torch.allclose(ours, theirs, atol=1e-13)


# ---------------------------------------------------------------------------
# shared voxel grid
# ---------------------------------------------------------------------------
@pytest.mark.contract
@pytest.mark.parametrize("shape,spacing", [((5, 5, 5), 1.0), ((7, 4, 3), 0.5), ((2, 2, 2), 2.0)])
def test_voxel_grids_coincide_with_midas_dfxm(shape, spacing):
    """chi(r) and F(r) must live on the same voxels, to the last bit."""
    ours = regular_grid(shape, spacing, dtype=DT)
    theirs = make_uniform_field(shape=shape, spacing_um=spacing, dtype=DT).positions
    assert ours.shape == theirs.shape
    assert torch.equal(ours, theirs)
