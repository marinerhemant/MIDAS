"""Phase 1 tests: the forward model (TT topographs, DCT frames).

Each test fixes an analytic prediction *before* looking at the model output: a
chord length, a Gaussian acceptance factor, a lever-arm displacement, a selection
rule. A forward model that only reproduced its own conventions would pass none of
them.
"""
import math

import pytest
import torch
from midas_dfxm.field import DeformationField
from midas_dfxm.forward import structure_factor_intensity
from midas_dfxm.io import fcc_reference_crystal
from midas_dfxm.optics import detector_basis

from midas_dct_tt import (
    PlaneDetector,
    angle_between_deg,
    attach_uniform_field,
    bragg_flashes,
    dct_frames,
    dct_omega_scan,
    expand_subvoxels,
    parallel_projection,
    psi_scan,
    sphere_grain,
    topograph_image,
    topograph_stack,
    tt_alignment,
    tt_resolution,
)

DT = torch.float64
LAMBDA_A = 0.172979          # ~71.7 keV
HKL = (1, 1, 1)


def _grain(radius_um=5.0, spacing_um=0.5):
    return attach_uniform_field(sphere_grain(radius_um, spacing_um=spacing_um))


def _aligned(grain, hkl=HKL):
    return tt_alignment(grain.field.reference_G(hkl), LAMBDA_A)


def _with_F(grain, F33):
    """Same grain with a uniform deformation gradient."""
    n = grain.n_voxels
    f = grain.field
    return grain.with_field(
        DeformationField(
            positions=f.positions,
            F=F33.to(dtype=f.F.dtype).expand(n, 3, 3).clone(),
            reference_orientation=f.reference_orientation,
            lattice_params=f.lattice_params,
            shape=f.shape,
        )
    )


# ---------------------------------------------------------------------------
# the undeformed limit: a pure Radon transform
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_topograph_of_a_perfect_grain_is_its_chord_map():
    """No deformation, no acceptance -> the pixel reads path length: 2R at centre."""
    R = 6.0
    g = sphere_grain(R, spacing_um=0.4, edge_width_um=0.2)
    al = tt_alignment(torch.tensor([0.1, 0.2, 1.0], dtype=DT) * 3.0, LAMBDA_A)
    det = PlaneDetector(pixel_um=0.4, shape=(81, 81), distance_um=1000.0)
    img = topograph_image(g, al, 0.0, detector=det)
    assert abs(float(img[40, 40]) - 2.0 * R) / (2.0 * R) < 0.03


@pytest.mark.unit
def test_topograph_conserves_projected_mass_through_the_psi_sweep():
    """A rigid rotation cannot change how much material the beam sees."""
    g = _grain()
    al = _aligned(g)
    det = PlaneDetector(pixel_um=0.5, shape=(96, 96), distance_um=1000.0)
    masses = [
        float(topograph_image(g, al, float(p), detector=det).sum()) * 0.25
        for p in (0.0, 45.0, 90.0, 180.0, 275.0)
    ]
    want = float(g.occupancy.sum()) * g.voxel_volume_um3
    for m in masses:
        assert abs(m - want) / want < 1e-9


@pytest.mark.unit
def test_topograph_matches_a_direct_parallel_projection():
    g = _grain()
    al = _aligned(g)
    det = PlaneDetector(pixel_um=0.5, shape=(96, 96), distance_um=2000.0)
    img = topograph_image(g, al, 33.0, detector=det)
    R = al.sample_rotation(33.0)
    ref = parallel_projection(
        g.positions @ R.T, g.occupancy, normal=al.beam_direction(),
        voxel_volume_um3=g.voxel_volume_um3, pixel_um=0.5, detector_shape=(96, 96),
    )
    assert torch.allclose(img, ref, atol=1e-12)


@pytest.mark.unit
def test_stack_has_one_image_per_angle():
    g = _grain(4.0)
    al = _aligned(g)
    det = PlaneDetector(pixel_um=0.5, shape=(64, 64), distance_um=1000.0)
    stack = topograph_stack(g, al, psi_scan(12), detector=det)
    assert stack.shape == (12, 64, 64)
    assert float(stack.sum()) > 0.0


@pytest.mark.unit
def test_offdetector_reporting():
    g = _grain(6.0)
    al = _aligned(g)
    big = PlaneDetector(pixel_um=0.5, shape=(96, 96), distance_um=1000.0)
    small = PlaneDetector(pixel_um=0.5, shape=(8, 8), distance_um=1000.0)
    _, f_big = topograph_image(g, al, 0.0, detector=big, return_offdetector=True)
    _, f_small = topograph_image(g, al, 0.0, detector=small, return_offdetector=True)
    assert f_big < 1e-12          # float residue of 1 - kept/total, not real loss
    assert f_small > 0.5


# ---------------------------------------------------------------------------
# acceptance: intensity responds to strain as the Gaussian predicts
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_perfect_crystal_sits_at_the_centre_of_the_acceptance():
    """F = I must cost nothing: weight 1 at every voxel."""
    g = _grain(4.0)
    al = _aligned(g)
    res = tt_resolution(al)
    det = PlaneDetector(pixel_um=0.5, shape=(64, 64), distance_um=1000.0)
    with_acc = topograph_image(g, al, 0.0, detector=det, hkl=HKL, resolution=res)
    without = topograph_image(g, al, 0.0, detector=det)
    assert torch.allclose(with_acc, without, rtol=1e-9, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("strain", (2e-5, 5e-5, 1e-4))
def test_uniform_dilation_attenuates_by_the_predicted_gaussian(strain):
    """A dilation moves |Q| off the acceptance centre along q_nom.

    Q = F^-T G0 = G0/(1+e), so the offset is -e/(1+e) |G| purely longitudinal,
    and the intensity must fall by exp(-0.5 (offset/sigma_par)^2). Predicted
    before the model is run.
    """
    g = _grain(4.0)
    al = _aligned(g)
    res = tt_resolution(al)
    det = PlaneDetector(pixel_um=0.5, shape=(64, 64), distance_um=1000.0)

    strained = _with_F(g, (1.0 + strain) * torch.eye(3, dtype=DT))
    ref = float(topograph_image(g, al, 0.0, detector=det, hkl=HKL, resolution=res).sum())
    got = float(topograph_image(strained, al, 0.0, detector=det, hkl=HKL,
                                resolution=res).sum())

    g_mag = float(torch.linalg.vector_norm(al.G_lab))
    offset = strain / (1.0 + strain) * g_mag
    want = math.exp(-0.5 * (offset / float(res.sigma_par)) ** 2)
    assert abs(got / ref - want) < 1e-6


@pytest.mark.unit
def test_a_large_dilation_extinguishes_the_reflection():
    """Far enough out of the acceptance and the grain disappears.

    "Far enough" is further than intuition suggests at these energies: at
    theta = 2.4 deg the longitudinal acceptance is sigma_par/|G| ~ 6.4e-3, so a
    1% dilation is only ~1.5 sigma and still leaves ~30% of the intensity. Five
    percent is ~7 sigma and genuinely dark. See test_acceptance.py for why the
    width is what it is.
    """
    g = _grain(4.0)
    al = _aligned(g)
    res = tt_resolution(al)
    det = PlaneDetector(pixel_um=0.5, shape=(64, 64), distance_um=1000.0)
    ref = float(topograph_image(g, al, 0.0, detector=det, hkl=HKL, resolution=res).sum())

    mild = _with_F(g, 1.01 * torch.eye(3, dtype=DT))         # ~1.5 sigma
    mild_frac = float(topograph_image(mild, al, 0.0, detector=det, hkl=HKL,
                                      resolution=res).sum()) / ref
    assert 0.1 < mild_frac < 0.6        # attenuated, emphatically not extinguished

    far = _with_F(g, 1.05 * torch.eye(3, dtype=DT))          # ~7 sigma
    far_frac = float(topograph_image(far, al, 0.0, detector=det, hkl=HKL,
                                     resolution=res).sum()) / ref
    assert far_frac < 1e-9


# ---------------------------------------------------------------------------
# the distorted-volume displacement: position, not just intensity
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_lattice_rotation_displaces_the_topograph_by_the_lever_arm():
    """A tilted lattice diffracts in a tilted direction, so the image moves.

    Predicted independently as ``L * tan(delta)``, with ``delta`` the angle
    between the deflected beam ``k_in + Q'`` and the nominal one. This is the
    displacement that makes naive back-projection mislocate material -- the
    "pseudo distorted grain volume" -- and the reason the forward model carries
    per-voxel ray directions.
    """
    from midas_dfxm.conventions import rotation_matrix

    g = _grain(4.0, spacing_um=0.5)
    al = _aligned(g)
    L = 20000.0
    det = PlaneDetector(pixel_um=1.0, shape=(256, 256), distance_um=L)

    tilt_deg = 0.05
    F = rotation_matrix((0.0, 1.0, 0.0), tilt_deg).to(DT)     # F = R => Q = R G0
    tilted = _with_F(g, F)

    def centroid(img):
        nu, nv = img.shape
        iu = torch.arange(nu, dtype=DT).unsqueeze(1)
        iv = torch.arange(nv, dtype=DT).unsqueeze(0)
        s = img.sum()
        return torch.stack([(img * iu).sum() / s, (img * iv).sum() / s])

    c0 = centroid(topograph_image(g, al, 0.0, detector=det, hkl=HKL))
    c1 = centroid(topograph_image(tilted, al, 0.0, detector=det, hkl=HKL))
    moved_um = float(torch.linalg.vector_norm(c1 - c0)) * det.pixel_um

    Q_lab = tilted.field.local_Q(HKL)[0] @ al.sample_to_lab.T
    delta = float(angle_between_deg(al.k_in + Q_lab, al.beam_direction()))
    want = L * math.tan(math.radians(delta))
    assert abs(moved_um - want) / want < 0.02


@pytest.mark.unit
def test_no_displacement_without_a_deformation_field():
    """The undeformed control for the test above: same setup, zero shift."""
    g = _grain(4.0)
    al = _aligned(g)
    det = PlaneDetector(pixel_um=1.0, shape=(128, 128), distance_um=20000.0)
    a = topograph_image(g, al, 0.0, detector=det, hkl=HKL)
    b = topograph_image(g, al, 0.0, detector=det)          # no field path at all
    assert torch.allclose(a, b, atol=1e-12)


# ---------------------------------------------------------------------------
# selection rules: the structure-factor null
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_forbidden_reflection_is_dark():
    """fcc (100) is extinguished by the lattice centring: the topograph must be black.

    Uses the real structure factor from midas_dfxm/midas_hkls, not a hand-set
    zero -- so this tests the physics path a user actually gets.
    """
    crystal = fcc_reference_crystal()
    g = _grain(4.0)
    al = _aligned(g)
    det = PlaneDetector(pixel_um=0.5, shape=(64, 64), distance_um=1000.0)

    allowed = float(structure_factor_intensity(crystal, (1, 1, 1), wavelength_A=LAMBDA_A))
    forbidden = float(structure_factor_intensity(crystal, (1, 0, 0), wavelength_A=LAMBDA_A))
    assert allowed > 1e5 and forbidden < 1e-12          # the selection rule itself

    bright = topograph_image(g, al, 0.0, detector=det, sf2=allowed)
    dark = topograph_image(g, al, 0.0, detector=det, sf2=forbidden)
    assert float(bright.sum()) > 0.0
    assert float(dark.sum()) / float(bright.sum()) < 1e-15


# ---------------------------------------------------------------------------
# Q_sample override (the hook the Phase-3 model comparison needs)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_Q_sample_overrides_the_field():
    """Supplying Q_sample must bypass field.local_Q entirely.

    This is what lets midas_dct_tt.inverse swap the exact and linearised local-Q
    models on identical data; if the field leaked through, the two arms of that
    comparison would silently be the same calculation.
    """
    g = _grain(4.0)
    al = _aligned(g)
    det = PlaneDetector(pixel_um=0.5, shape=(64, 64), distance_um=1000.0)
    res = tt_resolution(al)

    from_field = topograph_image(g, al, 0.0, detector=det, hkl=HKL, resolution=res)
    same_Q = topograph_image(g, al, 0.0, detector=det, hkl=HKL, resolution=res,
                             Q_sample=g.field.local_Q(HKL))
    assert torch.allclose(from_field, same_Q, atol=1e-12)

    # a different Q must give a different image
    shifted = g.field.local_Q(HKL) * 1.0005
    other = topograph_image(g, al, 0.0, detector=det, hkl=HKL, resolution=res,
                            Q_sample=shifted)
    assert not torch.allclose(from_field, other, atol=1e-9)


@pytest.mark.unit
def test_Q_sample_works_without_a_deformation_field():
    """A shape-only grain plus an explicit Q is a valid, fully specified case."""
    g = sphere_grain(3.0, spacing_um=1.0)                      # field is None
    ref = attach_uniform_field(sphere_grain(3.0, spacing_um=1.0))
    al = tt_alignment(ref.field.reference_G(HKL), LAMBDA_A)
    det = PlaneDetector(pixel_um=0.5, shape=(64, 64), distance_um=1000.0)
    Q = ref.field.local_Q(HKL)
    img = topograph_image(g, al, 0.0, detector=det, hkl=HKL, Q_sample=Q)
    assert float(img.sum()) > 0.0


# ---------------------------------------------------------------------------
# sub-voxel expansion
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_subvoxel_expansion_conserves_value_and_stays_inside_the_voxel():
    pos = torch.zeros(1, 3, dtype=DT)
    val = torch.ones(1, dtype=DT)
    p, v = expand_subvoxels(pos, val, 2.0, 3)
    assert p.shape == (27, 3) and v.shape == (27,)
    assert abs(float(v.sum()) - 1.0) < 1e-15
    assert float(p.abs().max()) <= 1.0                 # within +-spacing/2


@pytest.mark.unit
def test_supersampling_preserves_the_projected_mass():
    """Finer sampling redistributes intensity; it must not create any."""
    g = _grain(4.0)
    al = _aligned(g)
    det = PlaneDetector(pixel_um=0.5, shape=(96, 96), distance_um=1000.0)
    a = float(topograph_image(g, al, 0.0, detector=det).sum())
    b = float(topograph_image(g, al, 0.0, detector=det, supersample=2).sum())
    assert abs(a - b) / a < 1e-9


# ---------------------------------------------------------------------------
# DCT frames
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_dct_spots_land_in_the_frames_of_their_flashes():
    g = _grain(4.0)
    flashes = bragg_flashes(g.field.reference_G(HKL), LAMBDA_A)
    assert len(flashes) == 2
    centres = dct_omega_scan(36)                       # 10 deg frames
    det = PlaneDetector(pixel_um=1.0, shape=(192, 192), distance_um=500.0)
    frames = dct_frames([g], [flashes], wavelength_A=LAMBDA_A, detector=det,
                        omega_centres=centres)

    lit = [i for i in range(frames.shape[0]) if float(frames[i].sum()) > 0]
    step = float(centres[1] - centres[0])
    want = sorted({int((f.omega_deg - (float(centres[0]) - step / 2)) // step)
                   for f in flashes})
    assert lit == want


@pytest.mark.unit
def test_dct_frames_conserve_mass():
    g = _grain(4.0)
    flashes = bragg_flashes(g.field.reference_G(HKL), LAMBDA_A)
    det = PlaneDetector(pixel_um=1.0, shape=(192, 192), distance_um=500.0)
    frames = dct_frames([g], [flashes], wavelength_A=LAMBDA_A, detector=det,
                        omega_centres=dct_omega_scan(36))
    got = float(frames.sum()) * det.pixel_um ** 2
    want = len(flashes) * float(g.occupancy.sum()) * g.voxel_volume_um3
    assert abs(got - want) / want < 1e-9


@pytest.mark.unit
def test_dct_spot_sits_at_the_two_theta_ring():
    """A grain at the origin projects to L*tan(2 theta) from the beam centre."""
    g = _grain(3.0)
    flashes = bragg_flashes(g.field.reference_G(HKL), LAMBDA_A)
    L = 500.0
    det = PlaneDetector(pixel_um=1.0, shape=(192, 192), distance_um=L)
    frames = dct_frames([g], [flashes[:1]], wavelength_A=LAMBDA_A, detector=det,
                        omega_centres=dct_omega_scan(36))
    img = frames.sum(dim=0)
    nu, nv = det.shape
    iu = torch.arange(nu, dtype=DT).unsqueeze(1)
    iv = torch.arange(nv, dtype=DT).unsqueeze(0)
    s = img.sum()
    cu = float((img * iu).sum() / s) - (nu - 1) / 2.0
    cv = float((img * iv).sum() / s) - (nv - 1) / 2.0
    got = math.hypot(cu, cv) * det.pixel_um
    want = L * math.tan(math.radians(2.0 * flashes[0].theta_deg))
    assert abs(got - want) / want < 0.02


@pytest.mark.unit
def test_dct_frames_rejects_mismatched_inputs():
    g = _grain(3.0)
    with pytest.raises(ValueError, match="one list of flashes per grain"):
        dct_frames([g, g], [[]], wavelength_A=LAMBDA_A,
                   omega_centres=dct_omega_scan(4))


@pytest.mark.unit
def test_a_grain_with_no_flashes_contributes_nothing():
    g = _grain(3.0)
    frames = dct_frames([g], [[]], wavelength_A=LAMBDA_A,
                        omega_centres=dct_omega_scan(8))
    assert float(frames.sum()) == 0.0


# ---------------------------------------------------------------------------
# autograd + device
# ---------------------------------------------------------------------------
@pytest.mark.autograd
def test_gradients_flow_to_occupancy_through_the_forward():
    g = _grain(3.0, spacing_um=1.0).requires_grad_(True)
    al = _aligned(g)
    det = PlaneDetector(pixel_um=1.0, shape=(32, 32), distance_um=1000.0)
    topograph_image(g, al, 17.0, detector=det).sum().backward()
    assert g.logits.grad is not None and float(g.logits.grad.abs().sum()) > 0


@pytest.mark.autograd
def test_gradients_flow_to_the_deformation_field():
    g = _grain(3.0, spacing_um=1.0)
    al = _aligned(g)
    res = tt_resolution(al)
    F = (1.0 + 3e-5) * torch.eye(3, dtype=DT)
    strained = _with_F(g, F)
    strained.field.F.requires_grad_(True)
    det = PlaneDetector(pixel_um=1.0, shape=(32, 32), distance_um=1000.0)
    topograph_image(strained, al, 0.0, detector=det, hkl=HKL, resolution=res).sum().backward()
    assert strained.field.F.grad is not None
    assert float(strained.field.F.grad.abs().sum()) > 0


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_forward_device_parity(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps unavailable")
    dt = torch.float32 if device == "mps" else DT   # MPS has no float64
    tol = 2e-3 if device == "mps" else 1e-10

    g = _grain(4.0)
    al = _aligned(g)
    det = PlaneDetector(pixel_um=0.5, shape=(64, 64), distance_um=1000.0)
    ref = float(topograph_image(g, al, 21.0, detector=det).sum())

    gd = g.to(device=device, dtype=dt)
    ald = al.to(device=device, dtype=dt)
    img = topograph_image(gd, ald, 21.0, detector=det)
    assert img.device.type == device
    assert abs(float(img.sum()) - ref) / ref < tol
