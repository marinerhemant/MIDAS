"""Phase 1 tests: the ray-to-plane projector.

The projector is where a geometry error would be least visible -- a wrong
obliquity or a dropped normalisation still produces a plausible-looking blob. So
it is tested three ways that cannot all be satisfied by the same bug: against an
analytic answer (a sphere's chord), against a conservation law (projected mass),
and against an independent NumPy implementation.
"""
import math

import pytest
import torch
from midas_dfxm.optics import ObjectiveOptics

from midas_dct_tt import (
    offdetector_fraction,
    parallel_projection,
    pixel_coordinates,
    project_rays_to_plane,
    sphere_grain,
    splat_bilinear,
)
from midas_dct_tt.validate import numpy_project_rays_to_plane

DT = torch.float64
BEAM = torch.tensor([0.0, 0.0, 1.0], dtype=DT)   # project along lab z


# ---------------------------------------------------------------------------
# splat
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_splat_conserves_weight_when_fully_on_detector():
    u = torch.tensor([4.3, 7.9, 2.0], dtype=DT)
    v = torch.tensor([5.5, 3.1, 6.0], dtype=DT)
    w = torch.tensor([1.0, 2.0, 3.0], dtype=DT)
    img = splat_bilinear(u, v, w, (16, 16))
    assert abs(float(img.sum()) - float(w.sum())) < 1e-12


@pytest.mark.unit
def test_splat_places_an_integer_coordinate_in_one_pixel():
    img = splat_bilinear(torch.tensor([3.0], dtype=DT), torch.tensor([5.0], dtype=DT),
                         torch.tensor([1.0], dtype=DT), (16, 16))
    assert abs(float(img[3, 5]) - 1.0) < 1e-15
    assert abs(float(img.sum()) - 1.0) < 1e-15


@pytest.mark.unit
def test_splat_divides_a_half_pixel_coordinate_evenly():
    img = splat_bilinear(torch.tensor([3.5], dtype=DT), torch.tensor([5.0], dtype=DT),
                         torch.tensor([1.0], dtype=DT), (16, 16))
    assert abs(float(img[3, 5]) - 0.5) < 1e-15
    assert abs(float(img[4, 5]) - 0.5) < 1e-15


@pytest.mark.unit
def test_splat_drops_off_detector_weight():
    img = splat_bilinear(torch.tensor([-50.0], dtype=DT), torch.tensor([5.0], dtype=DT),
                         torch.tensor([1.0], dtype=DT), (16, 16))
    assert float(img.sum()) == 0.0


# ---------------------------------------------------------------------------
# agreement with midas_dfxm's own projection
# ---------------------------------------------------------------------------
@pytest.mark.contract
def test_parallel_projection_matches_objectiveoptics_m1():
    """Our projector must reduce to midas_dfxm's M = 1 projection exactly.

    Same splat, same detector basis, same centre convention -- so this is the
    check that reimplementing the splat locally (for per-voxel rays, which
    ObjectiveOptics cannot express) did not silently fork it.
    """
    k = torch.tensor([0.9, 0.3, 0.2], dtype=DT)
    pos = torch.randn(300, 3, dtype=DT) * 3.0
    val = torch.rand(300, dtype=DT)

    ours = parallel_projection(pos, val, normal=k, voxel_volume_um3=1.0,
                               pixel_um=1.0, detector_shape=(48, 48))
    theirs = ObjectiveOptics.from_k_out(k, magnification=1.0, pixel_um=1.0,
                                        detector_shape=(48, 48)).render(pos, val)
    assert torch.allclose(ours, theirs, atol=1e-12)


@pytest.mark.unit
def test_detector_distance_is_irrelevant_to_a_parallel_projection():
    """No magnification, no propagation shift: distance must drop out entirely."""
    pos = torch.randn(200, 3, dtype=DT) * 2.0
    val = torch.rand(200, dtype=DT)
    kw = dict(normal=BEAM, voxel_volume_um3=1.0, pixel_um=1.0, detector_shape=(32, 32))
    a = project_rays_to_plane(pos, val, BEAM, distance_um=0.0, **kw)
    b = project_rays_to_plane(pos, val, BEAM, distance_um=1.0e5, **kw)
    assert torch.allclose(a, b, atol=1e-10)


# ---------------------------------------------------------------------------
# analytic: a sphere's chord
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sphere_projects_to_its_chord_length():
    """The central pixel of a projected sphere reads 2R micrometers.

    This is the whole normalisation convention in one number: the projector
    returns a line integral of chi, so a uniform sphere's central pixel is the
    chord through its centre.
    """
    R = 6.0
    g = sphere_grain(R, spacing_um=0.4, edge_width_um=0.2)
    img = parallel_projection(
        g.positions, g.occupancy, normal=BEAM,
        voxel_volume_um3=g.voxel_volume_um3, pixel_um=0.4, detector_shape=(81, 81),
    )
    centre = float(img[40, 40])
    assert abs(centre - 2.0 * R) / (2.0 * R) < 0.02


@pytest.mark.unit
def test_projected_mass_equals_grain_volume():
    """sum(image) * pixel_area == sum(chi) * voxel_volume, exactly.

    A conservation law, not an approximation: bilinear weights sum to 1. It
    catches any error in the voxel-volume / pixel-area normalisation, which a
    single-pixel test could miss by a constant factor.
    """
    g = sphere_grain(5.0, spacing_um=0.5)
    pixel = 0.5
    img = parallel_projection(
        g.positions, g.occupancy, normal=BEAM,
        voxel_volume_um3=g.voxel_volume_um3, pixel_um=pixel, detector_shape=(64, 64),
    )
    got = float(img.sum()) * pixel ** 2
    want = float(g.occupancy.sum()) * g.voxel_volume_um3
    assert abs(got - want) / want < 1e-12


@pytest.mark.unit
def test_projected_mass_is_invariant_to_projection_direction():
    """Rotating the view cannot create or destroy material."""
    g = sphere_grain(5.0, spacing_um=0.5)
    masses = []
    for n in (
        torch.tensor([0.0, 0.0, 1.0], dtype=DT),
        torch.tensor([1.0, 0.0, 0.0], dtype=DT),
        torch.tensor([0.6, 0.5, 0.62], dtype=DT),
    ):
        img = parallel_projection(g.positions, g.occupancy, normal=n,
                                  voxel_volume_um3=g.voxel_volume_um3,
                                  pixel_um=0.5, detector_shape=(64, 64))
        masses.append(float(img.sum()) * 0.25)
    assert max(masses) - min(masses) < 1e-9 * max(masses)


# ---------------------------------------------------------------------------
# independent oracle
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_matches_the_numpy_oracle_parallel():
    pos = torch.randn(400, 3, dtype=DT) * 3.0
    val = torch.rand(400, dtype=DT)
    n = torch.tensor([0.6, 0.5, 0.62], dtype=DT)
    kw = dict(normal=n, distance_um=250.0, voxel_volume_um3=0.125,
              pixel_um=0.8, detector_shape=(40, 40))
    ours = project_rays_to_plane(pos, val, n, **kw).numpy()
    theirs = numpy_project_rays_to_plane(pos, val, n, **kw)
    assert abs(ours - theirs).max() < 1e-13


@pytest.mark.unit
def test_matches_the_numpy_oracle_with_per_voxel_rays():
    """The deformed case: every voxel travelling in its own direction."""
    torch.manual_seed(0)
    pos = torch.randn(300, 3, dtype=DT) * 3.0
    val = torch.rand(300, dtype=DT)
    n = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    dirs = n + 0.02 * torch.randn(300, 3, dtype=DT)      # small per-voxel tilts
    kw = dict(normal=n, distance_um=800.0, voxel_volume_um3=0.125,
              pixel_um=1.0, detector_shape=(48, 48))
    ours = project_rays_to_plane(pos, val, dirs, **kw).numpy()
    theirs = numpy_project_rays_to_plane(pos, val, dirs, **kw)
    assert abs(ours - theirs).max() < 1e-13


# ---------------------------------------------------------------------------
# ray deflection physics
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a_tilted_ray_lands_at_the_predicted_offset():
    """Displacement = distance * tan(tilt): the lever arm that gives TT its
    orientation sensitivity."""
    n = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    tilt = math.radians(0.05)
    d = torch.tensor([math.sin(tilt), 0.0, math.cos(tilt)], dtype=DT)
    L = 5000.0
    kw = dict(normal=n, distance_um=L, pixel_um=1.0, detector_shape=(512, 512))
    straight = pixel_coordinates(torch.zeros(1, 3, dtype=DT), n, **kw)
    tilted = pixel_coordinates(torch.zeros(1, 3, dtype=DT), d, **kw)
    moved = float(torch.linalg.vector_norm(tilted - straight))
    assert abs(moved - L * math.tan(tilt)) < 1e-6


@pytest.mark.unit
def test_deflection_scales_with_detector_distance():
    n = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    d = torch.tensor([0.001, 0.0, 1.0], dtype=DT)
    kw = dict(normal=n, pixel_um=1.0, detector_shape=(512, 512))
    o = torch.zeros(1, 3, dtype=DT)
    m1 = float(torch.linalg.vector_norm(pixel_coordinates(o, d, distance_um=1000.0, **kw)
                                        - pixel_coordinates(o, n, distance_um=1000.0, **kw)))
    m2 = float(torch.linalg.vector_norm(pixel_coordinates(o, d, distance_um=2000.0, **kw)
                                        - pixel_coordinates(o, n, distance_um=2000.0, **kw)))
    assert abs(m2 / m1 - 2.0) < 1e-9


@pytest.mark.unit
def test_a_ray_pointing_away_from_the_detector_is_discarded():
    """Must not wrap round and deposit intensity behind the sample."""
    n = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    away = torch.tensor([0.0, 0.0, -1.0], dtype=DT)
    img = project_rays_to_plane(
        torch.zeros(1, 3, dtype=DT), torch.ones(1, dtype=DT), away,
        normal=n, distance_um=100.0, voxel_volume_um3=1.0,
        pixel_um=1.0, detector_shape=(32, 32),
    )
    assert float(img.sum()) == 0.0


# ---------------------------------------------------------------------------
# off-detector accounting
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_offdetector_fraction_is_zero_when_everything_lands():
    g = sphere_grain(4.0, spacing_um=0.5)
    f = offdetector_fraction(g.positions, g.occupancy, BEAM, normal=BEAM,
                             distance_um=0.0, pixel_um=0.5, detector_shape=(64, 64))
    assert f == 0.0


@pytest.mark.unit
def test_offdetector_fraction_catches_a_too_small_detector():
    """Silent truncation is the failure this guards: a cropped grain still
    produces a perfectly plausible image."""
    g = sphere_grain(6.0, spacing_um=0.5)
    kw = dict(normal=BEAM, distance_um=0.0, pixel_um=0.5, detector_shape=(8, 8))
    f = offdetector_fraction(g.positions, g.occupancy, BEAM, **kw)
    assert f > 0.5
    # Exact, not a bound: 1 - f is precisely the mass the projector retains,
    # partial-weight edge voxels included.
    img = project_rays_to_plane(g.positions, g.occupancy, BEAM,
                                voxel_volume_um3=g.voxel_volume_um3, **kw)
    kept = float(img.sum()) * 0.25 / (float(g.occupancy.sum()) * g.voxel_volume_um3)
    assert abs(kept - (1.0 - f)) < 1e-12


# ---------------------------------------------------------------------------
# autograd + device
# ---------------------------------------------------------------------------
@pytest.mark.autograd
def test_gradients_flow_to_values_and_positions():
    pos = torch.randn(50, 3, dtype=DT, requires_grad=True)
    val = torch.rand(50, dtype=DT, requires_grad=True)
    parallel_projection(pos, val, normal=BEAM, voxel_volume_um3=1.0,
                        pixel_um=1.0, detector_shape=(32, 32)).sum().backward()
    assert val.grad is not None and float(val.grad.abs().sum()) > 0
    assert pos.grad is not None and torch.isfinite(pos.grad).all()


@pytest.mark.autograd
def test_gradient_survives_an_all_off_detector_projection():
    """A fit whose parameters briefly throw the spot off the edge must not die.

    Regression: the splat built its image with torch.zeros and only acquired a
    grad_fn via index_add_, so if *every* ray missed the detector the image was a
    plain tensor and backward() raised "element 0 of tensors does not require
    grad". That is a crash in the middle of an optimisation, not a zero image --
    and it killed the Phase-3 deformation fit at |H| = 1e-2.
    """
    val = torch.rand(16, dtype=DT, requires_grad=True)
    pos = torch.full((16, 3), 1.0e6, dtype=DT)          # far off any detector
    img = parallel_projection(pos, val, normal=BEAM, voxel_volume_um3=1.0,
                              pixel_um=1.0, detector_shape=(16, 16))
    assert float(img.sum().detach()) == 0.0
    assert img.requires_grad
    img.sum().backward()
    assert val.grad is not None
    assert float(val.grad.abs().sum()) == 0.0           # zero gradient, not absent


@pytest.mark.unit
def test_offdetector_grad_path_does_not_perturb_a_normal_projection():
    """The keep-alive term is 0.0 * weights.sum(): it must change nothing."""
    torch.manual_seed(0)
    pos = torch.randn(64, 3, dtype=DT)
    plain = torch.rand(64, dtype=DT)
    tracked = plain.clone().requires_grad_(True)
    kw = dict(normal=BEAM, voxel_volume_um3=1.0, pixel_um=1.0, detector_shape=(32, 32))
    assert torch.equal(parallel_projection(pos, plain, **kw),
                       parallel_projection(pos, tracked, **kw).detach())


@pytest.mark.autograd
def test_gradcheck_projection_wrt_values():
    torch.manual_seed(1)
    pos = torch.randn(12, 3, dtype=DT)
    n = torch.tensor([0.0, 0.0, 1.0], dtype=DT)

    def f(v):
        return project_rays_to_plane(pos, v, n, normal=n, distance_um=10.0,
                                     voxel_volume_um3=1.0, pixel_um=2.0,
                                     detector_shape=(12, 12))

    v = torch.rand(12, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(f, (v,))


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_projector_device_parity(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps unavailable")
    dt = torch.float32 if device == "mps" else DT   # MPS has no float64
    tol = 2e-3 if device == "mps" else 1e-12

    g = sphere_grain(4.0, spacing_um=0.5)
    ref = float(parallel_projection(g.positions, g.occupancy, normal=BEAM,
                                    voxel_volume_um3=g.voxel_volume_um3,
                                    pixel_um=0.5, detector_shape=(48, 48)).sum())
    gd = g.to(device=device, dtype=dt)
    img = parallel_projection(gd.positions, gd.occupancy,
                              normal=BEAM.to(device=device, dtype=dt),
                              voxel_volume_um3=gd.voxel_volume_um3,
                              pixel_um=0.5, detector_shape=(48, 48))
    assert img.device.type == device
    assert abs(float(img.sum()) - ref) / ref < tol
