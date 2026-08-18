"""Phase 2 tests: shape reconstruction (SIRT baseline + differentiable).

Plant-and-recover on noiseless data, the adjoint dot-product test, and the
scoring metrics. The missing-cone study is in ``tests/test_missing_cone.py`` --
it turned out to need the operator rather than a reconstruction.
"""
import math

import pytest
import torch

from midas_dct_tt import (
    PlaneDetector,
    axial_elongation,
    backproject,
    dice,
    forward_operator,
    iou,
    psi_scan,
    reconstruct_differentiable,
    sirt,
    sphere_grain,
    total_variation,
    tt_alignment,
)

DT = torch.float64
LAMBDA_A = 0.172979


def _setup(radius_um=4.0, spacing_um=1.0, n_angles=36, theta_deg=None, det_px=1.0,
           det_shape=(40, 40)):
    """A planted sphere, an alignment, and the forward operator over a psi scan."""
    grain = sphere_grain(radius_um, spacing_um=spacing_um, edge_width_um=0.5 * spacing_um)
    # Choose |G| to hit a requested Bragg angle, so the missing cone is a knob.
    k = 2.0 * math.pi / LAMBDA_A
    theta = 7.5 if theta_deg is None else theta_deg
    g_mag = 2.0 * k * math.sin(math.radians(theta))
    g_sample = g_mag * torch.tensor([0.3, -0.5, 0.81], dtype=DT) / \
        torch.linalg.vector_norm(torch.tensor([0.3, -0.5, 0.81], dtype=DT))
    al = tt_alignment(g_sample, LAMBDA_A)
    det = PlaneDetector(pixel_um=det_px, shape=det_shape, distance_um=0.0)
    psi = psi_scan(n_angles)
    A = forward_operator(grain.positions, al, psi, det,
                         voxel_volume_um3=grain.voxel_volume_um3)
    return grain, al, g_sample, A


# ---------------------------------------------------------------------------
# the operator and its adjoint
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_forward_operator_is_linear():
    """SIRT and the adjoint both assume linearity; assert it rather than hope."""
    grain, _, _, A = _setup()
    x = torch.rand(grain.n_voxels, dtype=DT)
    y = torch.rand(grain.n_voxels, dtype=DT)
    a, b = 2.3, -0.7
    lhs = A(a * x + b * y)
    rhs = a * A(x) + b * A(y)
    assert torch.allclose(lhs, rhs, atol=1e-10)


@pytest.mark.unit
def test_adjoint_passes_the_dot_product_test():
    """<A x, y> == <x, A^T y>.

    The standard tomography check, and the one that catches a back-projector
    that is not actually the transpose. It should pass trivially here because
    the adjoint is obtained by differentiating the forward -- which is the point.
    """
    torch.manual_seed(0)
    grain, _, _, A = _setup()
    x = torch.rand(grain.n_voxels, dtype=DT)
    y = torch.rand_like(A(x))
    lhs = float((A(x) * y).sum())
    rhs = float((x * backproject(A, y, grain.n_voxels, dtype=DT)).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-12


@pytest.mark.unit
def test_backprojection_is_nonnegative_for_nonnegative_data():
    grain, _, _, A = _setup()
    y = torch.rand_like(A(torch.zeros(grain.n_voxels, dtype=DT)))
    bp = backproject(A, y, grain.n_voxels, dtype=DT)
    assert float(bp.min()) >= 0.0
    assert float(bp.max()) > 0.0


# ---------------------------------------------------------------------------
# plant and recover
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_sirt_recovers_a_planted_sphere():
    grain, _, _, A = _setup()
    truth = grain.occupancy.detach()
    chi = sirt(A(truth), A, grain.n_voxels, n_iter=60, relaxation=1.0, dtype=DT)
    assert dice(chi, truth) > 0.9
    assert iou(chi, truth) > 0.8


@pytest.mark.unit
def test_sirt_reduces_the_data_residual_monotonically_at_first():
    grain, _, _, A = _setup()
    truth = grain.occupancy.detach()
    hist = []
    sirt(A(truth), A, grain.n_voxels, n_iter=15, dtype=DT,
         callback=lambda i, r: hist.append(r))
    assert hist[0] > hist[-1]
    assert hist[-1] / hist[0] < 0.5


@pytest.mark.unit
def test_differentiable_reconstruction_recovers_a_planted_sphere():
    grain, _, _, A = _setup()
    truth = grain.occupancy.detach()
    chi, info = reconstruct_differentiable(A(truth), A, grain.n_voxels,
                                           steps=200, lr=0.4, dtype=DT)
    assert dice(chi, truth) > 0.9
    assert info["history"][-1] < info["history"][0]


@pytest.mark.unit
def test_this_reconstruction_is_truncated_not_oscillating():
    """Why ``reconstruct_differentiable`` passes ``lr_schedule="none"``.

    ``midas_invert.fit`` defaults to a cosine schedule because Adam at a fixed
    rate can end a run worse than a point it already visited.  That failure does
    not occur here -- the run is still descending steeply when the budget runs
    out -- and annealing on top of a truncated run only halves the effective
    rate: dice 0.973 -> 0.916.  The drop stays above the 0.9 gate above, so
    nothing would fail; this test is what stops the schedule being switched on
    'for consistency' and quietly costing 5.7 points of dice.
    """
    grain, _, _, A = _setup()
    chi, info = reconstruct_differentiable(A(grain.occupancy.detach()), A,
                                           grain.n_voxels, steps=200, lr=0.4,
                                           dtype=DT)
    assert info["lr_schedule"] == "none"
    assert info["final_over_min"] == 0.0        # never overshoots: nothing to fix
    assert info["tail_improvement"] > 0.1       # still descending: truncated
    assert info["settled"] is False             # and the diagnostic says so
    assert dice(chi, grain.occupancy.detach()) > 0.95


@pytest.mark.slow
def test_differentiable_vs_sirt_on_a_fair_comparison():
    """Report the comparison; do not assume the differentiable arm wins.

    Both arms get a generous budget on identical noiseless data. The plan's gate
    is explicit that a negative result here is a result -- so this test asserts
    only that *both* recover the shape well, and leaves which is better to be
    measured rather than baked into an assertion.
    """
    grain, _, _, A = _setup(n_angles=48)
    truth = grain.occupancy.detach()
    b = A(truth)
    d_sirt = dice(sirt(b, A, grain.n_voxels, n_iter=120, dtype=DT), truth)
    chi_diff, _ = reconstruct_differentiable(b, A, grain.n_voxels, steps=400,
                                             lr=0.4, dtype=DT)
    d_diff = dice(chi_diff, truth)
    assert d_sirt > 0.85 and d_diff > 0.85


# ---------------------------------------------------------------------------
# TV prior
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_total_variation_is_zero_for_a_uniform_field():
    shape = (5, 5, 5)
    assert float(total_variation(torch.full((125,), 0.3, dtype=DT), shape)) < 1e-15


@pytest.mark.unit
def test_total_variation_grows_with_roughness():
    torch.manual_seed(0)
    shape = (6, 6, 6)
    smooth = torch.full((216,), 0.5, dtype=DT)
    rough = torch.rand(216, dtype=DT)
    assert float(total_variation(rough, shape)) > float(total_variation(smooth, shape))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_dice_and_iou_endpoints():
    a = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=DT)
    assert dice(a, a) == 1.0 and iou(a, a) == 1.0
    b = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=DT)
    assert dice(a, b) == 0.0 and iou(a, b) == 0.0
    assert dice(torch.zeros(4, dtype=DT), torch.zeros(4, dtype=DT)) == 1.0


@pytest.mark.unit
def test_axial_elongation_of_a_sphere_is_one():
    """The null for the missing-cone measurement: an isotropic object scores 1."""
    g = sphere_grain(5.0, spacing_um=0.5)
    for axis in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.4, -0.5, 0.77)):
        e = axial_elongation(g.occupancy, g.positions, axis)
        assert abs(e - 1.0) < 0.02


@pytest.mark.unit
def test_axial_elongation_detects_a_planted_prolate_shape():
    """Positive control: an object we deliberately stretch must score > 1."""
    g = sphere_grain(5.0, spacing_um=0.5)
    stretched = g.positions.clone()
    stretched[:, 2] *= 0.5           # squeeze coordinates => object elongated in z
    e = axial_elongation(g.occupancy, stretched, (0.0, 0.0, 1.0))
    assert e < 0.6                   # measured along the squeezed axis
    e2 = axial_elongation(g.occupancy, stretched, (1.0, 0.0, 0.0))
    assert e2 > 1.2


@pytest.mark.unit
def test_axial_elongation_rejects_an_empty_reconstruction():
    g = sphere_grain(3.0, spacing_um=1.0)
    with pytest.raises(ValueError, match="empty"):
        axial_elongation(torch.zeros(g.n_voxels, dtype=DT), g.positions, (0, 0, 1.0))


# ---------------------------------------------------------------------------
# the missing-cone study lives in tests/test_missing_cone.py
# ---------------------------------------------------------------------------
# Measuring the cone via reconstruction elongation was tried and does not work
# (inverse crime -- see that module). The cone is measured on the operator
# instead. `axial_elongation` remains a validated metric here; it simply is not
# sensitive to this effect under noiseless, exactly-representable conditions.
