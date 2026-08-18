"""Phase 2 study: measuring the TT missing cone.

The plan proposed measuring the missing cone by planting a sphere, reconstructing
it, and looking for axial elongation. **That test does not work**, and the reason
is instructive enough to keep here rather than in a commit message -- see
``test_noiseless_reconstruction_does_not_reveal_the_cone`` below.

What does work is measuring the *operator* instead of a reconstruction. The
missing cone is a statement about which Fourier modes the forward model is blind
to, so the direct experiment is to feed it plane-wave modes at controlled angles
to the tomographic axis and watch the response collapse inside the cone. It does,
by an order of magnitude, and the width tracks ``theta``.

Measured 2026-08-03 (20^3 grid, 1 um voxels, 48 projections, 4 um mode):

    angle(k, G)     theta = 15 deg      theta = 30 deg
        0             5.09                0.71
       10            10.16                0.98
       15            12.54                2.19
       25            11.62                8.22
       45             7.90                9.38
       90             6.49                6.87

At ``theta = 30`` the response inside the cone is ~10x below its far-field value
and only recovers past ~25 deg; at ``theta = 15`` it has already recovered by
15 deg. The transition is smooth rather than a step because a windowed mode on a
finite grid has spectral leakage -- the cone edge is not resolvable to better
than the window's bandwidth.
"""
import math

import pytest
import torch

from midas_dct_tt import (
    PlaneDetector,
    forward_operator,
    psi_scan,
    regular_grid,
    tt_alignment,
)

DT = torch.float64
LAMBDA_A = 0.172979
MODE_WAVELENGTH_UM = 4.0


def _operator(theta_deg, *, n=20, spacing_um=1.0, n_angles=48, det=(64, 64)):
    """Forward operator on a bare grid, plus the tomographic axis in sample frame."""
    positions = regular_grid((n, n, n), spacing_um, dtype=DT)
    k = 2.0 * math.pi / LAMBDA_A
    d = torch.tensor([0.3, -0.5, 0.81], dtype=DT)
    d = d / torch.linalg.vector_norm(d)
    g_sample = 2.0 * k * math.sin(math.radians(theta_deg)) * d
    alignment = tt_alignment(g_sample, LAMBDA_A)
    detector = PlaneDetector(pixel_um=spacing_um, shape=det, distance_um=0.0)
    A = forward_operator(positions, alignment, psi_scan(n_angles), detector,
                         voxel_volume_um3=spacing_um ** 3)
    return A, positions, d


def _frame(axis):
    tmp = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    if abs(float(axis @ tmp)) > 0.9:
        tmp = torch.tensor([0.0, 1.0, 0.0], dtype=DT)
    e1 = torch.linalg.cross(axis, tmp)
    return e1 / torch.linalg.vector_norm(e1)


def _response(A, positions, axis, angle_deg):
    """||A m|| / ||m|| for a windowed plane wave at ``angle_deg`` from ``axis``."""
    a = math.radians(angle_deg)
    khat = math.cos(a) * axis + math.sin(a) * _frame(axis)
    window = torch.exp(-(torch.linalg.vector_norm(positions, dim=-1) / 6.0) ** 2)
    m = torch.cos((2.0 * math.pi / MODE_WAVELENGTH_UM) * (positions @ khat)) * window
    return float(torch.linalg.vector_norm(A(m))) / float(torch.linalg.vector_norm(m))


# ---------------------------------------------------------------------------
# the cone, measured on the operator
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_axial_modes_are_strongly_suppressed():
    """A mode along G is nearly invisible to the scan: ~10x down on transverse."""
    A, pos, axis = _operator(30.0)
    on_axis = _response(A, pos, axis, 0.0)
    transverse = _response(A, pos, axis, 90.0)
    assert on_axis / transverse < 0.2


@pytest.mark.unit
def test_response_recovers_outside_the_cone():
    """Crossing the cone edge restores sensitivity by an order of magnitude."""
    A, pos, axis = _operator(30.0)
    inside = _response(A, pos, axis, 0.0)
    outside = _response(A, pos, axis, 25.0)
    assert outside / inside > 5.0


@pytest.mark.unit
def test_cone_width_tracks_theta():
    """The reflection-selection criterion, measured.

    At 15 deg off axis a ``theta = 15`` geometry has already recovered full
    sensitivity while a ``theta = 30`` one is still deep inside its cone. This is
    the quantitative statement behind "low-theta reflections give more complete
    tomographic coverage".
    """
    out = {}
    for theta in (15.0, 30.0):
        A, pos, axis = _operator(theta)
        out[theta] = _response(A, pos, axis, 15.0) / _response(A, pos, axis, 90.0)
    assert out[15.0] > 3.0 * out[30.0]
    assert out[15.0] > 1.0          # recovered
    assert out[30.0] < 0.6          # still suppressed


@pytest.mark.unit
def test_suppression_is_not_an_artefact_of_the_window():
    """Control: the same windowed mode is *not* suppressed transverse to the axis.

    Without this, the low on-axis response could just mean the window kills the
    mode regardless of direction.
    """
    A, pos, axis = _operator(30.0)
    for angle in (45.0, 60.0, 90.0):
        assert _response(A, pos, axis, angle) > 3.0


# ---------------------------------------------------------------------------
# the negative result, kept deliberately
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_noiseless_reconstruction_does_not_reveal_the_cone():
    """Planting a sphere and looking for elongation does NOT detect the cone.

    Measured: SIRT recovers a planted sphere from a noiseless 36-angle TT
    sinogram with Dice = 1.0000 and axial elongation 0.9999 at theta = 3 deg and
    at theta = 30 deg alike -- no elongation, no theta dependence. Adding 1% and
    5% Gaussian noise leaves Dice at 0.99 and 0.95 and still produces no
    elongation.

    Why: it is an **inverse crime**. The data is generated with the same
    discretisation used to reconstruct, so the planted object lies exactly in the
    reconstruction basis and the data is exactly consistent. Compact support,
    non-negativity, and a heavily over-determined discrete system (36 x 64 x 64
    measurements for ~9000 voxels) then fill the cone in. The cone is a null
    space of the *continuous* operator; discretising a compact object on a coarse
    grid does not preserve it as an exact null space.

    Kept as a test so the finding is not rediscovered: measure the operator, not
    the reconstruction. A reconstruction-based cone study would need a finer grid
    than the data supports, an object with genuine content inside the cone, or a
    genuinely different forward discretisation for the data.
    """
    from midas_dct_tt import axial_elongation, dice, sirt, sphere_grain

    grain = sphere_grain(4.0, spacing_um=1.0, edge_width_um=0.5)
    k = 2.0 * math.pi / LAMBDA_A
    d = torch.tensor([0.3, -0.5, 0.81], dtype=DT)
    d = d / torch.linalg.vector_norm(d)
    g_sample = 2.0 * k * math.sin(math.radians(30.0)) * d
    al = tt_alignment(g_sample, LAMBDA_A)
    det = PlaneDetector(pixel_um=1.0, shape=(40, 40), distance_um=0.0)
    A = forward_operator(grain.positions, al, psi_scan(36), det,
                         voxel_volume_um3=grain.voxel_volume_um3)

    truth = grain.occupancy.detach()
    chi = sirt(A(truth), A, grain.n_voxels, n_iter=60, dtype=DT)

    assert dice(chi, truth) > 0.99                    # essentially perfect
    elong = axial_elongation(chi, grain.positions, g_sample)
    assert abs(elong - 1.0) < 0.02                    # and entirely round
