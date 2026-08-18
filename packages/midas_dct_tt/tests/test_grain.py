"""Phase 0 tests: grain container (occupancy chi(r), optional F(r)) and generators."""
import math

import pytest
import torch

from midas_dct_tt import (
    GrainShape,
    attach_uniform_field,
    box_grain,
    dct_sample_rotation,
    faceted_grain,
    logits_from_signed_distance,
    plate_grain,
    regular_grid,
    sphere_grain,
)

DT = torch.float64


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_occupancy_is_bounded():
    g = sphere_grain(5.0, spacing_um=1.0)
    chi = g.occupancy
    assert float(chi.min()) >= 0.0 and float(chi.max()) <= 1.0


@pytest.mark.unit
def test_signed_distance_is_half_occupied_on_the_surface():
    sdf = torch.tensor([0.0], dtype=DT)
    chi = torch.sigmoid(logits_from_signed_distance(sdf, 0.5))
    assert abs(float(chi) - 0.5) < 1e-15


@pytest.mark.unit
def test_hard_mask_is_rejected():
    with pytest.raises(ValueError, match="edge_width_um"):
        logits_from_signed_distance(torch.zeros(3, dtype=DT), 0.0)


@pytest.mark.unit
def test_mismatched_positions_and_logits_are_rejected():
    with pytest.raises(ValueError, match="disagree on voxel count"):
        GrainShape(positions=torch.zeros(5, 3, dtype=DT), logits=torch.zeros(4, dtype=DT))


@pytest.mark.unit
def test_regular_grid_is_centred_and_ordered():
    pos = regular_grid((5, 5, 5), 2.0, dtype=DT)
    assert pos.shape == (125, 3)
    assert torch.allclose(pos.mean(dim=0), torch.zeros(3, dtype=DT), atol=1e-14)
    assert abs(float(pos[:, 0].max()) - 4.0) < 1e-14   # (5-1)/2 * 2.0


# ---------------------------------------------------------------------------
# planted shapes -- volumes against the analytic answer
# ---------------------------------------------------------------------------
def _soft_boundary_bias(area_um2, curvature_trace, edge_width_um):
    """Predicted volume over-count of a sigmoid boundary: ``A (2H) w^2 pi^2/6``.

    See :func:`midas_dct_tt.grain.logits_from_signed_distance`.
    """
    return area_um2 * curvature_trace * edge_width_um ** 2 * math.pi ** 2 / 6.0


@pytest.mark.unit
def test_sphere_volume_matches_analytic_plus_known_boundary_bias():
    """Sharp volume + the *predicted* soft-boundary bias, to 0.5%.

    Testing against the sharp volume alone would fail by 1.7% and testing with a
    2% tolerance would hide a bias we can predict in closed form. So predict it.
    """
    r, w = 6.0, 0.25
    g = sphere_grain(r, spacing_um=0.5, edge_width_um=w)
    sharp = 4.0 / 3.0 * math.pi * r ** 3
    want = sharp + _soft_boundary_bias(4.0 * math.pi * r ** 2, 2.0 / r, w)
    assert abs(float(g.volume_um3()) - want) / sharp < 0.005


@pytest.mark.unit
def test_boundary_bias_falls_as_edge_width_squared():
    """The w^2 law, over the width range where the grid still samples the ramp."""
    r = 6.0
    sharp = 4.0 / 3.0 * math.pi * r ** 3
    excess = [
        float(sphere_grain(r, spacing_um=0.5, edge_width_um=w, shape=(41, 41, 41)).volume_um3()) - sharp
        for w in (0.5, 0.25)
    ]
    assert abs(excess[0] / excess[1] - 4.0) / 4.0 < 0.02


@pytest.mark.unit
def test_box_volume_is_edge_dominated_and_shrinks_with_edge_width():
    """A box has no face curvature, so its bias comes entirely from convex edges.

    No closed form is derived for the edge term (it is not needed: the forward
    model integrates chi, so the bias cancels in plant-and-recover). What is
    asserted is that it behaves -- shrinks steeply with w -- and that the default
    stays inside a stated bound rather than an unstated one.
    """
    size = (8.0, 6.0, 4.0)
    sharp = size[0] * size[1] * size[2]
    kw = dict(spacing_um=0.5, shape=(33, 29, 25))
    wide = float(box_grain(size, edge_width_um=0.5, **kw).volume_um3()) - sharp
    narrow = float(box_grain(size, edge_width_um=0.25, **kw).volume_um3()) - sharp
    assert wide > narrow > 0.0
    assert wide / narrow > 3.0
    assert narrow / sharp < 0.08   # measured 7.4%; edge-dominated, documented


@pytest.mark.unit
def test_plate_is_thin_along_z():
    g = plate_grain((10.0, 10.0, 1.5), spacing_um=0.5)
    chi = g.occupancy
    extent = []
    for ax in range(3):
        w = (chi * g.positions[:, ax] ** 2).sum() / chi.sum()
        extent.append(float(w) ** 0.5)
    assert extent[2] < 0.3 * extent[0]
    assert abs(extent[0] - extent[1]) / extent[0] < 0.05


@pytest.mark.unit
def test_faceted_grain_octahedron_volume():
    """{111}-type octahedron: V = 4/3 * (d*sqrt(3))^3 / ... checked against the
    exact half-space intersection volume 4 d^3 sqrt(3) for |x|+|y|+|z| <= d*sqrt(3)."""
    d = 5.0
    normals, dists = [], []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                normals.append((sx, sy, sz))
                dists.append(d)
    g = faceted_grain(normals, dists, spacing_um=0.25)
    # |x|+|y|+|z| <= d*sqrt(3)  =>  V = (4/3) * (d sqrt(3))^3 / 2 ... use the
    # cross-polytope volume (2a)^3/6 with a = d*sqrt(3).
    a = d * math.sqrt(3.0)
    want = (2.0 * a) ** 3 / 6.0
    assert abs(float(g.volume_um3()) - want) / want < 0.02


@pytest.mark.unit
def test_shapes_are_centred_on_the_origin():
    for g in (sphere_grain(5.0, spacing_um=0.5), box_grain((6.0, 8.0, 4.0), spacing_um=0.5)):
        c = g.centroid_um()
        assert float(torch.linalg.vector_norm(c)) < 1e-9


@pytest.mark.unit
def test_occupancy_image_reshapes_to_the_grid():
    g = sphere_grain(4.0, spacing_um=1.0)
    img = g.occupancy_image()
    assert img.shape == g.shape
    assert img.numel() == g.n_voxels


@pytest.mark.unit
def test_unstructured_cloud_refuses_to_reshape():
    g = GrainShape(positions=torch.randn(10, 3, dtype=DT), logits=torch.zeros(10, dtype=DT))
    with pytest.raises(ValueError, match="regular grid"):
        g.occupancy_image()


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_positions_lab_applies_the_rotation_the_right_way_round():
    """v_lab = R @ v_sample, with the (N, 3) row layout preserved."""
    g = box_grain((4.0, 2.0, 2.0), spacing_um=1.0)
    R = dct_sample_rotation(90.0)
    lab = g.positions_lab(R)
    want = (R @ g.positions.T).T
    assert torch.allclose(lab, want, atol=1e-13)


@pytest.mark.unit
def test_rotation_preserves_volume_and_distances():
    g = sphere_grain(4.0, spacing_um=0.5)
    R = dct_sample_rotation(41.0)
    lab = g.positions_lab(R)
    assert torch.allclose(
        torch.linalg.vector_norm(lab, dim=-1),
        torch.linalg.vector_norm(g.positions, dim=-1),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# deformation field pairing (reuse of midas_dfxm)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_attach_uniform_field_shares_the_voxel_grid():
    g = attach_uniform_field(sphere_grain(4.0, spacing_um=1.0))
    assert g.field is not None
    assert g.field.positions.shape[0] == g.n_voxels
    assert torch.allclose(g.field.positions, g.positions, atol=1e-12)


@pytest.mark.unit
def test_attached_field_is_a_perfect_crystal():
    g = attach_uniform_field(sphere_grain(3.0, spacing_um=1.0))
    eye = torch.eye(3, dtype=DT).expand(g.n_voxels, 3, 3)
    assert torch.allclose(g.field.F, eye, atol=1e-14)
    # And |Q| = |G0| everywhere: F = I means no shift of the scattering vector.
    Q = g.field.local_Q((1, 1, 1))
    G0 = g.field.reference_G((1, 1, 1))
    assert torch.allclose(Q, G0.expand_as(Q), atol=1e-12)


@pytest.mark.unit
def test_reference_G_magnitude_matches_dspacing():
    """|G| = 2*pi/d for fcc (111) at a = 3.6356 Angstrom -- the 2*pi convention."""
    g = attach_uniform_field(sphere_grain(3.0, spacing_um=1.5))
    G0 = g.field.reference_G((1, 1, 1))
    d = 3.6356 / math.sqrt(3.0)
    assert abs(float(torch.linalg.vector_norm(G0)) - 2.0 * math.pi / d) < 1e-9


@pytest.mark.unit
def test_field_on_a_mismatched_grid_is_rejected():
    g = sphere_grain(3.0, spacing_um=1.0)
    field = attach_uniform_field(sphere_grain(5.0, spacing_um=1.0)).field
    with pytest.raises(ValueError, match="share the grid"):
        GrainShape(
            positions=g.positions, logits=g.logits, spacing_um=1.0,
            shape=g.shape, field=field,
        )


# ---------------------------------------------------------------------------
# autograd + device
# ---------------------------------------------------------------------------
@pytest.mark.autograd
def test_gradient_flows_to_occupancy_logits():
    g = sphere_grain(4.0, spacing_um=1.0).requires_grad_(True)
    g.volume_um3().backward()
    assert g.logits.grad is not None
    assert float(g.logits.grad.abs().sum()) > 0.0


@pytest.mark.autograd
def test_gradient_flows_to_the_planted_radius():
    """Shape parameters stay differentiable through the SDF -- not just the logits."""
    r = torch.tensor(4.0, dtype=DT, requires_grad=True)
    vol = sphere_grain(r, spacing_um=0.5, shape=(29, 29, 29)).volume_um3()
    vol.backward()
    # dV/dr = 4 pi r^2 for a sphere.
    want = 4.0 * math.pi * 4.0 ** 2
    assert abs(float(r.grad) - want) / want < 0.05


@pytest.mark.autograd
def test_gradient_flows_to_facet_distances():
    d = torch.full((6,), 4.0, dtype=DT, requires_grad=True)
    normals = [(1.0, 0, 0), (-1.0, 0, 0), (0, 1.0, 0), (0, -1.0, 0), (0, 0, 1.0), (0, 0, -1.0)]
    faceted_grain(normals, d, spacing_um=0.5, shape=(25, 25, 25)).volume_um3().backward()
    assert d.grad is not None and float(d.grad.abs().min()) > 0.0


@pytest.mark.device
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_device_parity_of_the_grain_volume(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("mps unavailable")
    dt = torch.float32 if device == "mps" else DT   # MPS has no float64

    ref = float(sphere_grain(4.0, spacing_um=0.5).volume_um3())
    g = sphere_grain(4.0, spacing_um=0.5, device=device, dtype=dt)
    assert g.positions.device.type == device
    assert abs(float(g.volume_um3()) - ref) / ref < 1e-4


@pytest.mark.device
def test_grain_to_moves_the_field_with_it():
    g = attach_uniform_field(sphere_grain(3.0, spacing_um=1.0)).to(dtype=torch.float32)
    assert g.positions.dtype == torch.float32
    assert g.field.F.dtype == torch.float32
