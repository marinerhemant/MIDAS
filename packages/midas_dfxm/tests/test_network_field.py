"""DFXM deformation field from a discrete-dislocation network (ExaDiS / ParaDiS).

Replaces the one-off script in `dev/paper/runs/real_validation`, which took 12
segments of the FCC-Cu network and rebuilt each as an INFINITE straight Stroh
line. That is not an approximation of a finite-segment network -- for a closed
loop it is qualitatively wrong, because the loop's relaxation volume lives
entirely in the closure. `test_infinite_lines_are_not_a_substitute_for_a_loop`
pins the difference.
"""
from pathlib import Path

import pytest
import torch

ddd = pytest.importorskip("midas_ddd", reason="network path needs midas-ddd")
from midas_ddd import (  # noqa: E402
    cubic_stiffness,
    isotropic_stiffness,
    prismatic_loop,
    read_paradis,
    straight_line,
)
from midas_dfxm.dislocation import (  # noqa: E402
    dislocation_deformation_field,
    network_deformation_field,
    stroh_dislocation,
)
from midas_ddd.realspace import segment_dislocations  # noqa: E402

REAL_DATA = (Path(__file__).resolve().parents[2]
             / "midas_dfxm/dev/paper/runs/real_validation/data/fcc_cu.data")
CU = dict(lam=100.0, mu=75.0)


def _grid(n=8, half=0.05):
    g = torch.linspace(-half, half, n, dtype=torch.float64)
    X, Y = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1),
                        torch.zeros(n * n, dtype=torch.float64)], -1)


@pytest.mark.unit
def test_network_field_has_the_deformation_field_contract():
    net = prismatic_loop(radius_um=0.02, burgers=(0, 0, 1.0), n_segments=16)
    pts = _grid()
    f = network_deformation_field(pts, net, isotropic_stiffness(**CU),
                                  shape=(8, 8, 1))
    assert f.F.shape == (pts.shape[0], 3, 3)
    assert torch.isfinite(f.F).all()
    assert torch.allclose(f.positions, pts)
    # F = I + beta, and beta is small but not zero.
    dev = (f.F - torch.eye(3, dtype=torch.float64)).abs().max()
    assert 0.0 < float(dev) < 1.0


@pytest.mark.unit
def test_segment_objects_drop_into_the_existing_superposition_helper():
    """SegmentDislocation matches the StrohDislocation duck type, so the
    pre-existing `dislocation_deformation_field` consumes it unchanged."""
    net = prismatic_loop(radius_um=0.02, burgers=(0, 0, 1.0), n_segments=12)
    C6 = isotropic_stiffness(**CU)
    pts = _grid(n=6)
    via_helper = dislocation_deformation_field(pts, segment_dislocations(net, C6))
    via_network = network_deformation_field(pts, net, C6)
    assert torch.allclose(via_helper.F, via_network.F, rtol=1e-12)


@pytest.mark.unit
def test_infinite_lines_are_not_a_substitute_for_a_loop():
    """The mistake the old one-off script made, pinned.

    Rebuilding a closed loop's segments as infinite straight Stroh lines gives a
    materially different field. If these ever agree, one of the two paths has
    stopped doing what it claims.
    """
    net = prismatic_loop(radius_um=0.02, burgers=(0, 0, 1.0), n_segments=8)
    C6 = isotropic_stiffness(**CU)
    pts = _grid(n=6, half=0.05)

    finite = network_deformation_field(pts, net, C6).F

    segvec = net.segment_vectors_um()
    lines = []
    for s, (i, j) in enumerate(net.segments.tolist()):
        xi = segvec[s] / torch.linalg.norm(segvec[s])
        b = net.burgers_directions()[s]
        # Any plane containing both the line and b.
        nrm = torch.linalg.cross(xi, b)
        if float(torch.linalg.norm(nrm)) < 1e-9:
            nrm = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        lines.append(stroh_dislocation(
            cubic_stiffness(168.4, 121.4, 75.4),
            burgers=tuple(b.tolist()), slip_normal=tuple(nrm.tolist()),
            line=tuple(xi.tolist()),
            core_position=tuple(net.nodes_um[i].tolist()),
            burgers_length_A=2.556, core_radius_um=0.002))
    infinite = dislocation_deformation_field(pts, lines).F

    rel = float((finite - infinite).abs().max()
                / (finite - torch.eye(3, dtype=torch.float64)).abs().max())
    assert rel > 0.5, ("infinite-line superposition happened to match the finite "
                       "network; that would mean one path is not doing its job")


@pytest.mark.unit
def test_open_lines_image_fine_even_though_they_do_not_scatter_at_small_angle():
    """A deformation structure has no relaxation volume and no small-angle
    signature, but it has a perfectly good DFXM field."""
    net = straight_line(length_um=0.5, n_segments=8)
    f = network_deformation_field(_grid(n=6, half=0.05), net,
                                  isotropic_stiffness(**CU))
    dev = (f.F - torch.eye(3, dtype=torch.float64)).abs().max()
    assert float(dev) > 0


@pytest.mark.skipif(not REAL_DATA.exists(), reason="fcc_cu.data not present")
@pytest.mark.unit
def test_real_exadis_network_end_to_end():
    """The whole point of the exercise: a real 5930-segment ExaDiS network to a
    DFXM deformation field, in one call."""
    net = read_paradis(REAL_DATA, b_magnitude_A=2.556)
    assert net.n_segments == 5930
    f = network_deformation_field(_grid(n=6, half=0.3), net,
                                  cubic_stiffness(168.4, 121.4, 75.4))
    assert torch.isfinite(f.F).all()
    strain = float((f.F - torch.eye(3, dtype=torch.float64)).abs().max())
    # A ~1e12 m^-2 network: hundreds of microstrain, not percent and not zero.
    assert 1e-6 < strain < 1e-2


@pytest.mark.autograd
def test_network_field_is_differentiable():
    net = prismatic_loop(radius_um=0.02, burgers=(0, 0, 1.0), n_segments=8)
    net.nodes_um = net.nodes_um.clone().requires_grad_(True)
    f = network_deformation_field(_grid(n=4), net, isotropic_stiffness(**CU))
    f.F.abs().sum().backward()
    assert net.nodes_um.grad is not None
    assert torch.isfinite(net.nodes_um.grad).all()
    assert float(net.nodes_um.grad.abs().max()) > 0
