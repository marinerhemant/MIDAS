"""Gates on the line-integral small-angle amplitude -- closed loops AND open lines.

Fast versions of the gates registered in
``packages/midas_saxs/dev/paper/PREREGISTER_line_term.md``; the full registered
runs are ``line_term_gates.py`` beside it. Three tests carry the weight, because
each compares the code with something it was not derived from:

* `test_straight_edge_matches_thomson_eq6_on_the_sheet` -- Thomson, Levine &
  Long, NIST IR 6117 = Acta Cryst. A 55, 433 (1999), Eq. (6), coded from the page.
* `test_edge_dipole_equals_the_prismatic_strip_on_the_sheet` -- the exact
  Laue-corrected amplitude of a planar prismatic loop, ``(1-kappa) b sin^2 A~(q)``.
* `test_line_form_equals_surface_form_for_closed_loops` -- Stokes' theorem against
  the independently verified cut-surface kernel. An implementation identity, not
  evidence about the physics.

Plus the periodic-cell bookkeeping that ExaDiS networks need, including the bug it
exposed: `find_loops` used to call a line that closes only through the periodic
boundary a loop. Such lines are refused here; their treatment is in
`test_periodic_lines.py`.
"""
import math

import pytest
import torch

from midas_ddd import (
    DislocationNetwork,
    combine,
    find_loops,
    isotropic_stiffness,
    line_small_angle_amplitude,
    loop_area_vectors_um2,
    polygon_area_exact_um2,
    prismatic_loop,
    segment_components,
    small_angle_amplitude,
    straight_line,
    validate_network,
)

DT = torch.float64
LAM, MU = 100.0, 75.0
NU = LAM / (2.0 * (LAM + MU))
KAPPA = LAM / (LAM + 2.0 * MU)
B_UM = 2.556e-4
# Thomson, Levine & Long, NIST IR 6117, Eq. (5).
KAPPA_T = -(1.0 / (2.0 * math.pi)) * (1.0 - 2.0 * NU) / (1.0 - NU)
C6 = isotropic_stiffness(LAM, MU)


def _unit(v):
    v = torch.as_tensor(v, dtype=DT)
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)


def _azimuths_without_zeros():
    return torch.tensor([k for k in range(24) if k not in (0, 12)], dtype=DT) * (math.pi / 12)


def _periodic_line(*, wrap, n=40, L=0.2, direction=(1.0, 2.0, 3.0)):
    """One period of a straight (1,2,3) edge line: folded into a periodic cell as a
    torus-closed circuit (``wrap=True``), or built as an open chain (``False``)."""
    period = torch.tensor(direction, dtype=DT) * L
    t = period / torch.linalg.norm(period)
    e = _unit(torch.linalg.cross(t, torch.tensor([0.0, 0.0, 1.0], dtype=DT)))
    if wrap:
        s = torch.arange(n, dtype=DT) / n
        pts = s[:, None] * period - 0.5 * period
        nodes = pts - L * torch.round(pts / L)
        segs = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n], dim=1)
        pbc, half = (True, True, True), L / 2
    else:
        s = torch.arange(n + 1, dtype=DT) / n
        nodes = s[:, None] * period - 0.5 * period
        segs = torch.stack([torch.arange(n), torch.arange(1, n + 1)], dim=1)
        pbc, half = (False, False, False), 2.0
    m = segs.shape[0]
    net = DislocationNetwork(
        nodes_um=nodes, segments=segs, burgers_b=e.expand(m, 3).clone(),
        normals=torch.zeros(m, 3, dtype=DT), b_magnitude_A=2.556,
        cell_min_um=torch.full((3,), -half, dtype=DT), cell_max_um=torch.full((3,), half, dtype=DT),
        pbc=pbc, constraints=torch.zeros(nodes.shape[0], dtype=torch.int64))
    return net, t


# ---------------------------------------------------------------------------
# Periodic bookkeeping
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_find_loops_excludes_a_line_that_closes_only_through_the_periodic_boundary():
    """A periodic straight line is a degree-2 circuit on the torus, not a loop.

    Before this check a (1,2,3) line in a 0.2 um cell came back as one closed loop
    with a 1e-19 um^2 "area" -- a cut surface that does not exist.
    """
    net, _ = _periodic_line(wrap=True)
    loops, winding = find_loops(net, return_winding=True)
    assert loops == [] and len(winding) == 1
    summary = validate_network(net)
    assert summary["n_loops"] == 0 and summary["n_winding_lines"] == 1
    assert any("periodic boundary" in w for w in summary["warnings"])


@pytest.mark.unit
def test_a_real_loop_folded_across_a_periodic_face_is_still_a_loop():
    """The closure test must not throw out a genuine loop that merely straddles a face."""
    R, n = 0.01, 32
    loop = prismatic_loop(radius_um=R, burgers=(0, 0, 1.0), n_segments=n, center_um=(0.1, 0.0, 0.0))
    folded = loop.nodes_um - 0.2 * torch.round(loop.nodes_um / 0.2)
    assert float(folded[:, 0].min()) < 0.0 < float(folded[:, 0].max())     # really folded
    net = DislocationNetwork(
        nodes_um=folded, segments=loop.segments, burgers_b=loop.burgers_b, normals=loop.normals,
        b_magnitude_A=loop.b_magnitude_A, cell_min_um=torch.full((3,), -0.1, dtype=DT),
        cell_max_um=torch.full((3,), 0.1, dtype=DT), pbc=(True, True, True),
        constraints=loop.constraints)
    loops, winding = find_loops(net, return_winding=True)
    assert len(loops) == 1 and winding == []
    area = float(torch.linalg.vector_norm(loop_area_vectors_um2(net, loops)[0]))
    assert area == pytest.approx(polygon_area_exact_um2(R, n), rel=1e-10)


@pytest.mark.unit
def test_a_finite_line_folded_across_a_face_scatters_like_the_unfolded_one():
    """Unwrapping is exact for anything that does not wind. An open, bent line folded
    into the cell, with its nodes renumbered, must give the same |A|^2 as the same
    line built whole. Without unwrapping, the folded pieces sit a lattice vector apart
    and interfere."""
    n = 10
    s = torch.arange(n + 1, dtype=DT) / n
    whole = torch.stack([0.05 + 0.10 * s, -0.02 + 0.05 * s + 0.01 * torch.sin(3.0 * s),
                         0.04 * s ** 2], dim=1)
    segs = torch.stack([torch.arange(n), torch.arange(1, n + 1)], dim=1)
    burg = _unit(torch.tensor([1.0, 0.3, -0.2], dtype=DT)).expand(n, 3).clone()

    def build(nodes, segments, pbc, half):
        return DislocationNetwork(
            nodes_um=nodes, segments=segments, burgers_b=burg, normals=torch.zeros(n, 3, dtype=DT),
            b_magnitude_A=2.556, cell_min_um=torch.full((3,), -half, dtype=DT),
            cell_max_um=torch.full((3,), half, dtype=DT), pbc=pbc,
            constraints=torch.zeros(n + 1, dtype=torch.int64))

    perm = torch.randperm(n + 1, generator=torch.Generator().manual_seed(2))
    folded_nodes = torch.empty_like(whole)
    folded_nodes[perm] = whole - 0.2 * torch.round(whole / 0.2)
    assert float(folded_nodes[:, 0].min()) < 0.0 < float(folded_nodes[:, 0].max())   # really folded
    folded = build(folded_nodes, perm[segs], (True, True, True), 0.1)
    unfolded = build(whole, segs, (False, False, False), 2.0)
    q = _unit(torch.randn(40, 3, generator=torch.Generator().manual_seed(5), dtype=DT)) * 300.0
    I_folded = line_small_angle_amplitude(folded, q, C6).abs() ** 2
    I_whole = line_small_angle_amplitude(unfolded, q, C6).abs() ** 2
    assert torch.allclose(I_folded, I_whole, rtol=1e-10, atol=1e-12 * float(I_whole.max()))


@pytest.mark.unit
def test_a_line_that_closes_only_through_the_boundary_is_refused():
    """One period of a winding line has no amplitude off the cell's reciprocal lattice
    that the network alone defines (/verify claim 8305670629a7). The function used to
    return one anyway, and it changed with node numbering."""
    folded, _ = _periodic_line(wrap=True)
    with pytest.raises(ValueError, match="closes only"):
        line_small_angle_amplitude(folded, torch.tensor([[300.0, 10.0, 5.0]], dtype=DT), C6)


@pytest.mark.unit
def test_components_group_segments_that_share_nodes():
    loop = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=16, cell_size_um=1.0)
    line = straight_line(length_um=0.2, n_segments=5, center_um=(0.3, 0.0, 0.0), cell_size_um=1.0)
    both = combine([loop, line], cell_size_um=1.0)
    labels, n = segment_components(both)
    assert n == 2
    assert len(set(labels[:16].tolist())) == 1 and len(set(labels[16:].tolist())) == 1
    assert int(labels[0]) != int(labels[16])
    q = _unit(torch.tensor([[0.3, 0.2, 1.0], [1.0, 0.0, 0.1]], dtype=DT)) * 400.0
    per = line_small_angle_amplitude(both, q, C6, per_component=True)
    assert tuple(per.shape) == (2, 2)
    assert torch.allclose(per.sum(dim=0), line_small_angle_amplitude(both, q, C6), rtol=1e-12)


# ---------------------------------------------------------------------------
# The physics gates (fast versions)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("character", ["prismatic", "shear"])
def test_line_form_equals_surface_form_for_closed_loops(character):
    """Stokes: closed-int exp(-i q.r) dl = i q x A~(q). An implementation identity
    against the verified cut-surface kernel, not evidence about the physics."""
    R = 0.005
    net = prismatic_loop(radius_um=R, burgers=(0, 0, 1.0), n_segments=64)
    if character == "shear":
        net.burgers_b = torch.tensor([1.0, 0.0, 0.0], dtype=DT).expand(net.n_segments, 3).clone()
    g = torch.Generator().manual_seed(3)
    d = _unit(torch.randn(12, 3, generator=g, dtype=DT))
    for qR in (0.05, 1.0):
        q = d * (qR / R)
        surface = small_angle_amplitude(net, q, C6, method="surface", n_rings=20)
        line = small_angle_amplitude(net, q, C6, method="line")
        assert torch.allclose(line, surface, rtol=1e-9, atol=1e-9 * float(surface.abs().max()))


@pytest.mark.unit
def test_straight_edge_matches_thomson_eq6_on_the_sheet():
    """``a = 4 pi b_e kappa_T sin(xi) sin(q_z H)/(q_p q_z)``, NIST IR 6117 Eq. (6).

    Deliberately not axis-aligned and over the whole azimuth orbit. Exact at
    q_z = 0; at q_z = 0.01 q_p Eq. 6 omits a factor q_p^2/(q_p^2 + q_z^2) = 1 - 1e-4.
    """
    H = 0.1
    t = _unit(torch.tensor([0.3, -0.4, 0.866], dtype=DT))
    ex = _unit(torch.linalg.cross(t, torch.tensor([1.0, 0.0, 0.0], dtype=DT)))
    ey = torch.linalg.cross(t, ex)
    net = straight_line(length_um=2 * H, burgers=tuple(ex.tolist()), line=tuple(t.tolist()),
                        slip_normal=tuple(ey.tolist()), n_segments=5)
    xi = _azimuths_without_zeros()
    for qp in (100.0, 1000.0):
        for qz, rtol in ((0.0, 1e-10), (1e-2 * qp, 2e-4)):
            q = qp * (torch.cos(xi)[:, None] * ex + torch.sin(xi)[:, None] * ey) + qz * t
            got = line_small_angle_amplitude(net, q, C6).abs()
            length_factor = H if qz == 0 else abs(math.sin(qz * H) / qz)
            ref = 4 * math.pi * B_UM * abs(KAPPA_T) * torch.sin(xi).abs() * length_factor / qp
            assert torch.allclose(got, ref, rtol=rtol)


@pytest.mark.unit
def test_isotropic_screw_does_not_scatter_but_an_edge_does():
    """In isotropic elasticity a screw has no dilatation, so (q x b).t = 0 exactly.
    The edge through the same code path is the positive control."""
    H = 0.1
    t = _unit(torch.tensor([0.3, -0.4, 0.866], dtype=DT))
    edge_dir = _unit(torch.linalg.cross(t, torch.tensor([1.0, 0.0, 0.0], dtype=DT)))
    g = torch.Generator().manual_seed(4)
    q = _unit(torch.randn(20, 3, generator=g, dtype=DT)) * 300.0
    screw = straight_line(length_um=2 * H, burgers=tuple(t.tolist()), line=tuple(t.tolist()), n_segments=5)
    edge = straight_line(length_um=2 * H, burgers=tuple(edge_dir.tolist()), line=tuple(t.tolist()),
                         n_segments=5)
    a_edge = line_small_angle_amplitude(edge, q, C6).abs()
    a_screw = line_small_angle_amplitude(screw, q, C6).abs()
    assert float(a_edge.max()) > 0.0
    assert float(a_screw.max()) <= 1e-12 * float(a_edge.max())


@pytest.mark.unit
def test_edge_dipole_equals_the_prismatic_strip_on_the_sheet():
    """Two antiparallel edges at y = +/-d are a prismatic strip minus its short ends,
    and on the sheet q_z = 0 the ends contribute exactly nothing. The planar-loop
    amplitude ``(1-kappa) b sin^2(theta) A~(q)`` is exact at any q, so this holds at
    every q_p d. It is sign-sensitive: a same-sign pair would give a cosine."""
    H, d = 0.1, 0.001
    kw = dict(length_um=2 * H, line=(0, 0, 1.0), slip_normal=(0, 1.0, 0), n_segments=5, cell_size_um=1.0)
    up = straight_line(burgers=(1.0, 0, 0), center_um=(0.0, d, 0.0), **kw)
    down = straight_line(burgers=(-1.0, 0, 0), center_um=(0.0, -d, 0.0), **kw)
    pair = combine([up, down], cell_size_um=1.0)
    xi = _azimuths_without_zeros()
    s = torch.sin(xi)
    for qp_d in (0.01, 1.0):
        qp = qp_d / d
        q = torch.stack([qp * torch.cos(xi), qp * s, torch.zeros_like(xi)], dim=1)
        got = line_small_angle_amplitude(pair, q, C6).abs()
        ref = ((1 - KAPPA) * B_UM * s ** 2 * (2 * d) * (2 * H)
               * torch.sinc(qp * d * s / math.pi).abs())
        assert torch.allclose(got, ref, rtol=1e-9)


# ---------------------------------------------------------------------------
# Differentiability and refusals
# ---------------------------------------------------------------------------

@pytest.mark.autograd
def test_line_amplitude_gradient_matches_finite_difference():
    net = straight_line(length_um=0.2, burgers=(1.0, 0.0, 0.3), line=(0.2, 0.1, 1.0), n_segments=5)
    q = _unit(torch.tensor([[0.3, 0.2, 0.05], [1.0, -0.4, 0.02]], dtype=DT)) * 400.0

    def objective(nodes):
        n2 = DislocationNetwork(
            nodes_um=nodes, segments=net.segments, burgers_b=net.burgers_b, normals=net.normals,
            b_magnitude_A=net.b_magnitude_A, cell_min_um=net.cell_min_um,
            cell_max_um=net.cell_max_um, pbc=net.pbc, constraints=net.constraints)
        return (line_small_angle_amplitude(n2, q, C6).abs() ** 2).sum()

    nodes = net.nodes_um.clone().requires_grad_(True)
    objective(nodes).backward()
    i, ax, h = 2, 0, 1e-7
    plus, minus = net.nodes_um.clone(), net.nodes_um.clone()
    plus[i, ax] += h
    minus[i, ax] -= h
    fd = float(objective(plus) - objective(minus)) / (2 * h)
    assert float(nodes.grad[i, ax]) == pytest.approx(fd, rel=1e-5)


@pytest.mark.unit
def test_line_amplitude_refuses_q_equals_zero():
    with pytest.raises(ValueError, match="exactly zero"):
        line_small_angle_amplitude(straight_line(length_um=0.1), torch.zeros(1, 3, dtype=DT), C6)


@pytest.mark.unit
def test_small_angle_amplitude_rejects_an_unknown_method():
    loop = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1.0))
    with pytest.raises(ValueError, match="method"):
        small_angle_amplitude(loop, torch.tensor([[1.0, 0.0, 0.0]], dtype=DT), C6, method="volume")
