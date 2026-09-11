"""Gates on the periodic-cell treatment: the reciprocal lattice, and intensity between its points.

Fast versions of the gates registered in
``packages/midas_saxs/dev/paper/PREREGISTER_periodic_lines.md``; the full registered
runs are ``periodic_lines_gates.py`` beside it.

Why this exists: /verify claim 8305670629a7 refuted the single-window treatment of
lines that close through a periodic boundary. Renumbering the nodes of a bowed line
changed |A|^2 at generic q, because one period of the chain was cut wherever the walk
ended. The tests here use bowed lines and dipoles on purpose. A straight line is the
one shape that cannot show the defect, and it is what the refuted gate used.
"""
import math

import pytest
import torch

from midas_ddd import (
    DislocationNetwork,
    combine,
    isotropic_stiffness,
    lattice_small_angle_amplitude,
    line_small_angle_amplitude,
    periodic_small_angle_intensity,
    prismatic_loop,
    segment_components,
    small_angle_amplitude,
    winding_components,
)
from midas_ddd.elasticity import _voigt_to_tensor
from midas_ddd.fourier import _image_shifts, _lattice_average, _segment_sum

DT = torch.float64
LAM, MU = 100.0, 75.0
KAPPA = LAM / (LAM + 2.0 * MU)
B_UM = 2.556e-4
C6 = isotropic_stiffness(LAM, MU)
L = 0.2
ASTAR = 2.0 * math.pi / L


def _net(nodes, segs, burgers):
    m = segs.shape[0]
    return DislocationNetwork(
        nodes_um=nodes, segments=segs, burgers_b=burgers, normals=torch.zeros(m, 3, dtype=DT),
        b_magnitude_A=2.556, cell_min_um=torch.full((3,), -L / 2, dtype=DT),
        cell_max_um=torch.full((3,), L / 2, dtype=DT), pbc=(True, True, True),
        constraints=torch.zeros(nodes.shape[0], dtype=torch.int64))


def _wrap(p):
    return p - L * torch.round(p / L)


def _bowed_line(n=40, y0=0.0, b=(0.0, 0.0, 1.0)):
    """A bowed line along x that closes only through the x faces."""
    s = torch.arange(n, dtype=DT) / n
    pts = torch.stack([s * L - 0.5 * L,
                       y0 + 0.10 * L * torch.sin(2 * math.pi * s),
                       0.05 * L * torch.sin(4 * math.pi * s + 0.7)], dim=1)
    segs = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n], dim=1)
    return _wrap(pts), segs, torch.tensor(b, dtype=DT).expand(n, 3).clone()


def _same_network_differently_stored(nodes, segs, burgers, seed):
    """Permuted node labels, every node moved by a random lattice vector, half the
    segments reversed with b -> -b, and the segment list shuffled."""
    g = torch.Generator().manual_seed(seed)
    N, M = nodes.shape[0], segs.shape[0]
    perm = torch.randperm(N, generator=g)
    moved = torch.empty_like(nodes)
    moved[perm] = nodes + L * torch.randint(-2, 3, (N, 3), generator=g).to(DT)
    new_segs = perm[segs]
    flip = torch.rand(M, generator=g) < 0.5
    new_segs = torch.where(flip[:, None], new_segs.flip(1), new_segs)
    new_b = torch.where(flip[:, None], -burgers, burgers)
    order = torch.randperm(M, generator=g)
    return moved, new_segs[order], new_b[order]


def _random_q(n, seed, lo, hi):
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(n, 3, generator=g, dtype=DT)
    d = d / torch.linalg.vector_norm(d, dim=1, keepdim=True)
    return d * (lo + (hi - lo) * torch.rand(n, 1, generator=g, dtype=DT))


def _storage_difference(a, b):
    """``(relative, absolute)``: the worst ``|a-b|/max(a,b)`` over values at least 1e-6 of
    the peak, and the worst ``|a-b|/peak`` over all. At intensity nulls float64 cancellation
    alone gives relative differences above 1e-10 (PREREGISTER_periodic_lines.md, A2), so
    nulls are held to the absolute bound instead."""
    top = torch.maximum(a, b)
    peak = float(top.max())
    bright = top >= 1e-6 * peak
    relative = float(((a - b).abs()[bright] / top[bright]).max())
    return relative, float((a - b).abs().max()) / peak


def _single_window(net, q):
    """The REFUTED estimator, rebuilt from the kernel: one walked period, at pixel q."""
    shift, comp, n, _ = _image_shifts(net)
    pos = net.nodes_um + torch.as_tensor(shift, dtype=DT) * net.cell_size_um
    d = net.segment_vectors_um()
    mid = pos[net.segments[:, 0]] + 0.5 * d
    A = _segment_sum(q, d, mid, net.burgers_um(), comp, n, _voigt_to_tensor(C6), 4_000_000)
    return A.sum(dim=0).abs() ** 2


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_winding_is_a_property_of_the_network_not_of_how_it_is_stored():
    nodes, segs, b = _bowed_line()
    assert winding_components(_net(nodes, segs, b))[2].tolist() == [True]
    other = _net(*_same_network_differently_stored(nodes, segs, b, seed=1))
    assert winding_components(other)[2].tolist() == [True]
    loop = prismatic_loop(radius_um=0.01, burgers=(0, 0, 1.0), n_segments=24,
                          center_um=(0.095, 0.0, 0.0))
    folded = _net(_wrap(loop.nodes_um), loop.segments, loop.burgers_b)
    assert float(folded.nodes_um[:, 0].min()) < 0.0 < float(folded.nodes_um[:, 0].max())
    assert winding_components(folded)[2].tolist() == [False]


# ---------------------------------------------------------------------------
# On the lattice
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_lattice_amplitude_of_a_bowed_line_ignores_labels_orientation_and_images():
    nodes, segs, b = _bowed_line()
    g = torch.Generator().manual_seed(0)
    hkl = torch.randint(-6, 7, (60, 3), generator=g)
    G = hkl[(hkl != 0).any(dim=1)].to(DT) * ASTAR
    ref = lattice_small_angle_amplitude(_net(nodes, segs, b), G, C6)
    for seed in (1, 2):
        got = lattice_small_angle_amplitude(
            _net(*_same_network_differently_stored(nodes, segs, b, seed)), G, C6)
        assert torch.allclose(got, ref, rtol=1e-10, atol=1e-12 * float(ref.abs().max()))


@pytest.mark.unit
def test_straight_edge_dipole_on_the_lattice_equals_the_prismatic_strip():
    """On the sheet G_z = 0 the dipole is a strip of width 2d and length L, exactly,
    straddling a face or not. The screening lives in this coherent amplitude."""
    n, d = 20, 0.001
    z = torch.arange(n, dtype=DT) * L / n - L / 2
    ring = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n], dim=1)
    hk = torch.tensor([[h, k, 0] for h in range(-8, 9) for k in range(-8, 9) if (h, k) != (0, 0)],
                      dtype=DT)
    G = hk * ASTAR
    Gp = torch.linalg.vector_norm(G, dim=1)
    sin_xi = G[:, 1] / Gp
    ref = ((1 - KAPPA) * B_UM * sin_xi ** 2 * (2 * d) * L
           * torch.sinc(Gp * d * sin_xi / math.pi).abs())
    keep = ref > 1e-6 * ref.max()
    for y0 in (0.0, L / 2):
        up = torch.stack([torch.zeros(n, dtype=DT), torch.full((n,), y0 + d, dtype=DT), z], dim=1)
        down = torch.stack([torch.zeros(n, dtype=DT), torch.full((n,), y0 - d, dtype=DT), z], dim=1)
        burg = torch.cat([torch.tensor([1.0, 0.0, 0.0], dtype=DT).expand(n, 3),
                          torch.tensor([-1.0, 0.0, 0.0], dtype=DT).expand(n, 3)])
        net = _net(_wrap(torch.cat([up, down])), torch.cat([ring, ring + n]), burg)
        got = lattice_small_angle_amplitude(net, G, C6).abs()
        assert torch.allclose(got[keep], ref[keep], rtol=1e-9)


# ---------------------------------------------------------------------------
# Between lattice points
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_intensity_between_lattice_points_ignores_storage_where_the_single_window_does_not():
    nodes, segs, b = _bowed_line()
    q = _random_q(300, seed=11, lo=3.2 * ASTAR, hi=3.2 * ASTAR + 300.0)
    a = _net(nodes, segs, b)
    other = _net(*_same_network_differently_stored(nodes, segs, b, seed=3))
    Ia, info = periodic_small_angle_intensity(a, q, C6)
    Ib, _ = periodic_small_angle_intensity(other, q, C6)
    assert float(Ia.max()) > 0.0
    relative, absolute = _storage_difference(Ia, Ib)
    assert relative <= 1e-10 and absolute <= 1e-12
    # Planted control: the refuted estimator, same pair, same metric.
    assert _storage_difference(_single_window(a, q), _single_window(other, q))[0] > 1e-3


@pytest.mark.unit
def test_a_dipole_straddling_a_face_keeps_its_intensity_and_its_screening():
    """Two antiparallel bowed lines 2 nm apart, then the whole network shifted by L/2
    in y. The coherent intensity must not move, and at q d << 1 it must sit far below
    the per-line incoherent sum."""
    n = 40
    n1, s1, b1 = _bowed_line(n=n, y0=0.001, b=(1.0, 0.0, 0.0))
    n2, s2, b2 = _bowed_line(n=n, y0=-0.001, b=(-1.0, 0.0, 0.0))
    nodes, segs, burg = torch.cat([n1, n2]), torch.cat([s1, s2 + n]), torch.cat([b1, b2])
    shifted = _wrap(nodes + torch.tensor([0.0, L / 2, 0.0], dtype=DT))
    q = _random_q(200, seed=12, lo=3.2 * ASTAR, hi=150.0)
    I0, _ = periodic_small_angle_intensity(_net(nodes, segs, burg), q, C6)
    I1, _ = periodic_small_angle_intensity(_net(shifted, segs, burg), q, C6)
    relative, absolute = _storage_difference(I0, I1)
    assert relative <= 1e-10 and absolute <= 1e-12
    per_line, _ = periodic_small_angle_intensity(_net(nodes, segs, burg), q, C6, per_component=True)
    assert tuple(per_line.shape) == (2, 200)
    assert float(I0.sum()) < 0.1 * float(per_line.sum())


@pytest.mark.unit
@pytest.mark.parametrize("ratio", [0.25, 1.0, 2.0, 3.0])
def test_a_constant_field_returns_the_constant_within_the_stated_ripple(ratio):
    """Parseval normalisation: a constant lattice field comes back as c (1 +/- rho), rho the
    Poisson ripple of the lattice sum that info reports. At the default resolution (2 a*)
    it is 4e-6; at one lattice spacing 18 %; at a quarter the estimator shows the cell's
    own lattice peaks, which is what a narrow window sees. Orthorhombic cell."""
    astar = torch.tensor([2 * math.pi / 0.20, 2 * math.pi / 0.25, 2 * math.pi / 0.30], dtype=DT)
    # above q_floor: nearer the origin the excluded G = 0 term is missing, by design
    q = _random_q(200, seed=5, lo=5.0 * float(astar.max()), hi=400.0)
    I, info = _lattice_average(q, astar, ratio * float(astar.max()),
                               lambda hkl: torch.ones(1, hkl.shape[0], dtype=DT))
    qmin = float(torch.linalg.vector_norm(q, dim=1).min())
    assert qmin >= info["q_floor_inv_um"]
    # the excluded G = 0 term is missing from every value, by design: at most W(q_min)/<sum W>
    s = info["sigma_inv_um"]
    missing_origin = math.exp(-qmin ** 2 / (2 * s * s)) / info["parseval_norm"]
    dev = float((I[0] - 1.0).abs().max())
    assert dev <= info["ripple_bound"] + missing_origin + 1e-12
    if ratio >= 2.0:
        assert dev <= 1e-5
    if ratio == 0.25:
        assert dev > 1.0                      # lattice peaks, not a smooth field
    assert info["q_floor_inv_um"] >= 3.0 * float(astar.max())


@pytest.mark.unit
def test_the_same_structure_in_a_supercell_gives_the_same_intensity_per_cell():
    """The physics lens's attack on the earlier normalised form: storing the identical
    periodic network in a 2x1x1 supercell changed I/n by 5.7 % at FWHM = a*. The Parseval
    form is supercell-invariant by construction; the normalised form is kept as the
    planted control and must still show the defect."""
    nodes, segs, b = _bowed_line()
    n = nodes.shape[0]
    one = _net(nodes, segs, b)
    # Two copies along x. The line winds along x, so copy 0's last node must chain to copy
    # 1's first node and copy 1's last node back to copy 0's first: a segment whose head
    # is stored a lattice vector m away from its continuous position ends, in copy c, on
    # copy (c - m). Duplicating the segment list would close each copy on itself through
    # a 0.975 L segment, a different structure (which the long-segment guard refuses).
    d = one.segment_vectors_um()
    m = torch.round((nodes[segs[:, 1]] - nodes[segs[:, 0]] - d)[:, 0] / L).to(torch.int64)
    head_copy = torch.remainder(-m, 2), torch.remainder(1 - m, 2)
    two_segs = torch.cat([torch.stack([segs[:, 0], segs[:, 1] + head_copy[0] * n], 1),
                          torch.stack([segs[:, 0] + n, segs[:, 1] + head_copy[1] * n], 1)])
    two_nodes = torch.cat([nodes, nodes + torch.tensor([L, 0.0, 0.0], dtype=DT)])
    two_nodes[:, 0] -= 0.5 * L                                    # cell [-L, L) along x
    two = DislocationNetwork(
        nodes_um=two_nodes, segments=two_segs, burgers_b=torch.cat([b, b]),
        normals=torch.zeros(2 * n, 3, dtype=DT), b_magnitude_A=2.556,
        cell_min_um=torch.tensor([-L, -L / 2, -L / 2], dtype=DT),
        cell_max_um=torch.tensor([L, L / 2, L / 2], dtype=DT), pbc=(True, True, True),
        constraints=torch.zeros(2 * n, dtype=torch.int64))
    assert winding_components(two)[1] == 1                          # one line, period 2L
    q = _random_q(200, seed=8, lo=3.3 * ASTAR, hi=200.0)
    for ratio in (0.25, 1.0, 3.0):
        I1, _ = periodic_small_angle_intensity(one, q, C6, fwhm_inv_um=ratio * ASTAR)
        I2, _ = periodic_small_angle_intensity(two, q, C6, fwhm_inv_um=ratio * ASTAR)
        relative, absolute = _storage_difference(I1, I2 / 2)
        assert relative <= 1e-10 and absolute <= 1e-12, (ratio, relative, absolute)
    # planted control: the normalised average, same inputs, FWHM = a*
    astar1, astar2 = 2 * math.pi / one.cell_size_um, 2 * math.pi / two.cell_size_um

    def stack(net, astar, normalised):
        def values(hkl):
            A = lattice_small_angle_amplitude(net, hkl.to(DT) * astar, C6)
            return (A.abs() ** 2)[None]
        return _lattice_average(q, astar, ASTAR, values, normalised=normalised)[0][0]

    J1, J2 = stack(one, astar1, True), stack(two, astar2, True)
    assert _storage_difference(J1, J2 / 2)[0] > 1e-3


@pytest.mark.unit
def test_a_lattice_point_equals_the_explicit_neighbour_weighted_sum():
    """At G the estimator is |A(G)|^2 W(0)/<sum W> plus its neighbours' leakage, the
    explicit 7x7x7 Parseval-weighted sum; at narrow resolution the peak factor is
    prod a*_i / (s sqrt(2 pi))^3, about 53 at a quarter spacing."""
    nodes, segs, b = _bowed_line()
    net = _net(nodes, segs, b)
    g = torch.Generator().manual_seed(4)
    hkl = torch.randint(-5, 6, (30, 3), generator=g)
    hkl = hkl[(hkl != 0).any(dim=1)]
    G = hkl.to(DT) * ASTAR
    fwhm = 0.25 * ASTAR
    s = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    norm = (2 * math.pi * s * s) ** 1.5 / ASTAR ** 3
    box = torch.tensor([(h, k, l) for h in range(-3, 4) for k in range(-3, 4) for l in range(-3, 4)])
    expected = torch.empty(G.shape[0], dtype=DT)
    for i in range(G.shape[0]):
        nb = hkl[i] + box
        nb = nb[(nb != 0).any(dim=1)]
        Gn = nb.to(DT) * ASTAR
        An = lattice_small_angle_amplitude(net, Gn, C6).abs() ** 2
        w = torch.exp(-((G[i] - Gn) ** 2).sum(dim=1) / (2 * s * s))
        expected[i] = (w * An).sum() / norm
    I, info = periodic_small_angle_intensity(net, G, C6, fwhm_inv_um=fwhm)
    assert torch.allclose(I, expected, rtol=1e-12, atol=1e-12 * float(expected.max()))
    peak_factor = float(I.max() / (lattice_small_angle_amplitude(net, G, C6).abs() ** 2).max())
    assert 40.0 < peak_factor < 60.0


@pytest.mark.unit
def test_a_pure_screw_network_is_reported_as_roundoff_and_an_edge_is_not():
    """Isotropic elasticity: a screw has no dilatation, so its lattice intensity is float64
    residue. The floor (eps sum |b||d| / |G|)^2 is implementation-independent (reproduction
    lens of /verify 887b8869a9c4)."""
    n = 40
    s = torch.arange(n, dtype=DT) / n
    pts = _wrap(torch.stack([s * L - 0.5 * L, torch.zeros(n, dtype=DT), torch.zeros(n, dtype=DT)], 1))
    segs = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n], dim=1)
    q = _random_q(100, seed=9, lo=3.3 * ASTAR, hi=200.0)
    screw = _net(pts, segs, torch.tensor([1.0, 0.0, 0.0], dtype=DT).expand(n, 3).clone())
    edge = _net(pts, segs, torch.tensor([0.0, 1.0, 0.0], dtype=DT).expand(n, 3).clone())
    I_s, info_s = periodic_small_angle_intensity(screw, q, C6)
    I_e, info_e = periodic_small_angle_intensity(edge, q, C6)
    assert float((I_s / info_s["roundoff_floor"]).max()) < 1e3
    assert float((I_e / info_e["roundoff_floor"]).max()) > 1e20
    assert info_s["burgers"]["n_violating"] == 0


@pytest.mark.unit
def test_a_winding_line_whose_burgers_vector_changes_warns():
    nodes, segs, b = _bowed_line()
    bad = b.clone()
    bad[20:] = -bad[20:]
    q = torch.tensor([[130.0, 40.0, -25.0]], dtype=DT)
    with pytest.warns(UserWarning, match="not conserved"):
        _, info = periodic_small_angle_intensity(_net(nodes, segs, bad), q, C6)
    assert info["burgers"]["n_violating"] == 2
    _, info = periodic_small_angle_intensity(_net(nodes, segs, b), q, C6)
    assert info["burgers"]["n_violating"] == 0


@pytest.mark.unit
def test_segments_longer_than_a_quarter_cell_are_refused():
    """Minimum-image storage folds a segment of half a cell or longer into its mirror image
    with no trace; the guard sits at a quarter cell, five times any DDD discretisation."""
    b = torch.tensor([0.0, 0.0, 1.0], dtype=DT)
    q = torch.tensor([[130.0, 40.0, -25.0]], dtype=DT)
    # a period-2L line stored as three segments of 2L/3: every one folds to L/3
    pts = _wrap(torch.tensor([[0.0, 0.0, 0.0], [2 * L / 3, 0.0, 0.0], [4 * L / 3, 0.0, 0.0]], dtype=DT))
    three = _net(pts, torch.tensor([[0, 1], [1, 2], [2, 0]]), b.expand(3, 3).clone())
    with pytest.raises(ValueError, match="minimum-image"):
        winding_components(three)
    with pytest.raises(ValueError, match="minimum-image"):
        periodic_small_angle_intensity(three, q, C6)
    # two segments of exactly L/2
    pts = torch.tensor([[-L / 2, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=DT)
    two = _net(pts, torch.tensor([[0, 1], [1, 0]]), b.expand(2, 3).clone())
    with pytest.raises(ValueError, match="minimum-image"):
        lattice_small_angle_amplitude(two, torch.tensor([[ASTAR, 0.0, 0.0]], dtype=DT), C6)
    # a fine discretisation passes
    nodes, segs, bb = _bowed_line()
    assert winding_components(_net(nodes, segs, bb))[2].tolist() == [True]


@pytest.mark.unit
def test_labels_excluding_a_bad_segments_whole_component_are_not_refused():
    """A5/A6 /verify (claim 3d56329b9706), artifact lens's own example: the guard used
    to fire on the WHOLE network even when `labels` excluded the offending segment's
    entire component, so a well-behaved loop population sharing one DislocationNetwork
    with an unrelated mis-discretised line could not be computed on at all. Two
    SEPARATE components (no shared node): 3 loops, and an unrelated winding line with
    one segment at L/3 (over the 0.25 L limit; loops rather than a winding dipole,
    because line_small_angle_amplitude refuses winding components unconditionally
    regardless of this fix, so a winding "good" population would not isolate it)."""
    dt = torch.float64
    kw = dict(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=12, cell_size_um=L,
             pbc=(True, True, True), b_magnitude_A=2.556)
    loops = combine([prismatic_loop(center_um=c, **kw)
                     for c in ((0.0, 0.0, 0.0), (0.05, 0.02, -0.03), (-0.04, 0.03, 0.01))],
                    cell_size_um=L)
    n_loops = loops.n_segments

    # one winding via 3 segments of L/3 each (> the 0.25 L limit), same construction as
    # test_segments_longer_than_a_quarter_cell_are_refused's three_2L_over_3 case
    bad_n = 3
    s = torch.arange(bad_n, dtype=dt) / bad_n
    bad_pts = _wrap(torch.stack([s * L, torch.full((bad_n,), 0.4 * L, dtype=dt),
                                 torch.full((bad_n,), 0.4 * L, dtype=dt)], dim=1))
    bad_segs = torch.stack([torch.arange(bad_n), (torch.arange(bad_n) + 1) % bad_n], dim=1)
    assert float((bad_pts[bad_segs[:, 1]] - bad_pts[bad_segs[:, 0]])[:, 0].abs().max()) > 0.25 * L

    nodes = torch.cat([loops.nodes_um, bad_pts])
    segs = torch.cat([loops.segments, bad_segs + loops.n_nodes])
    burg = torch.cat([loops.burgers_b, torch.tensor([0.0, 0.0, 1.0], dtype=dt).expand(bad_n, 3)])
    net = _net(nodes, segs, burg)
    # segment_components has no long-segment guard (pure topology), so it can classify
    # this net even though winding_components(net) itself must refuse it unscoped (below)
    comp, n_comp = segment_components(net)
    assert n_comp == 4    # 3 loops + the bad line, all separate components

    loop_component_ids = torch.unique(comp[:n_loops])
    loop_labels = torch.where(torch.isin(comp, loop_component_ids), comp, torch.full_like(comp, -1))
    # sanity: loop_labels really does touch only the loop components, not the bad line's
    assert bool((loop_labels[n_loops:] < 0).all())

    q = torch.tensor([[130.0, 40.0, -25.0]], dtype=dt)
    G = torch.tensor([[ASTAR, 0.0, 0.0]], dtype=dt)
    # unscoped (no labels): the bad line's component is included, must still refuse
    with pytest.raises(ValueError, match="minimum-image"):
        winding_components(net)
    with pytest.raises(ValueError, match="minimum-image"):
        line_small_angle_amplitude(net, q, C6)
    with pytest.raises(ValueError, match="minimum-image"):
        lattice_small_angle_amplitude(net, G, C6)
    with pytest.raises(ValueError, match="minimum-image"):
        periodic_small_angle_intensity(net, q, C6)
    # scoped to the loops' labels: none of the three periodic entry points should raise
    n_groups = int(comp.max()) + 1
    line_small_angle_amplitude(net, q, C6, per_component=True, labels=loop_labels, n_groups=n_groups)
    lattice_small_angle_amplitude(net, G, C6, per_component=True, labels=loop_labels, n_groups=n_groups)
    periodic_small_angle_intensity(net, q, C6, per_component=True, labels=loop_labels, n_groups=n_groups)


@pytest.mark.unit
def test_a_long_segment_inside_a_kept_component_still_refuses():
    """The scoping fix must not become a loophole: if the bad segment shares a component
    with a KEPT segment (even if `labels` only selects a subset of that one component's
    segments), the shift walk can still propagate the fold error to the kept segment's
    position, so it must still refuse."""
    dt = torch.float64
    n = 10
    s = torch.arange(n, dtype=dt) / n
    pts = _wrap(torch.stack([s * 0.30 * L * n, 0.02 * L * torch.sin(2 * math.pi * s),
                             0.02 * L * torch.cos(2 * math.pi * s)], dim=1))
    segs = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n], dim=1)
    net = _net(pts, segs, torch.tensor([0.0, 0.0, 1.0], dtype=dt).expand(n, 3))
    frac = (pts[segs[:, 1]] - pts[segs[:, 0]])[:, 0].abs() / L
    bad_seg = int(frac.argmax())
    assert float(frac[bad_seg]) > 0.25          # this network really does trip the guard
    keep_this_one_only = segs[:, 0] == (bad_seg + n // 2) % n     # a DIFFERENT segment, same component
    labels = torch.where(keep_this_one_only, torch.zeros(n, dtype=torch.int64),
                         torch.full((n,), -1, dtype=torch.int64))
    q = torch.tensor([[130.0, 40.0, -25.0]], dtype=dt)
    with pytest.raises(ValueError, match="minimum-image"):
        line_small_angle_amplitude(net, q, C6, labels=labels, n_groups=1)


@pytest.mark.autograd
def test_periodic_intensity_gradient_matches_finite_difference():
    nodes, segs, b = _bowed_line(n=16)
    q = torch.tensor([[130.0, 40.0, -25.0], [-60.0, 170.0, 90.0]], dtype=DT)

    def objective(x):
        I, _ = periodic_small_angle_intensity(_net(x, segs, b), q, C6)
        return I.sum()

    x = nodes.clone().requires_grad_(True)
    objective(x).backward()
    i, ax, h = 3, 1, 1e-7
    plus, minus = nodes.clone(), nodes.clone()
    plus[i, ax] += h
    minus[i, ax] -= h
    fd = float(objective(plus) - objective(minus)) / (2 * h)
    assert float(x.grad[i, ax]) == pytest.approx(fd, rel=1e-5)


# ---------------------------------------------------------------------------
# Refusals and warnings
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_refuses_off_lattice_zero_out_of_range_and_non_periodic():
    nodes, segs, b = _bowed_line()
    net = _net(nodes, segs, b)
    with pytest.raises(ValueError, match="not on the cell's reciprocal lattice"):
        lattice_small_angle_amplitude(net, torch.tensor([[1.5 * ASTAR, 0.0, 0.0]], dtype=DT), C6)
    with pytest.raises(ValueError, match="G = 0"):
        lattice_small_angle_amplitude(net, torch.zeros(1, 3, dtype=DT), C6)
    with pytest.raises(ValueError, match="outside"):
        periodic_small_angle_intensity(net, torch.ones(1, 3, dtype=DT), C6, fwhm_inv_um=0.1 * ASTAR)
    loop = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1.0))
    with pytest.raises(NotImplementedError, match="periodic along all three"):
        periodic_small_angle_intensity(loop, torch.ones(1, 3, dtype=DT), C6)


@pytest.mark.unit
def test_a_coherent_sum_over_several_loops_in_a_periodic_cell_warns():
    kw = dict(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=12, cell_size_um=L,
              pbc=(True, True, True))
    net = combine([prismatic_loop(center_um=(0.0, 0.0, 0.0), **kw),
                   prismatic_loop(center_um=(0.05, 0.02, -0.03), **kw)], cell_size_um=L)
    q = torch.tensor([[300.0, 50.0, 20.0]], dtype=DT)
    with pytest.warns(UserWarning, match="several loops coherently"):
        small_angle_amplitude(net, q, C6)
