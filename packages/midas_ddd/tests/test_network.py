"""Network ingest, Burgers conservation, loop inventory, relaxation volume.

The `REAL_DATA` tests run against the 5055-node FCC-Cu ExaDiS network that has
been in this repo since the DFXM round-trip work. It is a *deformation*
structure -- open lines through junctions, no closed loops -- which makes it the
right thing to test ingest and conservation on, and the wrong thing to test the
loop machinery on. The loop gates use `generate.prismatic_loop`, whose area is
known in closed form.
"""
import math
from pathlib import Path

import pytest
import torch

from midas_ddd.generate import (
    combine,
    polygon_area_exact_um2,
    prismatic_loop,
    straight_line,
)
from midas_ddd.network import DislocationNetwork, read_paradis, write_paradis
from midas_ddd.validate import (
    check_burgers_conservation,
    find_loops,
    loop_area_vectors_um2,
    relaxation_volumes_um3,
    resolution_report,
    validate_network,
)

REAL_DATA = (Path(__file__).resolve().parents[2]
             / "midas_dfxm/dev/paper/runs/real_validation/data/fcc_cu.data")

B_CU_A = 2.556
B_CU_UM = B_CU_A * 1e-4


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_prismatic_loop_shape_and_closure():
    net = prismatic_loop(radius_um=0.005, n_segments=24)
    assert net.n_nodes == 24 and net.n_segments == 24
    # Every node has exactly two arms -> the circuit is closed.
    arms = net.node_arms()
    assert all(len(v) == 2 for v in arms.values())


@pytest.mark.unit
def test_prismatic_loop_is_planar_and_normal_to_burgers():
    b = (1.0, 1.0, 1.0)
    net = prismatic_loop(radius_um=0.01, burgers=b, n_segments=16)
    bhat = torch.tensor(b, dtype=torch.float64)
    bhat = bhat / torch.linalg.norm(bhat)
    centred = net.nodes_um - net.nodes_um.mean(dim=0, keepdim=True)
    # Prismatic: the loop plane is normal to b, so every vertex is in-plane.
    assert torch.allclose(centred @ bhat, torch.zeros(16, dtype=torch.float64), atol=1e-12)


@pytest.mark.unit
def test_prismatic_loop_conserves_burgers_exactly():
    net = prismatic_loop(radius_um=0.005, n_segments=24)
    res = check_burgers_conservation(net, raise_on_fail=True)
    assert res.ok and res.max_residual == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("bad", [dict(n_segments=2), dict(radius_um=0.0)])
def test_prismatic_loop_rejects_degenerate_input(bad):
    kw = dict(radius_um=0.005, n_segments=12)
    kw.update(bad)
    with pytest.raises(ValueError):
        prismatic_loop(**kw)


@pytest.mark.unit
def test_straight_line_has_pinned_ends_and_passes_conservation():
    net = straight_line(length_um=1.0, n_segments=20)
    assert int(net.constraints[0]) == 7 and int(net.constraints[-1]) == 7
    # An open line does not conserve flux at its termini; the check must skip
    # pinned nodes rather than flagging a physically fine configuration.
    assert check_burgers_conservation(net).ok
    # ...and it MUST flag them if we stop skipping, or the skip is untested.
    assert not check_burgers_conservation(net, ignore_constrained=False).ok


@pytest.mark.unit
def test_combine_reindexes_and_unions_the_cell():
    loop = prismatic_loop(radius_um=0.005, n_segments=12, center_um=(0.5, 0, 0))
    line = straight_line(length_um=0.4, n_segments=8, center_um=(-0.5, 0, 0))
    both = combine([loop, line])
    assert both.n_nodes == loop.n_nodes + line.n_nodes
    assert both.n_segments == loop.n_segments + line.n_segments
    assert int(both.segments.max()) < both.n_nodes
    assert check_burgers_conservation(both).ok
    assert len(find_loops(both)) == 1        # the line must not close a circuit


@pytest.mark.unit
def test_combine_refuses_mismatched_burgers_scale():
    a = prismatic_loop(radius_um=0.005, n_segments=8, b_magnitude_A=2.556)
    b = prismatic_loop(radius_um=0.005, n_segments=8, b_magnitude_A=2.480)
    with pytest.raises(ValueError, match="b_magnitude_A"):
        combine([a, b])


# ---------------------------------------------------------------------------
# Loop area and relaxation volume -- the primary correctness anchor
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("n_seg", [3, 6, 12, 24, 96])
def test_loop_area_matches_the_closed_form_polygon(n_seg):
    """|A| must equal (n/2) R^2 sin(2 pi / n) exactly -- not pi R^2.

    A 24-gon is 1.1 % smaller than its circumscribing circle. Testing against
    pi R^2 would bake that discretisation error in as a tolerance and hide a
    real 1 % bug in the kernel later.
    """
    R = 0.005
    net = prismatic_loop(radius_um=R, n_segments=n_seg)
    loops = find_loops(net)
    assert len(loops) == 1
    A = loop_area_vectors_um2(net, loops)[0]
    assert float(torch.linalg.norm(A)) == pytest.approx(
        polygon_area_exact_um2(R, n_seg), rel=1e-12)


@pytest.mark.unit
def test_loop_area_vector_points_along_the_loop_normal():
    net = prismatic_loop(radius_um=0.01, burgers=(0.0, 0.0, 1.0), n_segments=32)
    A = loop_area_vectors_um2(net, find_loops(net))[0]
    Ahat = A / torch.linalg.norm(A)
    assert abs(abs(float(Ahat[2])) - 1.0) < 1e-12


@pytest.mark.unit
@pytest.mark.parametrize("R_um", [0.001, 0.005, 0.02])
def test_relaxation_volume_equals_b_dot_A(R_um):
    """dV = b . A. This is what a loop scatters like at q -> 0, so it is the
    quantity every small-angle gate downstream is anchored to."""
    n_seg = 48
    net = prismatic_loop(radius_um=R_um, n_segments=n_seg, b_magnitude_A=B_CU_A)
    dV = relaxation_volumes_um3(net, find_loops(net))
    expected = B_CU_UM * polygon_area_exact_um2(R_um, n_seg)
    assert float(dV[0]) == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_relaxation_volume_scales_as_R_squared():
    n_seg = 64
    v = [float(relaxation_volumes_um3(
            prismatic_loop(radius_um=R, n_segments=n_seg),
            find_loops(prismatic_loop(radius_um=R, n_segments=n_seg)))[0])
         for R in (0.005, 0.010)]
    assert v[1] / v[0] == pytest.approx(4.0, rel=1e-10)


@pytest.mark.unit
def test_vacancy_loop_has_the_opposite_relaxation_volume():
    """Interstitial and vacancy loops differ by the sign of dV. If the kernel
    ever loses that sign, interstitial/vacancy typing becomes impossible."""
    kw = dict(radius_um=0.005, n_segments=24)
    inter = prismatic_loop(**kw, burgers_scale_b=+1.0)
    vac = prismatic_loop(**kw, burgers_scale_b=-1.0)
    dv_i = float(relaxation_volumes_um3(inter, find_loops(inter))[0])
    dv_v = float(relaxation_volumes_um3(vac, find_loops(vac))[0])
    assert dv_i == pytest.approx(-dv_v, rel=1e-12)
    assert dv_i != 0.0


@pytest.mark.unit
def test_open_line_has_no_loops_and_no_relaxation_volume():
    """The negative control for every loop gate: a line encloses no area."""
    net = straight_line(length_um=1.0, n_segments=32)
    loops = find_loops(net)
    assert loops == []
    assert float(relaxation_volumes_um3(net, loops).sum()) == 0.0


@pytest.mark.unit
def test_loop_area_is_independent_of_where_the_loop_sits():
    """Computed about the centroid, so translating the loop must not move |A|."""
    a = prismatic_loop(radius_um=0.005, n_segments=24, center_um=(0, 0, 0))
    b = prismatic_loop(radius_um=0.005, n_segments=24, center_um=(37.0, -12.0, 5.0))
    Aa = loop_area_vectors_um2(a, find_loops(a))[0]
    Ab = loop_area_vectors_um2(b, find_loops(b))[0]
    assert torch.allclose(Aa, Ab, rtol=1e-12, atol=1e-18)


# ---------------------------------------------------------------------------
# Burgers conservation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_conservation_catches_a_deliberately_broken_network():
    net = prismatic_loop(radius_um=0.005, n_segments=12)
    net.burgers_b[3] = net.burgers_b[3] * 1.5        # break one segment
    res = check_burgers_conservation(net)
    assert not res.ok
    assert res.n_violating == 2                       # the two nodes of that segment


@pytest.mark.unit
def test_conservation_raise_message_names_the_stokes_reason():
    net = prismatic_loop(radius_um=0.005, n_segments=12)
    net.burgers_b[0] = net.burgers_b[0] * 2.0
    with pytest.raises(ValueError, match="Stokes"):
        check_burgers_conservation(net, raise_on_fail=True)


@pytest.mark.unit
def test_junction_magnitudes_are_required_for_conservation():
    """A sqrt(2) junction only balances if its magnitude survives ingest.

    This reproduces, in miniature, the bug the real file caught: normalising
    Burgers vectors to unit directions makes a three-arm junction node fail
    conservation by exactly sqrt(2) - 1.
    """
    # Y-junction: two perfect arms b1, b2 meeting a junction arm b1 + b2.
    b1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    b2 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    nodes = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [-1.0, -1.0, 0]],
                         dtype=torch.float64)
    segs = torch.tensor([[0, 1], [0, 2], [3, 0]], dtype=torch.int64)
    burg = torch.stack([b1, b2, b1 + b2])            # node 0: -b1 -b2 +(b1+b2) = 0
    net = DislocationNetwork(
        nodes_um=nodes, segments=segs, burgers_b=burg,
        normals=torch.zeros(3, 3, dtype=torch.float64), b_magnitude_A=B_CU_A,
        cell_min_um=torch.full((3,), -5.0, dtype=torch.float64),
        cell_max_um=torch.full((3,), 5.0, dtype=torch.float64),
        pbc=(False, False, False),
        constraints=torch.tensor([0, 7, 7, 7], dtype=torch.int64),
    )
    assert check_burgers_conservation(net).ok
    assert float(net.burgers_magnitudes_b()[2]) == pytest.approx(math.sqrt(2.0))

    # Now normalise the junction arm, as the first version of the reader did.
    net.burgers_b[2] = net.burgers_b[2] / math.sqrt(2.0)
    res = check_burgers_conservation(net)
    assert not res.ok
    assert res.max_residual == pytest.approx(math.sqrt(2.0) - 1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Real ExaDiS / ParaDiS file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_DATA.exists(), reason="fcc_cu.data not present")
@pytest.mark.unit
def test_read_real_exadis_network():
    net = read_paradis(REAL_DATA, b_magnitude_A=B_CU_A)
    assert net.n_nodes == 5055
    assert net.n_segments == 5930
    # 15 um cube: +-29412 b at 2.556 A.
    assert float(net.cell_size_um[0]) == pytest.approx(2 * 29412 * B_CU_UM, rel=1e-9)
    # A plausible deformation density, ~1e12 m^-2.
    assert 0.1 < net.dislocation_density_um2() < 10.0


@pytest.mark.skipif(not REAL_DATA.exists(), reason="fcc_cu.data not present")
@pytest.mark.unit
def test_real_network_conserves_burgers():
    """Reading with validate=True (the default) would raise if it did not."""
    net = read_paradis(REAL_DATA, b_magnitude_A=B_CU_A)
    res = check_burgers_conservation(net)
    assert res.ok
    assert res.max_residual < 1e-9


@pytest.mark.skipif(not REAL_DATA.exists(), reason="fcc_cu.data not present")
@pytest.mark.unit
def test_real_network_retains_its_junction_segments():
    """54 of the 5930 segments are sqrt(2) junctions. Losing their magnitude is
    what broke conservation on the first read of this file."""
    net = read_paradis(REAL_DATA, b_magnitude_A=B_CU_A)
    mags = net.burgers_magnitudes_b()
    n_junction = int((torch.abs(mags - math.sqrt(2.0)) < 1e-5).sum())
    n_perfect = int((torch.abs(mags - 1.0) < 1e-5).sum())
    assert n_junction == 54
    assert n_perfect + n_junction == net.n_segments


@pytest.mark.skipif(not REAL_DATA.exists(), reason="fcc_cu.data not present")
@pytest.mark.unit
def test_real_network_is_a_deformation_structure_not_a_loop_population():
    """Documents what this file IS, so nobody anchors a loop gate to it."""
    net = read_paradis(REAL_DATA, b_magnitude_A=B_CU_A)
    assert find_loops(net) == []


@pytest.mark.skipif(not REAL_DATA.exists(), reason="fcc_cu.data not present")
@pytest.mark.unit
def test_real_network_is_too_coarse_for_saxs_and_says_so():
    """Median segment 651 nm -> q_max ~ 1e-3 1/A. A SAXS run at 0.1 1/A on this
    network would be reporting the polyline, not the dislocations."""
    net = read_paradis(REAL_DATA, b_magnitude_A=B_CU_A)
    rep = resolution_report(net, q_max_inv_A=0.1)
    assert rep.q_max_supported_inv_A < 1e-2
    assert any("exceeds what this discretisation supports" in w for w in rep.warnings)


@pytest.mark.skipif(not REAL_DATA.exists(), reason="fcc_cu.data not present")
@pytest.mark.unit
def test_paradis_round_trip(tmp_path):
    """read -> write -> read must reproduce the network."""
    net = read_paradis(REAL_DATA, b_magnitude_A=B_CU_A)
    out = tmp_path / "roundtrip.data"
    write_paradis(net, out)
    back = read_paradis(out, b_magnitude_A=B_CU_A)
    assert back.n_nodes == net.n_nodes
    assert back.n_segments == net.n_segments
    assert torch.allclose(back.nodes_um, net.nodes_um, atol=1e-9)
    assert float(torch.abs(back.burgers_magnitudes_b().sort().values
                           - net.burgers_magnitudes_b().sort().values).max()) < 1e-9


@pytest.mark.unit
def test_read_paradis_rejects_a_non_paradis_file(tmp_path):
    p = tmp_path / "nope.txt"
    p.write_text("this is not a restart file\n")
    with pytest.raises(ValueError, match="nodalData"):
        read_paradis(p)


# ---------------------------------------------------------------------------
# Geometry conventions
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_minimum_image_keeps_a_boundary_crossing_segment_short():
    """Without minimum image, a segment wrapping the cell reads as a spurious
    box-length dislocation and the density is wildly overestimated."""
    L = 10.0
    nodes = torch.tensor([[-4.9, 0.0, 0.0], [4.9, 0.0, 0.0]], dtype=torch.float64)
    net = DislocationNetwork(
        nodes_um=nodes,
        segments=torch.tensor([[0, 1]], dtype=torch.int64),
        burgers_b=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        normals=torch.zeros(1, 3, dtype=torch.float64),
        b_magnitude_A=B_CU_A,
        cell_min_um=torch.full((3,), -L / 2, dtype=torch.float64),
        cell_max_um=torch.full((3,), L / 2, dtype=torch.float64),
        pbc=(True, True, True),
        constraints=torch.tensor([7, 7], dtype=torch.int64),
    )
    assert float(net.segment_lengths_um()[0]) == pytest.approx(0.2, abs=1e-12)
    net.pbc = (False, False, False)
    assert float(net.segment_lengths_um()[0]) == pytest.approx(9.8, abs=1e-12)


@pytest.mark.unit
def test_loop_area_survives_a_periodic_boundary():
    """A loop straddling a boundary must still report its true area, because
    the circuit is rebuilt by walking minimum-image segment vectors."""
    R, n = 0.005, 24
    free = prismatic_loop(radius_um=R, n_segments=n, burgers=(0, 0, 1),
                          center_um=(0.0, 0.0, 0.0), cell_size_um=1.0)
    A_free = float(torch.linalg.norm(loop_area_vectors_um2(free, find_loops(free))[0]))

    straddle = prismatic_loop(radius_um=R, n_segments=n, burgers=(0, 0, 1),
                              center_um=(0.0, 0.0, 0.0), cell_size_um=1.0,
                              pbc=(True, True, True))
    straddle.nodes_um = straddle.nodes_um + torch.tensor([0.5, 0.0, 0.0],
                                                         dtype=torch.float64)
    A_str = float(torch.linalg.norm(loop_area_vectors_um2(straddle, find_loops(straddle))[0]))
    assert A_str == pytest.approx(A_free, rel=1e-12)


@pytest.mark.unit
def test_burgers_direction_magnitude_split_matches_the_dfxm_signature():
    """burgers_directions() x burgers_magnitudes_b() x b_magnitude_A must
    reconstruct the physical vector -- that split is what gets handed to
    stroh_dislocation(burgers=..., burgers_length_A=...)."""
    net = prismatic_loop(radius_um=0.005, n_segments=8, burgers=(1, 1, 0))
    d = net.burgers_directions()
    m = net.burgers_magnitudes_b()
    assert torch.allclose(d * m.unsqueeze(-1), net.burgers_b, atol=1e-15)
    assert torch.allclose(torch.linalg.vector_norm(d, dim=-1),
                          torch.ones(net.n_segments, dtype=torch.float64))
    assert torch.allclose(net.burgers_um(), net.burgers_b * (B_CU_A * 1e-4))


@pytest.mark.unit
def test_validate_network_summary_reports_every_check():
    net = combine([prismatic_loop(radius_um=0.005, n_segments=24),
                   straight_line(length_um=0.5, n_segments=10)])
    s = validate_network(net, q_max_inv_A=1.0)
    assert s["burgers_conserved"] is True
    assert s["n_loops"] == 1
    assert s["n_loops_uniform_b"] == 1
    assert s["total_relaxation_volume_um3"] > 0
    assert s["n_segments"] == net.n_segments


@pytest.mark.unit
def test_combine_warns_when_the_unioned_cell_is_not_the_real_box():
    """A density bug that survived until an adversarial lens went looking.

    Every generated loop carries its own cell centred on itself. Union those and
    the volume exceeds the box the loops were placed in -- 14 loops in a 3.2 um
    box unioned to 123.7 um^3 against the true 32.8, making
    `dislocation_density_um2()` 3.8x too low. Any density quoted off that path,
    and anything inferred from one, is wrong by that factor.
    """
    import warnings
    box = 3.2
    g = torch.Generator().manual_seed(7)
    parts = [prismatic_loop(
        radius_um=0.2, burgers=(0, 0, 1.0), n_segments=12,
        center_um=tuple(((torch.rand(3, generator=g, dtype=torch.float64) * 2 - 1)
                         * 0.3 * box).tolist()),
        cell_size_um=box) for _ in range(14)]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        unioned = combine(parts)
    assert any("NOT the box" in str(x.message) for x in w)
    assert unioned.cell_volume_um3 > 2.0 * box ** 3

    explicit = combine(parts, cell_size_um=box)
    assert explicit.cell_volume_um3 == pytest.approx(box ** 3, rel=1e-9)
    # The whole point: the density differs by the inflation factor.
    assert (explicit.dislocation_density_um2()
            > 2.0 * unioned.dislocation_density_um2())


@pytest.mark.unit
def test_combine_accepts_an_explicit_cell_and_does_not_warn():
    import warnings
    a = prismatic_loop(radius_um=0.1, burgers=(0, 0, 1.0), n_segments=8,
                       center_um=(0.5, 0, 0))
    b = prismatic_loop(radius_um=0.1, burgers=(0, 0, 1.0), n_segments=8,
                       center_um=(-0.5, 0, 0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net = combine([a, b], cell_size_um=4.0)
    assert not any(isinstance(x.message, RuntimeWarning) for x in w)
    assert net.cell_volume_um3 == pytest.approx(64.0, rel=1e-9)


@pytest.mark.unit
def test_combine_rejects_conflicting_cell_arguments():
    a = prismatic_loop(radius_um=0.1, burgers=(0, 0, 1.0), n_segments=8)
    with pytest.raises(ValueError, match="OR"):
        combine([a], cell_size_um=4.0, cell_min_um=(-1, -1, -1),
                cell_max_um=(1, 1, 1))
