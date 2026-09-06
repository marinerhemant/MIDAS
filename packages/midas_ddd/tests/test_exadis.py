"""The ExaDiS bridge, exercised against a real ExaDiS build.

Every test here skips without `pyexadis`, which needs a Kokkos/CMake build and
can never be a pip dependency. That makes it tempting to leave the bridge
untested -- and it was, for a while: 275 lines that had never executed, sitting
on the one path a collaborator would actually reach for.

The load-bearing test is
`test_in_process_and_file_bridge_give_the_same_network`. The two routes into
this package -- ExaDiS objects in memory, and ExaDiS's own `.data` file read
back by `read_paradis` -- must agree. If they diverge, one of them is wrong and
the file bridge is the one with independent evidence behind it (it is validated
against the 5055-node FCC-Cu network in the repo).
"""
import math

import pytest
import torch

from midas_ddd.exadis import have_pyexadis
from midas_ddd.network import read_paradis
from midas_ddd.validate import check_burgers_conservation, find_loops, validate_network

pytestmark = pytest.mark.skipif(
    not have_pyexadis(),
    reason="pyexadis not importable; build ExaDiS with -DEXADIS_PYTHON_BINDING=On")

B_CU_A = 2.556


def _exadis_write(utils, N, path):
    """Write an ExaDiS network to ParaDiS format, whatever type the generator gave.

    ExaDiS's own `write_data` calls `N.get_disnet(ExaDisNet)`, i.e. it expects a
    `DisNetManager` -- but `generate_prismatic_config` hands back a bare
    `ExaDisNet`, which has no `get_disnet`. Their generator and their writer do
    not compose directly. `ExaDisNet` does carry its own `write_data`, so use
    that when the wrapper is absent.
    """
    if hasattr(N, "get_disnet"):
        utils.write_data(N, str(path))
    else:
        N.write_data(str(path))


@pytest.mark.unit
def test_require_pyexadis_returns_the_modules():
    from midas_ddd.exadis import require_pyexadis
    pyexadis, utils = require_pyexadis()
    assert hasattr(utils, "generate_prismatic_config")
    assert hasattr(utils, "generate_line_config")
    assert hasattr(utils, "write_data")


@pytest.mark.unit
def test_generate_prismatic_config_in_process():
    """Irradiation loops, straight from ExaDiS into a DislocationNetwork."""
    from midas_ddd.exadis import generate_prismatic_config
    net = generate_prismatic_config(
        crystal="fcc", box_size_b=2000.0, num_loops=6, radius_b=60.0,
        b_magnitude_A=B_CU_A, maxseg_b=30, seed=1234)
    assert net.n_nodes > 0 and net.n_segments > 0
    # read_paradis-style validation must pass on an in-process network too.
    assert check_burgers_conservation(net).ok
    loops = find_loops(net)
    assert len(loops) >= 1, "a prismatic config must contain closed loops"
    assert all(lp.uniform_burgers for lp in loops)


@pytest.mark.unit
def test_generate_line_config_in_process():
    """Deformation lines: open, periodic, and carrying NO closed loops."""
    from midas_ddd.exadis import generate_line_config
    net = generate_line_config(
        crystal="fcc", box_size_b=2000.0, num_lines=4,
        b_magnitude_A=B_CU_A, maxseg_b=100, seed=7)
    assert net.n_segments > 0
    assert check_burgers_conservation(net).ok


@pytest.mark.unit
def test_in_process_and_file_bridge_give_the_same_network(tmp_path):
    """THE gate on the bridge.

    Build one config in ExaDiS, then take both routes into this package:
    ExaDiS objects converted in memory, and ExaDiS's own `write_data` output
    read back by `read_paradis`. They describe the same physical network, so
    every invariant that does not depend on node ordering must match.
    """
    from midas_ddd.exadis import from_pyexadis, require_pyexadis
    _, utils = require_pyexadis()

    N = utils.generate_prismatic_config("fcc", 2000.0, 5, 60.0, maxseg=30, seed=99)
    in_proc = from_pyexadis(N, b_magnitude_A=B_CU_A)

    path = tmp_path / "exadis_out.data"
    _exadis_write(utils, N, path)
    from_file = read_paradis(path, b_magnitude_A=B_CU_A)

    assert from_file.n_nodes == in_proc.n_nodes
    assert from_file.n_segments == in_proc.n_segments
    assert from_file.b_magnitude_A == in_proc.b_magnitude_A

    # Order-independent invariants.
    def _sorted_lengths(net):
        return torch.sort(net.segment_lengths_um()).values

    assert torch.allclose(_sorted_lengths(in_proc), _sorted_lengths(from_file),
                          rtol=1e-6, atol=1e-12)
    assert torch.allclose(torch.sort(in_proc.burgers_magnitudes_b()).values,
                          torch.sort(from_file.burgers_magnitudes_b()).values,
                          rtol=1e-6, atol=1e-12)
    assert in_proc.total_line_length_um() == pytest.approx(
        from_file.total_line_length_um(), rel=1e-6)
    assert len(find_loops(in_proc)) == len(find_loops(from_file))


@pytest.mark.unit
def test_our_writer_is_readable_by_exadis(tmp_path):
    """`write_paradis` must produce something ExaDiS itself can read back.

    The round trip inside this package was already tested; this is the half
    that needs ExaDiS present, and it is what makes the file bridge genuinely
    bidirectional rather than read-only.
    """
    from midas_ddd.exadis import from_pyexadis, require_pyexadis
    from midas_ddd.network import write_paradis
    _, utils = require_pyexadis()

    N = utils.generate_prismatic_config("fcc", 2000.0, 4, 60.0, maxseg=40, seed=5)
    ours = from_pyexadis(N, b_magnitude_A=B_CU_A)

    path = tmp_path / "ours.data"
    write_paradis(ours, path)
    back = utils.read_paradis(str(path))          # ExaDiS reading OUR file
    disnet = back.get_disnet() if hasattr(back, "get_disnet") else back
    assert len(disnet.get_nodes_data()["positions"]) == ours.n_nodes
    assert len(disnet.get_segs_data()["nodeids"]) == ours.n_segments


@pytest.mark.unit
def test_cell_bounds_survive_the_conversion():
    """`_cell_bounds_b` probes an API that has moved between ExaDiS revisions.

    If the probe fails it falls back to the node bounding box with PBC off,
    which is safe but silently loses periodicity. Catch that here rather than
    in a minimum-image bug three modules downstream.
    """
    from midas_ddd.exadis import generate_prismatic_config
    box_b = 2000.0
    net = generate_prismatic_config(
        crystal="fcc", box_size_b=box_b, num_loops=3, radius_b=50.0,
        b_magnitude_A=B_CU_A, maxseg_b=40, seed=11)
    expected_um = box_b * B_CU_A * 1e-4
    got_um = float(net.cell_size_um.max())
    assert got_um == pytest.approx(expected_um, rel=1e-6), (
        f"cell probe returned {got_um} um, expected {expected_um} um -- the "
        "pyexadis Cell API has probably moved and _cell_bounds_b fell back to "
        "the node bounding box")


@pytest.mark.unit
def test_junction_burgers_magnitudes_are_not_normalised():
    """Same invariant the file reader has: |b| != 1 for reaction junctions, and
    that magnitude is what balances conservation at the junction node."""
    from midas_ddd.exadis import generate_prismatic_config
    net = generate_prismatic_config(
        crystal="fcc", box_size_b=2000.0, num_loops=8, radius_b=60.0,
        b_magnitude_A=B_CU_A, maxseg_b=30, seed=3)
    mags = net.burgers_magnitudes_b()
    assert float(mags.min()) > 0
    assert check_burgers_conservation(net, tol=1e-6).ok


@pytest.mark.unit
def test_generated_loops_feed_the_scattering_kernel():
    """End to end: ExaDiS generator -> relaxation volumes -> q.u~, no file."""
    from midas_ddd import isotropic_stiffness, q_dot_u_tilde, relaxation_volumes_um3
    from midas_ddd.exadis import generate_prismatic_config
    net = generate_prismatic_config(
        crystal="fcc", box_size_b=2000.0, num_loops=4, radius_b=60.0,
        b_magnitude_A=B_CU_A, maxseg_b=25, seed=42)
    loops = find_loops(net)
    dV = relaxation_volumes_um3(net, loops)
    assert dV.numel() == len(loops)
    assert float(dV.abs().max()) > 0

    q = torch.tensor([[0.0, 0.0, 1e-2]], dtype=torch.float64)
    amp = q_dot_u_tilde(net, q, isotropic_stiffness(100.0, 75.0))
    assert torch.isfinite(amp).all()
    assert float(amp.abs()[0]) > 0


@pytest.mark.unit
def test_validate_network_summary_on_an_exadis_config():
    from midas_ddd.exadis import generate_prismatic_config
    net = generate_prismatic_config(
        crystal="fcc", box_size_b=2000.0, num_loops=5, radius_b=60.0,
        b_magnitude_A=B_CU_A, maxseg_b=30, seed=17)
    s = validate_network(net, q_max_inv_A=0.1)
    assert s["burgers_conserved"] is True
    assert s["n_loops"] >= 1
    assert s["total_relaxation_volume_um3"] != 0.0
