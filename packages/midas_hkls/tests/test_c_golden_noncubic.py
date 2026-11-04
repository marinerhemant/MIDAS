"""Non-cubic golden tests against the compiled C GetHKLList (see data/c_golden/README.md).

The B matrix, ``frame_rotation`` and the Python hkl emitter are all checked against real C output
for hexagonal, trigonal, tetragonal, orthorhombic, monoclinic, triclinic and rhombohedral cells.
Before these files existed the only C comparison of g vectors was for cubic Au, where the two
crystal frames coincide, so nothing in the repo pinned the non-cubic MIDAS frame.
"""
import io
from pathlib import Path

import numpy as np
import pytest

from midas_hkls import Lattice, SpaceGroup, emit_nf_hkls_csv, frame_rotation
from midas_hkls.nf_hkls import _b_matrix

D = Path(__file__).parent / "data" / "c_golden"
CELLS = {
    "hexagonal_194": (194, (2.95, 2.95, 4.68, 90, 90, 120)),
    "trigonal_hex_167": (167, (4.76, 4.76, 12.99, 90, 90, 120)),
    "tetragonal_123": (123, (3.0, 3.0, 4.5, 90, 90, 90)),
    "orthorhombic_62": (62, (3.0, 4.0, 5.0, 90, 90, 90)),
    "monoclinic_14_b105": (14, (4.0, 5.0, 6.0, 90, 105, 90)),
    "monoclinic_14_b95": (14, (4.0, 5.0, 6.0, 90, 95, 90)),
    "triclinic_2": (2, (4.0, 5.0, 6.0, 80, 95, 100)),
    "rhombohedral_167r": (167, (5.0, 5.0, 5.0, 55, 55, 55)),
}
ORTHOGONAL = {"tetragonal_123", "orthorhombic_62"}
WL, LSD, MAXR = 0.222910, 1.0e6, 300000.0


def _load(name):
    a = np.loadtxt(D / f"{name}_hkls.csv", skiprows=1)
    return a        # h k l d ring g1 g2 g3 theta 2theta radius


@pytest.fixture(params=list(CELLS))
def case(request):
    sg, cell = CELLS[request.param]
    return request.param, sg, cell, _load(request.param)


def test_b_matrix_reproduces_the_c_g_vectors(case):
    name, sg, cell, a = case
    hkl, g = a[:, :3], a[:, 5:8]
    B = _b_matrix(Lattice(*cell))
    np.testing.assert_allclose((B @ hkl.T).T, g, atol=1e-12)
    np.testing.assert_allclose(1.0 / np.linalg.norm(g, axis=1), a[:, 3], atol=1e-12)


def test_textbook_frame_does_not_reproduce_c_but_frame_rotation_maps_it(case):
    name, sg, cell, a = case
    hkl, g = a[:, :3], a[:, 5:8]
    lat = Lattice(*cell)
    g_text = (np.asarray(lat.reciprocal_cartesian_vectors(), float).T @ hkl.T).T
    off = np.abs(g_text - g).max()
    if name in ORTHOGONAL:
        assert off < 1e-12                       # all angles 90: the frames coincide
    else:
        assert off > 0.05                        # a real frame difference, not float noise
    np.testing.assert_allclose((frame_rotation(lat) @ g_text.T).T, g, atol=1e-12)


def test_python_emitter_matches_the_c_rows_inside_the_c_hkl_box(case):
    name, sg, cell, a = case
    buf = io.StringIO()
    rows = np.array(emit_nf_hkls_csv(
        SpaceGroup.from_number(sg), Lattice(*cell), wavelength_A=WL, lsd_um=LSD,
        max_ring_rad_um=MAXR, fp=buf), float)
    dmin = a[:, 3].min() - 1e-9
    py = rows[rows[:, 3] >= dmin]
    inbox = (np.abs(py[:, :3]) <= 10).all(axis=1)       # the C code only walks 0..10 in h, k, l
    py_in, py_out = py[inbox], py[~inbox]
    key = lambda M: M[np.lexsort((M[:, 2], M[:, 1], M[:, 0]))]
    A, Q = key(a), key(py_in)
    assert A.shape == Q.shape, f"{name}: row sets differ ({A.shape[0]} C vs {Q.shape[0]} Python)"
    np.testing.assert_array_equal(A[:, :3], Q[:, :3])
    cols = [3, 5, 6, 7, 8, 9, 10]                        # d, g1..g3, theta, 2theta, radius
    np.testing.assert_allclose(Q[:, cols], A[:, cols], rtol=1e-9, atol=1e-9)
    if name != "trigonal_hex_167":                       # ring numbers shift once the C box drops a ring
        np.testing.assert_array_equal(Q[:, 4], A[:, 4])
    # every Python-only row is outside the C box (the C limitation, not a Python extra)
    if name == "trigonal_hex_167":
        assert {tuple(int(x) for x in r[:3]) for r in py_out} >= {(0, 0, 12), (0, 0, -12)}
