"""The hkl-row permutation must be right for non-cubic cells.

``SymmetryTable.hkl_perm[s, k]`` is the table row of the image of reflection k
under symmetry operator s. A Cartesian rotation acts on the Cartesian g vectors
directly (``g' = R g``); it acts on the integer Miller indices only as
``B^-1 R B``. The old code rotated the integer indices by ``R``, which is exact
for a cubic cell and wrong for a non-orthogonal one (SG 194: 3052 of 4104
images wrong). These tests use closed hkl sets (all hkl inside a g sphere, so
every image exists) and check the defining property ``g[perm[s, k]] = R_s g[k]``,
which does not depend on how the integers are handled.
"""
import itertools
import warnings

import numpy as np
import pytest

pytest.importorskip("midas_hkls")
from midas_hkls.lattice import Lattice                    # noqa: E402
from midas_hkls.nf_hkls import _b_matrix                  # noqa: E402

from midas_process_grains.compute.symmetry import (       # noqa: E402
    apply_sym_to_hkl_int, build_symmetry_table, hkl_row_images, rotated_g_rows,
)
from midas_process_grains.io.hkls import HklTable         # noqa: E402

# (name, space group, cell)
CELLS = [
    ("cubic", 225, (3.6, 3.6, 3.6, 90, 90, 90)),
    ("tetragonal", 123, (3.0, 3.0, 4.5, 90, 90, 90)),
    ("orthorhombic", 62, (3.0, 4.0, 5.0, 90, 90, 90)),
    ("hexagonal", 194, (2.95, 2.95, 4.68, 90, 90, 120)),
    ("trigonal-hex", 167, (4.76, 4.76, 12.99, 90, 90, 120)),
    ("monoclinic", 14, (4.0, 5.0, 6.0, 90, 105, 90)),
]


def _table(cell, gmax=1.6):
    """All hkl with |g| < gmax (a closed set), g = B (h, k, l) in the MIDAS frame."""
    B = _b_matrix(Lattice(*cell))
    nmax = [int(gmax * x) + 1 for x in cell[:3]]        # closes the g sphere for long axes
    hkl = np.array([h for h in itertools.product(*[range(-n, n + 1) for n in nmax])
                    if h != (0, 0, 0)])
    g = (B @ hkl.T).T
    keep = np.linalg.norm(g, axis=1) < gmax
    hkl, g = hkl[keep], g[keep]
    mag = np.round(np.linalg.norm(g, axis=1), 6)
    ring = np.unique(mag, return_inverse=True)[1] + 1
    real = np.zeros((len(hkl), 7))
    real[:, :3] = g
    real[:, 3] = ring
    integers = np.column_stack([hkl, ring]).astype(np.int32)
    h2r = {tuple(int(x) for x in r[:3]): i for i, r in enumerate(integers)}
    return HklTable(real=real, integers=integers, hkl_to_row=h2r), B, hkl


@pytest.mark.parametrize("name,sg,cell", CELLS, ids=[c[0] for c in CELLS])
def test_perm_maps_g_vectors_by_the_operator(name, sg, cell):
    tbl, B, hkl = _table(cell)
    st = build_symmetry_table(sg, tbl, warn_missing=False)
    perm = st.hkl_perm.cpu().numpy()
    R = st.ops_R.cpu().numpy()
    g = tbl.real[:, :3]
    assert (perm >= 0).all(), f"{name}: closed set must have every image"
    for s in range(st.n_sym):
        assert sorted(perm[s]) == list(range(len(g))), f"{name} op {s} not a bijection"
        np.testing.assert_allclose(g[perm[s]], g @ R[s].T, atol=1e-6)


@pytest.mark.parametrize("name,sg,cell", CELLS, ids=[c[0] for c in CELLS])
def test_perm_is_a_group_homomorphism(name, sg, cell):
    """perm[u] = perm[s] o perm[t] whenever R_u = R_s R_t."""
    tbl, _, _ = _table(cell)
    st = build_symmetry_table(sg, tbl, warn_missing=False)
    perm = st.hkl_perm.cpu().numpy()
    R = st.ops_R.cpu().numpy()
    n = st.n_sym
    for s in range(n):
        for t in range(n):
            prod = R[s] @ R[t]
            u = [i for i in range(n) if np.allclose(R[i], prod, atol=1e-6)]
            assert len(u) == 1, f"{name}: operators are not closed"
            np.testing.assert_array_equal(perm[u[0]], perm[s][perm[t]])


def test_integer_index_rotation_is_not_valid_for_hexagonal():
    """Documents why the integer path was replaced."""
    tbl, B, hkl = _table(CELLS[3][2])
    st = build_symmetry_table(194, tbl, warn_missing=False)
    R = st.ops_R.cpu().numpy()
    wrong = 0
    for s in range(st.n_sym):
        old = apply_sym_to_hkl_int(R[s], hkl)
        true = np.rint((np.linalg.inv(B) @ R[s] @ B @ hkl.T).T).astype(int)
        wrong += int((old != true).any(axis=1).sum())
    assert wrong > 0.3 * st.n_sym * len(hkl)
    # the new permutation agrees with the exact integer image B^-1 R B hkl
    for s in range(st.n_sym):
        true = np.rint((np.linalg.inv(B) @ R[s] @ B @ hkl.T).T).astype(int)
        rows = [tbl.hkl_to_row[tuple(int(x) for x in t)] for t in true]
        np.testing.assert_array_equal(st.hkl_perm[s].cpu().numpy(), rows)


def test_extend_symmetry_table_with_twin_uses_g_vectors():
    from midas_process_grains.compute.twins import (
        default_hcp_twin_relations, extend_symmetry_table_with_twin)
    tbl, _, _ = _table(CELLS[3][2])
    st = build_symmetry_table(194, tbl, warn_missing=False)
    tw = default_hcp_twin_relations(1.587, systems=("tension_1012",))[0]
    ext = extend_symmetry_table_with_twin(st, tw, tbl.real, tbl.integers, tbl.hkl_to_row)
    g = tbl.real[:, :3]
    R = ext.ops_R.cpu().numpy()
    perm = ext.hkl_perm.cpu().numpy()
    for s in range(ext.n_sym):
        ok = perm[s] >= 0
        assert ok.any()
        np.testing.assert_allclose(g[perm[s][ok]], (g @ R[s].T)[ok], atol=1e-5)


def test_missing_g_vectors_fall_back_with_a_warning():
    tbl, _, _ = _table(CELLS[0][2])
    stub = HklTable(real=np.zeros_like(tbl.real), integers=tbl.integers,
                    hkl_to_row=tbl.hkl_to_row)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = hkl_row_images(np.eye(3), stub.real, stub.integers, stub.hkl_to_row)
    assert any("no Cartesian g vectors" in str(x.message) for x in w)
    np.testing.assert_array_equal(out, np.arange(len(out)))


def test_rotated_g_rows_marks_absent_images_minus_one():
    g = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    rot90 = np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    np.testing.assert_array_equal(rotated_g_rows(rot90, g), [1, -1])
