"""Symmetry table + hkl-row permutation tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_process_grains.compute.symmetry import (
    SymmetryTable,
    build_symmetry_table,
    apply_sym_to_hkl_int,
)


def test_build_symmetry_table_fcc_returns_24_ops(cubic_hkl_table):
    tbl = build_symmetry_table(225, cubic_hkl_table, warn_missing=False)
    assert isinstance(tbl, SymmetryTable)
    assert tbl.space_group == 225
    assert tbl.n_sym == 24
    assert tbl.ops_quat.shape == (24, 4)
    assert tbl.ops_R.shape == (24, 3, 3)
    assert tbl.hkl_perm.shape == (24, len(cubic_hkl_table))


def test_first_op_is_identity(cubic_hkl_table):
    tbl = build_symmetry_table(225, cubic_hkl_table, warn_missing=False)
    # The identity op (s=0) should leave every row unchanged.
    np.testing.assert_array_equal(
        tbl.hkl_perm[0].cpu().numpy(),
        np.arange(len(cubic_hkl_table)),
    )


def test_sym_ops_are_proper_rotations(cubic_hkl_table):
    tbl = build_symmetry_table(225, cubic_hkl_table, warn_missing=False)
    R = tbl.ops_R.cpu().numpy()
    for s in range(tbl.n_sym):
        assert np.isclose(np.linalg.det(R[s]), 1.0, atol=1e-9), (
            f"op {s} has det {np.linalg.det(R[s])}, expected +1"
        )
        np.testing.assert_allclose(R[s] @ R[s].T, np.eye(3), atol=1e-10)


def test_apply_sym_to_hkl_int_round_trip():
    """Applying a 90-degree rotation about z to (1,0,0) gives (0,1,0)."""
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    rot = apply_sym_to_hkl_int(R, hkl)
    assert rot.tolist() == [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]


def test_hkl_perm_is_a_permutation_for_complete_orbit(tmp_path):
    """If we hand a *complete* hkl orbit (e.g. {111}, all 8 sign perms), every
    op maps within the orbit and π is a bijection."""
    p = tmp_path / "hkls.csv"
    # Full {111} orbit: 8 reflections (all sign permutations).
    text = "h k l D-spacing RingNr g1 g2 g3 Theta(deg) 2Theta(deg) Radius\n"
    for h in (-1, 1):
        for k in (-1, 1):
            for l in (-1, 1):
                # real g vectors (|g| = 1/d, FCC a = 3.6 A): (h, k, l)·0.27819
                text += (f"{h} {k} {l} 2.0754 1 {h*0.27819} {k*0.27819} {l*0.27819} "
                         f"2.39 4.78 60000.0\n")
    p.write_text(text)

    from midas_process_grains.io.hkls import load_hkl_table
    tbl_hkl = load_hkl_table(p)
    sym_tbl = build_symmetry_table(225, tbl_hkl, warn_missing=False)

    perm = sym_tbl.hkl_perm.cpu().numpy()
    n = perm.shape[1]
    # Within {111}, every op must be a permutation (no -1s, every row hit once).
    for s in range(sym_tbl.n_sym):
        assert (perm[s] >= 0).all(), f"op {s} has -1 entries on a complete orbit"
        assert sorted(perm[s]) == list(range(n)), (
            f"op {s} hkl_perm is not a permutation: {perm[s]}"
        )


def test_hkl_perm_handles_partial_orbit_with_warning(cubic_hkl_table):
    """The fixture has only canonical (h,k,l), not the full orbit.
    Many ops therefore map rows to triples not in the table → -1 entries."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tbl = build_symmetry_table(225, cubic_hkl_table, warn_missing=True)
    perm = tbl.hkl_perm.cpu().numpy()
    # At least one cell should be -1, since the fixture is partial.
    assert (perm == -1).any()
    assert any("symmetry image" in str(warning.message) for warning in w)


@pytest.mark.parametrize("device", ["cpu"])
def test_torch_round_trip_to_device(cubic_hkl_table, device):
    tbl = build_symmetry_table(
        225, cubic_hkl_table, device=device, warn_missing=False,
    )
    assert tbl.ops_quat.device.type == device
    assert tbl.ops_R.device.type == device
    assert tbl.hkl_perm.device.type == device
