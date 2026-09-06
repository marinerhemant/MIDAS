"""Niggli reduction — the canonical form of a lattice.

The property that matters is INVARIANCE: any two bases of one lattice must
reduce to the same cell. That is tested against random unimodular
transformations, which is a test the implementation cannot pass by accident.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_hkls import Lattice
from midas_hkls.niggli import niggli_reduce, same_lattice
from midas_hkls.ub_refine import cell_from_metric

ORTHO = (5.1, 6.3, 7.7, 90.0, 90.0, 90.0)
TRICLINIC = (5.1, 6.3, 7.7, 82.0, 95.0, 103.0)
TETRA = (3.6116, 3.6116, 19.2516, 90.0, 90.0, 90.0)
CUBIC = (4.0, 4.0, 4.0, 90.0, 90.0, 90.0)


def _A(cell):
    lat = Lattice(a=cell[0], b=cell[1], c=cell[2],
                  alpha=cell[3], beta=cell[4], gamma=cell[5])
    return np.asarray(lat.cartesian_vectors(), float).T


def _unimodular(rng, n_ops=6):
    M = np.eye(3, dtype=np.int64)
    for _ in range(n_ops):
        i, j = rng.choice(3, 2, replace=False)
        E = np.eye(3, dtype=np.int64)
        E[i, j] = rng.integers(-2, 3)
        M = M @ E
    return M


@pytest.mark.parametrize("cell", [ORTHO, TRICLINIC, TETRA, CUBIC],
                         ids=["ortho", "triclinic", "tetragonal", "cubic"])
def test_INVARIANT_under_any_basis_of_the_same_lattice(cell):
    """THE test. 40 random unimodular re-bases must all reduce identically."""
    rng = np.random.default_rng(7)
    ref = niggli_reduce(cell)
    assert ref.converged
    n = 0
    for _ in range(60):
        M = _unimodular(rng)
        if abs(round(float(np.linalg.det(M)))) != 1:
            continue
        Ai = _A(cell) @ M
        r = niggli_reduce(cell_from_metric(Ai.T @ Ai))
        assert r.converged, f"did not converge after {r.n_iterations} iterations"
        assert np.allclose(r.cell[:3], ref.cell[:3], rtol=1e-6, atol=1e-6), \
            f"lengths {r.cell[:3]} != {ref.cell[:3]}"
        assert np.allclose(r.cell[3:], ref.cell[3:], atol=1e-4)
        n += 1
    assert n >= 30, f"only {n} valid unimodular draws"


def test_an_already_reduced_cell_is_a_fixed_point():
    r = niggli_reduce(ORTHO)
    again = niggli_reduce(r.cell)
    assert np.allclose(again.cell, r.cell, atol=1e-9)


def test_the_transformation_reproduces_the_reduced_cell():
    r = niggli_reduce(TRICLINIC)
    A_red = _A(TRICLINIC) @ r.transformation
    assert np.allclose(cell_from_metric(A_red.T @ A_red), r.cell, atol=1e-8)
    assert abs(round(float(np.linalg.det(r.transformation)))) == 1   # same lattice
    assert np.linalg.det(A_red) > 0                                  # right-handed


def test_reduction_preserves_the_volume():
    for cell in (ORTHO, TRICLINIC, TETRA):
        v0 = Lattice(a=cell[0], b=cell[1], c=cell[2], alpha=cell[3],
                     beta=cell[4], gamma=cell[5]).volume()
        c = niggli_reduce(cell).cell
        v1 = Lattice(a=c[0], b=c[1], c=c[2], alpha=c[3], beta=c[4],
                     gamma=c[5]).volume()
        assert v1 == pytest.approx(v0, rel=1e-9)


def test_a_degenerate_cell_is_refused():
    """Either Lattice's own validation or the volume guard must stop it."""
    with pytest.raises(ValueError):
        niggli_reduce((5.0, 5.0, 5.0, 90.0, 90.0, 180.0))      # caught by Lattice
    with pytest.raises(ValueError):
        niggli_reduce((5.0, 5.0, 5.0, 30.0, 30.0, 90.0))       # metrically flat


def test_same_lattice_sees_through_a_change_of_setting():
    rng = np.random.default_rng(3)
    M = _unimodular(rng)
    while abs(round(float(np.linalg.det(M)))) != 1:
        M = _unimodular(rng)
    Ai = _A(TRICLINIC) @ M
    other = cell_from_metric(Ai.T @ Ai)
    assert not np.allclose(other[:3], TRICLINIC[:3], rtol=1e-3)   # looks different
    assert same_lattice(TRICLINIC, other)                          # but is not


def test_same_lattice_is_INVARIANT_to_the_type_I_II_flip():
    """A near-90 angle lets noise decide acute vs obtuse; comparison must not care.

    Measured on real ab-initio output: the same lattice reduced to 79.9 deg at
    one tolerance and 100.1 deg at another. Comparing angles directly reports
    them as two lattices; comparing |90 - angle| does not.
    """
    acute = (3.618, 3.703, 9.921, 79.91, 79.81, 88.80)
    obtuse = (3.632, 3.734, 9.918, 99.96, 100.48, 91.71)
    assert same_lattice(acute, obtuse, rel_len=0.05, abs_ang=3.0)


def test_same_lattice_still_says_no_to_a_genuinely_different_one():
    assert not same_lattice(ORTHO, TRICLINIC)
    assert not same_lattice(ORTHO, (5.1, 6.3, 9.9, 90.0, 90.0, 90.0))


def test_the_transposed_matrix_trap_is_documented():
    """A transposed step-5 matrix looks exactly like non-convergence."""
    import midas_hkls.niggli as ng
    assert "columns" in ng.__doc__ and "oscillates forever" in ng.__doc__
