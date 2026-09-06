"""Crystal system from the lattice's own symmetry group.

The assertions are integers — the order of the holohedry — which is what makes
this better than comparing lengths against tolerances.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_hkls.lattice_symmetry import (holohedry, lattice_symmetry_operations,
                                  tolerance_from_fit, holohedry_from_fit,
                                  ORDER_TO_SYSTEM)

CASES = [
    ("cubic", (4.0, 4.0, 4.0, 90., 90., 90.), 48),
    ("hexagonal", (4.0, 4.0, 6.0, 90., 90., 120.), 24),
    ("tetragonal", (3.6116, 3.6116, 19.2516, 90., 90., 90.), 16),
    ("rhombohedral", (5.0, 5.0, 5.0, 70., 70., 70.), 12),
    ("orthorhombic", (5.1, 6.3, 7.7, 90., 90., 90.), 8),
    ("monoclinic", (5.1, 6.3, 7.7, 90., 103., 90.), 4),
    ("triclinic", (5.1, 6.3, 7.7, 82., 95., 103.), 2),
]


@pytest.mark.parametrize("name,cell,order", CASES, ids=[c[0] for c in CASES])
def test_every_crystal_system_gives_its_holohedry_order(name, cell, order):
    h = holohedry(cell)
    assert h.order == order, f"{name}: order {h.order}, expected {order}"
    assert h.system == name


def test_the_PRIMITIVE_cell_of_a_body_centred_lattice_is_still_tetragonal():
    """The case that matters for ab initio: it returns the primitive cell."""
    h = holohedry((3.6116, 3.6116, 9.9588, 79.5507, 79.5507, 90.0))
    assert h.order == 16 and h.system == "tetragonal"
    assert h.n_fold_axes.get(4, 0) >= 2          # the 4-fold axis is there


def test_operations_are_a_group_containing_the_identity_and_inversion():
    ops = lattice_symmetry_operations((5.1, 6.3, 7.7, 90., 90., 90.))
    assert any(np.array_equal(M, np.eye(3, dtype=np.int64)) for M in ops)
    assert any(np.array_equal(M, -np.eye(3, dtype=np.int64)) for M in ops)
    for M in ops:                                 # closed under inverse
        inv = np.round(np.linalg.inv(M)).astype(np.int64)
        assert any(np.array_equal(inv, N) for N in ops)


def test_every_operation_really_preserves_the_metric():
    from midas_hkls import Lattice
    cell = (3.6116, 3.6116, 19.2516, 90., 90., 90.)
    lat = Lattice(a=cell[0], b=cell[1], c=cell[2], alpha=cell[3],
                  beta=cell[4], gamma=cell[5])
    A = np.asarray(lat.cartesian_vectors(), float).T
    G = A.T @ A
    for M in lattice_symmetry_operations(cell):
        assert np.abs(M.T @ G @ M - G).max() < 1e-3 * np.abs(G).max()
        assert abs(abs(round(float(np.linalg.det(M)))) - 1) < 1e-12


def test_a_looser_tolerance_can_only_add_operations():
    cell = (3.60, 3.65, 9.91, 79.9, 100.0, 90.8)          # noisy, near-tetragonal
    n = [len(lattice_symmetry_operations(cell, rel_tol=t))
         for t in (1e-3, 1e-2, 4e-2)]
    assert n == sorted(n), f"symmetry count not monotone in tolerance: {n}"


def test_an_unrecognised_order_is_reported_not_rounded():
    """A non-crystallographic count means the tolerance is wrong. Say so."""
    assert set(ORDER_TO_SYSTEM) == {2, 4, 8, 12, 16, 24, 48}
    assert ORDER_TO_SYSTEM.get(7) is None
    assert "unrecognised" in holohedry.__doc__


def test_the_tolerance_comes_from_the_covariance_not_by_hand():
    """Closes the loop: the fit's own sigma decides what symmetry is supportable."""
    from midas_hkls import refine_ub_from_gvectors, Lattice
    cell = (3.6116, 3.6116, 19.2516, 90., 90., 90.)
    lat = Lattice(a=cell[0], b=cell[1], c=cell[2], alpha=90., beta=90., gamma=90.)
    B = np.asarray(lat.reciprocal_cartesian_vectors(), float).T
    rng = np.random.default_rng(4)
    h = np.array([(i, j, k) for i in range(-2, 3) for j in range(-2, 3)
                  for k in range(-4, 5) if (i, j, k) != (0, 0, 0)], float)
    g = h @ B.T + rng.normal(0, 3e-4, (len(h), 3))
    fit = refine_ub_from_gvectors(h, g)
    tol = tolerance_from_fit(fit)
    assert tol > 0
    assert holohedry_from_fit(fit).system == "tetragonal"


def test_the_tolerance_dependence_is_documented():
    import midas_hkls.lattice_symmetry as hh
    assert "do not choose the tolerance by hand" in hh.__doc__.lower()
    assert "rhombohedral" in hh.__doc__          # the accidental-symmetry example


def test_the_tolerance_is_algebraically_the_same_test_as_the_ab_split():
    """Pin the circularity so nobody re-discovers it as a 'confirmation'.

    holohedry accepts the a<->b 4-fold iff |b^2-a^2|/(ab) <= tolerance, and
    tolerance_from_fit returns n_sigma*2*max(sigma/len). Those are one test.
    """
    a, b = 3.6229, 3.6811
    sa, sb = 0.0201, 0.0263
    residual = abs(b * b - a * a) / (a * b)
    tol = 3.0 * 2.0 * max(sa / a, sb / b)
    assert abs(residual - 0.03189) < 1e-4
    assert abs(tol - 0.04291) < 1e-4
    assert residual < tol                      # -> "tetragonal"
    # ... but the same sigma makes the split only 1.8 sigma. One number, twice.
    z = abs(b - a) / math.hypot(sa, sb)
    assert 1.7 < z < 1.9
    import midas_hkls.lattice_symmetry as hh
    assert "CANNOT corroborate" in hh.tolerance_from_fit.__doc__


def test_a_noisier_fit_is_awarded_HIGHER_symmetry():
    """The perverse direction, pinned. Improving data should be able to change
    the answer; here more noise buys more symmetry."""
    from midas_hkls import holohedry
    cell = (3.6229, 3.6811, 9.9441, 100.543, 100.040, 91.220)
    tight = holohedry(cell, rel_tol=0.020).order
    loose = holohedry(cell, rel_tol=0.043).order
    assert loose > tight, (tight, loose)
