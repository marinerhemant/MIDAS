"""midas_defect must use ONE crystal frame for the seed basis and the refined basis.

``rows.refine_lattice`` returns the upper-triangular Cholesky ``B`` of ``G*`` (a* along x:
the MIDAS frame, that of Grains.csv). The full-seed basis in ``refine_to_convergence`` and
``selfcal._B`` used ``Lattice.reciprocal_cartesian_vectors`` (real a along x), which differs
for any cell with gamma != 90 (30 deg about c for hexagonal) or beta != 90, so ``Domain.U``
changed frame at the first refit and selfcal predicted spots with a 30 deg-rotated basis.
For all-90-deg seeds (the tetragonal path) the two coincide.
"""
import math

import numpy as np
import pytest

from midas_hkls import Lattice
from midas_defect.rows import _seed_basis
from midas_defect.selfcal import _B

CELLS = [
    (2.95, 2.95, 4.68, 90, 90, 120),
    (4.0, 5.0, 6.0, 90, 105, 90),
    (4.0, 5.0, 6.0, 80, 95, 100),
    (5.0, 5.0, 5.0, 55, 55, 55),
    (3.0, 3.0, 4.5, 90, 90, 90),
]


def _chol_B(cell):
    """Independent construction: upper-triangular B with B^T B = G* (no midas_hkls)."""
    a, b, c, al, be, ga = cell
    al, be, ga = np.radians([al, be, ga])
    G = np.array([[a * a, a * b * np.cos(ga), a * c * np.cos(be)],
                  [a * b * np.cos(ga), b * b, b * c * np.cos(al)],
                  [a * c * np.cos(be), b * c * np.cos(al), c * c]])
    return np.linalg.cholesky(np.linalg.inv(G)).T


@pytest.mark.parametrize("cell", CELLS)
def test_seed_basis_is_the_cholesky_basis_refine_lattice_uses(cell):
    lat = Lattice(*cell)
    np.testing.assert_allclose(_seed_basis(lat, two_pi=False), _chol_B(cell), atol=1e-12)
    np.testing.assert_allclose(_seed_basis(lat, two_pi=True), 2 * math.pi * _chol_B(cell), atol=1e-12)


@pytest.mark.parametrize("cell", CELLS)
def test_selfcal_basis_is_the_same_frame(cell):
    np.testing.assert_allclose(_B(*cell), 2 * math.pi * _chol_B(cell), atol=1e-12)


def test_the_textbook_basis_it_replaced_was_30_degrees_off_for_hexagonal():
    lat = Lattice(2.95, 2.95, 4.68, 90, 90, 120)
    old = np.asarray(lat.reciprocal_cartesian_vectors(), float).T
    new = _seed_basis(lat, two_pi=False)
    assert np.abs(old - new).max() > 0.1
    np.testing.assert_allclose(np.linalg.norm(old, axis=0), np.linalg.norm(new, axis=0))
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    np.testing.assert_allclose(np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]]) @ old, new, atol=1e-12)
