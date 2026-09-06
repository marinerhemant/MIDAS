"""Conventional-setting search.

Every case here is one this session got wrong at least once, so the tests are
regression pins as much as specification.
"""
from __future__ import annotations
import math
import numpy as np
import pytest

from midas_hkls import (to_conventional, to_conventional_from_fit,
                        metric_symmetry, refine_ub_from_gvectors, Lattice)
from midas_hkls.conventional import _is_standard_setting, ConventionalCell

# primitive cells of centred lattices, and what they should resolve to
CENTRED = [
    # name,                primitive cell,                                    system,       index
    ("I-tetragonal", (3.6116, 3.6116, 9.9588, 79.5507, 79.5507, 90.0), "tetragonal", 2),
    ("F-cubic",      (2.8284, 2.8284, 2.8284, 60.0, 60.0, 60.0),       "cubic",      4),
    ("I-cubic",      (2.4749, 2.4749, 2.4749, 109.4712, 109.4712, 109.4712), "cubic", 2),
]


@pytest.mark.parametrize("name,cell,system,index", CENTRED, ids=[c[0] for c in CENTRED])
def test_centred_lattices_recover_their_conventional_setting(name, cell, system, index):
    r = to_conventional(cell)
    assert r.system == system, f"{name}: got {r.system}"
    assert r.centring_index == index, f"{name}: index {r.centring_index}"
    assert r.is_standard
    assert abs(r.volume_ratio - index) < 0.05


def _bct_primitive(a: float, c: float):
    """Exact primitive cell of a body-centred tetragonal lattice.

    Built from the primitive vectors (a,0,0), (0,a,0), (a/2,a/2,c/2) rather than
    quoted to 4 dp: the rounded angle 79.5507 that circulated in our notes is
    0.0022 deg off the true 79.552941, and that error propagates straight into
    the recovered conventional angles.
    """
    A = np.array([[a, 0., 0.], [0., a, 0.], [a/2, a/2, c/2]], float)
    L = np.linalg.norm(A, axis=1)

    def ang(u, v):
        return math.degrees(math.acos(
            float(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))))
    return (L[0], L[1], L[2], ang(A[1], A[2]), ang(A[0], A[2]), ang(A[0], A[1]))


def test_the_I_tetragonal_conventional_cell_has_the_right_axes():
    r = to_conventional(_bct_primitive(3.6116, 19.2516))
    a, b, c = r.cell[:3]
    assert abs(a - 3.6116) < 1e-6 and abs(b - 3.6116) < 1e-6
    assert abs(c - 19.2516) < 1e-5, f"c = {c}"
    assert all(abs(v - 90.) < 1e-6 for v in r.cell[3:]), r.cell[3:]
    assert r.centring_index == 2


def test_a_P_tetragonal_cell_with_large_c_over_a_stays_tetragonal():
    """Regression: a global max|G| tolerance made c/a ~ 5.3 read as triclinic,
    because 4 % of c^2 exceeds a^2 itself. Per-entry scaling fixed it."""
    r = to_conventional((3.6116, 3.6116, 19.2516, 90., 90., 90.))
    assert r.system == "tetragonal"
    assert r.centring_index == 1


def test_a_genuinely_triclinic_lattice_returns_its_niggli_cell_unchanged():
    """Regression: every candidate trivially satisfies the empty triclinic
    shape test, so a tiebreak used to wander off to a skewed x4 supercell."""
    r = to_conventional((5.1, 6.3, 7.7, 82., 95., 103.))
    assert r.system == "triclinic"
    assert r.centring_index == 1
    assert abs(r.volume_ratio - 1.0) < 1e-9
    assert sorted(round(v, 4) for v in r.cell[:3]) == [5.1, 6.3, 7.7]


def test_conventional_axes_run_ascending_where_the_system_allows():
    """Regression: the tiebreak once preferred the LONG axis first (19.23 /
    3.68 / 3.62)."""
    r = to_conventional((5.1, 6.3, 7.7, 90., 90., 90.))
    assert r.cell[0] <= r.cell[1] <= r.cell[2] + 1e-9


@pytest.mark.parametrize("cell,expected", [
    ((4., 4., 4., 90., 90., 90.), "cubic"),
    ((4., 4., 7., 90., 90., 90.), "tetragonal"),
    ((4., 5., 7., 90., 90., 90.), "orthorhombic"),
    ((4., 4., 7., 90., 90., 120.), "hexagonal"),
    ((4., 5., 7., 90., 103., 90.), "monoclinic"),
    ((5.1, 6.3, 7.7, 82., 95., 103.), "triclinic"),
    ((5., 5., 5., 70., 70., 70.), "rhombohedral"),
])
def test_metric_symmetry_delegates_and_still_labels_every_system(cell, expected):
    assert metric_symmetry(cell) == expected


def test_metric_symmetry_is_a_thin_wrapper_not_a_second_implementation():
    import midas_hkls.conventional as cv
    doc = " ".join(cv.metric_symmetry.__doc__.split())
    assert "thin wrapper" in doc
    assert "only one crystal system determination" in doc


def test_standard_setting_asks_about_shape_not_identity():
    """The predicate takes the system as INPUT; it must never re-label."""
    assert _is_standard_setting((4., 4., 7., 90., 90., 90.), "tetragonal",
                                rel_len=0.02, abs_ang=1.5)
    # same cell, told it is orthorhombic: still a standard orthorhombic shape
    assert _is_standard_setting((4., 4., 7., 90., 90., 90.), "orthorhombic",
                                rel_len=0.02, abs_ang=1.5)
    # a primitive I-tetragonal cell is NOT a standard tetragonal setting
    assert not _is_standard_setting((3.6116, 3.6116, 9.9588, 79.55, 79.55, 90.),
                                    "tetragonal", rel_len=0.02, abs_ang=1.5)


def test_an_unreachable_setting_is_flagged_not_faked():
    r = to_conventional((3.6116, 3.6116, 9.9588, 79.5507, 79.5507, 90.0),
                        max_index=1)          # cannot reach the I cell
    assert r.is_standard is False
    assert "NOT a standard setting" in str(r)


def test_from_fit_uses_the_covariance_and_can_differ_from_the_default_window():
    """The documented 2604 case: 0.0400 gives orthorhombic, 0.0429 tetragonal."""
    cell = (3.6229, 3.6811, 9.9441, 100.543, 100.040, 91.220)
    default = to_conventional(cell)
    tight = to_conventional(cell, rel_tol=0.0400)
    loose = to_conventional(cell, rel_tol=0.0429)
    assert tight.system == "orthorhombic"
    assert loose.system == "tetragonal"
    assert default.system == tight.system      # default window is 0.0400
    import midas_hkls.conventional as cv
    doc = " ".join(cv.to_conventional_from_fit.__doc__.split())
    assert "0.0400" in doc and "0.0429" in doc


def test_to_conventional_from_fit_runs_on_a_real_refinement():
    lat = Lattice(a=3.6116, b=3.6116, c=19.2516, alpha=90., beta=90., gamma=90.)
    B = np.asarray(lat.reciprocal_cartesian_vectors(), float).T
    rng = np.random.default_rng(3)
    h = np.array([(i, j, k) for i in range(-2, 3) for j in range(-2, 3)
                  for k in range(-4, 5) if (i, j, k) != (0, 0, 0)], float)
    g = h @ B.T + rng.normal(0, 2e-4, (len(h), 3))
    fit = refine_ub_from_gvectors(h, g)
    r = to_conventional_from_fit(fit)
    assert isinstance(r, ConventionalCell)
    assert r.system in ("tetragonal", "orthorhombic")


def test_centring_hint_reads_as_a_lattice_type():
    r = to_conventional((2.8284, 2.8284, 2.8284, 60., 60., 60.))
    assert "F" in r.centring_hint
