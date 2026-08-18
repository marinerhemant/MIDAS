"""Tests for midas_hkls.point_group and the Cartesian lattice embedding.

The strongest available test of the general machinery is that it **reproduces a
known-good special case**: the octahedral group built independently as signed
permutation matrices. Both `point_group_rotations('432')` and
`proper_rotations_from_space_group(225, cubic)` must equal that set element for
element, which is a far tighter statement than "has 24 elements".

Then the two regressions that cost real debugging time:

* **matrix keying** -- closing the group on quaternions returns 28 elements for
  432, because ``q`` and ``-q`` are the same rotation and the sign tie-break is
  unstable at ``w = 0``, which is exactly where the 180-degree elements sit;
* **improper -> -R** -- discarding improper operations under-symmetrises the 73
  space groups with improper operations but no inversion centre.

And the hexagonal geometry, where a cubic shortcut is silently wrong: in hcp,
(hkl) is not a Cartesian direction.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from midas_hkls import Lattice
from midas_hkls.point_group import (
    EXPECTED_ORDER,
    LAUE_TO_PROPER_GROUP,
    as_lattice,
    laue_class,
    plane_normal,
    plane_normals,
    point_group_rotations,
    proper_group_symbol,
    proper_rotations_from_space_group,
)
from midas_hkls.space_group import SpaceGroup
from midas_hkls.tables import crystal_system_for

# hcp alpha-Ti and omega-Ti, from the DAC parameter files.
TI_ALPHA = Lattice(2.9505, 2.9505, 4.6826, 90.0, 90.0, 120.0)
CEO2 = Lattice(5.41165, 5.41165, 5.41165, 90.0, 90.0, 90.0)

# One lattice per crystal system for the 230-sweep. A cell of HIGHER shape
# symmetry than the group requires is safe -- conjugation by an invertible M is
# injective, so distinct integer operations stay distinct -- and it sidesteps
# having to know each setting's unique axis. Hexagonal/trigonal genuinely need
# gamma = 120, because a 3-fold conjugated through a 90-degree cell is not
# orthogonal (and the orthogonality gate in the code catches exactly that).
SWEEP_LATTICES = {
    "triclinic": (3.0, 4.0, 5.0, 80.0, 85.0, 95.0),
    "monoclinic": (3.0, 4.0, 5.0, 90.0, 90.0, 90.0),
    "orthorhombic": (3.0, 4.0, 5.0, 90.0, 90.0, 90.0),
    "tetragonal": (3.0, 3.0, 5.0, 90.0, 90.0, 90.0),
    "trigonal": (3.0, 3.0, 5.0, 90.0, 90.0, 120.0),
    "hexagonal": (3.0, 3.0, 5.0, 90.0, 90.0, 120.0),
    "cubic": (4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
}


def _octahedral_reference() -> np.ndarray:
    """The 24 proper cubic rotations, as signed permutation matrices.

    Written independently of `point_group.py` -- no generators, no closure --
    so agreement is evidence rather than a tautology.
    """
    mats = []
    for perm in itertools.permutations(range(3)):
        for sx, sy, sz in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            m[0, perm[0]], m[1, perm[1]], m[2, perm[2]] = sx, sy, sz
            if abs(np.linalg.det(m) - 1.0) < 1e-9:
                mats.append(m)
    assert len(mats) == 24
    return np.array(mats)


def _keys(rots: np.ndarray) -> set:
    return {tuple(np.round(m, 8).ravel() + 0.0) for m in rots}


# ------------------------------------------------- the cubic cross-check
def test_432_equals_octahedral_group_element_for_element():
    assert _keys(point_group_rotations("432")) == _keys(_octahedral_reference())


def test_space_group_225_equals_octahedral_group():
    """A cubic cell's Cartesian frame IS the canonical one, so these must agree.

    This is the join between the two entry points: if it holds, the space-group
    route inherits the point-group route's validation.
    """
    rots = proper_rotations_from_space_group(225, CEO2)
    assert _keys(rots) == _keys(_octahedral_reference())


def test_matrix_keying_regression_432_has_24_not_28():
    """Regression: a quaternion-keyed closure returns 28 elements here."""
    assert len(point_group_rotations("432")) == 24


@pytest.mark.parametrize("symbol,order", sorted(EXPECTED_ORDER.items()))
def test_point_group_orders_and_closure(symbol, order):
    rots = point_group_rotations(symbol)
    assert len(rots) == order
    # every element is a proper rotation
    for m in rots:
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(m) == pytest.approx(1.0)
    # closed under composition
    keys = _keys(rots)
    for a in rots:
        for b in rots:
            assert tuple(np.round(a @ b, 8).ravel() + 0.0) in keys


def test_unknown_point_group_raises():
    with pytest.raises(KeyError):
        point_group_rotations("mmm")          # a Laue class, not a proper group


# ------------------------------------------------------- all 230 space groups
def test_all_230_space_groups_give_the_right_laue_proper_order():
    failures = []
    for n in range(1, 231):
        lat = SWEEP_LATTICES[crystal_system_for(n)]
        want = EXPECTED_ORDER[LAUE_TO_PROPER_GROUP[laue_class(n)]]
        try:
            rots = proper_rotations_from_space_group(n, lat)
        except Exception as exc:                        # pragma: no cover
            failures.append((n, type(exc).__name__, str(exc)[:80]))
            continue
        if len(rots) != want:
            failures.append((n, len(rots), want))
    assert not failures, f"{len(failures)} space groups wrong: {failures[:10]}"


def test_all_230_elements_are_proper_rotations():
    for n in range(1, 231):
        lat = SWEEP_LATTICES[crystal_system_for(n)]
        for m in proper_rotations_from_space_group(n, lat):
            assert np.allclose(m @ m.T, np.eye(3), atol=1e-8), f"SG {n}"
            assert np.linalg.det(m) > 0, f"SG {n}"


def test_centrosymmetry_split_is_92_65_73():
    """The 73 in the module docstring, counted rather than asserted.

    92 centrosymmetric (-R already present, dropping improper ops is harmless),
    65 Sohncke (no improper ops to drop), 73 neither -- and it is those 73 that
    break if improper operations are discarded instead of mapped to -R.
    """
    centro = sohncke = mixed = 0
    for n in range(1, 231):
        sg = SpaceGroup.from_number(n)
        has_improper = any(op.determinant() == -1 for op in sg.symmetry_operations())
        if sg.is_centrosymmetric():
            centro += 1
        elif has_improper:
            mixed += 1
        else:
            sohncke += 1
    assert (centro, sohncke, mixed) == (92, 65, 73)


@pytest.mark.parametrize("number,order", [(1, 1), (6, 2), (25, 4), (186, 12)])
def test_non_centrosymmetric_groups_are_not_under_symmetrised(number, order):
    """Pm -> 2 (not 1), Pmm2 -> 4 (not 2), P6_3mc -> 12 (not 6)."""
    lat = SWEEP_LATTICES[crystal_system_for(number)]
    sg = SpaceGroup.from_number(number)
    assert not sg.is_centrosymmetric()
    assert len(proper_rotations_from_space_group(number, lat)) == order


def test_inconsistent_lattice_is_an_error_not_a_wrong_answer():
    """A hexagonal group with a 90-degree cell must fail loudly."""
    with pytest.raises(AssertionError, match="not orthogonal"):
        proper_rotations_from_space_group(194, (3.0, 3.0, 5.0, 90.0, 90.0, 90.0))


def test_proper_group_symbol_round_trips():
    assert proper_group_symbol(225) == "432"
    assert proper_group_symbol(194) == "622"
    assert proper_group_symbol(1) == "1"
    assert laue_class(194) == "6/mmm"


# --------------------------------------------------------- lattice embedding
def test_cartesian_vectors_reproduce_the_metric_tensor():
    """The embedding must be the same geometry the metric tensor describes."""
    for lat in (TI_ALPHA, CEO2, Lattice(3.0, 4.0, 5.0, 80.0, 85.0, 95.0)):
        d = lat.cartesian_vectors()
        assert np.allclose(d @ d.T, lat.metric_tensor(), atol=1e-10)
        assert d[0, 1] == pytest.approx(0.0)      # a1 along x
        assert d[0, 2] == pytest.approx(0.0)
        assert d[1, 2] == pytest.approx(0.0)      # a2 in the xy plane
        assert d[1, 1] > 0 and d[2, 2] > 0


def test_cartesian_volume_matches_metric_volume():
    for lat in (TI_ALPHA, CEO2, Lattice(3.0, 4.0, 5.0, 80.0, 85.0, 95.0)):
        d = lat.cartesian_vectors()
        assert float(np.dot(d[0], np.cross(d[1], d[2]))) == pytest.approx(
            lat.volume(), rel=1e-12)


def test_degenerate_cell_raises():
    with pytest.raises(ValueError, match="degenerate cell"):
        Lattice(3.0, 4.0, 5.0, 10.0, 10.0, 170.0).cartesian_vectors()


def test_plane_normal_agrees_with_d_spacing():
    """|h . b| == 1/d, independently of the Cartesian embedding chosen."""
    for lat in (TI_ALPHA, CEO2, Lattice(3.0, 4.0, 5.0, 80.0, 85.0, 95.0)):
        B = lat.reciprocal_cartesian_vectors()
        for hkl in ((1, 0, 0), (1, 0, 1), (1, 1, 2), (0, 0, 2)):
            g = np.asarray(hkl, dtype=float) @ B
            assert 1.0 / np.linalg.norm(g) == pytest.approx(
                lat.d_spacing(*hkl), rel=1e-12)


# ------------------------------------------------------- hexagonal geometry
def test_hexagonal_plane_normals_need_the_lattice():
    """(100) and (001) are 90 deg apart in hcp; the cubic shortcut says 90 too,
    but (101) is where the shortcut breaks -- so test that one."""
    n_cubic_shortcut = plane_normal((1, 0, 1))            # no lattice: wrong here
    n_correct = plane_normal((1, 0, 1), TI_ALPHA)
    assert not np.allclose(n_cubic_shortcut, n_correct, atol=1e-3)
    c = plane_normal((0, 0, 1), TI_ALPHA)
    ang = np.degrees(np.arccos(abs(float(n_correct @ c))))
    # arctan(c/a * ... ) for hcp (101) vs (001); value fixed by the measured cell
    assert ang == pytest.approx(61.3795, abs=1e-3)


def test_basal_and_prism_are_perpendicular_in_hcp():
    a = plane_normal((1, 0, 0), TI_ALPHA)
    c = plane_normal((0, 0, 1), TI_ALPHA)
    assert float(a @ c) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("hkl,mult", [((1, 0, 0), 6), ((0, 0, 2), 2),
                                      ((1, 0, 1), 12), ((1, 1, 0), 6)])
def test_hexagonal_family_multiplicities(hkl, mult):
    """Signed multiplicities of {hkl} normals for 622 + hcp metric."""
    group = proper_rotations_from_space_group("P6_3/mmc", TI_ALPHA)
    assert len(group) == 12
    assert len(plane_normals(hkl, group, TI_ALPHA)) == mult


@pytest.mark.parametrize("hkl,mult", [((1, 1, 1), 8), ((1, 0, 0), 6),
                                      ((1, 1, 0), 12), ((2, 1, 0), 24)])
def test_cubic_family_multiplicities(hkl, mult):
    group = point_group_rotations("432")
    assert len(plane_normals(hkl, group, None)) == mult


def test_plane_normals_are_unit_and_sign_closed():
    group = proper_rotations_from_space_group("P6_3/mmc", TI_ALPHA)
    N = plane_normals((1, 0, 1), group, TI_ALPHA)
    assert np.allclose(np.linalg.norm(N, axis=1), 1.0)
    keys = {tuple(np.round(v, 6) + 0.0) for v in N}
    for v in N:                                   # Friedel: -h is present too
        assert tuple(np.round(-v, 6) + 0.0) in keys


def test_zero_hkl_raises():
    with pytest.raises(ValueError, match=r"\(000\)"):
        plane_normal((0, 0, 0), TI_ALPHA)


# ------------------------------------------------------------- input forms
def test_as_lattice_accepts_tuple_and_lattice():
    assert as_lattice(TI_ALPHA) is TI_ALPHA
    lat = as_lattice((2.9505, 2.9505, 4.6826, 90.0, 90.0, 120.0))
    assert lat.a == pytest.approx(2.9505) and lat.gamma == pytest.approx(120.0)
    with pytest.raises(ValueError, match="6 values|a, b, c"):
        as_lattice((1.0, 2.0, 3.0))


def test_space_group_accepts_number_symbol_and_object():
    ref = _keys(proper_rotations_from_space_group(225, CEO2))
    assert _keys(proper_rotations_from_space_group("Fm-3m", CEO2)) == ref
    assert _keys(proper_rotations_from_space_group(
        SpaceGroup.from_number(225), CEO2)) == ref
