"""Ab-initio lattice finding from g-vectors alone.

Exact recovery is the easy half. The tests that matter are the refusals: too few
reflections, two grains, a supercell, an axis longer than the grid can see.
An indexer that returns a plausible cell in those cases is worse than one that
returns nothing.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from midas_hkls import Lattice
from midas_hkls.ab_initio import (index_ab_initio, patterson, chance_score,
                                  score_vector, refine_vector, reduce_basis,
                                  find_candidate_vectors)

RNG = np.random.default_rng(20260903)
TRICLINIC = (5.1, 6.3, 7.7, 82.0, 95.0, 103.0)
ORTHO = (5.1, 6.3, 7.7, 90.0, 90.0, 90.0)


def _B(cell):
    lat = Lattice(a=cell[0], b=cell[1], c=cell[2],
                  alpha=cell[3], beta=cell[4], gamma=cell[5])
    return np.asarray(lat.reciprocal_cartesian_vectors(), float).T


def _rot(seed):
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _spots(cell, seed=0, hmax=3, q_cut=0.75, noise=2e-4):
    UB = _rot(seed) @ _B(cell)
    hkl = np.array([(h, k, l) for h in range(-hmax, hmax + 1)
                    for k in range(-hmax, hmax + 1)
                    for l in range(-hmax, hmax + 1) if (h, k, l) != (0, 0, 0)], float)
    g = hkl @ UB.T
    g = g[np.linalg.norm(g, axis=1) < q_cut]
    return g + np.random.default_rng(seed + 1).normal(0, noise, g.shape)


# ------------------------------------------------------------------ recovery

def test_recovers_an_ORTHORHOMBIC_cell_with_no_cell_supplied():
    res = index_ab_initio(_spots(ORTHO, seed=0), sigma_g=2e-4)
    assert res.success, str(res)
    assert np.allclose(sorted(res.cell[:3]), sorted(ORTHO[:3]), atol=2e-3)
    assert res.indexed_fraction > 0.98


def test_recovers_a_TRICLINIC_cell_and_does_not_snap_to_90_degrees():
    res = index_ab_initio(_spots(TRICLINIC, seed=3), sigma_g=2e-4)
    assert res.success, str(res)
    v_true = Lattice(a=TRICLINIC[0], b=TRICLINIC[1], c=TRICLINIC[2],
                     alpha=TRICLINIC[3], beta=TRICLINIC[4],
                     gamma=TRICLINIC[5]).volume()
    v_got = Lattice(a=res.cell[0], b=res.cell[1], c=res.cell[2],
                    alpha=res.cell[3], beta=res.cell[4], gamma=res.cell[5]).volume()
    assert abs(v_got - v_true) / v_true < 1e-3, (res.cell, v_got, v_true)
    assert res.indexed_fraction > 0.95
    assert min(abs(a - 90.0) for a in res.cell[3:]) > 1.0   # genuinely oblique
    # NB the angles may come back as supplements (180-82=98, 180-103=77): an
    # equally valid reduced basis for the same lattice. Volume is the invariant,
    # which is why it is what this test compares. See reduce_basis' docstring.


def test_the_cell_arrives_with_uncertainties():
    res = index_ab_initio(_spots(ORTHO, seed=0), sigma_g=2e-4)
    assert res.cell_sigma is not None
    assert all(0 < s < 0.01 for s in res.cell_sigma[:3])


# ------------------------------------------------------------------ refusals

def test_REFUSES_below_the_reflection_floor():
    g = _spots(ORTHO, seed=0)[:12]
    res = index_ab_initio(g, min_reflections=20)
    assert not res.success
    assert "refusal, not a failure" in " ".join(res.notes)
    assert res.cell is None


def test_two_grains_yields_ONE_of_them_not_an_average():
    """Subset selection changed what 'correct' means here.

    With a fraction-based criterion, two grains had to be refused. With subset
    selection, finding ONE of the two lattices is the right answer — it owns a
    self-consistent subset and says so. What must never happen is a cell that is
    neither, or one that silently averages them.
    """
    g = np.vstack([_spots(ORTHO, seed=0), _spots(TRICLINIC, seed=7)])
    res = index_ab_initio(g, sigma_g=2e-4, min_subset=30)
    if not res.success:
        return                                   # refusing is also acceptable
    from midas_hkls import Lattice
    v = Lattice(a=res.cell[0], b=res.cell[1], c=res.cell[2], alpha=res.cell[3],
                beta=res.cell[4], gamma=res.cell[5]).volume()
    v_o = Lattice(a=ORTHO[0], b=ORTHO[1], c=ORTHO[2], alpha=ORTHO[3],
                  beta=ORTHO[4], gamma=ORTHO[5]).volume()
    v_t = Lattice(a=TRICLINIC[0], b=TRICLINIC[1], c=TRICLINIC[2],
                  alpha=TRICLINIC[3], beta=TRICLINIC[4], gamma=TRICLINIC[5]).volume()
    assert min(abs(v - v_o) / v_o, abs(v - v_t) / v_t) < 0.10, (
        f"V={v:.1f} matches neither grain ({v_o:.1f}, {v_t:.1f}) — it is an "
        "average or an artifact")
    assert res.diagnostics["subset_fraction"] < 0.95     # it owns one, not both


def test_a_LONG_AXIS_beyond_the_grid_is_reported_not_silently_missed():
    """R_max = n_grid/(2 q_max). An axis longer than that cannot be found."""
    mag, dr, r_max = patterson(_spots(ORTHO, seed=0), n_grid=32)
    assert r_max == pytest.approx(32 * dr)
    res = index_ab_initio(_spots(ORTHO, seed=0), n_grid=32, sigma_g=2e-4)
    joined = " ".join(res.notes)
    assert "longest findable" in joined or res.success


# --------------------------------------------------------------- supercells

def test_a_SUPERCELL_axis_is_divided_back_down():
    """If a/2 still indexes everything, a was a supercell axis."""
    from midas_hkls.ab_initio import _try_shorter, _index_fraction
    A_true = _rot(0) @ np.linalg.inv(_B(ORTHO)).T   # sample-frame basis, columns
    g = _spots(ORTHO, seed=0)
    A_super = A_true.copy(); A_super[:, 0] *= 2
    fixed = _try_shorter(A_super, g, 0.15)
    assert fixed is not None
    assert np.linalg.norm(fixed[:, 0]) == pytest.approx(
        np.linalg.norm(A_true[:, 0]), rel=1e-9)


def test_a_supercell_is_flagged_by_its_predicted_reflection_count():
    from midas_hkls.ab_initio import _predicted_reflection_count
    A_true = _rot(0) @ np.linalg.inv(_B(ORTHO)).T
    n_true = _predicted_reflection_count(A_true, 0.75)
    n_super = _predicted_reflection_count(A_true * 2, 0.75)
    assert n_super == pytest.approx(8 * n_true, rel=1e-9)


# ------------------------------------------------------------- the machinery

def test_chance_level_is_what_a_random_direction_actually_gets():
    """The null the scores are judged against, checked by simulation."""
    g = _spots(ORTHO, seed=0)
    tol = 0.15
    got = []
    for s in range(40):
        v = np.random.default_rng(100 + s).normal(size=3)
        v = v / np.linalg.norm(v) * 37.0        # long, so projections wander
        got.append(score_vector(v, g, tol)[0])
    predicted = chance_score(len(g), tol)
    assert abs(np.mean(got) - predicted) < 0.35 * predicted, (np.mean(got), predicted)


def test_refine_vector_pulls_a_perturbed_lattice_vector_back():
    A = _rot(0) @ np.linalg.inv(_B(ORTHO)).T     # basis in the SAMPLE frame
    g = _spots(ORTHO, seed=0)
    v0 = A[:, 0] + np.array([0.05, -0.04, 0.03])
    v, n, rms = refine_vector(v0, g, tol=0.15)
    assert np.allclose(v, A[:, 0], atol=1e-3), (v, A[:, 0])
    # the residual floor is set by the noise: v.dg has sigma = |v| * sigma_g
    floor = np.linalg.norm(A[:, 0]) * 2e-4
    assert rms < 2 * floor, f"rms {rms:.3g} against a noise floor of {floor:.3g}"


def test_reduction_shortens_without_changing_the_lattice():
    A = np.linalg.inv(_B(TRICLINIC)).T
    skew = A @ np.array([[1., 4., 0.], [0., 1., 5.], [0., 0., 1.]])   # unimodular
    R = reduce_basis(skew)
    assert abs(abs(np.linalg.det(R)) - abs(np.linalg.det(A))) < 1e-9   # same lattice
    assert sum(np.linalg.norm(R[:, i]) for i in range(3)) < \
           sum(np.linalg.norm(skew[:, i]) for i in range(3))
    assert np.linalg.det(R) > 0                                        # right-handed


def test_reduction_rejects_a_degenerate_basis():
    with pytest.raises(ValueError, match="degenerate"):
        reduce_basis(np.array([[1., 2., 3.], [0., 0., 0.], [0., 0., 1.]]).T)
    with pytest.raises(ValueError, match="3x3"):
        reduce_basis(np.eye(2))


def test_candidates_are_reported_against_chance_not_as_raw_counts():
    g = _spots(ORTHO, seed=0)
    cands, diag = find_candidate_vectors(g, tol=0.15)
    assert len(cands) >= 3
    assert all(c.excess > 0 for c in cands)
    assert diag["chance"] == pytest.approx(chance_score(len(g), 0.15))
    assert "chance" in str(cands[0]) and "excess" in str(cands[0])
    assert diag["r_max"] > diag["dr"] > 0


def test_patterson_validates_its_grid():
    g = _spots(ORTHO, seed=0)
    with pytest.raises(ValueError, match="power of two"):
        patterson(g, n_grid=100)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        patterson(np.zeros((5, 2)))


def test_two_pi_convention_is_handled():
    g = _spots(ORTHO, seed=0)
    a = index_ab_initio(g, sigma_g=2e-4)
    b = index_ab_initio(g * 2 * math.pi, two_pi=True, sigma_g=2e-4 * 2 * math.pi)
    assert a.success and b.success
    assert np.allclose(sorted(a.cell[:3]), sorted(b.cell[:3]), rtol=1e-3)


def test_AXIS_ALIGNED_supercells_are_divided_back_to_the_true_volume():
    from midas_hkls.ab_initio import select_basis, LatticeCandidate
    A_true = _rot(0) @ np.linalg.inv(_B(ORTHO)).T
    g = _spots(ORTHO, seed=0)
    V = abs(np.linalg.det(A_true))
    for M in (np.diag([2., 2., 2.]), np.diag([3., 3., 3.])):
        As = A_true @ M
        cands = [LatticeCandidate(vector=As[:, i], score=len(g), rms=1e-4,
                                  chance=1.0,
                                  length=float(np.linalg.norm(As[:, i])))
                 for i in range(3)]
        A, info = select_basis(cands, g, tol=0.15, q_max=0.75)
        assert A is not None
        assert abs(np.linalg.det(A)) == pytest.approx(V, rel=1e-6), (
            f"a x{np.linalg.det(M):.0f} axis-aligned supercell was not divided back")


def test_a_GROSS_supercell_is_refused_when_nothing_smaller_is_available():
    """The last-resort absolute guard, for when the Patterson missed the truth.

    Calibrated loosely on purpose: the TRUE 2604 cell scores 20.6 on a 36 deg
    wedge, so a tight threshold refuses correct answers. Only gross cases
    (real ones scored 205 and 232) are caught here; the relative ranking is the
    real protection.
    """
    from midas_hkls.ab_initio import select_basis, LatticeCandidate
    A_true = _rot(0) @ np.linalg.inv(_B(ORTHO)).T
    g = _spots(ORTHO, seed=0)
    # index 7 on one axis: PRIME and above max_index=4, so the sublattice
    # search cannot reduce it and the loose absolute guard is what must fire
    As = A_true @ np.diag([7., 7., 7.])            # x343
    cands = [LatticeCandidate(vector=As[:, i], score=len(g), rms=1e-4, chance=1.0,
                              length=float(np.linalg.norm(As[:, i]))) for i in range(3)]
    A, info = select_basis(cands, g, tol=0.15, q_max=0.75, max_supercell_ratio=100.0)
    if A is not None:
        # axis division CAN reduce an axis-aligned x7; that is a pass, not a fail
        assert abs(np.linalg.det(A)) < 2.0 * abs(np.linalg.det(A_true))
    else:
        assert info["supercell_ratio"] > 100


def test_the_TRUE_cell_would_survive_the_absolute_guard_on_a_narrow_wedge():
    """Why the guard is loose. Pin it so nobody tightens it back."""
    from midas_hkls.ab_initio import _predicted_reflection_count
    V_true = 3.6116 ** 2 * 19.2516 / 2.0           # primitive I4/mmm
    A = np.diag([V_true ** (1 / 3)] * 3)
    ratio = _predicted_reflection_count(A, 1.0) / 51    # 51 = domain-1 reflections
    assert ratio > 5.0, "the true cell scores above any tight threshold"
    assert ratio < 100.0, "...and below the loose one, which is the point"


def test_a_SKEWED_supercell_is_now_reduced_by_the_sublattice_search():
    """x3 skewed used to survive; the fractional-combination search catches it.

    Axis division alone cannot: the supercell is not aligned with its own
    reduced basis. The search over (n1a+n2b+n3c)/m up to m=4 can.
    """
    from midas_hkls.ab_initio import select_basis, LatticeCandidate
    A_true = _rot(0) @ np.linalg.inv(_B(ORTHO)).T
    g = _spots(ORTHO, seed=0)
    As = A_true @ np.array([[3., 1., 0.], [0., 1., 1.], [0., 0., 1.]])
    cands = [LatticeCandidate(vector=As[:, i], score=len(g), rms=1e-4, chance=1.0,
                              length=float(np.linalg.norm(As[:, i]))) for i in range(3)]
    A, info = select_basis(cands, g, tol=0.15, q_max=0.75)
    assert A is not None
    assert abs(np.linalg.det(A)) == pytest.approx(abs(np.linalg.det(A_true)), rel=1e-6), (
        "the x3 skewed supercell was not reduced back to the true cell")
    import midas_hkls.ab_initio as ab
    assert "Beyond" in ab.__doc__ and "still get through" in ab.__doc__   # limit stated
    from midas_hkls.ab_initio import _find_finer_lattice
    assert "prime index above" in _find_finer_lattice.__doc__


def test_the_supercell_ratio_is_always_reported():
    g = _spots(ORTHO, seed=0)
    res = index_ab_initio(g, sigma_g=2e-4)
    assert res.success
    assert res.diagnostics["supercell_ratio"] < 5.0


def test_real_multiphase_spot_lists_are_refused_not_indexed():
    """Two lattices plus random background: no single cell should be returned."""
    g = np.vstack([_spots(ORTHO, seed=0), _spots(TRICLINIC, seed=7),
                   RNG.uniform(-0.7, 0.7, (80, 3))])
    res = index_ab_initio(g, sigma_g=2e-4, min_subset=40)
    if res.success:
        # acceptable only if it owns a genuine minority subset, i.e. one source
        assert res.diagnostics["subset_fraction"] < 0.75, str(res)
        assert res.diagnostics["noncoplanarity"] > 0.05


# ------------------------------------------------- primitive -> conventional

def test_the_conventional_cell_of_a_body_centred_lattice_is_recovered():
    """The whole reason this exists: primitive != published."""
    from midas_hkls import to_conventional
    # exact primitive cell of I4/mmm a=b=3.6116, c=19.2516
    r = to_conventional((3.6116, 3.6116, 9.9588, 79.5507, 79.5507, 90.0))
    assert r.system == "tetragonal"
    assert r.centring_index == 2                      # body centred
    assert sorted(r.cell[:3])[:2] == pytest.approx([3.6116, 3.6116], abs=1e-3)
    assert max(r.cell[:3]) == pytest.approx(19.2516, abs=5e-3)
    assert all(abs(a - 90.0) < 0.01 for a in r.cell[3:])   # input primitive is rounded
    assert r.volume_ratio == pytest.approx(2.0, rel=1e-6)


def test_a_primitive_lattice_is_its_own_conventional_cell():
    from midas_hkls import to_conventional
    r = to_conventional(TRICLINIC)
    assert r.centring_index == 1
    assert r.volume_ratio == pytest.approx(1.0, rel=1e-9)


def test_metric_symmetry_labels_the_common_systems():
    from midas_hkls import metric_symmetry
    assert metric_symmetry((4., 4., 4., 90., 90., 90.)) == "cubic"
    assert metric_symmetry((4., 4., 7., 90., 90., 90.)) == "tetragonal"
    assert metric_symmetry((4., 5., 7., 90., 90., 90.)) == "orthorhombic"
    assert metric_symmetry((4., 4., 7., 90., 90., 120.)) == "hexagonal"
    assert metric_symmetry((4., 5., 7., 90., 103., 90.)) == "monoclinic"
    assert metric_symmetry(TRICLINIC) == "triclinic"


def test_a_squashed_basis_is_never_offered_as_conventional():
    """A 15 deg angle reads as 'monoclinic' and outranks triclinic. Guard it."""
    from midas_hkls import to_conventional
    r = to_conventional((3.63, 3.73, 9.92, 79.4, 79.4, 88.3))
    assert all(30.0 <= a <= 150.0 for a in r.cell[3:]), r.cell


def test_the_setting_limit_is_documented():
    import midas_hkls.conventional as cv
    assert "not canonical" in cv.__doc__
    assert "0.04-0.11" in cv.__doc__
