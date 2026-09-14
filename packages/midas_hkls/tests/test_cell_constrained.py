"""Constrained and joint cell refinement. Truth is planted, so a passing test
means the estimator recovered something real, not that it agreed with itself."""
from __future__ import annotations
import math
import numpy as np
import pytest

from midas_hkls import Lattice
from midas_hkls.cell_constrained import (FREE_PARAMS, DomainData,
                                         refine_cell_constrained,
                                         refine_cell_joint, refine_cell_radial,
                                         refine_cell_joint_robust,
                                         refine_cell_radial_robust,
                                         tukey_biweight, split_with_error)

A0, C0 = 3.6116, 19.2516
SPLIT = 0.34


def _cell(a, b, c):
    return (a, b, c, 90., 90., 90.)


def _spots(cell, hkl, sigma=3e-4, seed=0):
    lat = Lattice(a=cell[0], b=cell[1], c=cell[2],
                  alpha=cell[3], beta=cell[4], gamma=cell[5])
    B = np.asarray(lat.reciprocal_cartesian_vectors(), float).T
    return hkl @ B.T + np.random.default_rng(seed).normal(0, sigma, (len(hkl), 3))


def _hkl(mx=3, mz=6):
    return np.array([(i, j, k) for i in range(-mx, mx+1) for j in range(-mx, mx+1)
                     for k in range(-mz, mz+1)
                     if (i, j, k) != (0, 0, 0) and (i+j+k) % 2 == 0], float)


def test_a_planted_orthorhombic_split_is_recovered():
    b0 = A0 * (1 + SPLIT/100)
    h = _hkl()
    g = _spots(_cell(A0, b0, C0), h)
    f = refine_cell_constrained(h, g, system="orthorhombic",
                                cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    got, sig, z = split_with_error(f)
    assert abs(got - SPLIT) < 0.05, f"recovered {got}, planted {SPLIT}"
    assert z > 10


@pytest.mark.parametrize("system,n_free", [(k, len(v)) for k, v in FREE_PARAMS.items()])
def test_each_system_refines_only_its_free_parameters(system, n_free):
    h = _hkl()
    g = _spots(_cell(A0, A0, C0), h)
    f = refine_cell_constrained(h, g, system=system,
                                cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    assert len(f.free_names) == n_free
    c = f.cell
    if system in ("cubic",):
        assert abs(c[0]-c[1]) < 1e-9 and abs(c[1]-c[2]) < 1e-9
    if system in ("tetragonal", "hexagonal"):
        assert abs(c[0]-c[1]) < 1e-9
    if system in ("orthorhombic", "tetragonal", "cubic"):
        assert all(abs(v-90.) < 1e-9 for v in c[3:])
    if system == "hexagonal":
        assert abs(c[5]-120.) < 1e-9


def test_the_symmetry_constraint_is_itself_a_discriminator():
    """A tetragonal fit forbids the split, so it must fit measurably worse."""
    b0 = A0 * (1 + SPLIT/100)
    h = _hkl(); g = _spots(_cell(A0, b0, C0), h)
    o = refine_cell_constrained(h, g, system="orthorhombic",
                                cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    t = refine_cell_constrained(h, g, system="tetragonal",
                                cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    assert t.rms > 2.0 * o.rms


def test_a_tetragonal_fit_reports_the_split_as_imposed_not_measured():
    h = _hkl(); g = _spots(_cell(A0, A0, C0), h)
    f = refine_cell_constrained(h, g, system="tetragonal",
                                cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    split, sig, z = split_with_error(f)
    assert split == 0.0
    assert math.isnan(sig), "a==b by symmetry is imposed; sigma must not be a number"


def test_joint_refinement_shares_one_cell_across_domains():
    b0 = A0 * (1 + SPLIT/100)
    cell = _cell(A0, b0, C0)
    h = _hkl()
    doms = []
    for d in range(3):
        idx = np.random.default_rng(d).choice(len(h), 60, replace=False)
        hh = h[idx]
        g = _spots(cell, hh, seed=100+d)
        R = np.linalg.qr(np.random.default_rng(200+d).normal(size=(3, 3)))[0]
        if np.linalg.det(R) < 0: R[:, 2] *= -1
        doms.append(DomainData(hh, g @ R.T))
    f = refine_cell_joint(doms, system="orthorhombic",
                          cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    assert f.n_domains == 3 and f.n_reflections == 180
    assert len(f.rotvecs) == 3
    got, _, _ = split_with_error(f)
    assert abs(got - SPLIT) < 0.08, f"joint fit recovered {got}"


def test_more_domains_on_one_cell_beat_one_domain():
    b0 = A0 * (1 + SPLIT/100); cell = _cell(A0, b0, C0); h = _hkl()
    def err(nd, per, seed):
        doms = []
        for d in range(nd):
            idx = np.random.default_rng(seed+d).choice(len(h), per, replace=False)
            doms.append(DomainData(h[idx], _spots(cell, h[idx], sigma=3e-3,
                                                  seed=seed+50+d)))
        f = refine_cell_joint(doms, system="orthorhombic",
                              cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
        return f.cell_sigma[0]
    assert err(4, 40, 0) < err(1, 40, 0)


# ---------------------------------------------------------------------------
# refine_cell_radial -- the radial-only (orientation-free) half of the
# staged refit, PREREGISTER_staged_radial_refit_2026-09-13.md
# ---------------------------------------------------------------------------

def test_radial_fit_recovers_a_planted_split():
    b0 = A0 * (1 + SPLIT/100)
    h = _hkl()
    g = _spots(_cell(A0, b0, C0), h)
    f = refine_cell_radial([DomainData(h, g)], system="orthorhombic",
                           cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    got, sig, z = split_with_error(f)
    assert abs(got - SPLIT) < 0.05, f"recovered {got}, planted {SPLIT}"


def test_radial_fit_is_exactly_orientation_independent():
    """The whole premise of this function: |g_pred| = |B @ hkl| is preserved by
    any rotation, so rotating the SAME domain's g-vectors must not move the
    fitted cell at all -- unlike refine_cell_joint, which fits a rotation per
    domain and so is not invariant to how g arrived rotated."""
    b0 = A0 * (1 + SPLIT/100)
    h = _hkl()
    g = _spots(_cell(A0, b0, C0), h)
    R = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    f0 = refine_cell_radial([DomainData(h, g)], system="orthorhombic",
                            cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    f1 = refine_cell_radial([DomainData(h, g @ R.T)], system="orthorhombic",
                            cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    assert np.allclose(f0.cell, f1.cell, atol=1e-10)


def test_radial_fit_does_not_manufacture_a_split_on_a_tetragonal_truth():
    h = _hkl()
    g = _spots(_cell(A0, A0, C0), h)
    f = refine_cell_radial([DomainData(h, g)], system="orthorhombic",
                           cell0=(3.6, 3.6, 19.2, 90., 90., 90.),
                           n_bootstrap=200, rng=np.random.default_rng(0))
    got, sig, z = split_with_error(f)
    assert abs(z) < 3, f"spurious split on a=b truth: {got} +/- {sig} (z={z})"


def test_radial_fit_bootstrap_and_rotvecs():
    h = _hkl()
    g = _spots(_cell(A0, A0, C0), h)
    f = refine_cell_radial([DomainData(h, g)], system="orthorhombic",
                           cell0=(3.6, 3.6, 19.2, 90., 90., 90.),
                           n_bootstrap=50, rng=np.random.default_rng(1))
    assert f.cell_sigma_bootstrap is not None
    assert f.n_bootstrap > 4
    assert f.rotvecs == []


def test_radial_and_joint_bootstrap_are_paired_by_construction():
    """Same seed, same domain shape -> identical resample draws in both
    functions' bootstrap loops (both call rng.integers(0, len(hkl), len(hkl))
    once per domain per resample, in the same order) -- this is what makes a
    same-data sigma comparison between the two methods meaningful rather than
    just two independently noisy bootstraps."""
    h = _hkl()
    g = _spots(_cell(A0, A0, C0), h)
    doms = [DomainData(h, g)]
    fj = refine_cell_joint(doms, system="orthorhombic",
                           cell0=(3.6, 3.6, 19.2, 90., 90., 90.),
                           n_bootstrap=30, rng=np.random.default_rng(0))
    fr = refine_cell_radial(doms, system="orthorhombic",
                            cell0=(3.6, 3.6, 19.2, 90., 90., 90.),
                            n_bootstrap=30, rng=np.random.default_rng(0))
    assert fj.cell_bootstrap_samples.shape[0] == fr.cell_bootstrap_samples.shape[0]


def test_the_module_says_why_a_free_triclinic_cell_is_the_wrong_tool():
    import midas_hkls.cell_constrained as cc
    doc = " ".join(cc.__doc__.split())        # docstring wraps; compare flat
    assert "absorb noise into exactly the quantity being measured" in doc
    assert "bootstrap" in doc


# ---------------------------------------------------------------------------
# weighted fits + robust (IRLS) refinement -- generic outlier down-weighting,
# not hand-picking a suspicious hkl. Added after a real La3Ni2O7 DAC domain
# (2026-09-14) turned out to have two reflections ~5x the
# normal residual scale from what looked like a misindexed/contaminated blob.
# ---------------------------------------------------------------------------

def _hkl48(seed=0):
    """A fixed 48-reflection subset -- the size of a real gated domain, where
    a single bad reflection actually moves the fit (see the docstring of
    refine_cell_joint_robust: on the FULL ~300-reflection _hkl() set a single
    outlier barely registers under refine_cell_radial, which is itself a
    useful, checked fact, not an assumption)."""
    h = _hkl()
    idx = np.random.default_rng(seed).choice(len(h), 48, replace=False)
    return h[idx]


def test_tukey_biweight_basic():
    resid = np.array([0.0, 0.01, -0.01, 0.02, -0.02, 5.0])
    w, scale = tukey_biweight(resid)
    assert w[-1] == 0.0
    assert np.all(w[:-1] > 0.9)
    assert scale > 0


def test_weights_are_validated_not_silently_ignored():
    h = _hkl48()
    g = _spots(_cell(A0, A0, C0), h)
    with pytest.raises(ValueError):
        refine_cell_joint([DomainData(h, g)], system="orthorhombic",
                          cell0=(3.6, 3.6, 19.2, 90., 90., 90.),
                          weights=[np.ones(len(h) - 1)])
    with pytest.raises(ValueError):
        refine_cell_radial([DomainData(h, g)], system="orthorhombic",
                           cell0=(3.6, 3.6, 19.2, 90., 90., 90.),
                           weights=[np.ones(len(h)), np.ones(len(h))])


def test_a_weight_of_exactly_zero_is_equivalent_to_exclusion():
    """The whole point of the weighting mechanism: downweighting a reflection
    all the way to 0 must give the SAME answer as never having included it."""
    h = _hkl48()
    g = _spots(_cell(A0, A0, C0), h)
    bad = 3
    w = np.ones(len(h)); w[bad] = 0.0
    f_weighted = refine_cell_joint([DomainData(h, g)], system="orthorhombic",
                                   cell0=(3.6, 3.6, 19.2, 90., 90., 90.), weights=[w])
    keep = np.arange(len(h)) != bad
    f_excluded = refine_cell_joint([DomainData(h[keep], g[keep])], system="orthorhombic",
                                   cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    assert np.allclose(f_weighted.cell, f_excluded.cell, atol=1e-6)


def test_robust_joint_recovers_from_a_planted_contaminated_reflection():
    """One badly-misplaced reflection manufactures a spurious split under the
    plain joint fit; refine_cell_joint_robust must find it, drive its weight
    near zero, and recover something much closer to the a=b truth."""
    h = _hkl48()
    g = _spots(_cell(A0, A0, C0), h)          # a=b truth, no real split planted
    bad = 5
    g = g.copy()
    g[bad] += np.array([0.08, -0.05, 0.03])   # contamination-scale offset

    f_plain = refine_cell_joint([DomainData(h, g)], system="orthorhombic",
                                cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    split_plain, _, _ = split_with_error(f_plain)
    assert split_plain > 0.15, f"test setup didn't manufacture a spurious split ({split_plain})"

    f_rob = refine_cell_joint_robust([DomainData(h, g)], system="orthorhombic",
                                     cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    split_rob, _, _ = split_with_error(f_rob)
    assert f_rob.weights[0][bad] < 0.1, f"outlier not downweighted: w={f_rob.weights[0][bad]}"
    assert split_rob < split_plain / 2, f"robust fit didn't help: plain={split_plain} robust={split_rob}"
    assert f_rob.robust_n_iter >= 1


def test_robust_radial_recovers_from_a_planted_contaminated_reflection():
    h = _hkl48()
    g = _spots(_cell(A0, A0, C0), h)
    bad = 5
    g = g.copy()
    g[bad] += np.array([0.08, -0.05, 0.03])

    f_plain = refine_cell_radial([DomainData(h, g)], system="orthorhombic",
                                 cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    split_plain, _, _ = split_with_error(f_plain)
    assert split_plain > 0.03, f"test setup didn't manufacture a spurious split ({split_plain})"

    f_rob = refine_cell_radial_robust([DomainData(h, g)], system="orthorhombic",
                                      cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    split_rob, _, _ = split_with_error(f_rob)
    assert f_rob.weights[0][bad] < 0.1, f"outlier not downweighted: w={f_rob.weights[0][bad]}"
    assert split_rob < 0.75 * split_plain, f"robust fit didn't help enough: plain={split_plain} robust={split_rob}"


def test_robust_weights_are_mostly_high_on_clean_radial_data():
    """No planted outlier: the radial (scalar, near-Gaussian) residual should
    NOT show the chi-distributed-norm skew the joint residual does -- see
    refine_cell_joint_robust's calibration-caveat docstring."""
    h = _hkl48()
    g = _spots(_cell(A0, A0, C0), h)
    f = refine_cell_radial_robust([DomainData(h, g)], system="orthorhombic",
                                  cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    assert np.median(f.weights[0]) > 0.7


def test_robust_joint_does_not_manufacture_a_split_on_clean_data():
    """Even though the joint residual's norm-skew downweights a chunk of a
    clean domain (documented, not a bug), the recovered CELL must still sit
    at the a=b truth -- the skew changes which points get less say, not the
    physics being measured."""
    h = _hkl48()
    g = _spots(_cell(A0, A0, C0), h)
    f = refine_cell_joint_robust([DomainData(h, g)], system="orthorhombic",
                                 cell0=(3.6, 3.6, 19.2, 90., 90., 90.),
                                 n_bootstrap=200, rng=np.random.default_rng(0))
    got, sig, z = split_with_error(f)
    assert abs(z) < 3, f"spurious split on a=b truth: {got} +/- {sig} (z={z})"


def test_robust_wrappers_are_independent_scorers():
    """Joint and radial score different residual definitions -- they need not
    flag the same reflections. This is itself the useful signal (a reflection
    flagged by BOTH is a stronger suspect than one flagged by only one)."""
    h = _hkl48()
    g = _spots(_cell(A0, A0, C0), h)
    fj = refine_cell_joint_robust([DomainData(h, g)], system="orthorhombic",
                                  cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    fr = refine_cell_radial_robust([DomainData(h, g)], system="orthorhombic",
                                   cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    assert fj.weights[0].shape == (len(h),)
    assert fr.weights[0].shape == (len(h),)
    # not asserting they agree -- the point is they are allowed not to
