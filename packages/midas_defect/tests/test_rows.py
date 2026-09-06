"""Reciprocal-lattice rows and row-seeded indexing.

Every test here pins a failure that verification found in the field version.
"""
from __future__ import annotations
import math
import numpy as np
import pytest

from midas_defect.rows import (
    find_lattice_rows, allowed_multiple_parity, unique_by_hkl,
    refine_row_spacing, refine_cell_orientation, refine_lattice,
    index_from_pairs, index_by_grid, orientation_grid,
    index_from_row, cell_from_row, row_scan_range,
                               candidate_row_spacings, hkl_box_from_geometry,
                               index_from_ladder, match_mask,
                               omega_smear_duplicates)
from midas_defect.ingest import live_frames
from midas_hkls import Lattice

A, C = 3.6116, 19.2516
STEPS = [(0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1),
         (1, 0, 1), (0, 1, 2), (1, 1, 2), (1, 2, 1)]


def _B():
    lat = Lattice(a=A, b=A, c=C, alpha=90., beta=90., gamma=90.)
    return np.asarray(lat.reciprocal_cartesian_vectors(), float).T * 2 * math.pi


def _row_cloud(step, rng, n=8, noise=2e-3):
    nmul = allowed_multiple_parity(step, 139)
    g = _B() @ np.array(step, float)
    gn = float(np.linalg.norm(g)); u = g / gn
    mult = [nmul * m for m in range(1, n + 1)]
    q = np.array([m * gn * u for m in mult] + [-m * gn * u for m in mult])
    return q + rng.normal(0, noise, q.shape), gn, nmul


def test_parity_is_per_direction_not_a_blanket_even_rule():
    """Odd-sum steps show only even multiples under I-centring; even-sum steps
    show every multiple. A blanket rule is wrong by 2x for the latter."""
    assert allowed_multiple_parity((0, 0, 1), 139) == 2      # odd sum
    assert allowed_multiple_parity((1, 1, 0), 139) == 1      # even sum
    assert allowed_multiple_parity((1, 1, 1), 139) == 2
    # F-centring is a different rule again
    assert allowed_multiple_parity((1, 1, 1), 69) == 1
    assert allowed_multiple_parity((1, 0, 0), 69) == 2


@pytest.mark.parametrize("step", STEPS, ids=[str(s) for s in STEPS])
def test_every_direction_is_found_and_identified(step):
    """Before the per-direction parity fix only (0,0,1) of these 8 worked."""
    rng = np.random.default_rng(0)
    q, gn, nmul = _row_cloud(step, rng)
    I = np.full(len(q), 1e5)
    rows = find_lattice_rows(q, I, n_seed=40, min_rungs=3, a=A, c=C,
                             space_group_number=139)
    assert rows, f"{step}: no row found"
    r = rows[0]
    assert r.hkl_step is not None, f"{step}: unidentified"
    # (1,0,0) and (0,1,0) share |G| when a == b -- accept the equivalent
    got = tuple(sorted((abs(r.hkl_step[0]), abs(r.hkl_step[1])))) + (abs(r.hkl_step[2]),)
    want = tuple(sorted((abs(step[0]), abs(step[1])))) + (abs(step[2]),)
    assert got == want, f"{step}: identified as {r.hkl_step}"
    assert r.match_rel < 0.02


def test_identification_can_REFUSE():
    """Without a threshold, (0,0,1) absorbed 80 % of producible spacings and a
    1.90x-scaled cloud was still confidently labelled (0,0,1)."""
    rng = np.random.default_rng(1)
    q, _, _ = _row_cloud((0, 0, 1), rng)
    rows = find_lattice_rows(q * 1.90, np.full(len(q), 1e5), n_seed=40,
                             min_rungs=3, a=A, c=C, space_group_number=139,
                             max_match_rel=0.05)
    assert rows, "row detection itself should still work"
    assert rows[0].hkl_step is None, "a 1.9x-scaled row must NOT be identified"


def test_hkl_box_from_geometry_matches_the_detector():
    assert hkl_box_from_geometry(A, C) == (4, 21)


def test_omega_smear_duplicates_catches_a_split_reflection():
    """The main false-domain mechanism: one omega-smeared reflection cut into
    blobs and re-indexed as a second grain."""
    hkl_old = np.array([[0, 0, 6], [0, 1, -3], [1, 1, 2]])
    rc_old = np.array([[800., 700.], [850., 720.], [400., 300.]])
    fr_old = np.array([10, 12, 30])
    # same hkl, a few px and a few frames away -> duplicates
    hkl_new = np.array([[0, 0, 6], [0, 1, -3], [2, 2, 0]])
    rc_new = np.array([[818., 706.], [875., 726.], [900., 900.]])
    fr_new = np.array([19, 18, 5])
    dup = omega_smear_duplicates(hkl_new, rc_new, fr_new,
                                 hkl_old, rc_old, fr_old)
    assert dup.tolist() == [True, True, False]


def test_omega_smear_guard_does_not_fire_on_a_distant_same_hkl():
    """A genuine second domain may share an hkl LABEL, just not the position."""
    dup = omega_smear_duplicates(
        np.array([[0, 0, 6]]), np.array([[800., 700.]]), np.array([10]),
        np.array([[0, 0, 6]]), np.array([[200., 1300.]]), np.array([11]))
    assert not dup.any()


def test_index_from_ladder_reports_a_margin():
    """A flat phi landscape must be distinguishable from a sharp one."""
    rng = np.random.default_rng(2)
    q, _, _ = _row_cloud((0, 0, 1), rng, n=10)
    out = index_from_ladder(q, np.full(len(q), 1e5), np.array([0., 0., 1.]),
                            a=A, c=C, return_margin=True)
    assert len(out) == 4
    U, n, phi, margin = out
    assert U is not None and n > 0
    assert isinstance(margin, (int, np.integer))


def test_live_frames_uses_the_maximum_not_the_sum():
    """A dead frame still SUMS to ~3e5; only the maximum separates it."""
    rng = np.random.default_rng(3)
    frames = rng.uniform(0, 3, (10, 200, 200)).astype(np.float32)   # baseline
    frames[3, 100, 100] = 5e5                                       # live peaks
    for k in (0, 1, 2, 4, 5, 6, 7):
        frames[k, 50 + k, 50] = 2e5
    live, mx = live_frames(frames)
    assert not live[8] and not live[9], "flat frames must read as dead"
    assert live[3] and live[0]
    assert frames[8].sum() > 1e4, "the dead frame still has a large SUM"


def test_live_frames_keeps_a_FAINT_but_real_frame():
    """A 5 %-of-median cut discarded frames peaking at 5031 and 30660 counts
    against a 639779 median -- real signal. Truly dead frames sit ~0.008 %."""
    import numpy as np
    from midas_defect.ingest import live_frames
    rng = np.random.default_rng(5)
    frames = rng.uniform(0, 3, (10, 100, 100)).astype(np.float32)
    for k in range(8):
        frames[k, 50, 50] = 640000.0          # normal frames
    frames[1, 50, 50] = 5031.0                # faint but REAL: 0.8 % of median
    frames[2, 50, 50] = 30660.0               # faint but REAL: 4.8 %
    frames[8, 50, 50] = 44.0                  # genuinely dead
    frames[9, 50, 50] = 58.0                  # genuinely dead
    live, mx = live_frames(frames)
    assert live[1] and live[2], "faint real frames must be KEPT"
    assert not live[8] and not live[9], "dead frames must be dropped"


# ------------------------------------------------------- duplicate claims

def test_unique_by_hkl_keeps_the_best_fragment():
    """A streak cut into three fragments must count as ONE reflection."""
    claim = np.array([True, True, True, True])
    hkl = np.array([[0, 0, 6], [0, 0, 6], [0, 0, 6], [0, 0, 4]])
    resid = np.array([0.05, 0.02, 0.09, 0.03])
    keep = unique_by_hkl(claim, hkl, resid)
    assert keep.sum() == 2, "two distinct reflections, not four claims"
    assert keep[1] and not keep[0] and not keep[2], "closest to integer survives"
    assert keep[3]


def test_unique_by_hkl_never_promotes_an_unclaimed_spot():
    claim = np.array([False, True, False])
    hkl = np.array([[1, 0, 2], [1, 0, 2], [1, 0, 2]])
    resid = np.array([0.001, 0.08, 0.002])       # unclaimed ones fit better
    keep = unique_by_hkl(claim, hkl, resid)
    assert keep.tolist() == [False, True, False]


def test_unique_by_hkl_distinguishes_sign():
    """(0,0,6) and (0,0,-6) are different reflections, not a duplicate pair."""
    claim = np.array([True, True])
    hkl = np.array([[0, 0, 6], [0, 0, -6]])
    keep = unique_by_hkl(claim, hkl, np.array([0.01, 0.02]))
    assert keep.all()


def test_match_mask_residual_matches_the_tolerance_it_applied():
    rng = np.random.default_rng(0)
    U = np.eye(3)
    lat_a, lat_c = 3.6116, 19.2516
    B = np.diag([2*np.pi/lat_a, 2*np.pi/lat_a, 2*np.pi/lat_c])
    hkl = np.array([[0, 0, 2], [0, 0, 4], [1, 1, 0], [0, 0, 6]], float)
    q = (U @ B @ hkl.T).T
    q = q + rng.normal(scale=1e-4, size=q.shape)
    claim, hi, resid = match_mask(q, U, a=lat_a, c=lat_c,
                                  return_residual=True)
    assert claim.all()
    assert (resid[claim] < 0.10).all()
    claim2, hi2 = match_mask(q, U, a=lat_a, c=lat_c)
    assert np.array_equal(claim, claim2) and np.array_equal(hi, hi2), \
        "return_residual must not change the decision"


# ------------------------------------------ seeding a domain from ANY row

ROT422 = [
    np.eye(3),
    np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),      # 4+ about c
    np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]),     # 2 about c
    np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]),      # 4- about c
    np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]),     # 2 about a
    np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]),     # 2 about b
    np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]),      # 2 about [110]
    np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]]),    # 2 about [1-10]
]


def _misorientation_deg(Ua, Ub):
    """Smallest rotation between two orientations, modulo 422."""
    best = 180.0
    for R in ROT422:
        M = Ua.T @ (Ub @ R)
        ang = math.degrees(math.acos(max(-1., min(1., (np.trace(M) - 1.) / 2.))))
        best = min(best, ang)
    return best


def _synthetic(U, a=3.6116, c=19.2516, hmax=3, lmax=12):
    B = np.diag([2*math.pi/a, 2*math.pi/a, 2*math.pi/c])
    hkl = np.array([(h, k, l)
                    for h in range(-hmax, hmax+1)
                    for k in range(-hmax, hmax+1)
                    for l in range(-lmax, lmax+1)
                    if (h, k, l) != (0, 0, 0) and (h+k+l) % 2 == 0], float)
    q = (U @ B @ hkl.T).T
    return q, hkl, B


def test_row_scan_range_matches_the_hardcoded_ladder_case():
    assert row_scan_range((0, 0, 1)) == (90.0, (1,))
    assert row_scan_range((0, 0, 4)) == (90.0, (1,))
    assert row_scan_range((1, 0, 0)) == (180.0, (1,))
    assert row_scan_range((1, 1, 0)) == (180.0, (1,))
    # no rotation of 422 reverses a fully general axis, so BOTH senses of the
    # row must be tried -- a row is an axis, it carries no direction
    assert row_scan_range((1, 2, 1)) == (360.0, (1, -1))
    assert row_scan_range((0, 1, 3)) == (360.0, (1,))
    with pytest.raises(ValueError):
        row_scan_range((0, 0, 0))


def test_cell_from_row_round_trips():
    a, c = 3.7, 19.9
    for step in [(0, 0, 1), (1, 1, 0), (1, 0, 0), (0, 1, 1), (1, 1, 2)]:
        n = allowed_multiple_parity(step, 139)
        h, k, l = step
        g = 2*math.pi*math.sqrt((h*h + k*k)/a**2 + l*l/c**2)
        ra, rc = cell_from_row(step, n*g, a0=a, c0=c)
        assert abs(ra - a) < 1e-6 and abs(rc - c) < 1e-6, step


def test_cell_from_row_solves_the_axis_the_row_constrains():
    """A (0,1,1) row is 28x more informative about a than about c."""
    a, c = 3.6116, 19.2516
    n = allowed_multiple_parity((0, 1, 1), 139)
    g = 2*math.pi*math.sqrt(1/a**2 + 1/c**2)
    ra, rc = cell_from_row((0, 1, 1), n*g, a0=a, c0=c*1.02)   # c held off-truth
    assert rc == pytest.approx(c*1.02), "c must be held, not fitted"
    assert ra != pytest.approx(a, abs=1e-9), "a must absorb the row"


def test_index_from_row_recovers_an_orientation_from_a_110_row():
    """The case index_from_ladder structurally cannot do: no (00L) seed."""
    ang = math.radians(37.0)
    U = np.array([[math.cos(ang), -math.sin(ang), 0.],
                  [math.sin(ang), math.cos(ang), 0.], [0., 0., 1.]])
    tilt = math.radians(23.0)
    U = np.array([[1., 0., 0.], [0., math.cos(tilt), -math.sin(tilt)],
                  [0., math.sin(tilt), math.cos(tilt)]]) @ U
    a, c = 3.6116, 19.2516
    q, hkl, B = _synthetic(U)
    u_row = U @ B @ np.array([1., 1., 0.]); u_row /= np.linalg.norm(u_row)
    Ur, n, phi = index_from_row(q, np.ones(len(q)), u_row, (1, 1, 0), a=a, c=c)
    assert Ur is not None
    assert n > 0.9*len(q), f"only {n} of {len(q)} planted spots matched"
    assert _misorientation_deg(U, Ur) < 0.5


def test_index_from_row_agrees_with_index_from_ladder_on_a_00L_seed():
    ang, tilt = math.radians(51.0), math.radians(17.0)
    U = np.array([[math.cos(ang), -math.sin(ang), 0.],
                  [math.sin(ang), math.cos(ang), 0.], [0., 0., 1.]])
    U = np.array([[1., 0., 0.], [0., math.cos(tilt), -math.sin(tilt)],
                  [0., math.sin(tilt), math.cos(tilt)]]) @ U
    a, c = 3.6116, 19.2516
    q, hkl, B = _synthetic(U)
    u = U @ np.array([0., 0., 1.])
    _, n_lad, _ = index_from_ladder(q, np.ones(len(q)), u, a=a, c=c)
    _, n_row, _ = index_from_row(q, np.ones(len(q)), u, (0, 0, 1), a=a, c=c)
    assert n_row >= n_lad, f"generalised seed lost spots: {n_row} < {n_lad}"


def test_index_from_row_tries_both_axis_senses_when_symmetry_does_not():
    """For a (1,2,1) row the two senses are different crystals; if only one
    were scanned, half of all such domains would be silently unindexable."""
    ang, tilt = math.radians(12.0), math.radians(64.0)
    U = np.array([[math.cos(ang), -math.sin(ang), 0.],
                  [math.sin(ang), math.cos(ang), 0.], [0., 0., 1.]])
    U = np.array([[1., 0., 0.], [0., math.cos(tilt), -math.sin(tilt)],
                  [0., math.sin(tilt), math.cos(tilt)]]) @ U
    a, c = 3.6116, 19.2516
    q, hkl, B = _synthetic(U)
    g = U @ B @ np.array([1., 2., 1.]); g /= np.linalg.norm(g)
    for sense in (+1.0, -1.0):
        _, n, _ = index_from_row(q, np.ones(len(q)), sense*g, (1, 2, 1),
                                 a=a, c=c, phi_step=0.5)
        assert n > 0.9*len(q), f"sense {sense:+.0f} matched only {n}"


def test_index_from_row_margin_is_reported():
    ang = math.radians(9.0)
    U = np.array([[math.cos(ang), -math.sin(ang), 0.],
                  [math.sin(ang), math.cos(ang), 0.], [0., 0., 1.]])
    q, hkl, B = _synthetic(U)
    u = U @ B @ np.array([1., 1., 0.]); u /= np.linalg.norm(u)
    Ur, n, phi, margin = index_from_row(q, np.ones(len(q)), u, (1, 1, 0),
                                        return_margin=True)
    assert margin > 0, "a planted crystal must beat its own rivals"


def test_candidate_spacings_are_one_per_symmetry_orbit():
    """a = b, so (0,1,1) and (1,0,1) are one reflection with one |G|.

    Listing both made every mixed-index row look ambiguous against its own
    symmetry partner, and p=66's (0,1,1) and (0,1,3) rows were thrown away for
    it. Two distinct orbits at the same spacing stay separate -- that ambiguity
    is real.
    """
    cands = candidate_row_spacings(3.6116, 19.2516)
    steps = [c[0] for c in cands]
    assert len(steps) == len(set(steps))
    for h, k, l in steps:
        assert h >= k >= 0 and l >= 0, f"{(h, k, l)} is not the orbit rep"
    assert (1, 0, 1) in steps and (0, 1, 1) not in steps
    # and no two entries are the symmetry partner of one another
    reps = {(max(h, k), min(h, k), abs(l)) for h, k, l in steps}
    assert len(reps) == len(steps)


def test_orbit_reduction_makes_a_mixed_row_identifiable():
    """Regression for the p=66 rejection: rel and rel2 were equal at 0.0108."""
    a, c = 3.6116, 19.2516
    cands = candidate_row_spacings(a, c)
    g = 2*math.pi*math.sqrt(1/a**2 + 1/c**2)          # a (1,0,1) row
    n = allowed_multiple_parity((1, 0, 1), 139)
    scored = sorted((abs(n*g - allowed_multiple_parity(hh, 139)*gp)
                     / (allowed_multiple_parity(hh, 139)*gp), hh)
                    for hh, gp in cands)
    assert scored[0][1] == (1, 0, 1)
    assert scored[1][0] > 10*max(scored[0][0], 1e-6), \
        "runner-up must be a genuinely different orbit, not the partner"


# ------------------------------------------- row spacing must be a FIT

def _noisy_rungs(D, n_steps, sigma, seed):
    rng = np.random.default_rng(seed)
    n = np.array(n_steps, float)
    return n * D + rng.normal(scale=sigma, size=n.size)


def _min_admissible_pair(proj, lo=0.30, hi=8.0):
    pv = np.sort(proj)
    return min(abs(pv[i] - pv[j]) for i in range(len(pv))
               for j in range(i + 1, len(pv))
               if lo <= abs(pv[i] - pv[j]) <= hi)


def test_spacing_is_a_fit_not_the_worst_pair():
    """Regression: the spacing used to BE a single pairwise difference.

    Candidates were tried in ascending order with ties broken by the first, so
    among equally-good candidates the SMALLEST won -- a minimum statistic. At
    La3Ni2O7 p=114 a 14-rung row was set by its one outlier pair, giving
    c = 19.4397 where the slope of the same rungs gives 19.1485, and the 1.5%
    error broke the ladder-seeded search on a domain worth 43 reflections.
    """
    D = 0.65626
    proj = _noisy_rungs(D, [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
                        0.006, 4)
    worst = _min_admissible_pair(proj)
    # precondition: this case actually exercises the bug
    assert worst < 0.99 * D, \
        f"test is toothless -- worst pair {worst:.5f} is not >1% below {D}"
    d, ok, resid = refine_row_spacing(proj, worst)
    assert abs(d - D) < 0.004 * D, f"LS gave {d:.5f}, truth {D:.5f}"
    assert abs(d - D) < abs(worst - D) / 4, "must beat the raw pair decisively"
    assert ok.sum() >= 12 and resid < 0.05


def test_refine_refuses_a_drift_past_max_drift():
    """The guard returns the input unchanged rather than a distant fit."""
    D = 0.65626
    proj = _noisy_rungs(D, [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
                        0.006, 4)
    worst = _min_admissible_pair(proj)
    assert worst < 0.99 * D                        # the fit wants to move ~2%
    d, ok, _ = refine_row_spacing(proj, worst, max_drift=0.005)
    assert ok is None and d == pytest.approx(worst), \
        "a refinement beyond max_drift must be refused, not returned"


def test_a_submultiple_start_is_a_fixed_point_not_a_drift():
    """Documents a real limit: this function cannot fix a parity error.

    Start at D/2 and every rung sits at an even multiple, so the fit is exactly
    self-consistent and returns D/2. That is not a failure of the refinement --
    D/2 genuinely describes the rungs -- and it is why parity is decided by
    `allowed_multiple_parity` from the space group, not by fitting. The
    candidate search is protected separately: candidates are pairwise
    differences of the projections, which are integer multiples of D, so D/2
    never becomes a candidate in the first place.
    """
    D = 0.656
    proj = _noisy_rungs(D, list(range(-8, 0)) + list(range(1, 9)), 0.002, 7)
    d, ok, resid = refine_row_spacing(proj, D / 2.0)
    assert d == pytest.approx(D / 2.0, rel=1e-3) and resid < 0.02
    pv = np.sort(proj)
    pairs = [abs(pv[i] - pv[j]) for i in range(len(pv))
             for j in range(i + 1, len(pv)) if 0.30 <= abs(pv[i] - pv[j]) <= 8.0]
    assert min(abs(p - D / 2.0) for p in pairs) > 0.05 * D, \
        "D/2 must not be reachable as a candidate spacing"


def test_refine_is_idempotent_on_a_clean_row():
    D = 1.7654
    proj = _noisy_rungs(D, [-3, -2, -1, 1, 2, 3], 1e-6, 11)
    d1, _, _ = refine_row_spacing(proj, D)
    d2, _, _ = refine_row_spacing(proj, d1)
    assert d1 == pytest.approx(D, rel=1e-4)
    assert d2 == pytest.approx(d1, rel=1e-12)


def test_find_lattice_rows_reports_the_fitted_spacing():
    """End to end: the Row must carry the fit, and its residual."""
    D = 0.65626
    steps = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    proj = _noisy_rungs(D, steps, 0.006, 4)
    u = np.array([0.3, -0.5, 0.81]); u /= np.linalg.norm(u)
    q = np.outer(proj, u)
    rows = find_lattice_rows(q, np.full(len(q), 1000.0), min_rungs=3,
                             identify=False)
    assert rows, "no row found"
    R = max(rows, key=lambda r: r.n_rungs)
    worst = _min_admissible_pair(proj)
    assert abs(R.spacing - D) < 0.004 * D, \
        f"row spacing {R.spacing:.5f} vs truth {D:.5f} (worst pair {worst:.5f})"
    assert np.isfinite(R.spacing_resid) and R.spacing_resid < 0.05


def test_interlopers_must_not_halve_the_row_spacing():
    """Regression: two spots at half positions used to halve a 14-rung ladder.

    A sub-multiple describes the same rungs with doubled integers, so it can
    never lose a "most distinct steps" contest, and two interlopers are enough
    to make it win. At p=114 the (00L) ladder came back as 16 rungs at 0.32810
    with 14 of 16 steps even and the two odd ones (+/-3) doing the damage; the
    real spacing is 0.6562, and the halved value feeds straight into c.
    """
    D = 0.65626
    real = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 9]
    rng = np.random.default_rng(3)
    proj = np.array(real, float)*D + rng.normal(scale=0.004, size=len(real))
    # two interlopers at half positions -- another domain, or noise
    proj = np.concatenate([proj, np.array([1.5, -1.5])*D])
    u = np.array([0.2, 0.3, 0.93]); u /= np.linalg.norm(u)
    q = np.outer(proj, u)
    rows = find_lattice_rows(q, np.full(len(q), 1000.0), min_rungs=3,
                             identify=False)
    assert rows
    R = max(rows, key=lambda r: r.n_rungs)
    assert R.spacing == pytest.approx(D, rel=0.01), \
        f"spacing {R.spacing:.5f} -- halved to {D/2:.5f}?" \
        if R.spacing < 0.75*D else f"spacing {R.spacing:.5f} vs {D:.5f}"
    odd = [x for x in R.steps if x % 2]
    assert len(odd) > 0, "the real row has odd steps; an all-even set means halved"


def test_common_factor_needs_a_majority_not_unanimity():
    from midas_defect.rows import _robust_common_factor
    assert _robust_common_factor([-12, -10, -8, -6, -4, -3, -2, 2, 3, 4,
                                  6, 8, 10, 12, 14, 18]) == 2
    assert _robust_common_factor([-3, -2, -1, 1, 2, 3, 5, 7]) == 1
    assert _robust_common_factor([2, 4, 6]) == 2
    # too few survivors after the collapse -> refuse
    assert _robust_common_factor([2, 4, 1, 3, 5, 7, 9], min_rungs=3) == 1


# --------------------------------------------- refining the cell after indexing

def test_refine_cell_recovers_a_planted_cell_from_a_wrong_seed():
    """The seed cell is a guess; with indices assigned the cell is a fit."""
    a_true, c_true = 3.6042, 19.1503
    ang, tilt = math.radians(29.0), math.radians(41.0)
    U = np.array([[math.cos(ang), -math.sin(ang), 0.],
                  [math.sin(ang), math.cos(ang), 0.], [0., 0., 1.]])
    U = np.array([[1., 0., 0.], [0., math.cos(tilt), -math.sin(tilt)],
                  [0., math.sin(tilt), math.cos(tilt)]]) @ U
    B = np.diag([2*math.pi/a_true, 2*math.pi/a_true, 2*math.pi/c_true])
    hkl = np.array([(h, k, l) for h in range(-2, 3) for k in range(-2, 3)
                    for l in range(-8, 9)
                    if (h, k, l) != (0, 0, 0) and (h+k+l) % 2 == 0], float)
    q = (U @ B @ hkl.T).T
    Ur, a_r, c_r = refine_cell_orientation(q, hkl, a0=3.6116, c0=19.4397)
    assert Ur is not None
    assert a_r == pytest.approx(a_true, rel=1e-6)
    assert c_r == pytest.approx(c_true, rel=1e-6), \
        f"c refined to {c_r:.5f}, planted {c_true:.5f}, seeded 19.4397"
    assert _misorientation_deg(U, Ur) < 1e-4


def test_refine_cell_survives_realistic_centroid_noise():
    a_true, c_true = 3.6042, 19.1503
    rng = np.random.default_rng(5)
    U = np.eye(3)
    B = np.diag([2*math.pi/a_true, 2*math.pi/a_true, 2*math.pi/c_true])
    hkl = np.array([(h, k, l) for h in range(-2, 3) for k in range(-2, 3)
                    for l in range(-8, 9)
                    if (h, k, l) != (0, 0, 0) and (h+k+l) % 2 == 0], float)
    q = (U @ B @ hkl.T).T + rng.normal(scale=0.004, size=(len(hkl), 3))
    _, a_r, c_r = refine_cell_orientation(q, hkl, a0=a_true, c0=c_true)
    assert abs(a_r/a_true - 1) < 0.005 and abs(c_r/c_true - 1) < 0.005


def test_refine_cell_refuses_coplanar_indices():
    """(h,k,0) alone cannot determine c; the fit must decline, not invent one."""
    hkl = np.array([(h, k, 0) for h in range(-3, 4) for k in range(-3, 4)
                    if (h, k) != (0, 0) and (h+k) % 2 == 0], float)
    B = np.diag([2*math.pi/3.6, 2*math.pi/3.6, 2*math.pi/19.2])
    q = (B @ hkl.T).T
    U, a_r, c_r = refine_cell_orientation(q, hkl, a0=3.6, c0=19.2)
    assert U is None and a_r == 3.6 and c_r == 19.2


def test_refine_cell_refuses_a_wild_jump():
    """A big move means the indices are wrong, not that the cell was."""
    hkl = np.array([(h, k, l) for h in range(-2, 3) for k in range(-2, 3)
                    for l in range(-6, 7)
                    if (h, k, l) != (0, 0, 0) and (h+k+l) % 2 == 0], float)
    B = np.diag([2*math.pi/3.6, 2*math.pi/3.6, 2*math.pi/19.2])
    q = (B @ hkl.T).T
    U, a_r, c_r = refine_cell_orientation(q, hkl, a0=3.6, c0=30.0,
                                          max_cell_change=0.10)
    assert U is None and c_r == 30.0, "a 36% jump in c must be refused"


# ------------------------------------------- the fit must not erase the signal

def _sheared_cell(a, c, gamma_deg):
    ca = math.cos(math.radians(gamma_deg))
    G = np.array([[a*a, a*a*ca, 0.], [a*a*ca, a*a, 0.], [0., 0., c*c]])
    return np.linalg.cholesky(np.linalg.inv(G)).T * 2*math.pi


def test_refine_lattice_recovers_a_planted_gamma_shear():
    """Regression for the defect that made the fit blind to the deliverable.

    In a Ruddlesden-Popper subcell the Fmmm distortion is a GAMMA SHEAR, not
    a != b, and a diagonal B cannot express it. The previous diagonal fit
    returned a_x = a_y unchanged to 5e-5 with the whole distortion in the
    discarded off-diagonal term, reporting a sheared cell as clean tetragonal.
    """
    a_t, c_t, gam = 3.6116, 19.2516, 90.5
    B = _sheared_cell(a_t, c_t, gam)
    hkl = np.array([(h, k, l) for h in range(-2, 3) for k in range(-2, 3)
                    for l in range(-8, 9)
                    if (h, k, l) != (0, 0, 0) and (h+k+l) % 2 == 0], float)
    q = (B @ hkl.T).T
    r = refine_lattice(q, hkl)
    assert r is not None
    assert r.gamma == pytest.approx(gam, abs=1e-3), \
        f"gamma came back {r.gamma}, planted {gam} -- the shear was erased"
    assert r.a == pytest.approx(a_t, rel=1e-6)
    assert r.b == pytest.approx(a_t, rel=1e-6)
    assert abs(r.ab_split) < 1e-6, "a gamma shear must NOT show up as an a/b split"
    # and the constrained wrapper must still lose it, on purpose and visibly
    _, a_c, c_c = refine_cell_orientation(q, hkl)
    assert a_c == pytest.approx(a_t, rel=1e-6) and c_c == pytest.approx(c_t, rel=1e-6)


def test_refine_lattice_recovers_a_planted_ab_split():
    """The other distortion mode: a genuinely orthorhombic a != b."""
    a_t, b_t, c_t = 3.6000, 3.6250, 19.2516
    B = np.diag([2*math.pi/a_t, 2*math.pi/b_t, 2*math.pi/c_t])
    hkl = np.array([(h, k, l) for h in range(-3, 4) for k in range(-3, 4)
                    for l in range(-8, 9)
                    if (h, k, l) != (0, 0, 0) and (h+k+l) % 2 == 0], float)
    q = (B @ hkl.T).T
    r = refine_lattice(q, hkl)
    assert r.a == pytest.approx(a_t, rel=1e-6) and r.b == pytest.approx(b_t, rel=1e-6)
    assert r.gamma == pytest.approx(90.0, abs=1e-6)
    assert r.ab_split == pytest.approx(2*(b_t-a_t)/(b_t+a_t), rel=1e-4)


def test_refine_lattice_reports_leverage_per_axis():
    """p=329 dom2 averaged a 7-reflection axis with a 68-reflection one."""
    B = np.diag([2*math.pi/3.6116, 2*math.pi/3.6116, 2*math.pi/19.2516])
    hkl = np.array([(h, k, l) for h in range(-1, 2) for k in range(-3, 4)
                    for l in range(-8, 9)
                    if (h, k, l) != (0, 0, 0) and (h+k+l) % 2 == 0], float)
    q = (B @ hkl.T).T
    r = refine_lattice(q, hkl)
    assert r.leverage[0] < r.leverage[1] < r.leverage[2], \
        "leverage must expose that h is far less constrained than k or l"


def test_refine_lattice_refuses_coplanar_indices():
    hkl = np.array([(h, k, 0) for h in range(-3, 4) for k in range(-3, 4)
                    if (h, k) != (0, 0) and (h+k) % 2 == 0], float)
    B = np.diag([2*math.pi/3.6, 2*math.pi/3.6, 2*math.pi/19.2])
    assert refine_lattice((B @ hkl.T).T, hkl) is None


def test_match_mask_q_tolerance_is_isotropic():
    """A fractional-index tolerance is 5x looser in plane than along c here."""
    a, c = 3.6116, 19.2516
    B = np.diag([2*math.pi/a, 2*math.pi/a, 2*math.pi/c])
    U = np.eye(3)
    base = np.array([[2., 0., 0.], [0., 0., 6.]])
    q = (B @ base.T).T
    # push each reflection off by the SAME distance in q
    dq = 0.10
    q_off = q + np.array([[dq, 0., 0.], [0., 0., dq]])
    frac, _ = match_mask(q_off, U, a=a, c=c, tol=0.10)
    assert frac[0] != frac[1], \
        "fractional tolerance should treat equal q errors unequally (the defect)"
    iso, _ = match_mask(q_off, U, a=a, c=c, tol_q=0.05)
    assert not iso.any(), "an isotropic tolerance must reject both equally"
    iso2, _ = match_mask(q_off, U, a=a, c=c, tol_q=0.15)
    assert iso2.all()


def test_robust_spacing_rejects_a_contaminant_rung():
    """p=66: one rung 0.064 off integer dragged c from 19.263 to 19.184."""
    proj = np.array([-1.2956, -0.6458, 0.6532, 1.3065, 1.9563, 2.6105,
                     3.3173, 3.9141])          # the real p=66 rungs
    d_plain, _, _ = refine_row_spacing(proj, 0.65506, robust=False)
    d_rob, _, _ = refine_row_spacing(proj, 0.65506, robust=True)
    c_plain, c_rob = 4*math.pi/d_plain, 4*math.pi/d_rob
    assert abs(c_rob - 19.320) < abs(c_plain - 19.320), \
        f"robust {c_rob:.4f} must be closer to the match-count optimum than " \
        f"plain {c_plain:.4f}"
    assert abs(c_rob - 19.320) < 0.35


def test_index_from_pairs_finds_a_domain_with_no_row_at_all():
    """Row seeding needs a row; two reflections are enough once c is known.

    In the residual after earlier domains are removed, rows are scarce -- at
    p=66 only 3-rung stubs remain and they die at the gates -- while pairs of
    spots are plentiful. This is what lets a later domain be found there.
    """
    a, c = 3.6116, 19.2516
    B = np.diag([2*math.pi/a, 2*math.pi/a, 2*math.pi/c])
    ang, tilt = math.radians(33.0), math.radians(47.0)
    U = np.array([[math.cos(ang), -math.sin(ang), 0.],
                  [math.sin(ang), math.cos(ang), 0.], [0., 0., 1.]])
    U = np.array([[1., 0., 0.], [0., math.cos(tilt), -math.sin(tilt)],
                  [0., math.sin(tilt), math.cos(tilt)]]) @ U
    hkl = np.array([(h, k, l) for h in range(-1, 2) for k in range(-2, 3)
                    for l in range(-6, 7)
                    if (h, k, l) != (0, 0, 0) and (h+k+l) % 2 == 0], float)
    q = (U @ B @ hkl.T).T
    rng = np.random.default_rng(2)
    noise = rng.normal(scale=0.004, size=q.shape)
    Ur, n, pair, mg = index_from_pairs(q + noise, np.full(len(q), 1e6),
                                       B, a=a, c=c)
    assert mg > 0, 'a planted domain must beat its distant rivals'
    assert Ur is not None, f"found only {n} reflections"
    assert n > 0.8*len(hkl), f"{n} of {len(hkl)}"
    assert _misorientation_deg(U, Ur) < 1.0


def test_index_from_pairs_declines_on_a_random_cloud():
    """It must be able to fail: a cloud with no crystal must return nothing."""
    a, c = 3.6116, 19.2516
    B = np.diag([2*math.pi/a, 2*math.pi/a, 2*math.pi/c])
    rng = np.random.default_rng(9)
    v = rng.normal(size=(120, 3))
    q = v/np.linalg.norm(v, axis=1)[:, None]*rng.uniform(1.0, 4.0, 120)[:, None]
    U, n, _, _ = index_from_pairs(q, rng.uniform(1e5, 1e7, 120), B, a=a,
                                  c=c, min_reflections=12)
    assert U is None, f"indexed {n} reflections out of a random cloud"


def test_per_reflection_sigma_helps_only_at_high_contrast():
    """Per-reflection sigma is supported, and is NOT free.

    Down-weighting some reflections costs leverage, so it only pays when the
    down-weighted ones are much worse than the rest. Measured on this synthetic:
    at 16x contrast it makes the a/b estimate WORSE (ratio 1.31), at 40x it
    halves the error (0.50). That is why the pipeline does not switch it on for
    the omega-smear case -- the measured contrast there is 1.106/0.270 = 4.1x,
    far below break-even. Recorded as a test so the trade-off is not rediscovered
    by someone enabling it and wondering why the answer got worse.
    """
    a_t, c_t = 3.6116, 19.2516
    B = np.diag([2*math.pi/a_t, 2*math.pi/a_t, 2*math.pi/c_t])
    hkl = np.array([(h, k, l) for h in range(-2, 3) for k in range(-2, 3)
                    for l in range(-8, 9)
                    if (h, k, l) != (0, 0, 0) and (h+k+l) % 2 == 0], float)
    q0 = (B @ hkl.T).T
    Z = np.array([0., 0., 1.])

    def trial(s_bad, n_draw=60):
        ef, ep = [], []
        for seed in range(n_draw):
            rng = np.random.default_rng(seed)
            bad = rng.random(len(q0)) < 0.3
            sig = np.where(bad, s_bad, 0.005)
            pooled = float(np.sqrt(np.mean(sig**2)))
            q = q0.copy()
            for i in range(len(q0)):
                tv = np.cross(Z, q0[i]); n = np.linalg.norm(tv)
                if n < 1e-9:
                    continue
                q[i] = q[i] + (tv/n)*rng.normal(0, sig[i])
            f = refine_lattice(q, hkl, sigma_rtn=(0.005, pooled, 0.005))
            p = refine_lattice(q, hkl, sigma_rtn=(0.005, sig, 0.005))
            ef.append(abs(f.ab_split)); ep.append(abs(p.ab_split))
        return np.mean(ep)/np.mean(ef)

    assert trial(0.20) < 0.8, "at 40x contrast per-reflection sigma must win"
    assert trial(0.08) > 1.0, "at 16x contrast it must lose -- leverage is not free"
    with pytest.raises(ValueError):
        refine_lattice(q0, hkl, sigma_rtn=(0.005, np.ones(3), 0.005))


def test_index_from_pairs_finds_a_domain_100x_fainter():
    """A weak crystallite can diffract ~100x less than a dominant one.

    With intensity-ranked anchors the search is blind to exactly the domains it
    exists to find, so the anchor list is stratified across the intensity range.
    """
    a, c = 3.6116, 19.2516
    B = np.diag([2*math.pi/a, 2*math.pi/a, 2*math.pi/c])
    hkl = np.array([(h, k, l) for h in range(-1, 2) for k in range(-2, 3)
                    for l in range(-6, 7)
                    if (h, k, l) != (0, 0, 0) and (h+k+l) % 2 == 0], float)
    rng = np.random.default_rng(4)

    def dom(angle, tilt):
        U = np.array([[math.cos(angle), -math.sin(angle), 0.],
                      [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        return np.array([[1., 0., 0.],
                         [0., math.cos(tilt), -math.sin(tilt)],
                         [0., math.sin(tilt), math.cos(tilt)]]) @ U

    U_bright = dom(math.radians(11.0), math.radians(21.0))
    U_faint = dom(math.radians(64.0), math.radians(53.0))
    q_b = (U_bright @ B @ hkl.T).T
    q_f = (U_faint @ B @ hkl.T).T
    q = np.vstack([q_b, q_f]) + rng.normal(scale=0.004, size=(2*len(hkl), 3))
    I = np.concatenate([np.full(len(hkl), 1e8), np.full(len(hkl), 1e6)])

    # the bright domain is already taken; look for the faint one in the residual
    exclude = np.zeros(len(q), bool)
    exclude[:len(hkl)] = True
    Ur, n, _, _ = index_from_pairs(q, I, B, a=a, c=c, exclude=exclude)
    assert Ur is not None, f"faint domain not found ({n} reflections)"
    assert _misorientation_deg(U_faint, Ur) < 1.0


def test_match_mask_anisotropic_acceptance_matches_the_error():
    """An isotropic cut must be set by the worst direction and loses good spots.

    The measured error is radial 0.0071, transverse 0.0145. A reflection pushed
    0.012 transversely is a 0.8-sigma event and should be kept; the same 0.012
    pushed radially is 1.7 sigma. An isotropic tol_q cannot tell them apart.
    """
    a, c = 3.6116, 19.2516
    B = np.diag([2*math.pi/a, 2*math.pi/a, 2*math.pi/c])
    U = np.eye(3)
    hkl = np.array([[1., 1., 0.], [1., -1., 0.]])
    q = (B @ hkl.T).T
    Z = np.array([0., 0., 1.])
    sig = (0.0071, 0.0145, 0.0094)
    push = 0.012
    qq = q.copy()
    qh0 = q[0]/np.linalg.norm(q[0])
    t0 = np.cross(Z, qh0); t0 /= np.linalg.norm(t0)
    qq[0] = q[0] + t0*push                        # transverse: 0.8 sigma
    qh1 = q[1]/np.linalg.norm(q[1])
    qq[1] = q[1] + qh1*push                       # radial: 1.7 sigma
    iso, _ = match_mask(qq, U, a=a, c=c, tol_q=0.010)
    assert not iso.any(), "an isotropic cut tight enough for radial kills both"
    ani, _ = match_mask(qq, U, a=a, c=c, tol_sigma=1.2, sigma_rtn=sig)
    assert ani[0] and not ani[1], \
        "the transverse one must survive and the radial one must not"


def test_orientation_grid_is_proper_and_scales():
    for step in (6.0, 3.0):
        g = orientation_grid(step)
        assert np.allclose([np.linalg.det(x) for x in g], 1.0)
        assert np.allclose([x @ x.T for x in g[::200]], np.eye(3), atol=1e-9)
    assert len(orientation_grid(3.0)) > 5*len(orientation_grid(6.0))


def test_index_by_grid_finds_the_dominant_grain_and_refuses_noise():
    """Documents both what it can do and what it cannot.

    Exhaustive orientation search is the WRONG tool for subdominant grains here
    and this test records why, so nobody wires it into the pipeline expecting
    otherwise. It recovers the dominant grain exactly and returns nothing on a
    random cloud; on a multi-grain cloud it returns spurious low-count peaks
    instead of the weaker grains, because a grid fine enough to match |q| ~ 5
    within tol_q = 0.05 has millions of orientations and that many trials find
    chance matches the margin gate cannot reject.
    """
    a, c = 3.6116, 19.2516
    B = np.diag([2*math.pi/a, 2*math.pi/a, 2*math.pi/c])
    ang, tilt = math.radians(11.0), math.radians(21.0)
    U = np.array([[math.cos(ang), -math.sin(ang), 0.],
                  [math.sin(ang), math.cos(ang), 0.], [0., 0., 1.]])
    U = np.array([[1., 0., 0.], [0., math.cos(tilt), -math.sin(tilt)],
                  [0., math.sin(tilt), math.cos(tilt)]]) @ U
    hkl = np.array([(h, k, l) for h in range(-1, 2) for k in range(-2, 3)
                    for l in range(-5, 6)
                    if (h, k, l) != (0, 0, 0) and (h+k+l) % 2 == 0], float)
    rng = np.random.default_rng(0)
    q = (U @ B @ hkl.T).T + rng.normal(scale=0.004, size=(len(hkl), 3))
    got = index_by_grid(q, B, a=a, c=c, step_deg=6.0, max_domains=1)
    assert got, "the dominant grain must be found"
    assert got[0][1] > 0.8*len(hkl)
    assert _misorientation_deg(U, got[0][0]) < 2.0

    v = rng.normal(size=(200, 3))
    qr = v/np.linalg.norm(v, axis=1)[:, None]*rng.uniform(1.0, 4.0, 200)[:, None]
    assert index_by_grid(qr, B, a=a, c=c, step_deg=6.0,
                         min_reflections=12) == [], "must refuse a random cloud"


# ---------------------------------------------------------------------------
# Regressions for two bugs that reached a production analysis, found by
# adversarial verification of a La3Ni2O7 result rather than by the suite.
# ---------------------------------------------------------------------------

def test_misorientation_matches_midas_stress():
    """The hard-coded 422 table must not drift from the canonical routine.

    `rows._misorientation_422` keeps a local symmetry table purely as a fast
    path (~8x) inside the pair seeder. `midas_stress.misorientation_om` is
    canonical. If they ever disagree, the local table is wrong, not the other.
    """
    from scipy.spatial.transform import Rotation as R
    from midas_stress import misorientation_om
    from midas_defect.rows import _misorientation_422
    A = R.random(120, random_state=11).as_matrix()
    B = R.random(120, random_state=12).as_matrix()
    worst = max(abs(_misorientation_422(a, b)
                    - np.degrees(float(misorientation_om(a, b, 139)[0])))
                for a, b in zip(A, B))
    assert worst < 1e-9, f"local 422 table drifted from midas_stress: {worst} deg"


def _planted_domain(a=3.6116, c=19.2516, seed=0, n_hkl=40, noise=0.0):
    """A tetragonal domain at a known cell, as observed q with a known U.

    ``noise`` matters: at 0.0 the lattice is EXACT, every subset of reflections
    determines the identical cell, and any test that compares two subsets is
    vacuous. That is precisely why the first version of the regression tests here
    passed against the buggy code.
    """
    from scipy.spatial.transform import Rotation as R
    rng = np.random.default_rng(seed)
    U = R.random(random_state=seed).as_matrix()
    hs = []
    for h in range(-2, 3):
        for k in range(-2, 3):
            for l in range(-6, 7, 2):
                if (h + k + l) % 2 == 0 and (h, k, l) != (0, 0, 0):
                    hs.append((h, k, l))
    hs = [hs[i] for i in rng.permutation(len(hs))[:n_hkl]]
    B = np.diag([2*np.pi/a, 2*np.pi/a, 2*np.pi/c])
    q = np.array([U @ (B @ np.array(h, float)) for h in hs])
    if noise:
        q = q + rng.normal(0.0, noise, q.shape)
    return U, q, a, c


def test_refine_to_convergence_postcondition_holds():
    """CONTRACT test: `lat` is the LS fit to `claim`. NOT a regression test.

    Verified 2026-09-06 that this CANNOT FAIL against the historical buggy code:
    the loop refines immediately after setting `claim`, so the invariant holds by
    construction and the trailing refit is a measured no-op (0.00e+00 change over
    40 randomised scenes). Kept as a guard for anyone who reorders the loop, and
    labelled so nobody mistakes it for evidence that the bug it was written for is
    covered. That bug is covered by the trap below.
    """
    from midas_defect.rows import refine_to_convergence, refine_lattice
    U, q, a, c = _planted_domain(seed=3)
    res = refine_to_convergence(q, U, a0=a, c0=c)
    assert res is not None
    direct = refine_lattice(q[res.claim], res.hkl[res.claim], a0=a, c0=c,
                            sigma_rtn=(0.0071, 0.0145, 0.0094))
    assert direct is not None
    assert abs(direct.c - res.lat.c) < 1e-9
    assert abs(direct.a - res.lat.a) < 1e-9


def test_post_hoc_claim_filtering_breaks_the_postcondition():
    """THE ACTUAL BUG, as a trap: narrowing `claim` after convergence needs a refit.

    The defect was never inside the loop -- it was a caller dropping reflections
    from `claim` (duplicate rejection against an already-accepted domain) AFTER
    the cell was fitted, then storing the old cell against the new reflection
    list. 26% of one run's `c` values were inconsistent that way, and a later
    placement error left 3 of 4 non-first domains up to 0.0271 A out -- the size
    of the effect under study.

    Any caller that narrows `claim` MUST refit. This shows the two disagree, so
    the requirement is not theoretical.
    """
    from midas_defect.rows import refine_to_convergence, refine_lattice
    # NOISE IS LOAD-BEARING: on an exact lattice every subset gives the same cell
    # and this test is vacuous -- which is how the first attempt at it passed
    # against the bug.
    U, q, a, c = _planted_domain(seed=3, noise=0.004)
    res = refine_to_convergence(q, U, a0=a, c0=c)
    assert res is not None and int(res.claim.sum()) >= 12

    kept = res.claim.copy()
    kept[np.flatnonzero(kept)[:4]] = False       # as a deduper would
    stale = res.lat                              # what a careless caller stores
    refit = refine_lattice(q[kept], res.hkl[kept], a0=a, c0=c,
                           sigma_rtn=(0.0071, 0.0145, 0.0094))
    assert refit is not None
    assert abs(refit.c - stale.c) > 1e-9, (
        "post-hoc filtering did not move the cell; this trap has stopped biting "
        "and the test needs a scene where it does")


def test_refine_to_convergence_is_monotone():
    """A kept round never loses reflections -- the loop is safe to run blind."""
    from midas_defect.rows import refine_to_convergence
    U, q, a, c = _planted_domain(seed=7)
    res = refine_to_convergence(q, U, a0=a, c0=c, max_iter=6)
    assert res is not None
    one = refine_to_convergence(q, U, a0=a, c0=c, max_iter=1)
    assert int(res.claim.sum()) >= int(one.claim.sum())


def test_seed_referenced_gate_manufactures_correlation():
    """A gate on |c/c_seed - 1| fabricates the anchoring it would be read as proving.

    This is a TRAP, not a feature of any function here -- it lives in the test
    suite because it invalidated a real measurement and cost a full verification
    cycle. Accepting a domain only when its cell is within `tol` of the SEED's is
    a truncation band around the REGRESSOR of the obvious anchoring test, so it
    produces a positive slope from data with zero anchoring in it. A gate
    referenced to a fixed nominal cell does not.

    Reference cell-plausibility gates to a nominal cell, never to a neighbour's.
    """
    from scipy import stats
    rng = np.random.default_rng(20260906)
    n, c_nom = 20000, 19.2516
    # ZERO anchoring by construction: seed and domain cells are independent.
    seed_c = c_nom * (1.0 + rng.normal(0, 0.008, n))
    dom_c = c_nom * (1.0 + rng.normal(0, 0.008, n))

    keep_seed = np.abs(dom_c/seed_c - 1.0) < 0.01        # seed-referenced
    keep_nom = np.abs(dom_c/c_nom - 1.0) < 0.015         # nominal-referenced
    assert keep_seed.sum() > 500 and keep_nom.sum() > 500

    r_seed = stats.linregress(seed_c[keep_seed], dom_c[keep_seed])
    r_nom = stats.linregress(seed_c[keep_nom], dom_c[keep_nom])

    # the seed-referenced gate invents a strongly positive slope out of noise
    assert r_seed.slope > 5 * r_seed.stderr, (
        f"expected the truncation trap to fire, got {r_seed.slope:.3f}")
    # the nominal-referenced gate does not
    assert abs(r_nom.slope) < 3 * r_nom.stderr, (
        f"nominal gate should be unbiased, got {r_nom.slope:.3f} +- {r_nom.stderr:.3f}")
    assert r_seed.slope > 4 * abs(r_nom.slope)
