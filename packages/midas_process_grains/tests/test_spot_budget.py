"""Tests for compute.spot_budget — spot-budget enforcement + decomposition."""

import numpy as np

from midas_process_grains.compute.spot_budget import (
    enforce_spot_budget, enforce_spot_budget_coverage,
)


def test_clean_population_all_claimed_once():
    # Three grains, disjoint spot sets -> every spot claimed once, nothing dropped.
    spot_sets = [set(range(0, 10)), set(range(10, 20)), set(range(20, 30))]
    q = np.array([1.0, 1.0, 1.0])
    fam = np.array([-1, -1, -1])  # all singletons
    r = enforce_spot_budget(spot_sets=spot_sets, quality_score=q,
                            twin_family_id=fam, min_owned_spots=5,
                            min_owned_fraction=0.5)
    assert r.n_observed_spots == 30
    assert r.n_claimed_once == 30
    assert r.n_shared_within_family == 0
    assert r.n_contested_between_families == 0
    assert r.keep_mask.all()
    assert abs(r.mean_multiplicity_grain - 1.0) < 1e-9


def test_twin_sharing_is_legitimate_not_contested():
    # Two grains in the SAME twin family share 5 of 10 spots -> within-family,
    # both co-own all their spots, both kept.
    a = set(range(0, 10))
    b = set(range(5, 15))            # overlaps a on {5..9}
    spot_sets = [a, b]
    q = np.array([1.0, 0.9])
    fam = np.array([7, 7])           # same family
    r = enforce_spot_budget(spot_sets=spot_sets, quality_score=q,
                            twin_family_id=fam, min_owned_spots=5,
                            min_owned_fraction=0.5)
    assert r.n_contested_between_families == 0
    assert r.n_shared_within_family == 5
    assert r.n_claimed_once == 10
    # both grains co-own their shared spots -> full ownership, both kept
    assert r.keep_mask.all()
    assert r.owned_spots.tolist() == [10, 10]


def test_contested_between_unrelated_families_drops_overlap():
    # A real grain (q=1) and an over-split duplicate (q=0.1) in DIFFERENT
    # families share 8 of the duplicate's 10 spots. The duplicate loses the
    # contested spots, retains only 2/10 owned -> dropped.
    real = set(range(0, 20))
    dup = set(range(0, 8)) | {100, 101}   # 8 contested + 2 unique
    spot_sets = [real, dup]
    q = np.array([1.0, 0.1])
    fam = np.array([1, 2])                  # unrelated families
    r = enforce_spot_budget(spot_sets=spot_sets, quality_score=q,
                            twin_family_id=fam, min_owned_spots=5,
                            min_owned_fraction=0.5)
    assert r.n_contested_between_families == 8
    assert r.keep_mask[0]            # real grain kept
    assert not r.keep_mask[1]        # over-split duplicate dropped
    assert r.owned_spots[0] == 20    # real grain wins all
    assert r.owned_spots[1] == 2     # duplicate keeps only its unique spots


def test_overcount_ratio_reflects_oversubscription():
    # 10 grains each claim the same 100 spots -> 10x leaf overcount.
    spot_sets = [set(range(100)) for _ in range(10)]
    q = np.ones(10)
    fam = np.arange(10)              # all distinct singleton families
    r = enforce_spot_budget(spot_sets=spot_sets, quality_score=q,
                            twin_family_id=fam, min_owned_spots=5,
                            min_owned_fraction=0.5)
    assert r.n_observed_spots == 100
    assert abs(r.mean_multiplicity_grain - 10.0) < 1e-9
    # every spot contested between 10 unrelated families
    assert r.n_contested_between_families == 100
    # only the single highest-quality grain (ties -> first best) owns them;
    # the rest own 0 -> dropped
    assert r.keep_mask.sum() == 1


def test_empty():
    r = enforce_spot_budget(spot_sets=[], quality_score=np.zeros(0),
                            twin_family_id=np.zeros(0))
    assert r.n_kept == 0 and r.n_observed_spots == 0


# ---------------------------------------------------------------------------
# coverage-greedy enforcement (multiplicity-allowance variant)
# ---------------------------------------------------------------------------

def test_coverage_drops_redundant_oversplit_copies():
    # One real grain (q=1, 100 spots) + 9 over-split duplicates (q<1) that
    # re-claim the SAME 100 spots. With cap=2 the first duplicate is admitted
    # (brings each spot to mult 2), the rest are fully saturated -> dropped.
    real = set(range(100))
    spot_sets = [set(real) for _ in range(10)]
    q = np.array([1.0] + [0.5 - 0.01 * i for i in range(9)])
    r = enforce_spot_budget_coverage(spot_sets=spot_sets, quality_score=q,
                                     multiplicity_cap=2, min_new_spots=5,
                                     min_new_fraction=0.5)
    assert r.n_kept == 2                      # cap=2 admits exactly two
    assert r.n_covered_spots == 100
    assert r.coverage_fraction == 1.0
    assert abs(r.mean_multiplicity_kept - 2.0) < 1e-9


def test_coverage_keeps_distinct_grains_full_coverage():
    # 50 grains each owning a disjoint 100-spot block -> all kept, 100% cover,
    # multiplicity 1 (no sharing).
    spot_sets = [set(range(b * 100, b * 100 + 100)) for b in range(50)]
    q = np.ones(50)
    r = enforce_spot_budget_coverage(spot_sets=spot_sets, quality_score=q,
                                     multiplicity_cap=2)
    assert r.n_kept == 50
    assert r.coverage_fraction == 1.0
    assert abs(r.mean_multiplicity_kept - 1.0) < 1e-9


def test_coverage_cap_controls_admitted_sharing():
    real = set(range(100))
    spot_sets = [set(real) for _ in range(6)]
    q = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    r3 = enforce_spot_budget_coverage(spot_sets=spot_sets, quality_score=q,
                                      multiplicity_cap=3)
    assert r3.n_kept == 3                      # cap=3 admits three
