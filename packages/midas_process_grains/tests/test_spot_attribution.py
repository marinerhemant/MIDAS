"""``midas_process_grains.spot_attribution``.

The load-bearing test is :func:`test_twin_pair_both_survive` — a shared spot two
grains fit equally well must be kept for BOTH. Measured on 1-ID LSHR, deleting
twin-shared spots made a refit 48.7 % worse (worse even than deleting the same
number of spots at random, +32.8 %), because those spots fit at 0.62× the
grain's own uncontested residual: they are the best spots present, not the worst.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_process_grains.spot_attribution import (
    DEFAULT_MIN_SPOTS,
    SpotAttribution,
    attribute_spots,
    twin_agreement,
)


def _claims(rows):
    """rows = [(grain, spot, residual_um), ...] -> the three parallel arrays."""
    g = [r[0] for r in rows]
    s = [r[1] for r in rows]
    d = [r[2] for r in rows]
    return g, s, d


def _pad(rows, grain, start_spot, n, resid=38.0):
    """Give a grain n private spots so it clears the floor.

    38 um is the measured median uncontested residual on the LSHR layer, so
    the fixtures sit at a realistic scale: twin-shared spots fit BETTER than
    private ones (23 um, 0.62x) and accidental overlaps far worse (152 um).
    """
    return rows + [(grain, start_spot + i, resid) for i in range(n)]


# ---------------------------------------------------------------------------
#  The regression this module exists to prevent
# ---------------------------------------------------------------------------

def test_twin_pair_both_survive():
    """Two grains fitting one shared spot EQUALLY WELL: neither may be dropped.

    This is the twin case. A 'give the spot to whoever fits best' rule cuts at
    responsibility 0.5, and a twin pair sits at exactly 0.5/0.5 — so such a rule
    deletes one twin of every pair, at random, by floating-point noise.
    """
    rows = [(1, 999, 23.0), (2, 999, 24.0)]          # twin-shared: fits BETTER
    rows = _pad(rows, 1, 1000, 40)
    rows = _pad(rows, 2, 2000, 40)
    att = attribute_spots(*_claims(rows))

    shared = (att.spot_id == 999)
    assert shared.sum() == 2
    assert att.keep[shared].all(), (
        "both halves of a twin pair must keep the shared spot; "
        f"rel_likelihood={att.rel_likelihood[shared]}")
    # and they are near-equal in BOTH statistics
    assert att.rel_likelihood[shared].min() > 0.9
    np.testing.assert_allclose(att.responsibility[shared], [0.5, 0.5], atol=0.02)


def test_responsibility_half_would_have_killed_a_twin():
    """Documents WHY the filter is relative, not responsibility-based.

    Not testing production behaviour — testing that the naive alternative
    really does fail, so nobody 'simplifies' the rule back into it.
    """
    rows = _pad([(1, 999, 23.0), (2, 999, 24.0)], 1, 1000, 40)
    rows = _pad(rows, 2, 2000, 40)
    att = attribute_spots(*_claims(rows))
    shared = att.spot_id == 999
    naive_keep = att.responsibility[shared] >= 0.5
    assert not naive_keep.all(), (
        "a responsibility>=0.5 cut should drop one twin — if it no longer does, "
        "the test fixture has drifted and the rationale needs rechecking")
    assert att.keep[shared].all(), "the real rule must keep both"


def test_accidental_overlap_drops_the_bad_claimant():
    """One grain fits the spot at 30 um, the other at 400: the second goes."""
    rows = [(1, 999, 30.0), (2, 999, 400.0)]
    rows = _pad(rows, 1, 1000, 40)
    rows = _pad(rows, 2, 2000, 40)
    att = attribute_spots(*_claims(rows))
    shared = att.spot_id == 999
    keep = dict(zip(att.grain_id[shared], att.keep[shared]))
    assert keep[1], "the good claimant keeps its spot"
    assert not keep[2], "the 400 um claimant must lose it"


def test_uncontested_spots_are_always_kept():
    """However badly a private spot fits, nobody else claims it — keep it."""
    rows = [(1, 500, 900.0)] + [(1, 1000 + i, 10.0) for i in range(40)]
    att = attribute_spots(*_claims(rows))
    lone = att.spot_id == 500
    assert att.n_claimants[lone][0] == 1
    assert att.keep[lone].all()


# ---------------------------------------------------------------------------
#  Scale
# ---------------------------------------------------------------------------

def test_sigma_comes_from_uncontested_spots_only():
    """Contamination must not set the scale it is judged against.

    Two grains share 30 spots at a terrible 500 um; each has 40 private spots at
    10 um. Sigma must track the 10, not be dragged up by the 500 — otherwise the
    filter dissolves exactly when contamination is worst.
    """
    rows = []
    for i in range(30):
        rows += [(1, 900 + i, 500.0), (2, 900 + i, 520.0)]
    rows = _pad(rows, 1, 1000, 40, resid=10.0)
    rows = _pad(rows, 2, 2000, 40, resid=10.0)
    att = attribute_spots(*_claims(rows))
    assert att.sigma == pytest.approx(10.0, abs=1.0), (
        f"sigma {att.sigma} was dragged toward the contested 500 um population")
    assert att.provenance["sigma_from"] == "median uncontested residual"


def test_explicit_sigma_is_recorded_as_from_the_caller():
    rows = _pad([], 1, 1000, 40)
    att = attribute_spots(*_claims(rows), sigma=42.0)
    assert att.sigma == 42.0
    assert att.provenance["sigma_from"] == "caller"


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_bad_sigma_is_rejected(bad):
    rows = _pad([], 1, 1000, 40)
    with pytest.raises(ValueError):
        attribute_spots(*_claims(rows), sigma=bad)


# ---------------------------------------------------------------------------
#  Guards
# ---------------------------------------------------------------------------

def test_min_spots_floor_protects_an_over_stripped_grain():
    """An under-determined refit is worse than a contaminated one."""
    rows = [(2, 900 + i, 10.0) for i in range(30)]          # rival, fits well
    rows += [(1, 900 + i, 800.0) for i in range(30)]        # grain 1 loses all
    rows += [(1, 1000 + i, 12.0) for i in range(5)]         # only 5 private
    att = attribute_spots(*_claims(rows), min_spots=25)
    kept = att.keep[att.grain_id == 1].sum()
    assert kept >= 25, f"floor not honoured: kept only {kept}"
    assert att.n_protected > 0

    loose = attribute_spots(*_claims(rows), min_spots=0)
    assert loose.keep[loose.grain_id == 1].sum() == 5, (
        "with the floor off, grain 1 should keep only its 5 private spots")


def test_nonfinite_residual_is_always_dropped():
    """Matched=0 rows are predictions, not claims."""
    rows = [(1, 500, np.nan), (1, 501, np.inf)]
    rows = _pad(rows, 1, 1000, 40)
    att = attribute_spots(*_claims(rows), min_spots=0)
    assert not att.keep[att.spot_id == 500][0]
    assert not att.keep[att.spot_id == 501][0]


def test_empty_input():
    att = attribute_spots([], [], [])
    assert len(att.grain_id) == 0 and att.n_dropped == 0
    assert att.kept_spot_sets() == {}


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        attribute_spots([1, 2], [1], [1.0])


# ---------------------------------------------------------------------------
#  Outputs
# ---------------------------------------------------------------------------

def test_kept_spot_sets_and_weight_map_agree_with_keep():
    rows = [(1, 999, 30.0), (2, 999, 400.0)]
    rows = _pad(rows, 1, 1000, 40)
    rows = _pad(rows, 2, 2000, 40)
    att = attribute_spots(*_claims(rows))
    sets = att.kept_spot_sets()
    assert 999 in sets[1] and 999 not in sets[2]
    assert sets[1] == sorted(sets[1]), "spot ids must come out sorted"

    wm = att.weight_map()
    assert wm[(1, 999)] > wm[(2, 999)]
    # consistency is UNNORMALISED: a well-fitting spot is ~1 for every grain
    # that fits it, which is what a weighted refit needs.
    assert wm[(1, 1000)] == pytest.approx(np.exp(-0.5), rel=0.3)
    assert att.per_grain_dropped() == {2: 1}


def test_summary_mentions_the_scale_and_the_drops():
    rows = [(1, 999, 30.0), (2, 999, 400.0)]
    rows = _pad(rows, 1, 1000, 40)
    rows = _pad(rows, 2, 2000, 40)
    txt = attribute_spots(*_claims(rows)).summary()
    assert "sigma" in txt and "dropped" in txt and "contested" in txt


# ---------------------------------------------------------------------------
#  The validation diagnostic
# ---------------------------------------------------------------------------

def test_twin_agreement_recovers_labels_it_never_saw():
    """The residual-only rule should keep twin claims and drop accidental ones.

    Grains 1 and 2 are twins sharing well-fitting spots; grain 3 accidentally
    overlaps grain 1 with badly-fitting ones. twin_agreement is handed the
    labels only to SCORE the result — attribute_spots never sees them.
    """
    rows = []
    for i in range(10):
        rows += [(1, 700 + i, 23.0), (2, 700 + i, 24.0)]      # twin-shared, better
    for i in range(10):
        rows += [(1, 800 + i, 30.0), (3, 800 + i, 450.0)]     # accidental, far worse
    for gr, base in ((1, 1000), (2, 2000), (3, 3000)):
        rows = _pad(rows, gr, base, 40)
    att = attribute_spots(*_claims(rows))

    ta = twin_agreement(att, {1: {2}, 2: {1}})
    assert ta["n_twin_claims"] == 20
    assert ta["n_accidental_claims"] == 20
    assert ta["keep_rate_twin"] == 1.0, "twin-shared claims must all survive"
    assert ta["keep_rate_accidental"] < 0.75, (
        "the accidental claims should largely be dropped; "
        f"kept {ta['keep_rate_accidental']:.2f}")


def test_twin_agreement_with_no_contested_claims():
    rows = _pad([], 1, 1000, 40)
    ta = twin_agreement(attribute_spots(*_claims(rows)), {})
    assert ta["n_twin_claims"] == 0
    assert np.isnan(ta["keep_rate_twin"])
