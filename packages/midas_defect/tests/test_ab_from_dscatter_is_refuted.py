"""KNOWN LIMIT, kept executable: a/b splitting is not recoverable from d alone.

This file tests no shipped code. It is a **non-identifiability proof**, kept
here so the method below is not re-invented — a deleted negative result is
worth nothing; an executable one is worth the week it cost.

The seductive idea
------------------
If a crystal has a != b, swapping a and b sends d(hkl) to d(khl), so the
a/b-SENSITIVE families (h != k) should show a wider spread of d than the blind
ones. Fit that extra spread and you get the splitting with **no orientation and
no indexing** — precisely what you cannot obtain when a sample is multi-grain
and will not index. Three estimators were built on this idea (family variance;
family variance with a measured d-dependent error floor; a five-parameter
unbinned mixture maximum likelihood). All three returned a large, significant,
FALSE splitting on a tetragonal control where a = b is established from 45
indexed reflections. Those specific numbers (mixture ML: delta = 0.630 % at
p = 0.0098 on a crystal with a = b) live in the source analysis' RUNNING_LOG
and are **not** re-derived here.

What IS proved here, and it is stronger than any one estimator failing
---------------------------------------------------------------------
The degeneracy is structural, so no estimator can escape it:

    A crystal with (a != b, no strain) and a crystal with (a = b, strain)
    produce the SAME distribution of d values. The branch label -- which spot
    is (200) and which is (020) -- is the only thing that separates them, and
    an unindexed spot list does not carry it.

So this is not a precision problem and no amount of data fixes it. What fixes
it is an **orientation**, which supplies the branch label; the final test shows
the same two ensembles becoming cleanly separable the moment it is available.

Related: `midas_hkls.distortion_mode` for the companion trap — whether the cell
setting you are fitting in can express the distortion at all.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
from scipy import stats

A_BAR, C = 3.7167, 19.25
DELTA = 0.0034                    # the real La3Ni2O7 supercell value, 0.34 %


def _families(max_index=4):
    return [(h, k, l)
            for h in range(0, max_index + 1)
            for k in range(0, max_index + 1)
            for l in range(0, 9)
            if not (h == k == l == 0)]


def _d(h, k, l, a, b, c=C):
    return 1.0 / np.sqrt((h / a) ** 2 + (k / b) ** 2 + (l / c) ** 2)


def ensemble(*, delta, strain, n_per_family=6, seed=0, max_index=4,
             labelled=False):
    """Observed d values from a crystal with this splitting and this strain.

    The branch (whether a spot is (hkl) or (khl)) is chosen at random and, by
    default, **not returned** — that is the premise of the orientation-free
    method. With ``labelled=True`` the branch comes back too, which is what an
    orientation would give you.
    """
    rng = np.random.default_rng(seed)
    a, b = A_BAR * (1 + delta), A_BAR * (1 - delta)
    ds, labs = [], []
    for (h, k, l) in _families(max_index):
        for _ in range(n_per_family):
            swap = rng.random() < 0.5
            hh, kk = (k, h) if swap else (h, k)
            ds.append(_d(hh, kk, l, a, b) * (1.0 + rng.normal(0.0, strain)))
            labs.append((hh, kk, l))          # the indices ACTUALLY recorded
    ds = np.asarray(ds)
    return (ds, labs) if labelled else ds


def _fractional_spread(delta, strain, seed=0):
    """Spread of d about the a = b prediction — the only observable here."""
    d_obs = ensemble(delta=delta, strain=strain, seed=seed)
    d_ref = np.asarray([_d(h, k, l, A_BAR, A_BAR)
                        for (h, k, l) in _families()
                        for _ in range(6)])
    return (d_obs - d_ref) / d_ref


# --------------------------------------------------------------------------
# the proof
# --------------------------------------------------------------------------

def test_a_real_splitting_and_pure_strain_give_the_same_d_distribution():
    """THE non-identifiability. Two different crystals, one d distribution."""
    split_only = ensemble(delta=DELTA, strain=0.0, seed=1)
    # choose the strain that matches the spread the splitting produces
    matched = float(np.std(_fractional_spread(DELTA, 0.0, seed=1)))
    strain_only = ensemble(delta=0.0, strain=matched, seed=2)

    ks = stats.ks_2samp(split_only, strain_only)
    assert ks.pvalue > 0.05, (
        f"KS p = {ks.pvalue:.4f} — the two ensembles separated on d alone. "
        "If this is reproducible the degeneracy may be weaker than believed; "
        "investigate before deleting.")


def test_the_matched_strain_is_an_ordinary_amount_not_a_contrivance():
    """A degeneracy that needed absurd strain would not matter in practice."""
    matched = float(np.std(_fractional_spread(DELTA, 0.0, seed=1)))
    assert 1e-4 < matched < 1e-2, (
        f"matched strain {matched*100:.3f} % is outside the ordinary range; "
        "the degeneracy would then be contrived rather than practical")


def test_the_spread_does_not_identify_WHICH_mechanism_produced_it():
    """Spread is a one-number summary of a two-parameter family."""
    matched = float(np.std(_fractional_spread(DELTA, 0.0, seed=1)))
    s_split = np.std(_fractional_spread(DELTA, 0.0, seed=3))
    s_strain = np.std(_fractional_spread(0.0, matched, seed=4))
    assert abs(s_split - s_strain) / max(s_split, s_strain) < 0.25, (
        f"spreads {s_split:.5f} vs {s_strain:.5f} differ by more than 25 % — "
        "recheck the matching before trusting the conclusion")


def test_sensitive_families_are_not_broader_once_strain_is_present():
    """The estimator's whole premise, tested and found not to hold."""
    matched = float(np.std(_fractional_spread(DELTA, 0.0, seed=1)))
    fams = _families()
    res = _fractional_spread(DELTA, matched, seed=5)
    per_fam = res.reshape(len(fams), 6)
    sens = np.array([h != k for (h, k, l) in fams])
    broad_sensitive = per_fam[sens].std(axis=1).mean()
    broad_blind = per_fam[~sens].std(axis=1).mean()
    ratio = broad_sensitive / broad_blind
    # the premise says this should be comfortably > 1; with realistic strain
    # present it is not, which is why the estimators absorbed the strain
    assert ratio < 2.0, (
        f"sensitive/blind width ratio {ratio:.2f} — if this is now large and "
        "stable the premise may hold after all; re-verify against a control")


def test_MORE_DATA_DOES_NOT_HELP():
    """Not a precision problem: 10x the spots leaves the degeneracy intact."""
    matched = float(np.std(_fractional_spread(DELTA, 0.0, seed=1)))
    big_split = ensemble(delta=DELTA, strain=0.0, n_per_family=60, seed=6)
    big_strain = ensemble(delta=0.0, strain=matched, n_per_family=60, seed=7)
    assert stats.ks_2samp(big_split, big_strain).pvalue > 0.01, (
        "the ensembles separated at 10x the data — the limit would then be "
        "statistical, not structural")


# --------------------------------------------------------------------------
# what DOES break it
# --------------------------------------------------------------------------

def test_an_ORIENTATION_breaks_the_degeneracy_immediately():
    """The constructive half: the branch label separates them at once.

    This is exactly what indexing supplies, and exactly what the orientation-free
    workaround was trying to avoid needing.
    """
    matched = float(np.std(_fractional_spread(DELTA, 0.0, seed=1)))
    d_split, lab_split = ensemble(delta=DELTA, strain=matched, seed=8,
                                  labelled=True)
    d_flat, lab_flat = ensemble(delta=0.0, strain=matched, seed=8,
                                labelled=True)

    def signed_asymmetry(d, lab):
        """With the branch known, every sensitive spot can be sign-aligned.

        Under a > b a spot whose FIRST index is the larger sits at larger d,
        and one whose second index is larger sits at smaller d. Aligning on
        ``sign(h - k)`` of the indices ACTUALLY recorded therefore turns the
        splitting into a coherent mean. Strain, having no such handedness,
        averages to zero however much of it there is.
        """
        out = []
        for dd, (h, k, l) in zip(d, lab):
            if h == k:
                continue
            ref = _d(h, k, l, A_BAR, A_BAR)
            out.append((dd - ref) / ref * np.sign(h - k))
        return np.asarray(out)

    a_split = signed_asymmetry(d_split, lab_split)
    a_flat = signed_asymmetry(d_flat, lab_flat)
    t = stats.ttest_ind(a_split, a_flat, equal_var=False)
    assert t.pvalue < 1e-6, (
        f"even WITH the branch label the two were not separable (p = "
        f"{t.pvalue:.3g}); the constructive claim needs rechecking")
    assert abs(np.mean(a_split)) > 3 * abs(np.mean(a_flat))
