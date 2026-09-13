"""a/b splitting helpers. Rewritten 2026-09-04 after a verify lens REFUTED the
partner criterion these originally encoded."""
from __future__ import annotations
import numpy as np
import pytest
from midas_hkls import (hkl_box_from_geometry, distortion_rank,
                        distortion_condition, ab_separable, shear_separable,
                        partner_multiplicity)


def test_the_box_comes_from_the_detector_not_from_a_guess():
    h, l = hkl_box_from_geometry(3.6116, 19.2516, wavelength_A=0.42459,
                                 tth_max_deg=26.4)
    assert (h, l) == (4, 21)
    assert l > 16, "the hand-picked lmax=16 deleted the (1,0,L) partners"


def test_a_lone_family_is_rank_1_however_many_l_values():
    """With c known, l enters as a known offset -- more l does NOT help."""
    lone = np.array([[2, -1, l] for l in range(-16, 17, 2)])
    assert distortion_rank(lone) == 1
    assert not ab_separable(lone)
    assert distortion_condition(lone) > 1e6


def test_a_SECOND_FAMILY_separates_a_from_b_with_NO_partner():
    """The correction. (1,1) gives 1/a^2+1/b^2, (2,-1) gives 4/a^2+1/b^2 --
    two independent rows. A partner criterion called this BLIND; on real 2604
    data 104 of 104 such domains are separable."""
    mixed = np.vstack([[[2, -1, l] for l in range(-6, 7, 2)],
                       [[1, 1, l] for l in range(-4, 5, 2)]])
    assert distortion_rank(mixed) == 2
    assert ab_separable(mixed)
    assert distortion_condition(mixed) < 20


def test_the_partner_improves_CONDITIONING_not_identifiability():
    lone_plus = np.vstack([[[2, -1, l] for l in range(-6, 7, 2)],
                           [[1, 1, l] for l in range(-4, 5, 2)]])
    with_partner = np.vstack([[[2, -1, l] for l in range(-6, 7, 2)],
                              [[1, -2, l] for l in range(-6, 7, 2)]])
    assert ab_separable(lone_plus) and ab_separable(with_partner)
    assert distortion_condition(with_partner) < distortion_condition(lone_plus)


def test_the_gamma_shear_needs_hk_and_SIGNS_must_survive():
    """(0,1)/(1,0) has hk=0 and is EXACTLY blind to a gamma shear -- the mode a
    Ruddlesden-Popper subcell actually has. (1,1)/(1,-1) carries it, and taking
    |h|,|k| would collapse them into one family and destroy the contrast."""
    no_shear = np.array([[0, 1, 2], [1, 0, 2], [0, 1, 4], [1, 0, 4]])
    assert ab_separable(no_shear)
    assert not shear_separable(no_shear), "hk=0 cannot see a shear"
    with_shear = np.vstack([no_shear, [[1, 1, 2], [1, -1, 2]]])
    assert shear_separable(with_shear)
    assert distortion_rank(with_shear) == 3


def test_signs_are_not_collapsed():
    """(1,1) and (1,-1) must NOT be treated as the same family."""
    same = np.array([[1, 1, l] for l in range(-4, 5, 2)])
    both = np.vstack([same, [[1, -1, l] for l in range(-4, 5, 2)]])
    assert distortion_rank(same) < distortion_rank(both)


def test_partner_multiplicity_still_reports_support():
    """Retained as a QUALITY measure -- it is not a gate."""
    hkl = np.array([[0, 1, l] for l in (-6, -4, -2, 2, 4)] + [[1, 0, -17]])
    mult = partner_multiplicity(hkl)
    assert min(mult[(0, 1)]) == 1


def test_box_rejects_a_nonsense_two_theta():
    with pytest.raises(ValueError):
        hkl_box_from_geometry(3.6, 19.2, wavelength_A=0.42, tth_max_deg=0.0)


def test_index_asymmetry_flags_a_sparse_column():
    """The systematic behind a fake splitting of consistent sign: in a limited
    wedge |k|>|h| outnumbered |h|>|k| 5:1, so 1/a^2 rides a sparse column and
    any radial systematic pushes a one way in EVERY domain."""
    from midas_hkls import index_asymmetry
    lopsided = np.array([[0, 1, 2]]*20 + [[0, 2, 4]]*15 + [[2, 0, 2]]*1)
    r = index_asymmetry(lopsided)
    assert r["n_k_gt_h"] > 5*r["n_h_gt_k"]
    assert r["sigma_ratio"] > 2, "a should be far worse determined than b here"
    balanced = np.array([[0, 1, 2]]*10 + [[1, 0, 2]]*10 + [[0, 2, 4]]*5 + [[2, 0, 4]]*5)
    rb = index_asymmetry(balanced)
    assert abs(rb["ratio"] - 1.0) < 0.01
    assert rb["sigma_ratio"] < 1.5


# ---------------------------------------------------------------- ab_sensitive_mask
# Restored 2026-09-09: three La3Ni2O7 analysis scripts imported this symbol and it
# had never been committed, so they could not run. The expectations below are the
# contract those call sites documented, not a fresh invention.

def test_ab_sensitive_mask_matches_the_documented_contract():
    """The exact case the nickelate call site printed and reasoned about."""
    from midas_hkls import ab_sensitive_mask
    hkl = np.array([[1, 0, 3], [0, 1, 3], [1, 1, 2], [1, -1, 2], [2, 1, 1], [1, 2, 1]])
    assert list(ab_sensitive_mask(hkl)) == [True, True, False, False, True, True]


def test_ab_sensitive_mask_requires_the_partner():
    """|h| != |k| is not enough on its own -- without the (k,h) partner a shift in
    |G| is degenerate with the scale."""
    from midas_hkls import ab_sensitive_mask
    lonely = np.array([[1, 0, 3], [2, 0, 1], [3, 0, 2]])       # no (0,h) anywhere
    assert not ab_sensitive_mask(lonely).any()
    paired = np.array([[1, 0, 3], [0, 1, 5]])                   # partner at another l
    assert ab_sensitive_mask(paired).all()


def test_ab_sensitive_mask_is_blind_to_the_gamma_shear():
    """h==k is never credited -- which is exactly why this mask must NOT be used
    to judge an Fmmm-supercell shear, whose splitting pair is (1,1)/(1,-1)."""
    from midas_hkls import ab_sensitive_mask
    shear_pair = np.array([[1, 1, 2], [1, -1, 2]])
    assert not ab_sensitive_mask(shear_pair).any()


def test_ab_sensitive_mask_agrees_with_partner_multiplicity():
    """The mask is the per-reflection form of partner_multiplicity: a reflection is
    sensitive iff its (|h|,|k|) pair appears in that dict."""
    from midas_hkls import ab_sensitive_mask, partner_multiplicity
    rng = np.random.default_rng(0)
    hkl = rng.integers(-3, 4, size=(200, 3))
    mask, pairs = ab_sensitive_mask(hkl), partner_multiplicity(hkl)
    for row, sens in zip(hkl, mask):
        x, y = abs(int(row[0])), abs(int(row[1]))
        assert sens == ((min(x, y), max(x, y)) in pairs)


def test_ab_sensitive_mask_edge_cases():
    from midas_hkls import ab_sensitive_mask
    assert ab_sensitive_mask(np.zeros((0, 3), dtype=int)).shape == (0,)
    assert not ab_sensitive_mask(np.array([[0, 0, 2]])).any()      # h==k==0
    assert ab_sensitive_mask(np.array([[2, -1, 0], [1, 2, 0]])).all()   # sign-insensitive


# ---------------------------------------------------------------- asymmetry_sign_test
# Added 2026-09-13: the raster-wide magnitude-only check in
# `06_raster_lattice_batch.ipynb` cannot tell a real crystal-to-crystal a/b trend from the
# SAME index-asymmetry artifact expressing itself with different severity per position --
# both produce a spread of |a-b|. This tests the mechanism `index_asymmetry` documents
# directly: whether sign(a-b) tracks which of h, k dominates that SAME domain's own
# reflections. The null (random, unrelated signs) must be able to pass, and a forced
# correlation (the documented mechanism, planted) must be caught -- both checked below.

def _lopsided_hkl(rng, h_dominant, n=20):
    """n reflections with |h|>|k| if h_dominant else |k|>|h|, random l."""
    lo = rng.integers(-15, 16, n)
    if h_dominant:
        return np.column_stack([np.full(n, 3), np.ones(n, int), lo])
    return np.column_stack([np.ones(n, int), np.full(n, 3), lo])


def test_asymmetry_sign_test_null_is_not_significant_when_signs_are_independent():
    """A null that cannot fail licenses nothing: domains whose a>b is UNRELATED to their own
    h/k dominance must not be flagged, however many of them there are."""
    from midas_hkls import asymmetry_sign_test
    rng = np.random.default_rng(0)
    domains = []
    for _ in range(60):
        h_dom = bool(rng.integers(0, 2))
        a_bigger = bool(rng.integers(0, 2))          # INDEPENDENT of h_dom, by construction
        a, b = (4.0, 3.9) if a_bigger else (3.9, 4.0)
        domains.append((_lopsided_hkl(rng, h_dom), a, b))
    r = asymmetry_sign_test(domains)
    assert r["n_used"] == 60
    assert r["p_value"] > 0.05, f"a genuinely independent null must not read as significant: {r}"
    assert 0.35 < r["agree_fraction"] < 0.65, r


def test_asymmetry_sign_test_catches_the_documented_mechanism_when_planted():
    """The positive control: force EXACTLY the mechanism index_asymmetry's docstring
    describes (h-dominant domains consistently fitted a>b) and confirm it is caught hard,
    not just noted."""
    from midas_hkls import asymmetry_sign_test
    rng = np.random.default_rng(1)
    domains = []
    for _ in range(40):
        h_dom = bool(rng.integers(0, 2))
        a, b = (4.0, 3.9) if h_dom else (3.9, 4.0)    # PLANTED: tracks h_dom every time
        domains.append((_lopsided_hkl(rng, h_dom), a, b))
    r = asymmetry_sign_test(domains)
    assert r["n_used"] == 40
    # planted: h-dominant domains get a>b, i.e. the MORE populous index's own axis comes out
    # LONGER -- "opposite sense" in the function's naming (see its docstring).
    assert r["direction"] == -1
    assert r["agree_fraction"] > 0.95
    assert r["p_value"] < 1e-6, f"a perfect planted correlation must be caught decisively: {r}"


def test_asymmetry_sign_test_excludes_uninformative_domains():
    """A tied index count or a == b carries no directional information -- must be excluded,
    not silently coerced into a sign."""
    from midas_hkls import asymmetry_sign_test
    tied_hkl = np.array([[2, 1, 0], [1, 2, 0]])       # one |h|>|k|, one |k|>|h|: tied
    r = asymmetry_sign_test([(tied_hkl, 4.0, 3.9), (np.array([[3, 1, 0]]), 4.0, 4.0)])
    assert r["n_used"] == 0
    assert r["direction"] == 0
    assert r["p_value"] != r["p_value"]               # NaN, not a fabricated number


def test_asymmetry_sign_test_is_a_two_sided_binomial_not_a_gaussian_approximation():
    """Small-N sanity: 3 of 3 agreeing should already read as somewhat unusual, without
    needing dozens of domains -- confirms the exact test isn't silently returning 1.0."""
    from midas_hkls import asymmetry_sign_test
    domains = [(_lopsided_hkl(np.random.default_rng(i), True), 4.0, 3.9) for i in range(3)]
    r = asymmetry_sign_test(domains)
    assert r["n_used"] == 3 and r["agree_fraction"] == 1.0
    assert r["p_value"] == pytest.approx(0.25, abs=1e-9)   # exact binomial: 2*(0.5)**3
