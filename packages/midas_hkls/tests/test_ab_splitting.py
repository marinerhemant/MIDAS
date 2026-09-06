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
