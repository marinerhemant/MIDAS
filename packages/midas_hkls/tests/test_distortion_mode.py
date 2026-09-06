"""Which distortion a cell setting can express.

The central test is the IDENTITY that made the RP finding a proof rather than
an estimate: with gamma = 90 a diagonal subcell metric gives (110) and (-110)
exactly equal d for ANY a and b. If that ever stops holding numerically, a
diagonal B-matrix has silently become able to fit a gamma shear.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from midas_hkls.distortion_mode import (
    supercell_to_subcell, subcell_to_supercell, supercell_hkl_to_subcell,
    splits_under, diagonal_B_can_express, can_distinguish_modes,
)

# La3Ni2O7 S5, the collaborators' hand index on the Fmmm supercell
A_SUP, B_SUP, C = 5.2739, 5.2384, 20.5


def test_supercell_orthorhombicity_is_a_gamma_shear():
    s = supercell_to_subcell(A_SUP, B_SUP)
    assert s.a == pytest.approx(3.7167, abs=1e-3)
    assert s.b == pytest.approx(s.a, abs=1e-12)          # metrically tetragonal
    assert s.gamma_deg == pytest.approx(89.613, abs=1e-2)
    assert s.delta_supercell == pytest.approx(0.00338, abs=2e-5)


def test_round_trip_supercell_subcell():
    s = supercell_to_subcell(A_SUP, B_SUP)
    A2, B2 = subcell_to_supercell(s.a, s.gamma_deg)
    assert A2 == pytest.approx(A_SUP, rel=1e-12)
    assert B2 == pytest.approx(B_SUP, rel=1e-12)


@pytest.mark.parametrize("a,b", [(3.0, 4.2), (3.6116, 3.6116), (3.5, 3.9),
                                 (2.5, 5.0), (4.0, 4.0001)])
def test_THE_IDENTITY_diagonal_metric_cannot_split_110(a, b):
    """|G(110)| == |G(-110)| EXACTLY at gamma = 90, for any a and b."""
    q1 = 1.0 / a ** 2 + 1.0 / b ** 2
    assert splits_under((1, 1, 0), (-1, 1, 0), a=a, b=b, c=C,
                        gamma_deg=90.0) is False
    # and state it as the closed form, so the reason is visible
    from midas_hkls.distortion_mode import _d_star_sq
    assert _d_star_sq(1, 1, 0, a, b, C, 90.0) == pytest.approx(q1, rel=1e-15)
    assert _d_star_sq(-1, 1, 0, a, b, C, 90.0) == pytest.approx(q1, rel=1e-15)


def test_the_two_modes_are_complementary():
    a = 3.7167
    # a != b, gamma = 90: (100)/(010) splits, (110)/(-110) does NOT
    assert splits_under((1, 0, 0), (0, 1, 0), a=3.70, b=3.74, c=C, gamma_deg=90.0)
    assert not splits_under((1, 1, 0), (-1, 1, 0), a=3.70, b=3.74, c=C, gamma_deg=90.0)
    # a == b, gamma != 90: the reverse
    assert not splits_under((1, 0, 0), (0, 1, 0), a=a, b=a, c=C, gamma_deg=89.613)
    assert splits_under((1, 1, 0), (-1, 1, 0), a=a, b=a, c=C, gamma_deg=89.613)


def test_supercell_200_020_maps_to_subcell_110():
    assert supercell_hkl_to_subcell(2, 0, 0) == (1.0, 1.0, 0)
    assert supercell_hkl_to_subcell(0, 2, 0) == (-1.0, 1.0, 0)
    assert supercell_hkl_to_subcell(2, 2, 0) == (0.0, 2.0, 0)
    assert supercell_hkl_to_subcell(-2, 2, 0) == (-2.0, 0.0, 0)


def test_diagonal_B_cannot_express_a_gamma_shear():
    assert diagonal_B_can_express("a_ne_b") is True
    assert diagonal_B_can_express("gamma_shear") is False
    with pytest.raises(ValueError):
        diagonal_B_can_express("triclinic")


def test_2604_index_cannot_distinguish_the_modes():
    """The real case: (00L),(11L),(21L),(22L) has gamma-sensitive, not a/b."""
    hkls = [(0, 0, 4), (1, 1, 3), (2, 1, 5), (2, 2, 0), (1, 1, 9)]
    v = can_distinguish_modes(hkls)
    assert v["has_gamma_sensitive"] is True
    assert v["has_ab_sensitive"] is False
    assert v["can_distinguish"] is False
    assert "indistinguishable" in v["note"]


def test_adding_the_missing_family_makes_it_distinguishable():
    hkls = [(0, 0, 4), (1, 1, 3), (2, 2, 0), (1, 0, 5), (0, 1, 5)]
    v = can_distinguish_modes(hkls)
    assert v["can_distinguish"] is True
    assert "both families" in v["note"]


def test_supercell_conversion_rejects_implausible_axes():
    """cos gamma is always in range, so the guard must be on PLAUSIBILITY."""
    with pytest.raises(ValueError, match="not a supercell pair"):
        supercell_to_subcell(10.0, 1.0)              # a 79 deg shear
    with pytest.raises(ValueError, match="positive"):
        supercell_to_subcell(-1.0, 5.0)
    # ...but it converts fine when explicitly allowed
    s = supercell_to_subcell(10.0, 1.0, max_shear_deg=90.0)
    assert 0.0 < s.gamma_deg < 90.0


def test_cos_gamma_is_always_in_domain():
    """Pin the closed form, so nobody re-adds an unreachable domain guard."""
    for A, B in [(10.0, 1.0), (1.0, 10.0), (5.0, 5.0), (1e3, 1e-3)]:
        cg = (A * A - B * B) / (A * A + B * B)
        assert -1.0 < cg < 1.0
