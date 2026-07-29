"""Tests for `midas_defect.burgers_population`.

Anchored to the deformed-Ti worked example of Dragomir & Ungár (2002) §5:
measured (q₁ᵐ, q₂ᵐ) = (−0.05, 0.18) ⇒ ⟨a⟩:⟨c⟩:⟨c+a⟩ = 75(10):20(5):5(3) %.
"""

from __future__ import annotations

import pytest

from midas_hkls.crystal import Atom, Crystal
from midas_hkls.lattice import Lattice
from midas_hkls.space_group import SpaceGroup

from midas_defect.contrast_factor_hex import hexagonal_stiffness
from midas_defect.burgers_population import (
    BURGERS_TYPES, burgers_magnitude_A, burgers_type_parameters,
    solve_burgers_population,
)


def _ti():
    cr = Crystal(
        lattice=Lattice(a=2.951, b=2.951, c=4.684, alpha=90.0, beta=90.0, gamma=120.0),
        space_group=SpaceGroup.from_number(194),
        atoms=[Atom(element="Ti", fract=(1/3, 2/3, 0.25), occupancy=1.0, label="Ti")])
    C6 = hexagonal_stiffness(c11=162.4, c12=92.0, c13=69.0, c33=180.7, c44=46.7)
    return C6, cr


@pytest.mark.unit
def test_burgers_magnitudes():
    _, cr = _ti()
    assert burgers_magnitude_A(cr, "a") == pytest.approx(2.951)
    assert burgers_magnitude_A(cr, "c") == pytest.approx(4.684)
    assert burgers_magnitude_A(cr, "c+a") == pytest.approx((2.951**2 + 4.684**2) ** 0.5)


@pytest.mark.slow
def test_type_parameters_cover_all_three_types():
    C6, cr = _ti()
    tp = burgers_type_parameters(C6, cr, n_phi=240)
    assert set(tp) == set(BURGERS_TYPES)
    # ⟨a⟩ averages 4 sub-slip-systems (BE, PrE, PyE, S1); S3 excluded from ⟨c⟩
    assert set(tp["a"].subsystems) == {"BE", "PrE", "PyE", "S1"}
    assert "S3" not in tp["c"].subsystems
    assert all(tp[t].cbar_hk0 > 0 for t in BURGERS_TYPES)


@pytest.mark.slow
def test_deformed_ti_burgers_population():
    """Reproduce the paper: (q₁ᵐ,q₂ᵐ)=(−0.05,0.18) ⇒ ~75/20/5 % ⟨a⟩/⟨c⟩/⟨c+a⟩."""
    C6, cr = _ti()
    tp = burgers_type_parameters(C6, cr, n_phi=240)
    pop = solve_burgers_population(-0.05, 0.18, tp)
    assert pop.all_nonnegative
    assert pop.dominant() == "a"
    # within the paper's stated experimental error (±10/±5/±3 %)
    assert pop.fractions["a"] == pytest.approx(0.75, abs=0.10)
    assert pop.fractions["c"] == pytest.approx(0.20, abs=0.05)
    assert pop.fractions["c+a"] == pytest.approx(0.05, abs=0.03)
    assert sum(pop.fractions.values()) == pytest.approx(1.0)
    assert pop.P_A2 > 0.0


@pytest.mark.unit
def test_solve_requires_all_three_types():
    from midas_defect.burgers_population import TypeContrastParameters
    bad = {"a": TypeContrastParameters("a", 0.2, -0.6, 0.1, 2.95, ("BE",))}
    with pytest.raises(ValueError, match="missing"):
        solve_burgers_population(-0.05, 0.18, bad)
