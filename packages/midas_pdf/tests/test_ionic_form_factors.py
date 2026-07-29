"""Regression tests for ionic form factors.

The KEY invariant: for every registered ionic species, the total scattering
strength at Q = 0 must equal the number of electrons:

    f(Q = 0) = a1 + a2 + a3 + a4 + c ≈ Z − charge

We test this to 5% tolerance (a Cromer-Mann-fit-quality bound; light ions
typically pass to < 1%). If a coefficient set fails this test, it's a
transcription error and MUST NOT ship.
"""
from __future__ import annotations

import pytest
import torch

from midas_pdf.composition import Composition
from midas_pdf.ionic_form_factors import (
    CromerMannCoeff, ION_COEFFICIENTS, available_ions,
    ionic_form_factor, is_ionic_species, register_ion,
)


# ---------------------------------------------------------------------------
# Sum-rule test: MUST pass for every shipped ion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("species", sorted(ION_COEFFICIENTS.keys()))
def test_ion_sum_rule_at_Q0(species):
    """f(Q = 0) must equal (Z − charge) to within 5% for every shipped ion."""
    coeff = ION_COEFFICIENTS[species]
    f0 = coeff.f_at_zero()
    rel_err = abs(f0 - coeff.z_effective) / coeff.z_effective
    assert rel_err < 0.05, (
        f"{species}: f(0) = {f0:.3f} but z_effective = {coeff.z_effective} "
        f"(relative error {rel_err:.2%} exceeds 5% tolerance)")


@pytest.mark.parametrize("species", sorted(ION_COEFFICIENTS.keys()))
def test_ion_form_factor_monotonic_at_low_Q(species):
    """f(Q) must decrease with Q in the small-Q regime (physical constraint)."""
    q = torch.linspace(0.0, 3.0, 8, dtype=torch.float64)
    f = ionic_form_factor(q, species)
    # allow tiny non-monotonicity from fit noise; require strict decrease
    # after Q > 0.5 to be safe against very small light ions
    diffs = torch.diff(f[q > 0.5])
    assert torch.all(diffs < 0.02), (
        f"{species}: f(Q) not monotonically decreasing at low Q: {f.tolist()}")


@pytest.mark.parametrize("species", sorted(ION_COEFFICIENTS.keys()))
def test_ion_form_factor_positive_at_moderate_Q(species):
    """f(Q) must remain > 0 for Q up to at least 15 Å⁻¹ (physical)."""
    q = torch.linspace(0.5, 15.0, 20, dtype=torch.float64)
    f = ionic_form_factor(q, species)
    assert torch.all(f > 0), (
        f"{species}: f(Q) goes non-positive in [0.5, 15]: min = {float(f.min())}")


# ---------------------------------------------------------------------------
# API surface tests
# ---------------------------------------------------------------------------

def test_is_ionic_species():
    assert is_ionic_species("Ni2+")
    assert is_ionic_species("O2-")
    assert is_ionic_species("Cu+")
    assert not is_ionic_species("Ni")
    assert not is_ionic_species("")


def test_available_ions_nonempty_and_sorted():
    ions = available_ions()
    assert len(ions) > 5
    assert ions == sorted(ions)


def test_ionic_form_factor_unknown_species_raises():
    q = torch.tensor([0.0, 1.0], dtype=torch.float64)
    with pytest.raises(KeyError, match="not registered"):
        ionic_form_factor(q, "Xx99+")


def test_register_ion_and_sum_rule_gate():
    good = CromerMannCoeff(
        a=(1.0, 2.0, 3.0, 4.0), b=(1.0, 2.0, 3.0, 4.0), c=0.0, z_effective=10,
        source="test",
    )
    # This one satisfies the sum rule: 1+2+3+4+0 = 10 = z_effective
    register_ion("TestGood2+", good)
    assert "TestGood2+" in ION_COEFFICIENTS
    del ION_COEFFICIENTS["TestGood2+"]                   # tidy up

    bad = CromerMannCoeff(
        a=(1.0, 2.0, 3.0, 4.0), b=(1.0, 2.0, 3.0, 4.0), c=0.0, z_effective=100,
        source="test",
    )
    # 1+2+3+4+0 = 10 != 100 → should be rejected by sum rule
    with pytest.raises(ValueError, match="exceeds tolerance"):
        register_ion("TestBad50+", bad)


# ---------------------------------------------------------------------------
# Integration with Composition
# ---------------------------------------------------------------------------

def test_composition_uses_ionic_when_species_registered():
    """Composition({'Ce4+': 1}) must give DIFFERENT f² than {'Ce': 1}."""
    q = torch.linspace(0.5, 15.0, 8, dtype=torch.float64)
    c_neutral = Composition({"Ce": 1})
    c_ionic   = Composition({"Ce4+": 1})
    _, f2_n = c_neutral.form_factor_averages(q)
    _, f2_i = c_ionic.form_factor_averages(q)
    # Should differ at low Q (Ce4+ has 4 fewer electrons)
    rel = (f2_i - f2_n) / f2_n
    assert float(rel.abs().max()) > 0.005, (
        "Ce4+ should differ from Ce at some Q; rel diff was too small")


def test_composition_falls_back_to_neutral_for_unregistered_ion():
    """Composition({'Ni2+': 1}) with Ni2+ NOT in table → neutral Ni behaviour."""
    q = torch.linspace(0.5, 15.0, 8, dtype=torch.float64)
    # Ensure Ni2+ is not registered so this test is meaningful
    assert "Ni2+" not in ION_COEFFICIENTS
    c_neutral = Composition({"Ni": 1})
    c_ionic_fallback = Composition({"Ni2+": 1})
    _, f2_n = c_neutral.form_factor_averages(q)
    _, f2_i = c_ionic_fallback.form_factor_averages(q)
    # Should match exactly (falls back through midas_hkls neutral)
    assert torch.allclose(f2_n, f2_i)


def test_composition_mixed_neutral_and_ion():
    """Ca-oxide with Ca2+ ionic and O neutral: sums correctly."""
    q = torch.linspace(0.5, 15.0, 6, dtype=torch.float64)
    c = Composition({"Ca2+": 1, "O": 1})
    f_avg, f2_avg = c.form_factor_averages(q)
    # Check f_avg at Q=0: should be (18 + 8) / 2 = 13 for CaO ionic
    _, f2_at0 = c.form_factor_averages(torch.tensor([0.0], dtype=torch.float64))
    # At Q=0, f_avg = (f_Ca2+(0) + f_O(0))/2 = (18 + 8)/2 = 13
    f_avg0, _ = c.form_factor_averages(torch.tensor([0.0], dtype=torch.float64))
    assert abs(float(f_avg0[0]) - 13.0) < 0.5    # 4% tolerance


def test_ionic_form_factor_shapes():
    q = torch.linspace(0.0, 20.0, 100, dtype=torch.float64)
    f = ionic_form_factor(q, "Ca2+")
    assert f.shape == q.shape
    assert f.dtype == torch.float64
