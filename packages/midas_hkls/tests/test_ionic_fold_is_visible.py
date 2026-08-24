"""An ion that scatters like a neutral atom must say so.

``midas_hkls`` ships the IT92 **neutral-atom** table. Before 2026-08-22 an
ionic species string was accepted, stripped, and silently evaluated as the
neutral atom: ``O2-`` returned f(0) = 7.999 where the ion has 10 electrons,
and ``Li1+`` returned 3.000 where the ion has 2. Nothing said anything, so a
correctly-written CIF produced quietly wrong structure factors — the |F|^2 for
an NMC811 cell was bit-identical with and without ion labels.

Inventing coefficients would be worse than folding, because a wrong number
looks right. So the fold stays, and becomes loud; real coefficients arrive
through :func:`register_ion`, which enforces the one check that catches a
transcription error without a second source — f(0) *is* the electron count.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_hkls.form_factors import (
    _FOLD_WARNED,
    _ION_TABLE,
    coefficients,
    form_factor,
    register_ion,
    registered_ions,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts from the shipped state (no ions, nothing warned)."""
    saved_ions = dict(_ION_TABLE)
    saved_warned = set(_FOLD_WARNED)
    _ION_TABLE.clear()
    _FOLD_WARNED.clear()
    yield
    _ION_TABLE.clear()
    _ION_TABLE.update(saved_ions)
    _FOLD_WARNED.clear()
    _FOLD_WARNED.update(saved_warned)


def test_neutral_species_do_not_warn():
    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error")
        coefficients("Ni")
        coefficients("O")


@pytest.mark.parametrize("species,neutral_z,ion_e", [
    ("O2-", 8, 10), ("Li1+", 3, 2), ("Ni2+", 28, 26), ("Fe(III)", 26, 23),
])
def test_ionic_fold_warns_and_names_the_error(species, neutral_z, ion_e):
    with pytest.warns(RuntimeWarning, match="NEUTRAL"):
        a, b, c = coefficients(species)
    f0 = float(a.sum() + c)
    assert f0 == pytest.approx(neutral_z, abs=0.05), (
        "the fold must still return the neutral atom — the warning is the fix, "
        "not a behaviour change"
    )
    assert f0 != pytest.approx(ion_e, abs=0.05)


def test_the_warning_fires_once_per_species_not_once_per_call():
    """A per-call warning in a structure-factor loop is a million lines."""
    import warnings as w
    with w.catch_warnings(record=True) as rec:
        w.simplefilter("always")
        for _ in range(50):
            coefficients("O2-")
    assert len(rec) == 1, f"expected 1 warning, got {len(rec)}"


def test_registering_an_ion_silences_the_fold_and_changes_f0():
    """Round trip with a synthetic but sum-rule-valid entry."""
    # O2- has 10 electrons; build coefficients that sum to exactly that.
    a = np.array([3.0, 3.0, 2.0, 1.0])
    b = np.array([13.0, 5.9, 0.6, 32.0])
    c = 1.0
    assert a.sum() + c == 10.0
    register_ion("O2-", a, b, c, source="synthetic-for-test")
    assert "O2-" in registered_ions()

    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error")          # must NOT warn any more
        got_a, got_b, got_c = coefficients("O2-")
    assert float(got_a.sum() + got_c) == pytest.approx(10.0)
    assert float(np.asarray(form_factor(np.array([0.0]), "O2-")).ravel()[0]) \
        == pytest.approx(10.0)


def test_register_ion_enforces_the_electron_count_sum_rule():
    """The check that catches a mistranscribed row without a second source."""
    bad_a = np.array([3.0, 3.0, 2.0, 1.0])       # sums to 9 + c
    with pytest.raises(ValueError, match="sum rule"):
        register_ion("O2-", bad_a, np.ones(4), 5.0)   # f(0)=14, want 10


def test_register_ion_rejects_a_neutral_species():
    with pytest.raises(ValueError, match="no charge"):
        register_ion("Ni", np.ones(4) * 7.0, np.ones(4), 0.0)


def test_register_ion_rejects_wrong_coefficient_count():
    with pytest.raises(ValueError, match="4 a and 4 b"):
        register_ion("O2-", np.ones(3), np.ones(3), 1.0)


def test_charge_parsing():
    from midas_hkls.form_factors import _parse_species
    assert _parse_species("Ni") == ("Ni", 0)
    assert _parse_species("Ni2+") == ("Ni", 2)
    assert _parse_species("O2-") == ("O", -2)
    assert _parse_species("Li1+") == ("Li", 1)
    assert _parse_species("Na+") == ("Na", 1)
    assert _parse_species("Cl-") == ("Cl", -1)
    assert _parse_species("Fe(III)") == ("Fe", 3)


def test_midas_pdf_publishes_its_ions_into_this_registry():
    """The dependency runs pdf -> hkls, so the data is pushed, not copied."""
    pdf = pytest.importorskip("midas_pdf.ionic_form_factors")
    n = pdf.publish_to_midas_hkls()
    assert n > 0, "midas_pdf shipped ions but none registered"
    for species in pdf.available_ions():
        if species in registered_ions():
            import warnings as w
            with w.catch_warnings():
                w.simplefilter("error")   # a published ion must not warn
                coefficients(species)
            break
    else:  # pragma: no cover
        pytest.fail("no midas_pdf ion reached the midas_hkls registry")
