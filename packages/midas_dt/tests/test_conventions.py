"""The conventions that silently produce wrong answers when got wrong."""

from __future__ import annotations

import numpy as np
import pytest

from midas_dt.conventions import (
    ADDITIVE_FIT_OUTPUTS,
    FIT_OUTPUT_NAMES,
    RECON_SIGN,
    U3O8_ENERGY_KEV,
    ScanKnownLimits,
    aps_1id_omega,
    fit_output_index,
    is_additive,
    recon_size,
    unsnake,
)


# ------------------------------------------------------------- fit outputs
def test_canonical_channel_order_matches_the_c():
    """Pinned against IntegratorPeakFitOMP.c valTypes[] and PeakFit.c Rfit[].

    Both agree, which is what makes this canonical. Every legacy Python script
    omits MaxIntensityObs from slot 5 and shifts everything after it.
    """
    assert FIT_OUTPUT_NAMES == (
        "RMEAN", "MixFactor", "SigmaG", "SigmaL", "MaxInt", "MaxIntensityObs",
        "BGFit", "BGSimple", "MeanError", "FitIntegratedIntensity",
        "TotalIntensity", "TotalIntensityBackgroundCorr",
    )
    assert len(FIT_OUTPUT_NAMES) == 12


def test_slot_5_is_maxintensityobs_not_bgfit():
    """The single index the legacy scripts get wrong."""
    assert fit_output_index("MaxIntensityObs") == 5
    assert fit_output_index("BGFit") == 6
    # PeakFit.c: Rfit[9] = CalcIntegratedIntensity(...)
    assert fit_output_index("FitIntegratedIntensity") == 9


def test_unknown_output_name_lists_the_valid_ones():
    with pytest.raises(KeyError, match="RMEAN"):
        fit_output_index("Rcen")


@pytest.mark.parametrize("name", sorted(ADDITIVE_FIT_OUTPUTS))
def test_additive_outputs_are_intensity_like(name):
    assert is_additive(name)


@pytest.mark.parametrize("name", ["RMEAN", "SigmaG", "SigmaL", "MixFactor", "MaxInt"])
def test_shape_parameters_are_not_additive(name):
    """These must not be back-projected directly.

    A projection's fitted RMEAN is the intensity-weighted mean along the ray,
    not the sum, so Radon inversion of it has no physical meaning.
    """
    assert not is_additive(name)


def test_is_additive_validates_the_name():
    with pytest.raises(KeyError):
        is_additive("NotAnOutput")


# --------------------------------------------------------------- recon sign
def test_recon_sign_is_negative():
    """recon_peak_all_mul.py negates before fitting; doLog 0 back-projects
    intensity, so gridrec returns a negative-going image."""
    assert RECON_SIGN == -1.0


# -------------------------------------------------------------------- omega
def test_omega_is_negated_by_default():
    """1-ID aerotech turns the opposite way. Standing site rule."""
    nominal = np.array([180.25, 180.0, 179.75])
    np.testing.assert_allclose(aps_1id_omega(nominal), [-180.25, -180.0, -179.75])


def test_omega_negation_can_be_disabled_explicitly():
    nominal = np.array([0.0, 1.0])
    np.testing.assert_allclose(aps_1id_omega(nominal, negate=False), nominal)


# -------------------------------------------------------------------- snake
def test_unsnake_reverses_only_odd_rows():
    data = np.array([[0, 1, 2, 3],
                     [0, 1, 2, 3],
                     [0, 1, 2, 3],
                     [0, 1, 2, 3]])
    out = unsnake(data, axis=0, frame_axis=1)
    np.testing.assert_array_equal(out[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(out[1], [3, 2, 1, 0])
    np.testing.assert_array_equal(out[2], [0, 1, 2, 3])
    np.testing.assert_array_equal(out[3], [3, 2, 1, 0])


def test_unsnake_is_its_own_inverse():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(6, 9))
    np.testing.assert_allclose(unsnake(unsnake(data)), data)


def test_unsnake_does_not_mutate_its_input():
    data = np.arange(12).reshape(3, 4)
    before = data.copy()
    unsnake(data)
    np.testing.assert_array_equal(data, before)


# ----------------------------------------------------------------- geometry
def test_recon_size_matches_the_2023_runs():
    """55 translations with ExtraPadForTomo 1 gave reconSize 128."""
    assert recon_size(55, extra_pad=True) == 128
    assert recon_size(55, extra_pad=False) == 64


@pytest.mark.parametrize("n,expected", [(1, 2), (32, 64), (33, 128), (64, 128)])
def test_recon_size_powers_of_two(n, expected):
    assert recon_size(n, extra_pad=True) == expected


def test_recon_size_rejects_nonpositive():
    with pytest.raises(ValueError, match="must be positive"):
        recon_size(0)


def test_u3o8_energy_is_90_5_not_the_commented_55_6():
    """The parameter files comment 55.618 keV next to lambda = 0.136994 A.

    12.398 / 0.136994 = 90.5 keV. The comment is stale; the wavelength is what
    the geometry was refined with. Confirmed with the beamtime owner.
    """
    assert U3O8_ENERGY_KEV == pytest.approx(90.5, abs=0.1)
    assert 12.398419843320026 / 0.136994 == pytest.approx(90.5, abs=0.1)


# ------------------------------------------------------------- known limits
def test_uncorrected_limits_are_reported():
    limits = ScanKnownLimits(snake_corrected=True, omega_negated=True)
    warnings = limits.warnings()
    assert any("Self-absorption" in w for w in warnings)
    assert any("Texture" in w for w in warnings)
    assert not any("Omega" in w for w in warnings)


def test_missing_omega_negation_is_flagged():
    limits = ScanKnownLimits(snake_corrected=True, omega_negated=False)
    assert any("mirrors the reconstruction" in w for w in limits.warnings())
