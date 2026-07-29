import numpy as np
import pytest

from midas_defect.distributions import (
    friedel_pair_asymmetry,
    jensen_shannon_divergence,
    kl_divergence,
    mackenzie_pdf,
)
from midas_defect.types import CrystalPhase


# -- Mackenzie --------------------------------------------------------------

def test_mackenzie_cubic_zero_at_zero_angle():
    pdf = mackenzie_pdf(np.array([0.0]), phase=CrystalPhase.FCC)
    assert pdf[0] == pytest.approx(0.0, abs=1e-12)


def test_mackenzie_cubic_zero_outside_cutoff():
    # Cubic cutoff is ~62.8 deg.
    pdf = mackenzie_pdf(np.array([70.0, 80.0, 90.0]), phase=CrystalPhase.FCC)
    np.testing.assert_allclose(pdf, 0.0, atol=1e-12)


def test_mackenzie_cubic_integrates_to_unity():
    theta = np.linspace(0.0, 64.0, 4001)
    pdf = mackenzie_pdf(theta, phase=CrystalPhase.FCC)
    # pdf is per-radian; convert step to radians for the integral.
    integral = np.trapezoid(pdf, theta * np.pi / 180.0)
    assert integral == pytest.approx(1.0, rel=0.05)


def test_mackenzie_cubic_peak_near_45_deg():
    # Cubic Mackenzie peak is canonically near 45 deg.
    theta = np.linspace(0.0, 64.0, 641)
    pdf = mackenzie_pdf(theta, phase=CrystalPhase.FCC)
    peak_deg = float(theta[int(np.argmax(pdf))])
    assert 40.0 < peak_deg < 50.0


def test_mackenzie_hex_integrates_to_unity():
    theta = np.linspace(0.0, 95.0, 4001)
    pdf = mackenzie_pdf(theta, phase=CrystalPhase.HCP)
    integral = np.trapezoid(pdf, theta * np.pi / 180.0)
    assert integral == pytest.approx(1.0, rel=0.05)


def test_mackenzie_bcc_matches_fcc():
    theta = np.linspace(0.0, 62.0, 200)
    np.testing.assert_allclose(
        mackenzie_pdf(theta, phase=CrystalPhase.FCC),
        mackenzie_pdf(theta, phase=CrystalPhase.BCC),
    )


def test_mackenzie_hex_zero_outside_cutoff():
    pdf = mackenzie_pdf(np.array([100.0, 120.0]), phase=CrystalPhase.HCP)
    np.testing.assert_allclose(pdf, 0.0, atol=1e-12)


def test_mackenzie_hex_nonzero_below_cutoff():
    pdf = mackenzie_pdf(np.array([30.0, 60.0]), phase=CrystalPhase.HCP)
    assert (pdf > 0).all()


# -- Divergence -------------------------------------------------------------

def test_kl_divergence_zero_for_identical_distributions():
    centers = np.linspace(0, 60, 30)
    h = np.exp(-((centers - 30.0) / 8.0) ** 2)
    assert kl_divergence(h, h, centers) == pytest.approx(0.0, abs=1e-9)


def test_kl_divergence_strictly_positive_for_different():
    centers = np.linspace(0, 60, 30)
    p = np.exp(-((centers - 30.0) / 8.0) ** 2)
    q = np.exp(-((centers - 45.0) / 8.0) ** 2)
    assert kl_divergence(p, q, centers) > 0.0


def test_jensen_shannon_symmetric_and_bounded():
    centers = np.linspace(0, 60, 30)
    p = np.exp(-((centers - 30.0) / 8.0) ** 2)
    q = np.exp(-((centers - 45.0) / 8.0) ** 2)
    jpq = jensen_shannon_divergence(p, q, centers)
    jqp = jensen_shannon_divergence(q, p, centers)
    assert jpq == pytest.approx(jqp, abs=1e-12)
    assert 0.0 <= jpq <= np.log(2.0)


def test_kl_rejects_zero_mass_histogram():
    centers = np.linspace(0, 60, 30)
    h = np.zeros_like(centers)
    with pytest.raises(ValueError, match="zero total mass"):
        kl_divergence(h, h + 1, centers)


# -- Friedel ----------------------------------------------------------------

def test_friedel_perfectly_symmetric_pair_gives_zero_asymmetry():
    I = {
        (0, (1, 1, 1)): 100.0,
        (0, (-1, -1, -1)): 100.0,
    }
    out = friedel_pair_asymmetry(I)
    assert out["asymmetry_per_pair"].size == 1
    assert out["asymmetry_per_pair"][0] == pytest.approx(0.0, abs=1e-12)
    assert out["mean_asymmetry"] == 0.0


def test_friedel_pair_asymmetry_value():
    I = {
        (0, (1, 1, 1)): 100.0,
        (0, (-1, -1, -1)): 60.0,
        (1, (2, 0, 0)): 50.0,
        (1, (-2, 0, 0)): 50.0,
    }
    out = friedel_pair_asymmetry(I)
    assert out["asymmetry_per_pair"].shape == (2,)
    # Pair 0: |100-60|/(160) = 0.25; pair 1: 0.0.
    expected = sorted(out["asymmetry_per_pair"].tolist())
    assert expected[0] == pytest.approx(0.0)
    assert expected[1] == pytest.approx(0.25)
    assert out["mean_asymmetry"] == pytest.approx(0.125)


def test_friedel_unpaired_entries_skipped():
    I = {
        (0, (1, 1, 1)): 100.0,
        # no (-1,-1,-1) mate
        (1, (2, 2, 0)): 5.0,
        (1, (-2, -2, 0)): 4.0,
    }
    out = friedel_pair_asymmetry(I)
    assert out["asymmetry_per_pair"].size == 1


def test_friedel_empty_input_returns_nan_summary():
    out = friedel_pair_asymmetry({})
    assert np.isnan(out["mean_asymmetry"])
    assert np.isnan(out["median_asymmetry"])
    assert out["asymmetry_per_pair"].size == 0
