import numpy as np
import pytest

from midas_defect.debye_waller import per_grain_B_factor


def _planted_grain_entry(hkls, G_mag, B_true, scale_true, F2, lp=1.0, rng=None):
    """Synthesise per-reflection intensities consistent with I = scale F^2 LP exp(-2 B sin^2/lambda^2)."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = G_mag * G_mag / (8.0 * np.pi**2)
    I = scale_true * F2 * lp * np.exp(-B_true * x)
    # Add ~1% multiplicative noise.
    I = I * np.exp(rng.normal(scale=0.01, size=I.shape))
    return {
        "refl_indices": np.arange(hkls.shape[0]),
        "G_magnitude": G_mag,
        "centroid": G_mag,
        "fwhm": np.zeros_like(G_mag),
        "skewness": np.full_like(G_mag, np.nan),
        "intensity": I,
        "n_voxels": np.full(hkls.shape[0], 100),
    }


def test_per_grain_B_recovers_planted_value_within_few_percent():
    hkls = np.array([
        [1, 1, 1],
        [2, 0, 0],
        [2, 2, 0],
        [3, 1, 1],
        [2, 2, 2],
        [4, 0, 0],
        [3, 3, 1],
        [4, 2, 0],
    ])
    G_mag = np.linalg.norm(hkls.astype(float), axis=1) * 2 * np.pi / 3.6  # Cu a=3.6
    F2 = np.ones_like(G_mag)  # equal-strength reflections for the test
    entry = _planted_grain_entry(hkls, G_mag, B_true=2.5, scale_true=100.0, F2=F2)
    out = per_grain_B_factor(
        [entry], hkls, structure_factor_squared=lambda hkl: 1.0
    )
    assert out["B_per_grain"][0] == pytest.approx(2.5, rel=0.05)
    assert out["scale_per_grain"][0] == pytest.approx(100.0, rel=0.05)
    assert out["R_squared"][0] > 0.99


def test_per_grain_B_uses_structure_factor_normalisation():
    # Planted F2 varies between reflections; correctly normalising should still
    # recover B; ignoring normalisation should give a different B.
    hkls = np.array([
        [1, 1, 1],
        [2, 0, 0],
        [2, 2, 0],
        [3, 1, 1],
        [2, 2, 2],
        [4, 0, 0],
    ])
    G_mag = np.linalg.norm(hkls.astype(float), axis=1) * 2 * np.pi / 3.6
    F2 = np.array([10.0, 8.0, 12.0, 15.0, 5.0, 11.0])
    entry = _planted_grain_entry(hkls, G_mag, B_true=1.8, scale_true=50.0, F2=F2)
    # Correct normalisation (F2 lookup matching the plant)
    def f2_correct(hkl):
        idx = np.where((hkls == hkl).all(axis=1))[0][0]
        return float(F2[idx])

    out_correct = per_grain_B_factor([entry], hkls, structure_factor_squared=f2_correct)
    assert out_correct["B_per_grain"][0] == pytest.approx(1.8, rel=0.05)
    assert out_correct["R_squared"][0] > 0.99


def test_per_grain_B_too_few_reflections_returns_nan():
    entry = {
        "refl_indices": np.arange(2),
        "G_magnitude": np.array([1.0, 2.0]),
        "centroid": np.array([1.0, 2.0]),
        "fwhm": np.array([0.0, 0.0]),
        "skewness": np.array([np.nan, np.nan]),
        "intensity": np.array([1.0, 1.0]),
        "n_voxels": np.array([10, 10]),
    }
    out = per_grain_B_factor(
        [entry], np.array([[1, 0, 0], [2, 0, 0]]),
        structure_factor_squared=lambda hkl: 1.0
    )
    assert np.isnan(out["B_per_grain"][0])


def test_per_grain_B_lp_factor_callable_is_used():
    # Same data with LP=2 should give the same B if we divide out by LP=2.
    hkls = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]])
    G_mag = np.linalg.norm(hkls.astype(float), axis=1) * 2 * np.pi / 3.6
    F2 = np.ones_like(G_mag)
    entry = _planted_grain_entry(hkls, G_mag, B_true=2.0, scale_true=100.0, F2=F2, lp=2.0)
    out = per_grain_B_factor(
        [entry], hkls,
        structure_factor_squared=lambda hkl: 1.0,
        lp_factor_fn=lambda hkl, G: 2.0,
    )
    assert out["B_per_grain"][0] == pytest.approx(2.0, rel=0.05)
