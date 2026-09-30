"""One grain, one sine: midas_pipeline.diagnostics.sine_consistency."""
import numpy as np
from midas_pipeline.diagnostics.sine_consistency import fit_one_sine, sine_consistency


def _sino(omegas, centres, n_scans=121, width=6):
    s = np.zeros((len(omegas), n_scans))
    for h, c in enumerate(centres):
        lo = int(round(c - width / 2)); s[h, max(lo, 0):max(lo, 0) + width] = 1.0
    return s


def test_one_grain_fits_one_sine():
    w = np.linspace(-180, 180, 90, endpoint=False)
    st, _ = fit_one_sine(_sino(w, 60 + 25 * np.sin(np.radians(w + 30))), w)
    assert st["resid_median"] < 1.0 and abs(st["amplitude"] - 25) < 1.0 and abs(st["centre"] - 60) < 1.0


def test_two_regions_of_one_orientation_are_flagged(tmp_path):
    w = np.linspace(-180, 180, 90, endpoint=False)
    c = np.where(np.abs(w) < 90, 60 + 25 * np.sin(np.radians(w + 30)), 60 + 25 * np.sin(np.radians(w + 150)))
    st, _ = fit_one_sine(_sino(w, c), w)
    assert st["resid_median"] > 5
    # stage-level wrapper: two grains, one good, one merged
    good = _sino(w, 60 + 20 * np.sin(np.radians(w)))
    S = np.stack([good, _sino(w, c)]); nH = S.shape[1]
    S.tofile(tmp_path / f"sinos_raw_2_{nH}_121.bin"); np.tile(w, (2, 1)).tofile(tmp_path / f"omegas_2_{nH}.bin")
    np.array([nH, nH], np.int32).tofile(tmp_path / "nrHKLs_2.bin")
    summ = sine_consistency(tmp_path)
    assert summ["flagged"] == [1]
    tab = np.loadtxt(tmp_path / "SineConsistency.csv", delimiter=",", skiprows=1)
    assert tab.shape == (2, 8) and tab[1, 7] == 1 and tab[0, 7] == 0


def test_too_few_rows_is_nan_not_flagged():
    st, _ = fit_one_sine(np.ones((3, 10)), np.array([0.0, 10, 20]))
    assert np.isnan(st["resid_median"])
