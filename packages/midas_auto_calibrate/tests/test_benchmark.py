"""benchmark.py — pyFAI-optional harness.

Tests split into two tiers:
  * pure-Python primitives (_edges_to_centers, _pseudo_voigt, BenchmarkResult
    serialisation) — run everywhere, no pyFAI required.
  * end-to-end `benchmark()` against the bundled CeO2 image — requires
    both the MIDASCalibrant binary AND pyFAI installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_auto_calibrate import (
    BenchmarkResult,
    DetectorGeometry,
    MidasBinaryNotFoundError,
    benchmark,
    data as mac_data,
    midas_bin,
)
from midas_auto_calibrate.benchmark import (
    _edges_to_centers,
    _fit_ring_strain,
    _pseudo_voigt,
)


class TestPseudoVoigt:
    def test_peak_is_centered_and_normalized(self):
        x = np.linspace(-5, 5, 200)
        y = _pseudo_voigt(x, amp=10.0, center=0.0, fwhm=1.0, mixing=0.5, bg=0.0)
        # Max at center, no negative values (bg=0), Lorentzian tails > Gaussian.
        assert y[np.argmin(np.abs(x))] == pytest.approx(y.max(), rel=1e-3)
        assert (y >= 0).all()

    def test_bg_offsets_everywhere(self):
        x = np.linspace(-5, 5, 100)
        y0 = _pseudo_voigt(x, 1.0, 0.0, 1.0, 0.5, bg=0.0)
        y1 = _pseudo_voigt(x, 1.0, 0.0, 1.0, 0.5, bg=3.0)
        np.testing.assert_allclose(y1 - y0, 3.0)


class TestEdgesToCenters:
    def test_typical_case_edges_one_longer(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        centers = _edges_to_centers(edges, n_bins=3)
        np.testing.assert_allclose(centers, [0.5, 1.5, 2.5])

    def test_when_edges_already_centers(self):
        # Older pyFAI versions sometimes return n bins, not n+1 edges.
        centers_in = np.array([0.5, 1.5, 2.5])
        out = _edges_to_centers(centers_in, n_bins=3)
        np.testing.assert_allclose(out, centers_in)


class TestFitRingStrainEdgeCases:
    def test_returns_empty_when_mask_too_narrow(self):
        # Fake I_2d where no tth_centers fall within window_deg.
        I_2d = np.zeros((10, 50))
        tth_centers = np.linspace(10.0, 20.0, 50)

        from scipy.optimize import curve_fit
        strains = _fit_ring_strain(
            I_2d, tth_centers,
            tth_expected=100.0, d_expected=1.0,
            wavelength_A=0.172, curve_fit=curve_fit,
            window_deg=0.5,
        )
        assert strains.size == 0


class TestBenchmarkResult:
    def test_as_dict_with_only_midas_fields(self):
        result = BenchmarkResult(
            midas_pseudo_strain=4.2,
            midas_pseudo_strain_std=1.1,
            midas_seconds=12.3,
            midas_geometry=DetectorGeometry(),
        )
        d = result.as_dict()
        assert d == {"midas_ustrain": 4.2, "midas_seconds": 12.3}
        assert "pyfai_ustrain" not in d
        assert "speedup" not in d

    def test_as_dict_with_pyfai_fields_includes_ratios(self):
        result = BenchmarkResult(
            midas_pseudo_strain=4.2,
            midas_pseudo_strain_std=1.1,
            midas_seconds=12.3,
            midas_geometry=DetectorGeometry(),
            pyfai_pseudo_strain=127.0,
            pyfai_pseudo_strain_std=18.2,
            pyfai_seconds=76.2,
        )
        d = result.as_dict()
        assert d["pyfai_ustrain"] == 127.0
        assert d["accuracy_ratio"] == pytest.approx(127.0 / 4.2)
        assert d["speedup"] == pytest.approx(76.2 / 12.3)


# ---------------------------------------------------------------------------
# End-to-end — requires MIDASCalibrant binary + pyFAI.
# ---------------------------------------------------------------------------

def _has_pyfai() -> bool:
    try:
        import pyFAI  # noqa: F401
        return True
    except ImportError:
        return False


def _has_midas_binary() -> bool:
    try:
        midas_bin("MIDASCalibrant")
        return True
    except MidasBinaryNotFoundError:
        return False


@pytest.mark.skipif(
    not mac_data.CEO2_PILATUS.exists(),
    reason="Bundled CeO2 data missing.",
)
@pytest.mark.skipif(
    not _has_midas_binary(),
    reason="MIDASCalibrant not discoverable — set MIDAS_BIN.",
)
def test_benchmark_midas_only_no_pyfai(tmp_path):
    """Without include_pyfai=True pyFAI is never imported; runs anywhere."""
    import shutil
    image = tmp_path / "CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif"
    shutil.copy(mac_data.CEO2_PILATUS, image)
    shutil.copy(mac_data.CEO2_PILATUS_DARK,
                tmp_path / "dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif")
    shutil.copy(mac_data.CEO2_PILATUS_MASK, tmp_path / "mask_upd.tif")

    # Use a reduced iteration count to keep the test under 30 s.
    result = benchmark(
        image,
        material="CeO2",
        wavelength=0.172973,
        pixel_size=172.0,
        nr_pixels_y=1475,
        nr_pixels_z=1679,
        lsd=657_436.9, ybc=685.5, zbc=921.0,
        n_iterations=1,
        n_cpus=4,
        include_pyfai=False,
        work_dir=tmp_path,
    )
    assert result.midas_seconds > 0
    assert result.pyfai_pseudo_strain is None
    d = result.as_dict()
    assert "midas_ustrain" in d
    assert "pyfai_ustrain" not in d
