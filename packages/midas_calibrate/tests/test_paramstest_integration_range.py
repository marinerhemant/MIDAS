"""A written paramstest must carry the radial range an integrator bins over.

`to_text()` emitted the bin *sizes* (RBinSize, EtaBinSize) but never the range
they span. midas_integrate_v2 defaults RMin and RMax to 0.0 and derives
nR = ceil((RMax - RMin) / RBinSize), so every file this writer produced from a
calibration that had no template range gave nR = 0 and died with
``ValueError: Invalid bins: nR=0``. Found 2026-08-21 on a 1-ID ge5 CeO2
calibration, where the geometry itself was good (6.87 microstrain) but the file
it wrote could not be handed to midas-integrate-v2 without hand-patching.

MaxRingRad had the same shape of bug in the other direction: parsed by
from_file and *required* by validate(), but never emitted, so a file this
writer produced failed to validate when read back.

Units matter here and are not uniform in this format: RMin/RMax/MaxRingRad are
in PIXELS (like RBinSize), while RhoD is in micrometres. The fallback therefore
divides by the pixel size.
"""
from __future__ import annotations

import math

import pytest

from midas_calibrate.params import CalibrationParams


def _params(**kw) -> CalibrationParams:
    """A calibration with a fitted geometry but no template binning range."""
    p = CalibrationParams()
    p.Lsd = 1246705.586748
    p.BC_y, p.BC_z = 1047.457903, 1017.053839
    p.pxY = p.pxZ = 200.0
    p.NrPixelsY = p.NrPixelsZ = 2048
    p.Wavelength = 0.136960
    p.SpaceGroup = 225
    p.LatticeConstant = (5.4116, 5.4116, 5.4116, 90.0, 90.0, 90.0)
    p.RhoD = 280638.08          # micrometres
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _keys(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out.setdefault(parts[0], parts[1])
    return out


def test_written_file_carries_a_radial_range():
    keys = _keys(_params().to_text())
    assert "RMin" in keys, "RMin missing: an integrator cannot bin without it"
    assert "RMax" in keys, "RMax missing: this is the nR=0 bug"
    assert float(keys["RMax"]) > float(keys["RMin"])


def test_range_falls_back_to_the_detector_edge_in_pixels():
    """RhoD is micrometres; the emitted range must be pixels."""
    p = _params()
    keys = _keys(p.to_text())
    assert float(keys["RMax"]) == pytest.approx(p.RhoD / p.pxY, abs=1e-6)
    # A whole-detector default, not a micrometre value that would be 200x too big
    assert float(keys["RMax"]) < p.NrPixelsY * 2


def test_the_written_file_actually_yields_positive_bins():
    """The regression, expressed the way the integrator computes it."""
    p = _params()
    keys = _keys(p.to_text())
    r_min, r_max = float(keys["RMin"]), float(keys["RMax"])
    r_bin = float(keys["RBinSize"])
    n_r = int(math.ceil((r_max - r_min) / r_bin))
    assert n_r > 0, f"nR={n_r}: file cannot be integrated"


def test_an_explicit_range_is_preserved_not_overwritten():
    p = _params(RMin=120.0, RMax=1401.0)
    keys = _keys(p.to_text())
    assert float(keys["RMin"]) == 120.0
    assert float(keys["RMax"]) == 1401.0


def test_range_round_trips_through_from_file(tmp_path):
    p = _params(RMin=120.0, RMax=1401.0)
    f = tmp_path / "paramstest.txt"
    p.write(f)
    back = CalibrationParams.from_file(f)
    assert back.RMin == 120.0
    assert back.RMax == 1401.0
    # and is not also duplicated into the pass-through extras
    assert "RMin" not in back.extra
    assert "RMax" not in back.extra


def test_maxringrad_survives_a_write_read_round_trip(tmp_path):
    """It is required by validate(); dropping it made written files invalid."""
    p = _params(MaxRingRad=1401.0)
    f = tmp_path / "paramstest.txt"
    p.write(f)
    back = CalibrationParams.from_file(f)
    assert back.MaxRingRad == 1401.0
    back.validate()          # would raise "MaxRingRad must be positive (px)"


def test_written_file_validates_even_with_no_explicit_maxringrad(tmp_path):
    f = tmp_path / "paramstest.txt"
    _params().write(f)
    CalibrationParams.from_file(f).validate()


def test_end_to_end_the_integrator_gets_positive_bins(tmp_path):
    """The regression through the real chain, not a local re-derivation.

    to_text() -> spec_from_v1_paramstest() -> IntegrationSpec.n_r_bins, which
    is what actually raised ``Invalid bins: nR=0``.
    """
    from_v1 = pytest.importorskip("midas_integrate_v2.compat.from_v1")

    f = tmp_path / "paramstest.txt"
    _params().write(f)
    spec = from_v1.spec_from_v1_paramstest(f)
    assert spec.RMax > spec.RMin
    assert spec.n_r_bins > 0
    assert spec.n_eta_bins > 0

    # and the old output -- the same file minus the lines this fix adds -- is
    # still shown to fail, so the test cannot pass for the wrong reason.
    stripped = "\n".join(
        ln for ln in f.read_text().splitlines()
        if ln.split()[0] not in ("RMin", "RMax", "MaxRingRad")
    ) + "\n"
    g = tmp_path / "old_style.txt"
    g.write_text(stripped)
    assert from_v1.spec_from_v1_paramstest(g).n_r_bins == 0
