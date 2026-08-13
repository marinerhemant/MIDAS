"""The two branches, and the linearity rule that separates them."""

from __future__ import annotations

import numpy as np
import pytest

from midas_dt.branches import compare, format_comparison
from midas_dt.channels import Channel
from midas_dt.conventions import ADDITIVE_FIT_OUTPUTS, FIT_OUTPUT_NAMES, is_additive
from midas_dt.peakfit import fit_lineout


# ---------------------------------------------------------------- peakfit
def _peak(r, centre=118.0, width=3.0, amp=100.0, bg=5.0):
    return bg + amp * np.exp(-0.5 * ((r - centre) / width) ** 2)


def test_fit_recovers_a_known_peak_centre():
    r = np.linspace(105, 130, 100)
    f = fit_lineout(r, _peak(r, centre=118.0))
    assert f.get("RMEAN") == pytest.approx(118.0, abs=0.5)


def test_fit_returns_all_twelve_canonical_channels():
    r = np.linspace(105, 130, 100)
    f = fit_lineout(r, _peak(r))
    assert f.values.shape == (1, 12)
    assert set(f.as_dict()) == set(FIT_OUTPUT_NAMES)


def test_legacy_shared_width_makes_sigmag_equal_sigmal():
    """PeakFit.c sets both from one parameter; the default reproduces that."""
    r = np.linspace(105, 130, 100)
    f = fit_lineout(r, _peak(r), shared_width=True)
    assert f.get("SigmaG") == pytest.approx(f.get("SigmaL"))
    assert f.shared_width is True


def test_maxintensityobs_is_the_observed_max():
    r = np.linspace(105, 130, 100)
    y = _peak(r)
    f = fit_lineout(r, y)
    assert f.get("MaxIntensityObs") == pytest.approx(float(y.max()))


def test_totalintensity_is_the_raw_sum():
    r = np.linspace(105, 130, 100)
    y = _peak(r)
    assert fit_lineout(r, y).get("TotalIntensity") == pytest.approx(float(y.sum()))


def test_fit_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 5"):
        fit_lineout(np.arange(3.0), np.arange(3.0))


def test_fit_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match"):
        fit_lineout(np.arange(10.0), np.arange(9.0))


# -------------------------------------------------------------- linearity
def test_only_intensity_like_outputs_are_additive():
    """The rule Branch A depends on."""
    assert ADDITIVE_FIT_OUTPUTS == {
        "TotalIntensity", "TotalIntensityBackgroundCorr", "FitIntegratedIntensity",
    }
    for name in ("RMEAN", "SigmaG", "SigmaL", "MixFactor", "MaxInt"):
        assert not is_additive(name)


# ---------------------------------------------------------------- compare
def _result(branch, maps, linearity=None):
    from midas_dt.branches import BranchResult
    from midas_dt.conventions import ScanKnownLimits
    return BranchResult(
        maps=maps, branch=branch, channel=Channel(105, 125),
        limits=ScanKnownLimits(snake_corrected=True, omega_negated=True),
        linearity=linearity or {k: "exact" for k in maps},
    )


def test_compare_reports_zero_for_identical_maps():
    m = {"RMEAN": np.full((8, 8), 118.0)}
    stats = compare(_result("a", m), _result("b", {k: v.copy() for k, v in m.items()}))
    assert stats["RMEAN"]["rel_rms"] == pytest.approx(0.0)
    assert stats["RMEAN"]["n"] == 64


def test_compare_measures_a_real_discrepancy():
    a = {"RMEAN": np.full((8, 8), 120.0)}
    b = {"RMEAN": np.full((8, 8), 118.0)}
    stats = compare(_result("a", a), _result("b", b))
    assert stats["RMEAN"]["rel_rms"] == pytest.approx(2.0 / 118.0, rel=1e-6)


def test_compare_ignores_voxels_masked_in_either_branch():
    a = {"RMEAN": np.array([[118.0, np.nan], [118.0, 118.0]])}
    b = {"RMEAN": np.array([[118.0, 118.0], [np.nan, 118.0]])}
    assert compare(_result("a", a), _result("b", b))["RMEAN"]["n"] == 2


def test_compare_handles_no_overlap():
    a = {"RMEAN": np.full((2, 2), np.nan)}
    b = {"RMEAN": np.full((2, 2), 118.0)}
    stats = compare(_result("a", a), _result("b", b))
    assert stats["RMEAN"]["n"] == 0
    assert np.isnan(stats["RMEAN"]["rel_rms"])


def test_approximate_outputs_are_surfaced():
    r = _result("fit-then-recon[none]", {"RMEAN": np.zeros((4, 4))},
                linearity={"RMEAN": "approximate"})
    assert r.approximate_outputs() == ["RMEAN"]
    assert "APPROXIMATE" in r.describe()


def test_weighted_moment_is_not_marked_exact():
    r = _result("fit-then-recon[intensity]", {"RMEAN": np.zeros((4, 4))},
                linearity={"RMEAN": "weighted-moment"})
    assert r.approximate_outputs() == ["RMEAN"]


def test_format_comparison_carries_the_caveats():
    m = {"RMEAN": np.full((4, 4), 118.0)}
    a = _result("A", m, linearity={"RMEAN": "weighted-moment"})
    b = _result("B", {k: v.copy() for k, v in m.items()})
    text = format_comparison(compare(a, b), a, b)
    assert "rel_rms" in text
    assert "Approximate in at least one branch" in text
    assert "Self-absorption" in text     # limits travel with the result
