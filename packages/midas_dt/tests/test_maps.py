"""Detector radius -> d-spacing -> strain, and relative phase fractions."""

from __future__ import annotations

import numpy as np
import pytest

from midas_dt.branches import BranchResult
from midas_dt.channels import Channel
from midas_dt.conventions import ScanKnownLimits
from midas_dt.geometry import DTGeometry
from midas_dt.maps import (
    d_spacing_map,
    phase_fraction_map,
    radius_to_d_spacing,
    radius_to_two_theta,
    strain_map,
)

# The real 2022 U3O8 geometry.
GEO = DTGeometry(lsd_um=1071098.336, bc_y_px=790.3118888, bc_z_px=864.5394861,
                 px_um=172.0, n_pixels_y=1475, n_pixels_z=1679,
                 wavelength_a=0.136994)


def _result(maps, linearity=None, n_eta_channel=None):
    ch = n_eta_channel or Channel(105, 125, eta_min=-180, eta_max=180, eta_bin=360)
    return BranchResult(maps=maps, branch="test", channel=ch,
                        limits=ScanKnownLimits(snake_corrected=True, omega_negated=True),
                        linearity=linearity or {k: "exact" for k in maps})


# ------------------------------------------------------------- geometry
def test_two_theta_is_small_at_high_energy():
    """90.5 keV, Lsd 1.07 m: a 500 px radius is only a few degrees."""
    tt = float(radius_to_two_theta(500.0, GEO))
    assert 4.0 < tt < 5.5, tt


def test_d_spacing_round_trips_through_bragg():
    """d from radius, then radius back from d, must return the same radius."""
    r = 118.0
    d = float(radius_to_d_spacing(r, GEO))
    theta = np.arcsin(GEO.wavelength_a / (2 * d))
    r_back = np.tan(2 * theta) * GEO.lsd_um / GEO.px_um
    assert r_back == pytest.approx(r, rel=1e-9)


def test_larger_radius_is_smaller_d():
    """Bragg: higher angle, smaller spacing. Getting this backwards would
    invert every strain map."""
    assert radius_to_d_spacing(200.0, GEO) < radius_to_d_spacing(100.0, GEO)


def test_zero_radius_is_not_a_number():
    assert np.isnan(radius_to_d_spacing(0.0, GEO))


def test_d_spacing_is_physically_plausible_for_u3o8():
    """The 2022 runs fitted rings near 105-125 px; those must land in a
    sensible lattice-spacing range, not microns or picometres."""
    d = float(radius_to_d_spacing(118.0, GEO))
    assert 1.0 < d < 10.0, f"d = {d} A is not a crystal spacing"


# --------------------------------------------------------------- strain
def test_strain_is_zero_at_the_reference():
    d = float(radius_to_d_spacing(118.0, GEO))
    res = _result({"RMEAN": np.full((4, 4), 118.0)})
    sm = strain_map(res, GEO, d0_a=d)
    np.testing.assert_allclose(sm.strain, 0.0, atol=1e-12)


def test_strain_sign_follows_d():
    """Larger d than reference is tensile (positive)."""
    d0 = float(radius_to_d_spacing(118.0, GEO))
    res = _result({"RMEAN": np.full((2, 2), 117.0)})   # smaller radius -> larger d
    assert strain_map(res, GEO, d0_a=d0).strain.mean() > 0


def test_missing_d0_gives_a_relative_map_and_says_so(caplog):
    res = _result({"RMEAN": np.array([[117.0, 118.0], [119.0, 118.0]])})
    with caplog.at_level("WARNING"):
        sm = strain_map(res, GEO)
    assert "RELATIVE strain map" in caplog.text
    assert np.nanmedian(sm.strain) == pytest.approx(0.0, abs=1e-9)


def test_strain_map_states_it_is_not_a_tensor():
    res = _result({"RMEAN": np.full((2, 2), 118.0)})
    sm = strain_map(res, GEO, d0_a=2.0)
    assert sm.is_tensor is False
    assert any("not the tensor" in c for c in sm.caveats())
    assert any("single eta bin" in c for c in sm.caveats())


def test_multi_eta_channel_drops_the_single_bin_caveat():
    ch = Channel(105, 125, eta_min=-180, eta_max=180, eta_bin=45)   # 8 bins
    res = _result({"RMEAN": np.full((2, 2), 118.0)}, n_eta_channel=ch)
    sm = strain_map(res, GEO, d0_a=2.0)
    assert sm.n_eta_bins == 8
    assert not any("single eta bin" in c for c in sm.caveats())


def test_strain_rejects_a_nonpositive_reference():
    res = _result({"RMEAN": np.full((2, 2), 118.0)})
    with pytest.raises(ValueError, match="d0 must be positive"):
        strain_map(res, GEO, d0_a=0.0)


def test_strain_rejects_an_all_nan_map():
    res = _result({"RMEAN": np.full((2, 2), np.nan)})
    with pytest.raises(ValueError, match="no finite d-spacings"):
        strain_map(res, GEO)


def test_d_spacing_warns_when_the_input_was_back_projected_directly(caplog):
    res = _result({"RMEAN": np.full((2, 2), 118.0)},
                  linearity={"RMEAN": "approximate"})
    with caplog.at_level("WARNING"):
        d_spacing_map(res, GEO)
    assert "not a physically meaningful" in caplog.text


# -------------------------------------------------------- phase fractions
def test_phase_fractions_sum_to_one():
    a = _result({"TotalIntensityBackgroundCorr": np.full((3, 3), 30.0)})
    b = _result({"TotalIntensityBackgroundCorr": np.full((3, 3), 10.0)})
    frac = phase_fraction_map({"A": a, "B": b})
    np.testing.assert_allclose(frac["A"] + frac["B"], 1.0)
    np.testing.assert_allclose(frac["A"], 0.75)


def test_phase_fractions_reject_a_non_additive_weight():
    """RMEAN does not add along a ray, so it cannot weight a fraction."""
    a = _result({"RMEAN": np.ones((2, 2))})
    b = _result({"RMEAN": np.ones((2, 2))})
    with pytest.raises(ValueError, match="does not add along a ray"):
        phase_fraction_map({"A": a, "B": b}, output="RMEAN")


def test_phase_fractions_need_two_phases():
    a = _result({"TotalIntensityBackgroundCorr": np.ones((2, 2))})
    with pytest.raises(ValueError, match="at least 2 phases"):
        phase_fraction_map({"A": a})


def test_empty_voxels_are_nan_not_zero():
    a = _result({"TotalIntensityBackgroundCorr": np.array([[0.0, 5.0]])})
    b = _result({"TotalIntensityBackgroundCorr": np.array([[0.0, 5.0]])})
    frac = phase_fraction_map({"A": a, "B": b})
    assert np.isnan(frac["A"][0, 0])
    assert frac["A"][0, 1] == pytest.approx(0.5)
