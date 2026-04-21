"""Progressive (multi-stage) calibration tests.

Pure-Python unit tests run everywhere. End-to-end tests against the
bundled Pilatus CeO₂ + MIDAS Example Varex Ceria are skipped without
the MIDASCalibrant binary.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from midas_auto_calibrate import (
    CalibrationConfig,
    MidasBinaryNotFoundError,
    calibrate_progressive,
    data as mac_data,
    midas_bin,
)
from midas_auto_calibrate.progressive import (
    DEFAULT_FIT_P_MODELS,
    ProgressiveResult,
    _apply_autocal_defaults,
    _normalize_models,
    _p_seeds,
    _stage1_config,
    _stage2_config,
)
from midas_auto_calibrate import DetectorGeometry


class TestNormalizeModels:
    def test_default_all_expands(self):
        m = _normalize_models("all")
        assert "dipole" in m
        assert "trefoil" in m
        assert "pentafoil5" in m
        assert "hexafoil6" in m

    def test_csv_string(self):
        m = _normalize_models("tilt,spherical,dipole")
        assert m == frozenset({"tilt", "spherical", "dipole"})

    def test_iterable(self):
        m = _normalize_models(["tilt", "spherical"])
        assert m == frozenset({"tilt", "spherical"})

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _normalize_models("")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            _normalize_models("tilt,ninefold")


class TestPSeeds:
    def test_all_mode_seeds_all_harmonics(self):
        modes = _normalize_models("all")
        s = _p_seeds(modes)
        assert s[0] == pytest.approx(1e-4)     # spherical amplitude
        assert s[3] == 0.0                     # phase starts 0
        assert s[7] == pytest.approx(1e-4)     # dipole amplitude
        assert s[8] == pytest.approx(45.0)     # dipole phase
        assert s[13] == pytest.approx(1e-4)    # hexafoil amp
        assert s[14] == pytest.approx(45.0)    # hexafoil phase

    def test_only_tilt_seeds_nothing(self):
        s = _p_seeds(_normalize_models("tilt"))
        assert all(v == 0.0 for v in s.values())

    def test_spherical_only_seeds_p0_p5(self):
        s = _p_seeds(_normalize_models("spherical"))
        assert s[0] == pytest.approx(1e-4)
        assert s[7] == 0.0      # dipole not active


class TestStageConfigs:
    def _base_cfg(self):
        return CalibrationConfig(
            material="CeO2", wavelength=0.17, pixel_size=172,
            lsd=1_000_000, ybc=100, zbc=100, rho_d=200_000,
            nr_pixels_y=512, nr_pixels_z=512,
            extra_params={"tolTilts": 3, "NPanelsY": 6, "NPanelsZ": 8,
                          "p0": 0.999, "tolBC": 20},
        )

    def test_stage1_locks_distortion(self):
        s1 = _stage1_config(self._base_cfg())
        assert s1.extra_params["tolP"] == 0
        assert s1.extra_params["tolP4"] == 0
        # p0 user-override stripped from extra_params (Stage 1 runs
        # monolithic with p=0 regardless of user's initial value).
        assert "p0" not in s1.extra_params

    def test_stage1_strips_panels(self):
        s1 = _stage1_config(self._base_cfg())
        for key in ("NPanelsY", "NPanelsZ", "PanelShiftsFile"):
            assert key not in s1.extra_params

    def test_stage1_preserves_user_tolTilts(self):
        s1 = _stage1_config(self._base_cfg())
        assert s1.extra_params["tolTilts"] == 3

    def test_stage2_seeds_p7_to_p14_when_all(self):
        geom = DetectorGeometry(lsd=1e6, ybc=100, zbc=100, tx=0, ty=0.1, tz=0.2)
        s2 = _stage2_config(
            self._base_cfg(), geom, _normalize_models("all")
        )
        assert s2.extra_params["p7"] == pytest.approx(1e-4)
        assert s2.extra_params["p8"] == pytest.approx(45.0)
        assert s2.extra_params["p13"] == pytest.approx(1e-4)
        assert s2.extra_params["p14"] == pytest.approx(45.0)

    def test_stage2_carries_stage1_tilts(self):
        geom = DetectorGeometry(lsd=1_234_567, ybc=200, zbc=300,
                                 tx=0, ty=0.05, tz=-0.1)
        s2 = _stage2_config(
            self._base_cfg(), geom, _normalize_models("all")
        )
        assert s2.lsd == pytest.approx(1_234_567)
        assert s2.ybc == pytest.approx(200)
        assert s2.ty == pytest.approx(0.05)

    def test_stage2_respects_user_p_override(self):
        cfg = self._base_cfg()
        cfg.extra_params["p0"] = 5.5e-5   # user-picked seed
        s2 = _stage2_config(cfg, DetectorGeometry(), _normalize_models("all"))
        assert s2.extra_params["p0"] == pytest.approx(5.5e-5)

    def test_autocal_defaults_include_per_p_tolerances(self):
        extra = {}
        _apply_autocal_defaults(extra)
        assert extra["tolP3"] == 45          # phase, degrees
        assert extra["tolP8"] == 180         # phase, degrees
        assert extra["tolP7"] == pytest.approx(1e-3)  # dipole amplitude

    def test_autocal_defaults_respect_user_values(self):
        extra = {"tolP7": 9e-9, "DoubletSeparation": 99}
        _apply_autocal_defaults(extra)
        assert extra["tolP7"] == 9e-9
        assert extra["DoubletSeparation"] == 99


def _binary_available() -> bool:
    try:
        midas_bin("MIDASCalibrant")
        midas_bin("GetHKLList")
        return True
    except MidasBinaryNotFoundError:
        return False


needs_binary = pytest.mark.skipif(
    not _binary_available(),
    reason="MIDASCalibrant + GetHKLList not discoverable.",
)


@needs_binary
@pytest.mark.skipif(
    not mac_data.CEO2_PILATUS.exists(),
    reason="Bundled CeO2 data missing.",
)
def test_progressive_converges_on_bundled_pilatus(tmp_path):
    """Progressive calibration converges below 200 µε on bundled Pilatus.

    AutoCalibrateZarr with TPS spline hits ~17 µε; without the spline
    (which we don't implement in MVP progressive), we plateau around
    70-100 µε depending on initial tilts. The <200 µε threshold protects
    against algorithmic regressions.
    """
    img = tmp_path / "CeO2_00001.tif"
    shutil.copy(mac_data.CEO2_PILATUS, img)
    shutil.copy(mac_data.CEO2_PILATUS_DARK, tmp_path / "dark.tif")
    shutil.copy(mac_data.CEO2_PILATUS_MASK, tmp_path / "mask_upd.tif")

    cfg = CalibrationConfig(
        material="CeO2", lattice_params=(5.4116,) * 3 + (90,) * 3,
        wavelength=0.172973, pixel_size=172.0,
        lsd=657_436.9, ybc=685.485, zbc=921.034,
        rho_d=219964.42411013643,
        nr_pixels_y=1475, nr_pixels_z=1679,
        dark_file="dark.tif", mask_file="mask_upd.tif", im_trans_opt=[2],
        extra_params={
            "Width": 1000, "OmegaStart": -180, "OmegaStep": 0.25,
            "tolTilts": 3, "tolBC": 20, "tolLsd": 15000,
            "OutlierIterations": 3, "MultFactor": 5,
            "NormalizeRingWeights": 1, "WeightByRadius": 1,
            "WeightByFitSNR": 1, "L2Objective": 1,
            "MaxRingNumber": 14,
            # Pilatus mosaic — must be in extra_params for now (panels
            # aren't structured fields in CalibrationConfig).
            "NPanelsY": 6, "NPanelsZ": 8,
            "PanelSizeY": 243, "PanelSizeZ": 195,
            "PanelGapsY": [1, 7, 1, 7, 1],
            "PanelGapsZ": [17, 17, 17, 17, 17, 17, 17],
            "FixPanelID": 12, "tolShifts": 1, "tolRotation": 3,
            "PerPanelLsd": 1, "PerPanelDistortion": 1,
            "PanelShiftsFile": "panelshiftsCalibrant.txt",
            "RingsToExclude": [[n] for n in range(19, 34)],
        },
    )
    result = calibrate_progressive(
        cfg, img, work_dir=tmp_path,
        fit_p_models="all",
        n_iterations_stage1=5, n_iterations_stage2=5, n_cpus=2,
    )
    assert isinstance(result, ProgressiveResult)
    assert len(result.stages) == 2
    assert result.stages[0][0] == "stage1_geometry"
    assert result.stages[1][0] == "stage2_distortion"
    # Stage 2 should be < 200 µε (vs AutoCalibrateZarr's ~17 µε with TPS).
    assert 0 < result.pseudo_strain < 200, (
        f"Progressive Pilatus residual {result.pseudo_strain:.1f} µε "
        f"is outside the expected [0, 200) window. "
        f"Stage 1: {result.stages[0][1].pseudo_strain:.1f} µε"
    )


_VAREX_TIF = (
    Path(__file__).resolve().parents[3]
    / "FF_HEDM" / "Example" / "Calibration"
    / "Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif"
)


@needs_binary
@pytest.mark.skipif(
    not _VAREX_TIF.exists(),
    reason="Varex Ceria dataset missing.",
)
def test_progressive_converges_on_varex(tmp_path):
    """Progressive calibration converges below 100 µε on the Varex Ceria frame.

    Uses the converged-BC seed (matches AutoCalibrateZarr's Hough output).
    Auto-Hough BC-finder is deferred to v0.2 but users can seed manually.
    """
    shutil.copy(_VAREX_TIF, tmp_path / _VAREX_TIF.name)
    cfg = CalibrationConfig(
        material="CeO2", lattice_params=(5.4116,) * 3 + (90,) * 3,
        wavelength=0.196793, pixel_size=150.0,
        lsd=895_930, ybc=1446.97, zbc=1468.91,
        rho_d=309094.28,
        nr_pixels_y=2880, nr_pixels_z=2880,
        im_trans_opt=[2], r_max=1440,
        extra_params={
            "Width": 1000, "OmegaStart": -180, "OmegaStep": 0.25,
            "tolTilts": 3, "tolBC": 20, "tolLsd": 25000,
            "OutlierIterations": 3, "MultFactor": 5,
            "NormalizeRingWeights": 1, "WeightByRadius": 1,
            "WeightByFitSNR": 1, "L2Objective": 1,
            "MaxRingNumber": 14,
        },
    )
    result = calibrate_progressive(
        cfg, tmp_path / _VAREX_TIF.name, work_dir=tmp_path,
        fit_p_models="all",
        n_iterations_stage1=5, n_iterations_stage2=5, n_cpus=2,
    )
    assert 0 < result.pseudo_strain < 100, (
        f"Varex progressive residual {result.pseudo_strain:.1f} µε "
        f"should be < 100 (AutoCalibrateZarr hits ~4 µε with TPS spline)."
    )
    # Higher-order p's should have been seeded and refined, not left at zero.
    g = result.geometry
    active_p_count = sum(
        1 for i in range(7, 15) if abs(getattr(g, f"p{i}")) > 1e-8
    )
    assert active_p_count >= 4, (
        f"Expected at least 4 of p7..p14 non-zero (dipole/trefoil/pentafoil5/"
        f"hexafoil6 active); got {active_p_count}"
    )
