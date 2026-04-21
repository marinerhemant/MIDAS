"""CalibrationConfig + write_params_file primitives."""

from __future__ import annotations

import pytest

from midas_auto_calibrate import CalibrationConfig, write_params_file


class TestWriteParamsFile:
    def test_scalar_and_sequence(self, tmp_path):
        path = tmp_path / "P.txt"
        write_params_file(path, {
            "Lsd": 1_000_000.0,
            "BC": [1024.0, 1024.0],
            "ImTransOpt": (1, 3),
            "SpaceGroup": 225,
        })
        text = path.read_text()
        assert "Lsd 1000000.0\n" in text
        assert "BC 1024.0 1024.0\n" in text
        assert "ImTransOpt 1 3\n" in text
        assert "SpaceGroup 225\n" in text

    def test_none_value_skipped(self, tmp_path):
        path = tmp_path / "P.txt"
        write_params_file(path, {"Lsd": 1.0, "MaskFile": None})
        text = path.read_text()
        assert "Lsd" in text
        assert "MaskFile" not in text

    def test_bool_becomes_int(self, tmp_path):
        path = tmp_path / "P.txt"
        write_params_file(path, {"Normalize": True, "GradientCorrection": False})
        text = path.read_text()
        assert "Normalize 1\n" in text
        assert "GradientCorrection 0\n" in text

    def test_list_of_lists_repeats_key(self, tmp_path):
        # This mirrors AutoCalibrateZarr's "PeakLocation <r>\n" pattern where
        # the same key appears on many lines with different payloads.
        path = tmp_path / "P.txt"
        write_params_file(path, {
            "PeakLocation": [[45.12], [78.55], [120.8]],
        })
        lines = [l for l in path.read_text().splitlines() if l.startswith("PeakLocation")]
        assert lines == ["PeakLocation 45.12", "PeakLocation 78.55", "PeakLocation 120.8"]


class TestCalibrationConfig:
    def test_minimal_config_emits_required_keys(self):
        cfg = CalibrationConfig(
            material="CeO2",
            lattice_params=[5.4116, 5.4116, 5.4116, 90, 90, 90],
            wavelength=0.172979,
            pixel_size=200.0,
            nr_pixels_y=2048,
            nr_pixels_z=2048,
        )
        params = cfg.to_params()

        assert params["Wavelength"] == pytest.approx(0.172979)
        assert params["px"] == 200.0
        assert params["NrPixelsY"] == 2048
        assert params["SpaceGroup"] == 225
        assert params["LatticeConstant"] == [5.4116, 5.4116, 5.4116, 90, 90, 90]
        # BC packed as a list so write_params_file emits a single 2-value line.
        assert params["BC"] == [1024.0, 1024.0]

    def test_rmax_overrides_from_max_r_px(self):
        cfg = CalibrationConfig(
            wavelength=0.172979, pixel_size=200.0,
            nr_pixels_y=2048, nr_pixels_z=2048,
        )
        params = cfg.to_params(max_r_px=1200.3)
        # math.ceil(1200.3 + 50) = 1251
        assert params["RMax"] == 1251

    def test_rmax_absent_when_neither_provided(self):
        cfg = CalibrationConfig(
            wavelength=0.172979, pixel_size=200.0,
            nr_pixels_y=2048, nr_pixels_z=2048,
        )
        assert "RMax" not in cfg.to_params()

    def test_rmax_from_config_when_no_override(self):
        cfg = CalibrationConfig(
            wavelength=0.172979, pixel_size=200.0,
            nr_pixels_y=2048, nr_pixels_z=2048,
            r_max=2500.0,
        )
        assert cfg.to_params()["RMax"] == 2500.0

    def test_extra_merged_in(self):
        cfg = CalibrationConfig(wavelength=0.172979)
        params = cfg.to_params(extra={"DoPeakFit": 1, "FitROIPadding": 30})
        assert params["DoPeakFit"] == 1
        assert params["FitROIPadding"] == 30

    def test_end_to_end_writes_valid_parameters_txt(self, tmp_path):
        cfg = CalibrationConfig(
            material="CeO2",
            lattice_params=[5.4116, 5.4116, 5.4116, 90, 90, 90],
            wavelength=0.172979,
            pixel_size=200.0,
            lsd=1_000_000.0,
            ybc=1024.0, zbc=1024.0,
            nr_pixels_y=2048, nr_pixels_z=2048,
            im_trans_opt=[1, 3],
            folder=str(tmp_path),
        )
        path = tmp_path / "ps.txt"
        write_params_file(path, cfg.to_params(max_r_px=1200.0))

        text = path.read_text()
        for key in ("Lsd ", "BC ", "Wavelength ", "px ", "NrPixelsY ",
                    "NrPixelsZ ", "SpaceGroup ", "RMin ", "RMax ",
                    "RBinSize ", "EtaMin ", "EtaMax ", "EtaBinSize ",
                    "Normalize ", "GradientCorrection ", "ImTransOpt ",
                    "Folder ", "LatticeConstant "):
            assert key in text, f"missing key in output: {key}"
