"""calibrate.py — stdout parser + filename splitter + subprocess orchestration.

End-to-end tests that actually run ``MIDASCalibrant`` against a bundled
CeO2 image land in ``test_calibrate_ceo2.py`` (Week 4, skipped unless
``MIDAS_CEO2_DATA`` env points at the bundled fixture).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from textwrap import dedent

import pytest

from midas_auto_calibrate import CalibrationConfig, auto_calibrate, run_calibration
from midas_auto_calibrate.calibrate import (
    _load_convergence_history,
    _parse_final_geometry,
    _split_numbered_filename,
)


class TestSplitNumberedFilename:
    def test_six_digit_padding(self):
        stem, num, pad, ext = _split_numbered_filename(Path("CeO2_000042.h5"))
        assert stem == "CeO2"
        assert num == 42
        assert pad == 6
        assert ext == "h5"

    def test_four_digit_padding(self):
        stem, num, pad, ext = _split_numbered_filename(Path("/abs/path/Ni_0001.tif"))
        assert (stem, num, pad, ext) == ("Ni", 1, 4, "tif")

    def test_stem_with_underscores(self):
        stem, num, pad, ext = _split_numbered_filename(
            Path("Ceria_63keV_900mm_000007.edf"))
        assert stem == "Ceria_63keV_900mm"
        assert num == 7
        assert pad == 6
        assert ext == "edf"

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _split_numbered_filename(Path("calibrant.tif"))


# ---------------------------------------------------------------------------
# Fixture: a representative CalibrantIntegratorOMP stdout dump. Every field
# we parse out must appear here; re-paste the real binary's output if the
# format changes.
# ---------------------------------------------------------------------------
_FAKE_STDOUT = dedent("""
    CalibrantIntegratorOMP Version: midas-auto-calibrate v0.1.0
    Number of planes being considered: 12.
    Iterations: 30, RBinWidth: 1, EtaBinSize: 1.00
    ...
    Iteration 30/30 MeanStrain 5.123 ppm
    Mean Values
    Lsd 1002345.678
    BC 1021.44 1029.27
    ty -0.00123
    tz 0.05432
    p0 1.23e-06
    p1 -2.34e-07
    p2 3.45e-08
    p3 0.0
    p4 4.56e-09
    p5 0.0
    p6 0.0
    p7 0.0
    p8 0.0
    p9 0.0
    p10 0.0
    p11 0.0
    p12 0.0
    p13 0.0
    p14 0.0
    RhoD 180123.456
    MeanStrain 4.87
    StdStrain 1.23
    Copy to parameter file:
    """).strip()


class TestParseFinalGeometry:
    def test_extracts_all_geometry_fields(self):
        cfg = CalibrationConfig(
            wavelength=0.172979, pixel_size=200.0,
            nr_pixels_y=2048, nr_pixels_z=2048,
        )
        geom = _parse_final_geometry(_FAKE_STDOUT, cfg)

        assert geom.lsd == pytest.approx(1_002_345.678)
        assert geom.ybc == pytest.approx(1021.44)
        assert geom.zbc == pytest.approx(1029.27)
        assert geom.ty == pytest.approx(-0.00123)
        assert geom.tz == pytest.approx(0.05432)
        assert geom.p0 == pytest.approx(1.23e-06)
        assert geom.p4 == pytest.approx(4.56e-09)
        assert geom.rhod == pytest.approx(180_123.456)
        assert geom.mean_strain == pytest.approx(4.87)
        assert geom.std_strain == pytest.approx(1.23)
        # Defaults carried through from config
        assert geom.wavelength == pytest.approx(0.172979)
        assert geom.px == pytest.approx(200.0)
        assert geom.nr_pixels_y == 2048

    def test_uses_last_block_when_multiple(self):
        # The binary prints multiple "Mean Values" blocks during iteration;
        # we must pick the last one.
        first = _FAKE_STDOUT.replace("Lsd 1002345.678", "Lsd 999000.0")
        stdout = first + "\n" + _FAKE_STDOUT
        cfg = CalibrationConfig(wavelength=0.1)
        geom = _parse_final_geometry(stdout, cfg)
        assert geom.lsd == pytest.approx(1_002_345.678)

    def test_missing_mean_values_returns_config_defaults(self):
        cfg = CalibrationConfig(lsd=777.0, wavelength=0.1)
        geom = _parse_final_geometry("no geometry here", cfg)
        assert geom.lsd == pytest.approx(777.0)


class TestLoadConvergenceHistory:
    def test_parses_csv_with_typed_columns(self, tmp_path):
        csv = tmp_path / "hist.csv"
        csv.write_text(
            "Iter,MeanStrain_ppm,StdStrain_ppm,Lsd,ybc,zbc\n"
            "0,25.4,3.2,1000000.0,1024.0,1024.0\n"
            "1,8.7,1.9,1000123.4,1021.5,1029.2\n"
        )
        rows = _load_convergence_history(csv)
        assert rows is not None
        assert len(rows) == 2
        assert rows[0]["MeanStrain_ppm"] == pytest.approx(25.4)
        assert rows[1]["Lsd"] == pytest.approx(1_000_123.4)
        # Iter should be int-castable even though we store as float for simplicity.
        assert rows[0]["Iter"] == 0.0

    def test_missing_file_returns_none(self, tmp_path):
        assert _load_convergence_history(tmp_path / "nope.csv") is None


# ---------------------------------------------------------------------------
# End-to-end dispatch — uses a fake binary to verify we build the right
# argv + Parameters.txt + that the stdout parser hooks up.
# ---------------------------------------------------------------------------

def _write_fake_binary(bin_dir: Path, stdout_to_emit: str, exit_code: int = 0) -> Path:
    """Create a minimal executable that prints a fixed stdout and exits.

    Uses a /bin/sh heredoc so the fake binary doesn't depend on
    ``python3`` being on PATH (we empty PATH in the test to force binary
    discovery to go through the explicit ``bin_dir`` argument).
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = bin_dir / "_stdout.txt"
    stdout_path.write_text(stdout_to_emit)
    exe = bin_dir / "MIDASCalibrant"
    exe.write_text(
        "#!/bin/sh\n"
        f"cat {stdout_path}\n"
        f"exit {exit_code}\n"
    )
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return exe


class TestRunCalibrationDispatch:
    def test_invokes_binary_and_parses_geometry(self, tmp_path, monkeypatch):
        # Fake data file so file-existence check passes.
        (tmp_path / "CeO2_000001.h5").write_text("not real h5")
        fake_bin_dir = tmp_path / "bin"
        _write_fake_binary(fake_bin_dir, _FAKE_STDOUT, exit_code=0)

        # Isolate binary discovery to our fake.
        monkeypatch.delenv("MIDAS_BIN", raising=False)
        monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
        # Keep /usr/bin + /bin on PATH so shell utilities (cat, sh) are
        # found, but remove anything that could shadow MIDASCalibrant.
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        cfg = CalibrationConfig(
            material="CeO2",
            lattice_params=[5.4116, 5.4116, 5.4116, 90, 90, 90],
            wavelength=0.172979,
            pixel_size=200.0,
            nr_pixels_y=2048, nr_pixels_z=2048,
        )
        result = run_calibration(
            cfg,
            data_file=tmp_path / "CeO2_000001.h5",
            work_dir=tmp_path,
            n_cpus=1,
            bin_dir=fake_bin_dir,
        )

        # Parameters.txt was written and contains our config.
        params_text = (tmp_path / "Parameters.txt").read_text()
        assert "Wavelength 0.172979" in params_text
        assert "FileStem CeO2" in params_text
        assert "StartNr 1" in params_text
        assert "Padding 6" in params_text
        assert "Ext .h5" in params_text   # dot-prefixed per MIDAS convention

        # Geometry parsed.
        assert result.geometry.lsd == pytest.approx(1_002_345.678)
        assert result.pseudo_strain == pytest.approx(4.87)
        assert result.pseudo_strain_std == pytest.approx(1.23)
        assert result.stdout == _FAKE_STDOUT
        assert (tmp_path / "calibrant.stdout").exists()

    def test_exit_code_nonzero_raises_by_default(self, tmp_path, monkeypatch):
        (tmp_path / "X_0001.h5").write_text("stub")
        fake_bin_dir = tmp_path / "bin"
        _write_fake_binary(fake_bin_dir, _FAKE_STDOUT, exit_code=3)

        monkeypatch.delenv("MIDAS_BIN", raising=False)
        monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
        # Keep /usr/bin + /bin on PATH so shell utilities (cat, sh) are
        # found, but remove anything that could shadow MIDASCalibrant.
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        cfg = CalibrationConfig(wavelength=0.172979)
        with pytest.raises(RuntimeError, match="MIDASCalibrant exited 3"):
            run_calibration(
                cfg,
                data_file=tmp_path / "X_0001.h5",
                work_dir=tmp_path,
                bin_dir=fake_bin_dir,
            )

    def test_check_false_returns_result_on_failure(self, tmp_path, monkeypatch):
        (tmp_path / "X_0001.h5").write_text("stub")
        fake_bin_dir = tmp_path / "bin"
        _write_fake_binary(fake_bin_dir, _FAKE_STDOUT, exit_code=1)

        monkeypatch.delenv("MIDAS_BIN", raising=False)
        monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
        # Keep /usr/bin + /bin on PATH so shell utilities (cat, sh) are
        # found, but remove anything that could shadow MIDASCalibrant.
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        cfg = CalibrationConfig(wavelength=0.172979)
        result = run_calibration(
            cfg,
            data_file=tmp_path / "X_0001.h5",
            work_dir=tmp_path,
            bin_dir=fake_bin_dir,
            check=False,
        )
        # Still got a result object; can inspect stdout for debugging.
        assert result.stdout == _FAKE_STDOUT

    def test_auto_calibrate_is_alias(self):
        # Ergonomic top-level alias documented in the README.
        assert auto_calibrate is run_calibration
