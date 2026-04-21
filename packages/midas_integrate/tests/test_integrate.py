"""Integrator + make_zarr_zip + peakfit — pure-Python tier + end-to-end.

The end-to-end MIDASIntegrator run depends on proper zarr.zip prep and
a pre-built Map.bin; test_mapper already exercises Map.bin end-to-end.
Here we verify the wiring around Integrator + the pure-Python fit utilities.
"""

from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest

from midas_integrate import (
    IntegrationConfig,
    IntegrationResult,
    Integrator,
    Mapper,
    MapArtifacts,
    fit_peaks_1d,
    make_zarr_zip,
    pseudo_voigt,
)


# ---------------------------------------------------------------------------
# pseudo_voigt + fit_peaks_1d — pure-Python, no binaries.
# ---------------------------------------------------------------------------

class TestPseudoVoigt:
    def test_peak_is_centred(self):
        x = np.linspace(-5, 5, 200)
        y = pseudo_voigt(x, amp=10, center=0, fwhm=1.0, mixing=0.5, bg=0.0)
        # Peak is at x=0.
        assert abs(x[int(np.argmax(y))]) < x[1] - x[0]

    def test_bg_offsets(self):
        x = np.linspace(-5, 5, 50)
        y0 = pseudo_voigt(x, 1.0, 0.0, 1.0, 0.5, 0.0)
        y1 = pseudo_voigt(x, 1.0, 0.0, 1.0, 0.5, 7.5)
        np.testing.assert_allclose(y1 - y0, 7.5)


class TestFitPeaks1D:
    def test_recovers_synthetic_peak(self):
        rng = np.random.default_rng(0)
        r = np.linspace(0, 100, 1000)
        # Two clean peaks + a bit of Gaussian noise.
        intensity = (
            pseudo_voigt(r, 50.0, 20.0, 1.5, 0.4, 2.0)
            + pseudo_voigt(r, 30.0, 70.0, 2.0, 0.6, 0.0)
            + rng.normal(0, 0.1, len(r))
        )
        fits = fit_peaks_1d(r, intensity, initial_centers=[20.0, 70.0], window=5.0)
        assert len(fits) == 2
        assert fits[0]["center"] == pytest.approx(20.0, abs=0.05)
        assert fits[0]["fwhm"] == pytest.approx(1.5, rel=0.1)
        assert fits[1]["center"] == pytest.approx(70.0, abs=0.05)

    def test_empty_window_returns_nan(self):
        r = np.array([0.0, 1.0])         # too few points
        intensity = np.array([0.0, 0.0])
        fits = fit_peaks_1d(r, intensity, initial_centers=[10.0])
        assert len(fits) == 1
        assert np.isnan(fits[0]["center"])
        assert fits[0]["initial_center"] == 10.0


# ---------------------------------------------------------------------------
# make_zarr_zip — MIDAS-schema round-trip.
# ---------------------------------------------------------------------------

class TestMakeZarrZip:
    def _cfg(self):
        return IntegrationConfig(
            lsd=1_000_000, ybc=64, zbc=64,
            wavelength=0.172979, pixel_size=200.0,
            nr_pixels_y=128, nr_pixels_z=128,
            r_bin_size=0.25, eta_bin_size=1.0, r_max=50.0,
        )

    def test_writes_zip_with_required_entries(self, tmp_path):
        img = np.random.default_rng(0).integers(0, 1000, (128, 128)).astype(np.float32)
        out = tmp_path / "bundle.zarr.zip"
        make_zarr_zip(img, self._cfg(), out, chunk_y=64, chunk_z=64)
        assert out.exists()

        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        # Must carry exchange/data and the minimal analysis_parameters tree.
        assert any("exchange/data/.zarray" in n for n in names), names
        for key in ("Wavelength", "Lsd", "RMin", "RMax",
                    "RBinSize", "EtaBinSize", "PixelSize"):
            assert any(
                f"analysis/process/analysis_parameters/{key}/" in n for n in names
            ), f"missing parameter {key}"

    def test_extra_params_are_written(self, tmp_path):
        img = np.zeros((128, 128), dtype=np.float32)
        out = tmp_path / "bundle.zarr.zip"
        make_zarr_zip(img, self._cfg(), out, extra_params={
            "DoPeakFit": 1,
            "FitROIPadding": 30,
        })
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert any("analysis_parameters/DoPeakFit/" in n for n in names)
        assert any("analysis_parameters/FitROIPadding/" in n for n in names)

    def test_dark_subtraction(self, tmp_path):
        img = np.full((128, 128), 10, dtype=np.float32)
        dark = np.full((128, 128), 3, dtype=np.float32)
        out = tmp_path / "bundle.zarr.zip"
        make_zarr_zip(img, self._cfg(), out, dark=dark, chunk_y=64, chunk_z=64)

        # Pull the zarr back out and verify the subtraction.
        import zarr
        with zipfile.ZipFile(out) as zf:
            zf.extractall(tmp_path / "extracted")
        root = zarr.open(str(tmp_path / "extracted"), mode="r")
        data = root["exchange/data"][:]
        assert data.shape == (1, 128, 128)
        # 10 - 3 = 7
        assert float(data[0, 0, 0]) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Integrator dispatch — fake binary, no MIDASIntegrator needed.
# ---------------------------------------------------------------------------

def _fake_integrator(bin_dir: Path, *, emit_cake: bool, exit_code: int = 0) -> Path:
    """Fake MIDASIntegrator that touches the expected output file."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / "MIDASIntegrator"
    script = ["#!/bin/sh", 'STEM="${1%.*}"']
    if emit_cake:
        script.append('touch "${STEM}.caked.hdf"')
    script.append(f"exit {exit_code}")
    exe.write_text("\n".join(script) + "\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def _fake_map_artifacts(tmp_path: Path) -> MapArtifacts:
    mb = tmp_path / "Map.bin"
    nb = tmp_path / "nMap.bin"
    mb.write_bytes(b"\x00" * 512)
    nb.write_bytes(b"\x00" * 512)
    return MapArtifacts(work_dir=tmp_path, map_bin=mb, n_map_bin=nb)


class TestIntegratorDispatch:
    def test_validates_backend_arg(self):
        cfg = IntegrationConfig(nr_pixels_y=64, nr_pixels_z=64)
        with pytest.raises(ValueError, match="backend must be"):
            Integrator(cfg, MapArtifacts(Path("."), Path("."), Path(".")),
                       backend="xpu")

    def test_invokes_binary_and_returns_cake_path(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "fake_bin"
        _fake_integrator(bin_dir, emit_cake=True)

        monkeypatch.delenv("MIDAS_BIN", raising=False)
        monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        workdir = tmp_path / "work"
        workdir.mkdir()
        (workdir / "bundle.zarr.zip").write_bytes(b"fakezip")
        artifacts = _fake_map_artifacts(workdir)

        integ = Integrator(IntegrationConfig(nr_pixels_y=64, nr_pixels_z=64),
                           artifacts)
        result = integ.integrate(workdir / "bundle.zarr.zip", n_cpus=2,
                                 bin_dir=bin_dir)
        assert isinstance(result, IntegrationResult)
        # Path.stem of "bundle.zarr.zip" → "bundle.zarr", so the expected
        # .caked.hdf name preserves the .zarr suffix. This matches the
        # filenames MIDAS produces in practice.
        assert result.cake_path == workdir / "bundle.zarr.caked.hdf"
        assert result.backend == "cpu"

    def test_missing_cake_raises(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "fake_bin"
        _fake_integrator(bin_dir, emit_cake=False, exit_code=0)

        monkeypatch.delenv("MIDAS_BIN", raising=False)
        monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        workdir = tmp_path / "work"
        workdir.mkdir()
        (workdir / "bundle.zarr.zip").write_bytes(b"fakezip")
        artifacts = _fake_map_artifacts(workdir)

        integ = Integrator(IntegrationConfig(nr_pixels_y=64, nr_pixels_z=64),
                           artifacts)
        with pytest.raises(RuntimeError, match="no .caked.hdf was produced"):
            integ.integrate(workdir / "bundle.zarr.zip", bin_dir=bin_dir)

    def test_gpu_backend_raises_helpful_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIDAS_BIN", raising=False)
        monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        workdir = tmp_path
        (workdir / "bundle.zarr.zip").write_bytes(b"fakezip")
        artifacts = _fake_map_artifacts(workdir)

        integ = Integrator(IntegrationConfig(nr_pixels_y=64, nr_pixels_z=64),
                           artifacts, backend="gpu")
        with pytest.raises(RuntimeError, match="midas-integrate-gpu wheel"):
            integ.integrate(workdir / "bundle.zarr.zip")
