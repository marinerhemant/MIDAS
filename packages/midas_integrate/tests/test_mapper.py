"""Mapper.build() — invokes MIDASDetectorMapper, parses header."""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

from midas_integrate import (
    IntegrationConfig,
    MapArtifacts,
    Mapper,
    MidasBinaryNotFoundError,
    midas_bin,
)
from midas_integrate.mapper import _maybe_read_header


def _has_midas_bin() -> bool:
    try:
        midas_bin("MIDASDetectorMapper")
        return True
    except MidasBinaryNotFoundError:
        return False


needs_binary = pytest.mark.skipif(
    not _has_midas_bin(),
    reason="MIDASDetectorMapper not discoverable — set MIDAS_BIN.",
)


# ---------------------------------------------------------------------------
# Pure-Python: header parser, Mapper.build dispatch with a fake binary.
# ---------------------------------------------------------------------------

def test_header_parser_returns_none_for_missing_magic(tmp_path):
    p = tmp_path / "nothing.bin"
    p.write_bytes(b"\x00" * 512)
    assert _maybe_read_header(p) is None


def test_header_parser_returns_none_for_tiny_file(tmp_path):
    p = tmp_path / "tiny.bin"
    p.write_bytes(b"short")
    assert _maybe_read_header(p) is None


def _fake_mapper(bin_dir: Path, *, emit_map: bool, emit_nmap: bool,
                 exit_code: int = 0) -> Path:
    """Fake MIDASDetectorMapper that writes the requested output files.

    The fake executes ``touch Map.bin`` / ``touch nMap.bin`` when asked
    so we can exercise Mapper.build() without the real C binary.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = ["#!/bin/sh"]
    if emit_map:
        script.append("touch Map.bin")
    if emit_nmap:
        script.append("touch nMap.bin")
    script.append(f"exit {exit_code}")
    exe = bin_dir / "MIDASDetectorMapper"
    exe.write_text("\n".join(script) + "\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def _minimal_cfg():
    return IntegrationConfig(
        lsd=600_000, ybc=512, zbc=512,
        wavelength=0.17, pixel_size=172.0,
        nr_pixels_y=1024, nr_pixels_z=1024,
    )


class TestMapperDispatch:
    def test_writes_params_file_and_invokes_binary(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "fake_bin"
        _fake_mapper(bin_dir, emit_map=True, emit_nmap=True)

        monkeypatch.delenv("MIDAS_BIN", raising=False)
        monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        workdir = tmp_path / "work"
        mapper = Mapper(_minimal_cfg())
        artifacts = mapper.build(workdir, n_cpus=2, bin_dir=bin_dir)

        assert isinstance(artifacts, MapArtifacts)
        assert artifacts.map_bin.exists()
        assert artifacts.n_map_bin.exists()
        # Parameters.txt written in the work dir with the keys we emit.
        params = (workdir / "Mapper.Parameters.txt").read_text()
        assert "Lsd 600000" in params
        assert "BC 512 512" in params
        assert "Wavelength 0.17" in params

    def test_nonzero_exit_raises(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "fake_bin"
        _fake_mapper(bin_dir, emit_map=False, emit_nmap=False, exit_code=7)

        monkeypatch.delenv("MIDAS_BIN", raising=False)
        monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        with pytest.raises(RuntimeError, match="MIDASDetectorMapper exited 7"):
            Mapper(_minimal_cfg()).build(tmp_path / "work", bin_dir=bin_dir)

    def test_missing_map_bin_raises(self, tmp_path, monkeypatch):
        """Binary exited OK but didn't produce Map.bin — likely a silent
        parameter-file issue; surface it explicitly."""
        bin_dir = tmp_path / "fake_bin"
        _fake_mapper(bin_dir, emit_map=False, emit_nmap=True, exit_code=0)

        monkeypatch.delenv("MIDAS_BIN", raising=False)
        monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        with pytest.raises(RuntimeError, match="Map.bin was not produced"):
            Mapper(_minimal_cfg()).build(tmp_path / "work", bin_dir=bin_dir)


class TestFromGeometry:
    def test_shortcut_matches_explicit_construction(self):
        from midas_auto_calibrate import DetectorGeometry

        geom = DetectorGeometry(
            lsd=600_000, ybc=512, zbc=512, wavelength=0.17,
            px=172.0, nr_pixels_y=1024, nr_pixels_z=1024,
        )
        m1 = Mapper.from_geometry(geom, r_max=800)
        assert m1.config.lsd == pytest.approx(600_000)
        assert m1.config.r_max == 800


# ---------------------------------------------------------------------------
# End-to-end against the real MIDASDetectorMapper binary.
# ---------------------------------------------------------------------------

@needs_binary
def test_real_mapper_produces_map_bin(tmp_path):
    """Real MIDASDetectorMapper invocation with a symmetric synthetic geometry."""
    cfg = IntegrationConfig(
        lsd=1_000_000, ybc=1024, zbc=1024,
        wavelength=0.172979, pixel_size=200.0,
        nr_pixels_y=2048, nr_pixels_z=2048,
        r_bin_size=1.0, eta_bin_size=2.0,  # bigger bins = faster
    )
    artifacts = Mapper(cfg).build(tmp_path, n_cpus=2)

    assert artifacts.map_bin.exists()
    assert artifacts.n_map_bin.exists()
    assert artifacts.map_bin.stat().st_size > 256      # header + data
    # Header is optional (older MIDAS builds omit it); assert only when present.
    hdr = artifacts.header
    if hdr is not None:
        assert hdr.n_pixels_y == 2048
        assert hdr.n_pixels_z == 2048
        assert hdr.r_bin_size == pytest.approx(1.0)
        assert hdr.eta_bin_size == pytest.approx(2.0)
