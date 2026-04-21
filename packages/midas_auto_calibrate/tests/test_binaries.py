"""_binaries.py resolution-order tests."""

from __future__ import annotations

import os
import stat
import subprocess

import pytest

from midas_auto_calibrate import MidasBinaryNotFoundError, midas_bin


def _make_fake_exe(dirpath, name):
    path = dirpath / name
    path.write_text("#!/bin/sh\necho fake\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_binary_not_found_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("MIDAS_BIN", raising=False)
    monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(MidasBinaryNotFoundError) as excinfo:
        midas_bin("DoesNotExistXYZ")

    # The error message must enumerate every location tried so users can debug.
    msg = str(excinfo.value)
    assert "PATH(DoesNotExistXYZ)" in msg


def test_explicit_bin_dir_beats_env(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    explicit_exe = _make_fake_exe(explicit, "MIDASCalibrant")
    _make_fake_exe(env_dir, "MIDASCalibrant")

    monkeypatch.setenv("MIDAS_BIN", str(env_dir))

    resolved = midas_bin("MIDASCalibrant", bin_dir=explicit)
    assert resolved == explicit_exe.resolve()


def test_midas_bin_env_resolves(monkeypatch, tmp_path):
    monkeypatch.setenv("MIDAS_BIN", str(tmp_path))
    monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
    exe = _make_fake_exe(tmp_path, "MIDASCalibrant")

    resolved = midas_bin("MIDASCalibrant")
    assert resolved == exe.resolve()


def test_midas_install_dir_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("MIDAS_BIN", raising=False)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = _make_fake_exe(bindir, "MIDASCalibrant")
    monkeypatch.setenv("MIDAS_INSTALL_DIR", str(tmp_path))

    resolved = midas_bin("MIDASCalibrant")
    assert resolved == exe.resolve()
