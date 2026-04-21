"""_binaries.py resolution-order tests (mirror midas_auto_calibrate pattern)."""

from __future__ import annotations

import stat

import pytest

from midas_integrate import MidasBinaryNotFoundError, midas_bin


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

    assert "PATH(DoesNotExistXYZ)" in str(excinfo.value)


def test_explicit_bin_dir_beats_env(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    explicit_exe = _make_fake_exe(explicit, "MIDASIntegrator")
    _make_fake_exe(env_dir, "MIDASIntegrator")

    monkeypatch.setenv("MIDAS_BIN", str(env_dir))

    resolved = midas_bin("MIDASIntegrator", bin_dir=explicit)
    assert resolved == explicit_exe.resolve()


def _disable_package_bins(monkeypatch):
    """Hide wheel-shipped _bin dirs so env-var fallbacks are reachable."""
    from midas_integrate import _binaries as b
    monkeypatch.setattr(b, "_package_bin_dir", lambda _name: None)


def test_midas_bin_env_resolves(monkeypatch, tmp_path):
    _disable_package_bins(monkeypatch)
    monkeypatch.setenv("MIDAS_BIN", str(tmp_path))
    monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
    exe = _make_fake_exe(tmp_path, "MIDASIntegrator")

    resolved = midas_bin("MIDASIntegrator")
    assert resolved == exe.resolve()


def test_midas_install_dir_fallback(monkeypatch, tmp_path):
    _disable_package_bins(monkeypatch)
    monkeypatch.delenv("MIDAS_BIN", raising=False)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = _make_fake_exe(bindir, "MIDASIntegrator")
    monkeypatch.setenv("MIDAS_INSTALL_DIR", str(tmp_path))

    resolved = midas_bin("MIDASIntegrator")
    assert resolved == exe.resolve()


def test_auto_calibrate_bin_discovered(monkeypatch, tmp_path):
    """Binaries shipped by midas_auto_calibrate's wheel are discoverable.

    Validates the cross-package discovery path in ``_binaries._package_bin_dir``.
    Skipped when midas_auto_calibrate isn't installed with a populated _bin.
    """
    from midas_integrate._binaries import _package_bin_dir

    bin_dir = _package_bin_dir("midas_auto_calibrate")
    if bin_dir is None or not (bin_dir / "GetHKLList").exists():
        pytest.skip("midas_auto_calibrate not wheel-installed with GetHKLList.")

    # Ensure env / PATH can't provide it so the resolution must come from
    # the sister-package fallback.
    monkeypatch.delenv("MIDAS_BIN", raising=False)
    monkeypatch.delenv("MIDAS_INSTALL_DIR", raising=False)
    monkeypatch.setenv("PATH", "")

    resolved = midas_bin("GetHKLList")
    assert resolved == (bin_dir / "GetHKLList").resolve()
