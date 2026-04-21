"""GPU runtime diagnostics — pure-Python, no GPU required.

The actual CUDA path is exercised in test_gpu_loopback.py (skip-if-no-GPU).
"""

from __future__ import annotations

import platform
import subprocess
from unittest import mock

import pytest

import midas_integrate_gpu
from midas_integrate_gpu._runtime import (
    EnvironmentCheck,
    check_environment,
    is_gpu_available,
)


def test_version_exported():
    assert midas_integrate_gpu.__version__ == "0.1.0"


def test_environment_check_returns_dataclass():
    env = check_environment()
    assert isinstance(env, EnvironmentCheck)


def test_platform_reports_macos_as_unsupported(monkeypatch):
    """On macOS, the check correctly flags the platform mismatch."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    env = check_environment()
    assert not env.platform_ok
    assert any("requires Linux" in p for p in env.problems)


def test_platform_reports_windows_as_unsupported(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    env = check_environment()
    assert not env.platform_ok


def test_missing_nvidia_smi_produces_problem(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    env = check_environment()
    assert not env.nvidia_smi_found
    assert any("nvidia-smi not found" in p for p in env.problems)


def test_is_gpu_available_boolean():
    # Just check it returns a bool; truth value depends on test machine.
    assert isinstance(is_gpu_available(), bool)


def test_driver_version_parsed_when_smi_present(monkeypatch, tmp_path):
    """Mock nvidia-smi output; verify we parse the driver string."""
    fake_smi = tmp_path / "nvidia-smi"
    fake_smi.write_text("#!/bin/sh\necho 535.104.05\n")
    fake_smi.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda name: str(fake_smi) if name == "nvidia-smi" else None)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="535.104.05\n", stderr="",
        )
    monkeypatch.setattr(subprocess, "run", fake_run)

    env = check_environment()
    assert env.nvidia_smi_found
    assert env.driver_version == "535.104.05"
