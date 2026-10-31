"""supports_multidet(): does the installed binary apply per-panel geometry?

C is compiled when the package is installed, so a new Python package can sit on
an old binary and the version string cannot tell the two apart. The probe looks
for the marker the feature adds to the executable.
"""
from __future__ import annotations

from midas_index import backend_c


def test_marker_present_means_supported(tmp_path, monkeypatch):
    exe = tmp_path / "midas_indexer"
    exe.write_bytes(b"\x7fELF....MIDAS_MULTIDET_SUPPORT_PHASE9B....")
    monkeypatch.setattr(backend_c, "binary_path", lambda: exe)
    assert backend_c.supports_multidet() is True


def test_old_binary_without_the_marker_is_not_supported(tmp_path, monkeypatch):
    exe = tmp_path / "midas_indexer"
    exe.write_bytes(b"\x7fELF....Multi-detector mode: %d panels parsed....")
    monkeypatch.setattr(backend_c, "binary_path", lambda: exe)
    assert backend_c.supports_multidet() is False


def test_missing_binary_is_not_supported(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_c, "binary_path", lambda: tmp_path / "absent")
    assert backend_c.supports_multidet() is False
