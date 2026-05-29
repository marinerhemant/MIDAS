"""Unit tests for batch / multi-layer discovery."""
from __future__ import annotations

from pathlib import Path

from midas_pipeline.discovery import discover_layer_files


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_discover_layer_files_basic(tmp_path: Path):
    raw = tmp_path / "raw"
    for n in (1, 2, 3, 4, 5):
        _touch(raw / f"sample_{n:06d}.ge3")
    found = discover_layer_files(str(raw), ".ge3", 6, 2, 4)
    assert [n for n, _ in found] == [2, 3, 4]
    assert all(stem == "sample" for _, stem in found)


def test_discover_layer_files_skips_darks(tmp_path: Path):
    raw = tmp_path / "raw"
    _touch(raw / "sample_000001.ge3")
    _touch(raw / "dark_sample_000002.ge3")
    _touch(raw / "sample_000003.ge3")
    found = discover_layer_files(str(raw), ".ge3", 6, 1, 5)
    assert [n for n, _ in found] == [1, 3]


def test_discover_layer_files_handles_missing(tmp_path: Path):
    raw = tmp_path / "raw"
    _touch(raw / "sample_000001.ge3")
    _touch(raw / "sample_000005.ge3")
    found = discover_layer_files(str(raw), ".ge3", 6, 1, 5)
    assert [n for n, _ in found] == [1, 5]


def test_discover_layer_files_multiple_stems(tmp_path: Path):
    raw = tmp_path / "raw"
    _touch(raw / "stemA_000001.ge3")
    _touch(raw / "stemB_000002.ge3")
    found = discover_layer_files(str(raw), ".ge3", 6, 1, 2)
    stems = {s for _, s in found}
    assert stems == {"stemA", "stemB"}


def test_discover_layer_files_missing_dir(tmp_path: Path):
    found = discover_layer_files(str(tmp_path / "does_not_exist"),
                                 ".ge3", 6, 1, 5)
    assert found == []
