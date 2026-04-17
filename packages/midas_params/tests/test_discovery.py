"""Discovery tests — filename parsing, directory scan, merge semantics."""

from __future__ import annotations

import textwrap
from pathlib import Path as FsPath

import pytest

from midas_params.discovery import (
    DiscoveryResult,
    discover_from_calibration_file,
    discover_from_file,
    merge,
    parse_frame_filename,
    scan_directory_for_range,
)


MIDAS_ROOT = FsPath(__file__).resolve().parents[3]


# ─── Filename parsing ────────────────────────────────────────────────────────


def test_parse_frame_filename_basic():
    r = parse_frame_filename("/any/sample_000042.ge3")
    assert r == {"stem": "sample", "num": 42, "padding": 6, "ext": ".ge3"}


def test_parse_frame_filename_stem_with_underscore():
    r = parse_frame_filename("exp_run2_000100.tif")
    assert r == {"stem": "exp_run2", "num": 100, "padding": 6, "ext": ".tif"}


def test_parse_frame_filename_no_match():
    """Files without a trailing _NNN number don't match — good, they fall
    through to the HDF5/Zarr probe path."""
    assert parse_frame_filename("just_a_name.txt") is None  # no number
    assert parse_frame_filename("random_file.h5") is None
    assert parse_frame_filename("no_number.ge3") is None


def test_parse_frame_filename_padding_3():
    r = parse_frame_filename("s_042.tif")
    assert r["padding"] == 3
    assert r["num"] == 42


# ─── Directory scanning ──────────────────────────────────────────────────────


def test_scan_directory_finds_range(tmp_path):
    for i in [3, 4, 5, 6, 7]:
        (tmp_path / f"exp_{i:06d}.ge3").touch()
    r = scan_directory_for_range(tmp_path, "exp", 6, ".ge3")
    assert r == (3, 7, 5)


def test_scan_directory_ignores_other_stems(tmp_path):
    (tmp_path / "wanted_000001.ge3").touch()
    (tmp_path / "other_000042.ge3").touch()
    r = scan_directory_for_range(tmp_path, "wanted", 6, ".ge3")
    assert r == (1, 1, 1)


def test_scan_directory_ignores_other_exts(tmp_path):
    (tmp_path / "s_000001.ge3").touch()
    (tmp_path / "s_000002.tif").touch()
    r = scan_directory_for_range(tmp_path, "s", 6, ".ge3")
    assert r == (1, 1, 1)


def test_scan_directory_empty(tmp_path):
    assert scan_directory_for_range(tmp_path, "nothing", 6, ".ge3") is None


# ─── Top-level discovery ─────────────────────────────────────────────────────


def test_discover_from_file_populates_all_filename_fields(tmp_path):
    for i in range(10, 20):
        (tmp_path / f"mydata_{i:06d}.ge3").touch()
    r = discover_from_file(tmp_path / "mydata_000015.ge3")
    assert r.extracted["RawFolder"] == str(tmp_path)
    assert r.extracted["FileStem"] == "mydata"
    assert r.extracted["Padding"] == 6
    assert r.extracted["Ext"] == ".ge3"
    assert r.extracted["StartNr"] == 10
    assert r.extracted["EndNr"] == 19
    # Confidence tags
    assert r.confidence["FileStem"] == "high"
    assert "dir-scan" in r.source["StartNr"]


def test_discover_from_nonexistent_file():
    r = discover_from_file("/definitely/not/a/real/path_000001.ge3")
    assert not r.extracted
    assert r.warnings


def test_discover_from_calibration_file_reads_params():
    ff_ex = MIDAS_ROOT / "FF_HEDM" / "Example" / "Parameters.txt"
    if not ff_ex.exists():
        pytest.skip("FF Example not present")
    r = discover_from_calibration_file(ff_ex)
    assert r.extracted["SpaceGroup"] == 225
    assert r.extracted["Wavelength"] == 0.22291
    # Every value should be tagged with a param-file source
    assert all("param-file" in v for v in r.source.values())


# ─── merge() priority ────────────────────────────────────────────────────────


def test_merge_earlier_wins():
    a = DiscoveryResult(
        extracted={"Lsd": 1000000, "px": 200},
        confidence={"Lsd": "high", "px": "high"},
        source={"Lsd": "from-a", "px": "from-a"},
    )
    b = DiscoveryResult(
        extracted={"Lsd": 999999, "SpaceGroup": 225},
        confidence={"Lsd": "medium", "SpaceGroup": "high"},
        source={"Lsd": "from-b", "SpaceGroup": "from-b"},
    )
    m = merge(a, b)
    assert m.extracted["Lsd"] == 1000000          # a wins
    assert m.extracted["px"] == 200               # only in a
    assert m.extracted["SpaceGroup"] == 225       # only in b


def test_merge_preserves_warnings():
    a = DiscoveryResult(warnings=["warn-a"])
    b = DiscoveryResult(warnings=["warn-b"])
    m = merge(a, b)
    assert "warn-a" in m.warnings
    assert "warn-b" in m.warnings
