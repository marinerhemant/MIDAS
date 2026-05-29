"""Unit tests for FF seeding helpers (NF→FF resolver, GrainsFile patch,
RawFolder/Dark override).

Tests live for the ported helpers in ``midas_pipeline.ff_seeding`` —
renamed from the legacy ``midas_ff_pipeline.seeding`` to avoid a name
collision with the existing ``midas_pipeline.seeding/`` PF-merged-FF
package.
"""
from __future__ import annotations

from pathlib import Path

from midas_pipeline.ff_seeding import (
    apply_raw_dir_override,
    patch_params_with_grains,
    resolve_grains_file_for_layer,
)


def test_resolve_grains_file_prefers_nf(tmp_path: Path):
    nf = tmp_path / "nf"
    nf.mkdir()
    (nf / "GrainsLayer3.csv").write_text("seed3")
    explicit = tmp_path / "Grains.csv"
    explicit.write_text("explicit")
    seed = resolve_grains_file_for_layer(
        layer_nr=3,
        grains_file=str(explicit),
        nf_result_dir=str(nf),
    )
    assert seed is not None
    assert seed.endswith("GrainsLayer3.csv")


def test_resolve_grains_file_falls_back_to_explicit(tmp_path: Path):
    nf = tmp_path / "nf"
    nf.mkdir()
    explicit = tmp_path / "Grains.csv"
    explicit.write_text("explicit")
    seed = resolve_grains_file_for_layer(
        layer_nr=2,
        grains_file=str(explicit),
        nf_result_dir=str(nf),
    )
    assert seed == str(explicit)


def test_resolve_grains_file_returns_none_when_neither(tmp_path: Path):
    seed = resolve_grains_file_for_layer(
        layer_nr=1, grains_file=None, nf_result_dir=None,
    )
    assert seed is None


def test_patch_params_with_grains_appends(tmp_path: Path):
    p = tmp_path / "params.txt"
    p.write_text("Lsd 1000000\nBC 1024 1024\n")
    patch_params_with_grains(p, "/path/to/Grains.csv")
    text = p.read_text()
    assert "GrainsFile /path/to/Grains.csv" in text
    assert "MinNrSpots 1" in text


def test_patch_params_with_grains_replaces(tmp_path: Path):
    p = tmp_path / "params.txt"
    p.write_text("GrainsFile /old.csv\nMinNrSpots 4\nLsd 1000000\n")
    patch_params_with_grains(p, "/new.csv")
    lines = p.read_text().splitlines()
    grains_lines = [l for l in lines if l.startswith("GrainsFile")]
    minspots_lines = [l for l in lines if l.startswith("MinNrSpots")]
    assert grains_lines == ["GrainsFile /new.csv"]
    assert minspots_lines == ["MinNrSpots 1"]


def test_apply_raw_dir_override_replaces_rawfolder_and_dark(tmp_path: Path):
    new_raw = tmp_path / "new_raw"
    new_raw.mkdir()
    p = tmp_path / "params.txt"
    p.write_text(
        "RawFolder /old/raw\n"
        "Dark /old/raw/dark_00001.ge3\n"
        "Lsd 1000000\n"
    )
    apply_raw_dir_override(p, str(new_raw))
    text = p.read_text()
    assert f"RawFolder {new_raw}" in text
    assert f"Dark {new_raw}/dark_00001.ge3" in text


def test_apply_raw_dir_override_appends_when_missing(tmp_path: Path):
    new_raw = tmp_path / "raw"
    new_raw.mkdir()
    p = tmp_path / "params.txt"
    p.write_text("Lsd 1000000\n")
    apply_raw_dir_override(p, str(new_raw))
    assert f"RawFolder {new_raw}" in p.read_text()
