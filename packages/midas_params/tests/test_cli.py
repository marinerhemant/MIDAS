"""CLI smoke tests."""

from __future__ import annotations

import io
import json
import textwrap
from contextlib import redirect_stdout
from pathlib import Path as FsPath

import pytest

from midas_params.cli import main


MIDAS_ROOT = FsPath(__file__).resolve().parents[3]


def _run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_cli_validate_bad_file_exits_nonzero(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 999\n")  # out of range
    rc, out = _run(["validate", str(fn), "--path", "ff", "--no-color"])
    assert rc != 0
    assert "SpaceGroup=999 is not a valid" in out


def test_cli_validate_json_output(tmp_path):
    fn = tmp_path / "p.txt"
    fn.write_text("SpaceGroup 225\n")
    rc, out = _run(["validate", str(fn), "--path", "ff", "--json"])
    data = json.loads(out)
    assert "issues" in data
    assert data["path"] == "ff"


def test_cli_inspect_unknown_file():
    rc, out = _run(["inspect", "/not/a/real/path.ge3", "--no-color"])
    assert rc == 0  # inspect reports but doesn't fail
    assert "No parameters" in out or "Path does not exist" in out


def test_cli_inspect_finds_frames(tmp_path):
    for i in [1, 2, 3]:
        (tmp_path / f"sample_{i:06d}.ge3").touch()
    rc, out = _run(["inspect", str(tmp_path / "sample_000002.ge3"), "--no-color"])
    assert rc == 0
    assert "FileStem" in out
    assert "sample" in out


def test_cli_inspect_json(tmp_path):
    for i in [1, 2]:
        (tmp_path / f"s_{i:06d}.tif").touch()
    rc, out = _run(["inspect", str(tmp_path / "s_000001.tif"), "--json"])
    data = json.loads(out)
    assert data["extracted"]["FileStem"] == "s"
    assert data["extracted"]["EndNr"] == 2


def test_cli_wizard_non_interactive_writes_file(tmp_path):
    """Non-interactive wizard: feed seeds, get a file out, post-validate."""
    # Build a dataset so discovery succeeds
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    for i in range(1, 11):
        (rawdir / f"exp_{i:06d}.ge3").touch()

    # Calibration file that provides most of the required geometry
    calib = tmp_path / "calib.txt"
    calib.write_text(textwrap.dedent(f"""
        RawFolder {rawdir}
        FileStem exp
        Ext .ge3
        Padding 6
        StartNr 1
        EndNr 10
        Lsd 1000000
        BC 1022 1022
        px 200
        NrPixels 2048
        Wavelength 0.22291
        LatticeConstant 4.08 4.08 4.08 90 90 90
        SpaceGroup 225
        OmegaStart 180
        OmegaEnd -180
        OmegaStep -0.25
        OmegaRange -180 180
        BoxSize -1000000 1000000 -1000000 1000000
        RingThresh 1 10
        RingThresh 2 10
        OverAllRingToIndex 2
        Completeness 0.8
        StepSizeOrient 0.2
        StepSizePos 100
    """).strip())

    out = tmp_path / "generated.txt"
    rc, text = _run([
        "wizard", "--path", "ff",
        "--out", str(out),
        "--from-calibration", str(calib),
        "--non-interactive",
    ])
    assert out.exists()
    # Re-validate: should pass (no errors) since we supplied everything
    from midas_params import Path
    from midas_params.validator import validate
    r = validate(str(out), Path.FF)
    # Non-fatal infos (like SpaceGroup=225 smell) OK; but no errors allowed.
    assert r.ok, [i.message for i in r.errors]
