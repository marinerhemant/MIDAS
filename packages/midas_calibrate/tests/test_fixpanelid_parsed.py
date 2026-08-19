"""`FixPanelID` must reach CalibrationParams.

It was declared as a field (`FixedPanelID`) but never parsed, under either
spelling, so the anchored panel stayed 0 no matter what the parameter file
said. Found 2026-08-19 on a 48-panel Pilatus whose file says `FixPanelID 28`.

`FixPanelID` is the spelling used by the C (CalibrantIntegratorOMP.c,
FitMultipleGrains.c), by midas_params' registry, and by the files
AutoCalibrateZarr writes; `FixedPanelID` is accepted as an alias.
"""
from __future__ import annotations

import pytest

from midas_calibrate.params import CalibrationParams

_MIN = """\
Lsd 650000.0
BC 738.0 842.0
px 172.0
Wavelength 0.19582
SpaceGroup 225
LatticeConstant 5.4116 5.4116 5.4116 90.0 90.0 90.0
MaxRingRad 1000.0
NrPixelsY 1475
NrPixelsZ 1679
"""


def _write(tmp_path, extra: str):
    p = tmp_path / "ps.txt"
    p.write_text(_MIN + extra)
    return CalibrationParams.from_file(p)


def test_canonical_spelling_is_parsed(tmp_path):
    assert _write(tmp_path, "FixPanelID 28\n").FixedPanelID == 28


def test_legacy_spelling_still_accepted(tmp_path):
    assert _write(tmp_path, "FixedPanelID 17\n").FixedPanelID == 17


def test_absent_key_keeps_the_zero_default(tmp_path):
    assert _write(tmp_path, "").FixedPanelID == 0


def test_trailing_semicolon_and_comment_tolerated(tmp_path):
    """MIDAS C files carry `;` terminators and inline comments."""
    assert _write(tmp_path, "FixPanelID 28;  # anchored module\n").FixedPanelID == 28


@pytest.mark.parametrize("raw,want", [("28", 28), ("28.0", 28), ("0", 0)])
def test_value_forms(tmp_path, raw, want):
    assert _write(tmp_path, f"FixPanelID {raw}\n").FixedPanelID == want
