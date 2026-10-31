"""The refiner's position box follows the indexer's position grid (multi-detector FF, #75)."""
from __future__ import annotations

from pathlib import Path

import pytest

from midas_pipeline.stages.refinement import _widen_refiner_position_window


def _write(tmp_path: Path, rsample=100.0, step=100.0, extra="") -> Path:
    lines = ["Lsd 3300000.0;"]
    if rsample is not None:
        lines.append(f"Rsample {rsample:.6f};")
    if step is not None:
        lines.append(f"StepsizePos {step:.6f};")
    p = tmp_path / "paramstest_refine_comp.txt"
    p.write_text("\n".join(lines) + "\n" + extra)
    return p


def _rsample(path: Path) -> float:
    val = None
    for line in path.read_text().splitlines():            # last occurrence wins, as in C
        tok = line.replace(";", " ").split()
        if len(tok) >= 2 and tok[0] == "Rsample":
            val = float(tok[1])
    return val


def test_rsample_is_raised_to_three_steps(tmp_path, monkeypatch):
    monkeypatch.delenv("MIDAS_REFINER_POS_STEPS", raising=False)
    p = _write(tmp_path)
    _widen_refiner_position_window(p)
    assert _rsample(p) == 300.0
    assert "StepsizePos 100.000000;" in p.read_text()        # nothing else changed


def test_a_larger_rsample_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("MIDAS_REFINER_POS_STEPS", raising=False)
    p = _write(tmp_path, rsample=500.0)
    before = p.read_text()
    _widen_refiner_position_window(p)
    assert p.read_text() == before


def test_finer_grid_needs_less_reach(tmp_path, monkeypatch):
    monkeypatch.delenv("MIDAS_REFINER_POS_STEPS", raising=False)
    p = _write(tmp_path, rsample=100.0, step=25.0)            # 3 x 25 = 75 < 100
    before = p.read_text()
    _widen_refiner_position_window(p)
    assert p.read_text() == before


def test_missing_keys_do_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("MIDAS_REFINER_POS_STEPS", raising=False)
    for kw in ({"step": None}, {"rsample": None}):
        p = _write(tmp_path, **kw)
        before = p.read_text()
        _widen_refiner_position_window(p)
        assert p.read_text() == before


def test_environment_override_and_off_switch(tmp_path, monkeypatch):
    p = _write(tmp_path)
    monkeypatch.setenv("MIDAS_REFINER_POS_STEPS", "5")
    _widen_refiner_position_window(p)
    assert _rsample(p) == 500.0
    off = tmp_path / "off"
    off.mkdir()
    q = _write(off)
    before = q.read_text()
    monkeypatch.setenv("MIDAS_REFINER_POS_STEPS", "0")
    _widen_refiner_position_window(q)
    assert q.read_text() == before


def test_the_last_rsample_line_is_the_one_compared(tmp_path, monkeypatch):
    monkeypatch.delenv("MIDAS_REFINER_POS_STEPS", raising=False)
    p = _write(tmp_path, rsample=100.0, extra="Rsample 400.000000;\n")
    before = p.read_text()
    _widen_refiner_position_window(p)                          # 400 >= 300: unchanged
    assert p.read_text() == before
