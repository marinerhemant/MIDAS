"""Every pipeline that builds the residual-correction map must pass the MEAN pitch (``px_mean=``), derived from pxY and pxZ.

``build_residual_corr_map`` falls back to pxY when ``px_mean`` is omitted, which is wrong for non-square pixels (R_px is in mean-pixel
units), and an omitted or mis-derived argument would run silently. The callers need a full calibration fixture to exercise, so this pins
the call sites by AST: each call passes ``px_mean=<name>`` and that name is assigned ``0.5 * (pxY... + pxZ...)`` in the same module.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PIPE = Path(__file__).resolve().parents[1] / "midas_calibrate_v2" / "pipelines"


@pytest.mark.parametrize("fname", ["single.py", "multi.py"])
def test_build_residual_corr_map_callers_pass_the_mean_pitch(fname):
    tree = ast.parse((PIPE / fname).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None)) == "build_residual_corr_map"]
    assert calls, f"{fname}: no build_residual_corr_map call found"
    assigns = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            assigns.setdefault(n.targets[0].id, []).append(ast.unparse(n.value))
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "px_mean" in kw, f"{fname}:{c.lineno}: px_mean not passed"
        assert isinstance(kw["px_mean"], ast.Name), f"{fname}:{c.lineno}: px_mean should be a named mean-pitch variable"
        srcs = assigns.get(kw["px_mean"].id, [])
        assert any("pxY" in s and "pxZ" in s and "0.5" in s for s in srcs), (
            f"{fname}:{c.lineno}: {kw['px_mean'].id} is not 0.5*(pxY+pxZ): {srcs}")
