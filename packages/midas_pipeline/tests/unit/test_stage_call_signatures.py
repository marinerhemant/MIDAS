"""Every keyword a stage passes must exist on the function it calls.

Four PF paths shipped unrunnable because a stage passed a keyword the callee
does not take (reconstruct -> mlem_recon ``n_pixels``, reconstruct ->
voxelmap_recon ``nScans``/``nGrs``, fuse -> bayesian_fusion ``nGrs``,
em_refine -> run_em_spot_ownership ``opt_steps``). Each died with a TypeError
on first use and no test called it. This walks the stage sources and checks
the keywords statically, so it covers branches the unit tests never reach.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

import midas_pipeline.stages as stages_pkg

STAGE_DIR = Path(stages_pkg.__file__).parent


def _imported_callables(tree: ast.Module, modname: str) -> dict:
    """name -> (module, attr) for every ``from <rel> import name`` in the file,
    including imports nested inside functions."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            base = importlib.util.resolve_name("." * node.level + (node.module or ""),
                                               modname.rsplit(".", 1)[0])
            for a in node.names:
                out[a.asname or a.name] = (base, a.name)
    return out


def _calls():
    for src in sorted(STAGE_DIR.glob("*.py")):
        modname = f"midas_pipeline.stages.{src.stem}"
        tree = ast.parse(src.read_text())
        imported = _imported_callables(tree, modname)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id not in imported or not node.keywords:
                continue
            kws = [k.arg for k in node.keywords if k.arg is not None]
            yield pytest.param(src.name, node.lineno, imported[node.func.id], kws,
                               id=f"{src.stem}:{node.lineno}:{node.func.id}")


@pytest.mark.parametrize("fname,lineno,target,kws", list(_calls()))
def test_stage_keywords_exist_on_callee(fname, lineno, target, kws):
    modname, attr = target
    try:
        fn = getattr(importlib.import_module(modname), attr)
    except ImportError as e:            # optional backend not installed here
        pytest.skip(f"{modname}: {e}")
    if not callable(fn) or inspect.isclass(fn):
        pytest.skip("not a function")
    sig = inspect.signature(fn)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return
    bad = [k for k in kws if k not in sig.parameters]
    assert not bad, f"{fname}:{lineno} passes {bad} to {modname}.{attr}{sig}"
