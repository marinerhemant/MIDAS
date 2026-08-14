"""Reconstruction goes through midas-tomo, never scikit-image.

This is a standing constraint on the package, not a style preference:
``skimage.transform.iradon`` is a different reconstruction with different
filtering, different centring conventions and no relationship to what the
rest of MIDAS produces. Reaching for it -- as a convenience, a fallback, or a
test fixture -- silently forks the reconstruction and the two stop agreeing.

The failure mode this guards is a plausible one: someone adds
``except ImportError: from skimage.transform import iradon`` so the tests pass
on a machine without the engine built, and from then on those tests measure
scikit-image.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "midas_dt"
SOURCES = sorted(PKG.rglob("*.py"))


def test_there_are_sources_to_scan():
    # A glob that silently matches nothing turns every check below vacuous.
    assert len(SOURCES) > 5, f"only found {len(SOURCES)} modules under {PKG}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_module_does_not_import_skimage(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.split(".")[0] == "skimage", (
                f"{path.name}:{node.lineno} imports {name}. Reconstruction "
                f"goes through midas_tomo; see the module docstring here."
            )


def test_importing_midas_dt_does_not_pull_in_skimage():
    """The AST scan misses a runtime ``importlib.import_module('skimage')``."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, midas_dt; "
         "print(any(m == 'skimage' or m.startswith('skimage.') "
         "for m in sys.modules))"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False", out.stdout


def test_recon_module_calls_midas_tomo():
    """The positive half: assert what it *does* use, not only what it doesn't."""
    src = (PKG / "recon.py").read_text()
    assert "midas_tomo" in src, "recon.py no longer references midas_tomo"
