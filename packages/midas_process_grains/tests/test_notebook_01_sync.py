"""The shipped 01_ff_grain_consolidation.ipynb must match notebooks/_build.py.

The .ipynb is a derived artefact: ``_build.py`` is the source of truth and the
notebook is regenerated (and executed) from it. Editing one without the other
is how the shipped notebook came to carry stored output from a pre-2026-08-21
run -- "3 grains, 47 columns" -- while the code that produced it had started
raising ``ValueError: Length mismatch`` against a 53-column file.

These tests are cheap: they never execute the notebook, they only check that
the checked-in cell sources are the ones _build.py generates today and that
they read the two CSVs by name rather than by position.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
_NB = _NB_DIR / "01_ff_grain_consolidation.ipynb"


def _build_module():
    spec = importlib.util.spec_from_file_location("_pg_nb_build",
                                                  _NB_DIR / "_build.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(
    not _NB.exists() or not (_NB_DIR / "_build.py").exists(),
    reason="notebooks/ not present in this install")


def _cells():
    return json.loads(_NB.read_text())["cells"]


def _code_text() -> str:
    return "\n".join("".join(c["source"]) for c in _cells()
                     if c["cell_type"] == "code")


def _md_text() -> str:
    return "\n".join("".join(c["source"]) for c in _cells()
                     if c["cell_type"] == "markdown")


def test_shipped_notebook_matches_its_generator():
    """Regenerating must be a no-op on the cell sources."""
    mod = _build_module()
    want = [(kind, src) for kind, src in mod.NB_01]
    got = [("md" if c["cell_type"] == "markdown" else "py", "".join(c["source"]))
           for c in _cells()]
    assert len(got) == len(want), (
        f"notebook has {len(got)} cells, _build.py defines {len(want)} -- "
        f"run `python notebooks/_build.py`")
    for i, ((kw, sw), (kg, sg)) in enumerate(zip(want, got)):
        assert kg == kw, f"cell {i}: kind {kg!r} != {kw!r}"
        assert sg == sw, (
            f"cell {i} is stale; run `python notebooks/_build.py` "
            f"and re-execute the notebook")


def test_notebook_reads_both_csvs_by_name():
    """No hand-written positional column list may come back.

    ``grains.columns = cols[:grains.shape[1]]`` slices the NAME list down, so
    it survives a file with too many columns but not one with too few names:
    53 columns against 47 names raises. The SpotMatrix cell hardcoded 12 names
    against 28 columns.
    """
    code = _code_text()
    assert "read_grains_csv" in code
    assert "read_spot_matrix" in code
    assert "cols[:grains.shape[1]]" not in code
    assert "sm.columns = [" not in code
    assert "grains.columns = " not in code


def test_notebook_output_reports_the_current_widths():
    """The stored output must come from a run against the current format."""
    outs = []
    for c in _cells():
        for o in c.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain", "")
            outs.append("".join(t) if isinstance(t, list) else str(t))
    text = "\n".join(outs)
    assert text.strip(), "notebook has no stored output -- re-execute it"
    assert "3 grains, 53 columns" in text, (
        "stored output is from a pre-53-column run; re-execute the notebook")
    assert "SpotMatrix: 28 columns" in text


def test_notebook_prose_states_the_current_widths():
    md = _md_text()
    assert "| `Grains.csv` | 53 |" in md
    assert "| `SpotMatrix.csv` | 28 |" in md
    # The Matched == 0 rows are the reason matched_only defaults to True; the
    # prose has to say so or the default looks arbitrary.
    assert "Matched" in md and "matched_only" in md
