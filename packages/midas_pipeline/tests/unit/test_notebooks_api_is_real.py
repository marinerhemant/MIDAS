"""Every attribute a shipped notebook reaches for must exist on the object.

Notebooks are not executed in CI (most need real beamtime data), so an API
rename lands in them silently. ``13_ff_calibrate_then_multi_detector.ipynb``
had never been executed at all -- every ``execution_count`` was ``None`` --
and carried two calls that could not work:

* ``pipe.layer_result`` and ``LayerResult.grains_df()``. ``Pipeline.run()``
  returns a ``list[LayerResult]``; there is no ``layer_result`` attribute, and
  ``grains_df`` belongs to the deprecated ``midas_ff_pipeline``.
* ``pd.read_csv(SpotMatrix.csv)`` with the DEFAULT COMMA separator on a TAB
  file, then indexing ``ObsZ``/``ObsY``/``PredZ``/``PredY``/``DetectorID`` --
  five column names no SpotMatrix layout has ever had.

These tests are static: they parse the notebooks, compile every code cell and
check the names against the real dataclasses.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_NB_DIR = Path(__file__).resolve().parents[2] / "notebooks"

pytestmark = pytest.mark.skipif(not _NB_DIR.is_dir(),
                                reason="notebooks/ not present in this install")


def _code_cells(name: str) -> list[str]:
    nb = json.loads((_NB_DIR / name).read_text())
    return ["".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "code"]


def _all_notebooks() -> list[str]:
    return sorted(p.name for p in _NB_DIR.glob("*.ipynb"))


@pytest.mark.parametrize("name", _all_notebooks())
def test_every_code_cell_parses(name):
    """A cell that does not even compile cannot have been run."""
    for i, src in enumerate(_code_cells(name)):
        if src.lstrip().startswith(("%", "!")):
            continue        # IPython magic / shell escape
        try:
            ast.parse(src)
        except SyntaxError as exc:      # pragma: no cover - failure path
            pytest.fail(f"{name} code cell {i}: {exc}")


_NB13 = "13_ff_calibrate_then_multi_detector.ipynb"


def _nb13_ast() -> list[ast.Module]:
    return [ast.parse(src) for src in _code_cells(_NB13)]


def _attrs_and_strings(trees):
    """Attribute names and string literals in EXECUTABLE code.

    Parsing rather than substring-matching the raw cell text: the cells now
    carry comments naming the broken API so a reader knows what changed, and a
    grep would match those.
    """
    attrs, strings, names = set(), set(), set()
    for t in trees:
        for node in ast.walk(t):
            if isinstance(node, ast.Attribute):
                attrs.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                strings.add(node.value)
            elif isinstance(node, ast.Name):
                names.add(node.id)
    return attrs, strings, names


@pytest.mark.skipif(not (_NB_DIR / _NB13).exists(), reason="nb 13 absent")
def test_nb13_does_not_call_nonexistent_pipeline_api():
    from midas_pipeline.results import LayerResult

    attrs, _, names = _attrs_and_strings(_nb13_ast())
    assert not hasattr(LayerResult, "grains_df"), (
        "LayerResult grew a grains_df -- update this test and the notebook")
    assert "grains_df" not in attrs
    assert "layer_result" not in attrs
    # What it must use instead.
    assert "run" in attrs
    assert "read_grains_csv" in names


@pytest.mark.skipif(not (_NB_DIR / _NB13).exists(), reason="nb 13 absent")
def test_nb13_reads_spot_matrix_through_the_canonical_reader():
    attrs, strings, names = _attrs_and_strings(_nb13_ast())
    assert "read_spot_matrix" in names
    for phantom in ("ObsZ", "ObsY", "PredZ", "PredY", "DetectorID"):
        assert phantom not in strings, f"{phantom} is not a SpotMatrix column"
    # The columns it actually needs, and which do exist.
    assert "theorEta" in strings
    assert {"eta", "y_lab", "z_lab", "spot_id", "grain_id"} <= attrs


@pytest.mark.skipif(not (_NB_DIR / _NB13).exists(), reason="nb 13 absent")
def test_nb13_uses_only_real_result_fields():
    """Attribute access on result_p1 / result_p2 must resolve on LayerResult."""
    from midas_pipeline.results import (CrossDetMergeResult, LayerResult,
                                        ProcessGrainsResult)

    layer_fields = set(LayerResult.__dataclass_fields__) | {
        n for n in dir(LayerResult) if not n.startswith("_")}
    seen: set[str] = set()
    for src in _code_cells(_NB13):
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ("result_p1", "result_p2")):
                seen.add(node.attr)
    assert seen, "no result_p1/result_p2 attribute access found -- test stale"
    unknown = sorted(seen - layer_fields)
    assert not unknown, f"LayerResult has no {unknown}"

    # And the two nested results the notebook drills into.
    assert "grains_csv" in ProcessGrainsResult.__dataclass_fields__
    for f in ("n_total_spots", "n_per_detector"):
        assert f in CrossDetMergeResult.__dataclass_fields__
