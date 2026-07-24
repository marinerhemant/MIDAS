"""The .ipynb files are derived artefacts; ``notebooks/_build.py`` is the source.

Guards that convention: if someone hand-edits a notebook, or edits _build.py and
forgets to regenerate, the checked-in .ipynb no longer matches its source and
this fails.

Only the *source* cells are compared -- executed notebooks legitimately carry
outputs and execution counts that the builder does not produce.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def _load_builder():
    spec = importlib.util.spec_from_file_location("_nb_build", NB_DIR / "_build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not (NB_DIR / "_build.py").exists(),
                    reason="notebooks/ not present (not shipped with the wheel)")
def test_every_notebook_is_registered():
    """Every .ipynb on disk must come from a NOTEBOOKS entry."""
    builder = _load_builder()
    on_disk = {p.stem for p in NB_DIR.glob("*.ipynb")}
    registered = set(builder.NOTEBOOKS)
    assert on_disk <= registered, (
        f"unregistered notebook(s) on disk: {sorted(on_disk - registered)} -- "
        "add them to NOTEBOOKS in _build.py or delete them")


@pytest.mark.skipif(not (NB_DIR / "_build.py").exists(),
                    reason="notebooks/ not present (not shipped with the wheel)")
@pytest.mark.parametrize("name", [p.stem for p in sorted(NB_DIR.glob("*.ipynb"))])
def test_notebook_source_matches_builder(name):
    """Regenerating from _build.py must reproduce the checked-in cell sources."""
    builder = _load_builder()
    if name not in builder.NOTEBOOKS:
        pytest.skip(f"{name} not registered")

    on_disk = json.loads((NB_DIR / f"{name}.ipynb").read_text())
    disk_cells = [(c["cell_type"], "".join(c["source"])) for c in on_disk["cells"]]
    want_cells = [("markdown" if k == "md" else "code", s)
                  for k, s in builder.NOTEBOOKS[name]]

    assert len(disk_cells) == len(want_cells), (
        f"{name}.ipynb has {len(disk_cells)} cells, _build.py declares "
        f"{len(want_cells)} -- rerun `python _build.py {name}`")
    for i, (got, want) in enumerate(zip(disk_cells, want_cells)):
        assert got[0] == want[0], f"{name} cell {i}: type {got[0]} != {want[0]}"
        assert got[1] == want[1], (
            f"{name} cell {i}: source drifted from _build.py -- "
            f"rerun `python _build.py {name}`")
