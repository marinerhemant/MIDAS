"""Static checks on the shipped notebooks.

Not execution -- these need real data and a built engine. What they catch is
the cheap, common rot: a notebook still importing the pre-package module, or
calling a keyword that was renamed during the port. That drift is invisible
until a user opens the notebook, which is the worst time to find it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

# Names that existed before the port and must not reappear.
LEGACY = (
    "midas_tomo_python",  # the loose module, now a package
    "numCPUs",
    "filterNr",
    "doLog=",
    "extraPad",
    "autoCentering",
    "doCleanup",
    "ringRemoval=",
    "doStripeRemoval",
    "stripeSnr",
    "stripeLaSize",
    "stripeSmSize",
    "useGPU",
    "fftwBridge",
)


def _cells(nb: Path, kind: str) -> list[str]:
    doc = json.loads(nb.read_text())
    return [
        "".join(c["source"])
        for c in doc.get("cells", [])
        if c.get("cell_type") == kind
    ]


@pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks shipped")
@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_no_legacy_api_names(nb):
    text = "\n".join(_cells(nb, "code") + _cells(nb, "markdown"))
    found = sorted({name for name in LEGACY if name in text})
    assert not found, (
        f"{nb.name} still refers to pre-package names {found}. The port "
        f"renamed these; a notebook that keeps them fails only when a user "
        f"runs it."
    )


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_parse(nb):
    """Every code cell must be syntactically valid Python.

    Cheap guard against a broken edit -- including the scripted rewrite that
    repathed these from the legacy module.
    """
    for i, src in enumerate(_cells(nb, "code")):
        # IPython magics and shell escapes are not Python; skip those cells.
        if any(l.lstrip().startswith(("%", "!")) for l in src.splitlines()):
            continue
        try:
            ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(f"{nb.name} code cell {i} does not parse: {exc}")


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_imports_resolve_to_the_public_api(nb):
    """Anything imported from midas_tomo must actually be exported.

    This is the check that would have caught the rename drift in the other
    direction -- a notebook importing something the package no longer has.
    """
    import midas_tomo

    for src in _cells(nb, "code"):
        if any(l.lstrip().startswith(("%", "!")) for l in src.splitlines()):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "midas_tomo":
                for alias in node.names:
                    assert hasattr(midas_tomo, alias.name), (
                        f"{nb.name} imports midas_tomo.{alias.name}, which the "
                        f"package does not export"
                    )
