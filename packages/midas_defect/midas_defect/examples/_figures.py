"""Where example figures are written.

A developer clone keeps figures in the package's ``dev/paper/figures`` tree, which
is gitignored (``packages/*/dev``) and therefore absent from every sdist, wheel and
fresh clone. An installed copy has no such tree -- and creating one would write
inside site-packages -- so those runs fall back to ``./figures`` under the current
working directory. Set ``MIDAS_FIGDIR`` to override either default.
"""
from __future__ import annotations

import os

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
# packages/midas_defect/ -- the dev tree sits beside the importable package.
_PKG_ROOT = os.path.dirname(os.path.dirname(_EXAMPLES_DIR))
_DEV_FIGURES = os.path.join(_PKG_ROOT, "dev", "paper", "figures")


def figure_dir(out_dir: str | None = None) -> str:
    """Resolve and create the directory example figures are written to."""
    if out_dir is None:
        out_dir = os.environ.get("MIDAS_FIGDIR")
    if out_dir is None:
        out_dir = (_DEV_FIGURES if os.path.isdir(os.path.dirname(_DEV_FIGURES))
                   else os.path.join(os.getcwd(), "figures"))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
