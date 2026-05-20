"""Tiny builder for student-friendly notebooks.

Each notebook is built from an ordered list of (kind, content) tuples
where ``kind`` is ``"md"`` or ``"code"``. We emit a minimal nbformat-4
JSON file with ``language_info`` set to Python so JupyterLab and VSCode
both render it correctly out of the box.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def _cell(kind: str, source: str) -> dict:
    if kind == "md":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
    if kind == "code":
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }
    raise ValueError(f"unknown cell kind {kind!r}")


def write_notebook(path: Path, cells: List[Tuple[str, str]]) -> None:
    """Build and write a notebook from (kind, source) tuples."""
    nb = {
        "cells": [_cell(k, s) for k, s in cells],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, indent=1))


__all__ = ["write_notebook"]
