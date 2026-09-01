"""Header-driven readers for ``Grains.csv`` and ``SpotMatrix.csv``.

Both files have been widened repeatedly (Grains 19 -> 21 -> 23 -> 47 -> 53
columns; SpotMatrix 12 -> 28) and are written under two different header
tokens: ``%ID`` by ``midas_process_grains.io.csv`` and ``%GrainID`` by its
``c_parity`` emitter (which is what the CLI actually runs). Both spellings are
real and both are on disk. Every positional reader in the tree froze one
snapshot of that layout and drifted silently. The one rule here is: **resolve
every column by NAME, never by position.**

Two traps this module exists to close, for the fields ``midas_transforms``
consumes:

1. **A reader keyed on one header token sees zero grains on the other.**
   ``radius/theoretical.py`` used to delegate to
   ``midas_stress.io.read_grains_csv``, which located its header by
   ``line.startswith("%GrainID")`` and therefore raised ``ValueError`` on every
   ``%ID`` file the Python writer produces. (That parser has since been
   re-anchored on ``O11`` too. Keeping the read local means midas-transforms
   does not have to carry a midas-stress version floor for it, and gives one
   reader for both artefacts rather than two in two packages.)

2. **SpotMatrix carries predicted-but-never-found reflections.** Rows with
   ``Matched == 0`` are reflections a grain was predicted to produce and that
   were never observed. They hold ``-1`` in ``SpotID``/``RingNr`` and NaN in
   every observed column, and are ~3.3 % of rows on real data. Left in, they
   reach the per-grain volume/K refinement with ``intensity = 0`` and
   ``ring_idx = -1`` but a *valid* ``grain_idx``, so they dilute that grain's
   fit with phantom zero-intensity spots. ``matched_only`` therefore defaults
   to **True**.

Why this is not a call into ``midas_process_grains.io.read``
------------------------------------------------------------
That module is the canonical implementation and this one deliberately mirrors
its rules, but ``midas-process-grains`` **depends on** ``midas-transforms``
(for the ParamsTest reader), so importing it here would make the packaging
graph circular and would drag h5py / hdf5plugin / scipy / midas-index into a
package that is otherwise a lean transforms library.
``tests/test_ff_plumbing.py`` cross-checks the two readers against the same
fixture whenever ``midas_process_grains`` happens to be importable, which it
always is in the monorepo, so the two cannot drift apart unnoticed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np

__all__ = [
    "GrainsFormatError",
    "read_grains_columns",
    "read_spot_matrix_columns",
]

#: Grain-id column spellings, in preference order.
_ID_NAMES = ("GrainID", "ID")


class GrainsFormatError(ValueError):
    """Raised when a file is not a recognisable ProcessGrains output."""


def _split_row(raw: str) -> List[str]:
    """Split one data row.

    ``SpotMatrix.csv`` is written with ``newline='\\t\\n'`` so every row ends
    in a tab; a naive ``split('\\t')`` yields a trailing empty field and
    ``float('')`` raises. Stripping first handles that, and whitespace
    splitting handles the space-separated generations too.
    """
    return raw.rstrip().rstrip("\t").split()


def _read_named_columns(path: Path, anchor: str) -> Dict[str, np.ndarray]:
    """Parse a ``%``-headed MIDAS table into ``{column name: array}``.

    ``anchor`` is a column name that every generation of the file carries; the
    header line is found by looking for it rather than for the ID token, which
    is spelled two different ways.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text().splitlines()
    hidx = None
    for i, line in enumerate(lines):
        if line.startswith("%") and anchor in line.lstrip("%").split():
            hidx = i
            break
    if hidx is None:
        raise GrainsFormatError(
            f"{path}: no '%' header line containing {anchor!r} — not a "
            f"ProcessGrains output?"
        )
    cols = lines[hidx].lstrip("%").split()

    rows: List[List[str]] = []
    for raw in lines[hidx + 1:]:
        if not raw.strip() or raw.lstrip().startswith("%"):
            continue
        toks = _split_row(raw)
        if len(toks) < len(cols):
            continue        # short/truncated row: skip rather than mis-align
        rows.append(toks[:len(cols)])

    data: Dict[str, np.ndarray] = {}
    if rows:
        arr = np.array(rows, dtype=object)
        for j, name in enumerate(cols):
            try:
                data[name] = arr[:, j].astype(np.float64)
            except (TypeError, ValueError):
                data[name] = arr[:, j]
    else:
        for name in cols:
            data[name] = np.zeros(0, dtype=np.float64)
    data["__columns__"] = np.array(cols, dtype=object)
    return data


def _require(data: Dict[str, np.ndarray], names: Sequence[str], path: Path):
    missing = [n for n in names if n not in data]
    if missing:
        raise GrainsFormatError(
            f"{path}: required column(s) {missing} absent; header has "
            f"{list(data['__columns__'])!r}"
        )


def read_grains_columns(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read a ``Grains.csv`` of any width into ``{column name: array}``.

    The grain-id column is normalised to ``"GrainID"`` whichever token the
    file uses, so callers never have to branch on the header spelling.
    ``"__columns__"`` holds the header as written.
    """
    path = Path(path)
    data = _read_named_columns(path, anchor="O11")
    id_name = next((n for n in _ID_NAMES if n in data), None)
    if id_name is None:
        raise GrainsFormatError(
            f"{path}: header has neither GrainID nor ID: "
            f"{list(data['__columns__'])!r}"
        )
    data["GrainID"] = data[id_name]
    return data


def read_spot_matrix_columns(
    path: Union[str, Path], *, matched_only: bool = True
) -> Dict[str, np.ndarray]:
    """Read a ``SpotMatrix.csv`` of either width into ``{column name: array}``.

    Parameters
    ----------
    matched_only : default **True**. Drops the ``Matched == 0`` rows — the
        reflections predicted but never found. See the module docstring. A
        12-column legacy file has no ``Matched`` column; every row there is by
        definition an observation, so nothing is dropped.

    Extra keys ``"__n_rows_total__"`` and ``"__n_rows_unmatched__"`` record the
    completeness deficit that the filter removes, so a caller can report it
    instead of silently losing it.
    """
    path = Path(path)
    data = _read_named_columns(path, anchor="SpotID")
    id_name = next((n for n in _ID_NAMES if n in data), None)
    if id_name is None:
        raise GrainsFormatError(
            f"{path}: header has neither GrainID nor ID: "
            f"{list(data['__columns__'])!r}"
        )
    data["GrainID"] = data[id_name]

    cols = data.pop("__columns__")
    n_total = int(len(data[id_name]))
    matched = data.get("Matched")
    n_unmatched = int(np.sum(matched < 0.5)) if matched is not None else 0
    if matched_only and matched is not None and n_unmatched:
        keep = matched > 0.5
        data = {k: v[keep] for k, v in data.items()}
    data["__columns__"] = cols
    data["__n_rows_total__"] = n_total
    data["__n_rows_unmatched__"] = n_unmatched
    return data
