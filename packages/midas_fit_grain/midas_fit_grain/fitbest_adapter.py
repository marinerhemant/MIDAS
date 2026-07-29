"""Adapt the c-omp refiner's ``FitBest_<vox>_<sp>.csv`` output into the
``Result_OrientPos_voxel_<vox>.csv`` form that ``midas_pf_odf`` and
``midas_pipeline.stages.consolidation_pf`` consume.

Why an adapter and not a rename
-------------------------------
``FitUnified.c`` writes a **multi-block** CSV per (voxel, seed)::

    line 1 : header
    line 2 : the 39-col refined result  (OM 1-9, pos 11-13, lattice 15-20,
             strain 27-35, Euler 36-38)   <- the only line pf-odf reads
    line 3 : header again
    line 4+: one row per matched spot

``midas_pf_odf.io._read_voxel_result`` does ``np.loadtxt(skiprows=1)`` and
requires a single 1-D row, so it chokes on the trailing header + per-spot rows.
:func:`fitbest_to_result_orientpos` extracts just the refined-result line
(and, when a voxel has several ``FitBest_<vox>_<sp>`` solutions, keeps the
highest-completeness one), writing a clean 2-line
``Result_OrientPos_voxel_<vox>.csv``.

The extracted columns (OM in 1-9, lattice in 15-20) are exactly what pf-odf's
``_read_voxel_result`` reads (``row[1:10]`` / ``row[15:21]``), so this is the
permanent bridge from the fast C refiner to the peak-shape strain code.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_HEADER = (
    "SpotID O11 O12 O13 O21 O22 O23 O31 O32 O33 "
    "SpotID x y z "
    "SpotID a b c alpha beta gamma "
    "SpotID PosErr OmeErr InternalAngle Radius Completeness "
    "E11 E12 E13 E21 E22 E23 E31 E32 E33 "
    "Eul1 Eul2 Eul3\n"
)
_FN = re.compile(r"FitBest_(\d+)_(\d+)\.csv$")
_COMPLETENESS_COL = 26          # 0-indexed within the refined-result row


def _first_data_row(fitbest_path: str) -> list[str] | None:
    """Second physical line (the refined-result row) as whitespace tokens,
    or ``None`` if the file has no data line."""
    with open(fitbest_path) as fh:
        fh.readline()               # header
        data = fh.readline()
    if not data:
        return None
    toks = data.split()
    return toks if toks else None


def fitbest_to_result_orientpos(
    fitbest_dir: str | Path, results_dir: str | Path,
) -> int:
    """Convert every ``FitBest_*.csv`` in *fitbest_dir* into
    ``Result_OrientPos_voxel_<vox>.csv`` in *results_dir*.

    When a voxel has several ``FitBest_<vox>_<sp>`` solutions the
    highest-completeness one (col 26) is kept.

    Returns the number of per-voxel result files written.
    """
    fitbest_dir = Path(fitbest_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    best: dict[int, tuple[float, list[str]]] = {}   # voxNr -> (completeness, tokens)
    for ent in os.scandir(fitbest_dir):
        m = _FN.search(ent.name)
        if not m:
            continue
        vox = int(m.group(1))
        toks = _first_data_row(ent.path)
        if toks is None or len(toks) <= _COMPLETENESS_COL:
            continue
        try:
            comp = float(toks[_COMPLETENESS_COL])
        except ValueError:
            continue
        prev = best.get(vox)
        if prev is None or comp > prev[0]:
            best[vox] = (comp, toks)

    for vox, (_comp, toks) in best.items():
        out = results_dir / f"Result_OrientPos_voxel_{vox}.csv"
        with open(out, "w") as fh:
            fh.write(_HEADER)
            fh.write(" ".join(toks) + "\n")
    return len(best)
